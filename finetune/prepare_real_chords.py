"""Build a labelled train/validation set from REAL choir recordings of blocked
chords, using the score as the annotation.

Why this is possible at all
---------------------------
Fine-tuning has so far used soundfont renders because frame-synchronised labels
for real choir audio were assumed unavailable. For *blocked chords at the start
of a song* they are available:

  * homophonic writing means all voices enter and release together, so there is
    no per-part timeline to reconcile -- the thing that makes general
    score-to-audio alignment hard;
  * the pitches are known exactly from the MIDI and are constant for the
    duration of the chord;
  * trimming the attack and release leaves a steady state where the labels hold
    frame by frame.

The natural imbalance of a real ensemble then becomes a FEATURE of the data. A
voice sitting low in the mix carries the same label as any other, so the loss
penalises the model exactly where it under-reads. None of the synthetic
machinery -- victim levels, `undilute_quiet_recall`, balance guards -- is
needed, because that exists only to track imbalance which was manufactured.

Pitch is measured, not assumed
------------------------------
The target blur is sigma = 1 bin = 20 cents, so a choir 20-30 cents flat puts
the ridge a full sigma or more off the energy it marks, systematically. The
offset is therefore measured per take by `tuning.align_and_tune()` and applied
to the LABEL frequencies (never to the audio). On synthetic renders whose true
offset is 0 that estimator returns -1.5 cents and recovers injected offsets of
+-40 cents with constant error; on a real recording it returned -21.5 cents,
against -16.0 from `librosa.estimate_tuning`.

Takes whose alignment or tuning peak is too weak are SKIPPED rather than
corrected by a number that means nothing -- `--min_align_r` and
`--min_tuning_prominence`. Note that r ~ 0.5 is what a good fit looks like for
this sparse-mask metric: a synthetic render made from the very same MIDI scores
0.53, so do not expect 0.9.

Edges
-----
The HCQT is computed over the WHOLE file in one pass. Two reasons: no boundary
artifacts, and -- less obviously -- `compute_pump_features_segmented`
re-references the dB scale to the maximum of each 10 s segment, silently
applying a per-segment AGC. The whole-file pass uses one reference, which is
what the model was trained on.

Output
------
One npz per fixed-length window, `<take>_c<chord>_w<window>.npz`, with keys
`mag` (F, win, H), `dph` (F, win, H), `tgt` (F, win) -- the layout
`finetune.prepare()` writes, so the existing training loop consumes it
unchanged. A manifest CSV records every decision.

The split is BY TAKE, never by chord within a take: holding out chords from a
performance the model has already heard measures memorisation, not
generalisation across performances.

Example
-------
    python finetune/prepare_real_chords.py \\
        --audio "rec/*.wav" --midi "scores/*.mid" \\
        --out ./finetune/data/real --measures 1-2
"""
from __future__ import print_function

import os
import sys
import csv
import glob
import argparse

import numpy as np

import matplotlib
matplotlib.use('Agg')                 # headless: these are written, never shown
import matplotlib.pyplot as plt
import librosa.display

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils
from tuning import align_and_tune, chord_spans, librosa_tuning_cents, SR, HOP
from compare_voice_salience import parse_midi, midi_tempo_bpm, track_name, hz, note_name


# --------------------------------------------------------------------------
# Score -> chords
# --------------------------------------------------------------------------
def read_blocked_chords(midi_path, m_from, m_to, bpm=None, beats_per_measure=4,
                        onset_tol=0.06, merge_repeats=False):
    """Chords as the interval in which ALL of the chord's notes sound TOGETHER:
    t0 = max(onset), t1 = min(offset).

    Deliberately NOT `compare_voice_salience.read_score()`, which runs each
    chord to the next chord's ONSET. That is fine for scoring salience against
    a score, but here it would swallow the rests: in Parijs the notes of chord
    1 stop at 0.667 s while the next chord starts at 1.000 s, so read_score's
    span labels 333 ms of silence as five sounding voices, and the 100 ms trim
    then shaves the rest instead of the sustain.

    `merge_repeats` (off by default) joins consecutive chords with the same
    pitches that are contiguous -- a re-articulated chord. The pitch content is
    unchanged across the re-attack, so the labels stay valid, but the attack
    transient then sits mid-span where trimming cannot remove it. Enable only
    when the separate spans are too short to be usable.
    """
    div, tracks = parse_midi(midi_path)
    if bpm is None:
        bpm = midi_tempo_bpm(tracks)
    spb, tpm = 60.0 / bpm, div * beats_per_measure
    t_start, t_end = (m_from - 1) * tpm, m_to * tpm

    notes = []
    for ti, ev in enumerate(tracks):
        if not any(k == 'midi' and (s & 0xF0) == 0x90 and d[1] > 0
                   for _t, k, s, d in ev):
            continue                                  # conductor / empty track
        name = track_name(ev, 'trk%d' % ti)
        pending = {}
        for tick, kind, status, data in ev:
            if kind != 'midi':
                continue
            cmd = status & 0xF0
            if cmd == 0x90 and data[1] > 0:
                pending.setdefault(data[0], []).append(tick)
            elif cmd == 0x80 or (cmd == 0x90 and data[1] == 0):
                if pending.get(data[0]):
                    on = pending[data[0]].pop(0)
                    if on < t_end and tick > t_start:
                        notes.append((name, data[0],
                                      max(on, t_start) / div * spb,
                                      min(tick, t_end) / div * spb))
    if not notes:
        return []

    groups = []
    for n in sorted(notes, key=lambda x: x[2]):
        if groups and abs(n[2] - groups[-1][0][2]) <= onset_tol:
            groups[-1].append(n)
        else:
            groups.append([n])

    chords = []
    for g in groups:
        t0, t1 = max(n[2] for n in g), min(n[3] for n in g)
        if t1 > t0:
            chords.append([t0, t1,
                           sorted({(n[0], n[1]) for n in g}, key=lambda x: -x[1])])
    if merge_repeats:
        merged = []
        for c in chords:
            if (merged and {p for _v, p in merged[-1][2]} == {p for _v, p in c[2]}
                    and abs(c[0] - merged[-1][1]) <= onset_tol):
                merged[-1][1] = c[1]
            else:
                merged.append(c)
        chords = merged
    return [(a, b, v) for a, b, v in chords]


# --------------------------------------------------------------------------
# Verification plots
#
# Two things can go wrong silently and neither shows up in the manifest: the
# score can be aligned to the wrong part of the audio, and the tuning
# correction can put the labels somewhere other than the sung pitch. Both are
# obvious the moment you look at the CQT with the labels drawn on it, so every
# take gets a picture.
# --------------------------------------------------------------------------
def _specshow(ax, energy, cmap='inferno'):
    bpo, _n_oct, _h, sr, fmin, hop, _os = utils.get_hcqt_params()
    return librosa.display.specshow(
        energy, x_axis='time', y_axis='cqt_hz', sr=sr, hop_length=hop,
        fmin=fmin, bins_per_octave=bpo, cmap=cmap, ax=ax)


def _ylim_for(freqs, n_harm):
    lo = min(freqs) / 1.6
    hi = max(freqs) * (n_harm + 0.6)
    return lo, hi


def _ylim_fundamentals(freqs, margin=1.12):
    """Tight bounds on the fundamentals alone (~2 semitones of margin).

    The per-chord panels exist to check that a label sits on its ridge, and the
    harmonics above only cost vertical resolution: over a fixed panel height,
    dropping two octaves of harmonics roughly triples the pixels per octave,
    which is what makes a 20-cent tuning shift visible at all."""
    return min(freqs) / margin, max(freqs) * margin


def plot_take_overview(energy, chords, spans, fit, tune, cents, title, out_png,
                       n_harm=2, state=None):
    """Whole take: chord extent as fitted, the trimmed part actually used, and
    the corrected label frequencies drawn over the energy.

    With `state` the three label regions are shaded along the bottom, which is
    the thing to check: green where the chord's pitches are asserted, grey where
    nothing is (attack, release, reverb tail), blue where silence is asserted.
    """
    all_f = [f for _t0, _t1, fr in chords for f in fr]
    fig, ax = plt.subplots(figsize=(16, 7))
    _specshow(ax, energy)

    if state is not None:
        lo = min(all_f) / 1.55
        band = lo * 1.06
        t = np.arange(len(state)) * HOP / SR
        for val, colour in ((1, 'lime'), (2, 'deepskyblue'), (0, 'grey')):
            ax.fill_between(t, lo, band, where=(state == val), step='mid',
                            color=colour, alpha=0.85, linewidth=0)

    for ci, ((t0, t1, _fr), (i0, i1, cfr)) in enumerate(zip(chords, spans)):
        a0, a1 = fit['scale'] * t0 + fit['offset'], fit['scale'] * t1 + fit['offset']
        s0, s1 = i0 * HOP / SR, i1 * HOP / SR
        # full matched chord
        ax.axvline(a0, color='cyan', ls='--', lw=1.0, alpha=0.9)
        ax.axvline(a1, color='cyan', ls=':', lw=1.0, alpha=0.7)
        # trimmed (used) region
        ax.axvspan(s0, s1, color='lime', alpha=0.12)
        if i1 > i0:
            ax.hlines(cfr, s0, s1, color='lime', lw=1.6, alpha=0.95)
        ax.text(a0, max(all_f) * (n_harm + 0.3), ' c%d' % (ci + 1),
                color='cyan', fontsize=8, va='top')

    ax.set_ylim(*_ylim_for(all_f, n_harm))
    ax.set_title(title, fontsize=10)
    handles = [plt.Line2D([], [], color='cyan', ls='--', label='matched chord start/end'),
               plt.Line2D([], [], color='lime', lw=6, alpha=0.3, label='trimmed (used)'),
               plt.Line2D([], [], color='lime', lw=2, label='label freq (tuning-corrected)')]
    if state is not None:
        handles += [plt.Line2D([], [], color='lime', lw=6, label='supervised: chord'),
                    plt.Line2D([], [], color='deepskyblue', lw=6, label='supervised: silence'),
                    plt.Line2D([], [], color='grey', lw=6, label='masked (no gradient)')]
    ax.legend(handles=handles, loc='upper right', fontsize=8, framealpha=0.7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def plot_tuning_evidence(energy, fgrid, chords, spans, tune, cents, title, out_png,
                         span_cents=100.0, step=2.0):
    """The tuning estimate's own evidence, at a scale where it is visible.

    A 20-cent shift is about three pixels on a spectrogram spanning two
    octaves, so the overlay plots cannot actually show whether the correction
    is right. These two panels can:

      left  -- the shift-correlation curve the estimator maximised. A sharp,
               isolated peak means the offset is measured; a flat curve means
               it is unknown, and `prominence` is the height of that peak above
               the curve's median.
      right -- CQT energy as a function of cents from the nominal pitch, one
               thin line per voice and the mean in bold. If the choir is 20
               cents flat, the mean peaks at -20, and the estimate should sit
               on that peak.

    The two are independent views: the left uses the whole score as a rigid
    mask, the right just reads energy around each labelled pitch.
    """
    fig, (axc, axe) = plt.subplots(1, 2, figsize=(13, 4.6))

    if tune.get('grid') is not None:
        axc.plot(tune['grid'], tune['curve'], color='steelblue', lw=1.5)
        axc.axvline(tune['cents'], color='crimson', lw=1.5,
                    label='estimate %+.1f c' % tune['cents'])
        axc.axhline(np.nanmedian(tune['curve']), color='grey', ls=':', lw=1,
                    label='median (prominence base)')
        axc.axvline(0.0, color='black', ls='--', lw=1, alpha=0.5, label='nominal A440')
        axc.set_xlabel('label shift (cents)')
        axc.set_ylabel('correlation with CQT energy')
        axc.set_title('shift-correlation, prominence = %.3f' % tune['prominence'],
                      fontsize=9)
        axc.legend(fontsize=8)
        axc.grid(alpha=0.3)

    grid = np.arange(-span_cents, span_cents + 1e-9, step)
    logf = np.log(fgrid)
    curves = []
    for (t0, t1, nominal), (i0, i1, _c) in zip(chords, spans):
        if i1 <= i0:
            continue
        prof = energy[:, i0:i1].mean(axis=1)
        for f in nominal:
            vals = np.interp(np.log(f * 2.0 ** (grid / 1200.0)), logf, prof)
            curves.append(vals)
            axe.plot(grid, vals, color='grey', lw=0.6, alpha=0.45)
    if curves:
        mean = np.mean(curves, axis=0)
        axe.plot(grid, mean, color='seagreen', lw=2.4, label='mean over voices')
        axe.axvline(grid[int(np.argmax(mean))], color='seagreen', ls='--', lw=1.2,
                    label='energy peak %+.0f c' % grid[int(np.argmax(mean))])
    axe.axvline(cents, color='crimson', lw=1.5, label='applied %+.1f c' % cents)
    axe.axvline(0.0, color='black', ls='--', lw=1, alpha=0.5, label='nominal A440')
    axe.set_xlabel('cents from nominal pitch')
    axe.set_ylabel('CQT energy (normalised dB)')
    axe.set_title('energy around each labelled pitch (%d voice-chords)' % len(curves),
                  fontsize=9)
    axe.legend(fontsize=8)
    axe.grid(alpha=0.3)

    fig.suptitle(title, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def plot_chord_details(energy, chords, spans, fit, cents, title, out_png, n_harm=2):
    """One zoomed panel per chord. Corrected labels in green, uncorrected in
    red: if the green lines sit on ridges and the red ones do not, the tuning
    correction is doing its job. If neither does, the alignment is wrong."""
    n = len(chords)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 5.0), squeeze=False)
    shift = 2.0 ** (cents / 1200.0)
    for ci, ax in enumerate(axes[0]):
        t0, t1, nominal = chords[ci]
        i0, i1, cfr = spans[ci]
        a0, a1 = fit['scale'] * t0 + fit['offset'], fit['scale'] * t1 + fit['offset']
        s0, s1 = i0 * HOP / SR, i1 * HOP / SR
        _specshow(ax, energy)
        ax.axvspan(s0, s1, color='lime', alpha=0.12)
        ax.axvline(a0, color='cyan', ls='--', lw=1.0)
        ax.axvline(a1, color='cyan', ls=':', lw=1.0)
        if abs(cents) > 1e-6:
            ax.hlines(nominal, s0, s1, color='red', lw=1.2, ls=':', alpha=0.9)
        ax.hlines(cfr, s0, s1, color='lime', lw=1.6, alpha=0.95)
        pad = max(0.25, 0.35 * (a1 - a0))
        ax.set_xlim(max(0.0, a0 - pad), a1 + pad)
        ax.set_ylim(*_ylim_fundamentals(nominal))
        ax.set_title('chord %d  %s' % (ci + 1, ' '.join(
            note_name(int(round(69 + 12 * np.log2(f / 440.0)))) for f in nominal)),
            fontsize=8)
        if ci:
            ax.set_ylabel('')
    fig.suptitle(title + '   (green = corrected label, red dotted = uncorrected)',
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def pair_audio_with_midi(audio_paths, midi_paths):
    """Match each take to the score whose stem is the longest prefix of the
    take's stem, so `Parijs_take03.wav` pairs with `Parijs.mid` and repeated
    takes of one song need only a single score."""
    midis = {os.path.splitext(os.path.basename(m))[0]: m for m in midi_paths}
    pairs, unmatched = [], []
    for a in audio_paths:
        stem = os.path.splitext(os.path.basename(a))[0]
        cands = [k for k in midis if stem.lower().startswith(k.lower())]
        if cands:
            best = max(cands, key=len)
            pairs.append((a, midis[best], best, stem))
        else:
            unmatched.append(a)
    return pairs, unmatched


def main(args):
    audio = sorted(sum([glob.glob(os.path.expanduser(p)) for p in args.audio], []))
    midis = sorted(sum([glob.glob(os.path.expanduser(p)) for p in args.midi], []))
    if not audio:
        raise SystemExit("--audio matched no files")
    if not midis:
        raise SystemExit("--midi matched no files")

    pairs, unmatched = pair_audio_with_midi(audio, midis)
    for a in unmatched:
        print("  ! no score for %s -- skipped" % os.path.basename(a))
    if not pairs:
        raise SystemExit("No audio/score pairs. Name takes '<song>_takeNN.wav' "
                         "beside '<song>.mid'.")

    m_from, m_to = ((int(x) for x in args.measures.split('-'))
                    if '-' in args.measures else
                    (int(args.measures), int(args.measures)))

    # Split BY TAKE. Deterministic given --seed so a rerun reproduces the split.
    rng = np.random.RandomState(args.seed)
    if args.valid_takes:
        want = {t.lower() for t in args.valid_takes}
        in_valid = [t.lower() in want for _a, _m, _s, t in pairs]
        missing = want - {t.lower() for _a, _m, _s, t in pairs}
        for m in sorted(missing):
            print("  ! --valid_takes '%s' matches no take" % m)
    elif args.valid_songs:
        vs = set(args.valid_songs)
        in_valid = [song in vs for _a, _m, song, _t in pairs]
    elif args.valid_take_frac <= 0.0:
        # An explicit 0 means "training material only" -- e.g. building the set
        # up take by take, or holding out a whole song separately. The floor
        # below must not override it.
        in_valid = [False] * len(pairs)
    else:
        # At least one take, so a small --valid_take_frac cannot silently round
        # down to an empty validation set.
        n_valid = max(1, int(round(args.valid_take_frac * len(pairs))))
        pick = set(rng.permutation(len(pairs))[:n_valid].tolist())
        in_valid = [i in pick for i in range(len(pairs))]

    fgrid = utils.get_freq_grid()
    pump = utils.create_pump_object()
    out_dirs = {s: os.path.join(args.out, s) for s in ('train', 'valid')}
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)
    plot_dir = args.plot_dir or os.path.join(args.out, 'plots')
    if not args.no_plots:
        os.makedirs(plot_dir, exist_ok=True)

    trim = args.trim_ms / 1000.0
    hop = args.win_hop or args.win
    manifest, n_written = [], {'train': 0, 'valid': 0}
    processed_takes = set()

    print("\n%d take(s), measures %d-%d, trim %.0f ms/end, win %d frames\n"
          % (len(pairs), m_from, m_to, args.trim_ms, args.win))

    for pi, (wav, mid, song, take) in enumerate(pairs):
        split = 'valid' if in_valid[pi] else 'train'
        processed_takes.add(take)
        print("[%s] %-40s score=%s" % (split, os.path.basename(wav)[:40],
                                       os.path.basename(mid)))

        # Drop this take's windows from BOTH splits before rewriting them. A
        # re-run with different settings can produce fewer windows than before,
        # and the leftovers would otherwise stay in the training set carrying
        # labels built under the old settings. Both splits, because a take can
        # move between them.
        stale = sum((glob.glob(os.path.join(d, '%s_w*.npz' % take))
                     for d in out_dirs.values()), [])
        for f in stale:
            os.remove(f)
        if stale:
            print("    removed %d window(s) from a previous run" % len(stale))

        chords_raw = read_blocked_chords(mid, m_from, m_to, args.bpm,
                                         beats_per_measure=args.beats_per_measure,
                                         merge_repeats=args.merge_repeats)
        if not chords_raw:
            print("    ! no chords in measures %d-%d -- take skipped\n" % (m_from, m_to))
            continue
        if args.max_chords:
            chords_raw = chords_raw[:args.max_chords]
        chords = [(t0, t1, [hz(p) for _v, p in vs]) for t0, t1, vs in chords_raw]

        # ---- whole-file HCQT: one dB reference, no segment edges ----
        feats = utils.compute_pump_features(pump, wav)
        mag = np.transpose(feats['dphase/mag'][0], (2, 1, 0)).astype(np.float32)
        dph = np.transpose(feats['dphase/dphase'][0], (2, 1, 0)).astype(np.float32)
        T = mag.shape[2]
        tgrid = utils.get_time_grid(T)

        # ---- alignment + tuning, alternated ----
        if args.scale is not None:
            res = align_and_tune(mag, fgrid, tgrid, chords,
                                 scales=(args.scale, args.scale + 1e-9, 1.0),
                                 offsets=(args.offset, args.offset + 1e-9, 1.0))
        else:
            res = align_and_tune(mag, fgrid, tgrid, chords,
                                 scales=tuple(args.scale_range),
                                 offsets=tuple(args.offset_range))
        if res is None:
            print("    ! alignment/tuning failed -- take skipped\n")
            continue

        # A take that stops part-way through the passage leaves score chords with
        # no audio to correlate against, which drags `r` down and makes a
        # correctly-aligned take look misaligned. Drop the chords that fall off
        # the end and refit on what is actually present.
        dur = T * HOP / SR
        inside = [c for c in chords
                  if res['fit']['scale'] * c[0] + res['fit']['offset'] < dur - 0.05]
        if len(inside) < len(chords):
            print("    %d of %d chords fall beyond the audio (%.2f s) -- refitting "
                  "on the rest" % (len(chords) - len(inside), len(chords), dur))
            chords = inside
            chords_raw = chords_raw[:len(inside)]
            if not chords:
                print("    ! no chord inside the audio -- take skipped\n")
                continue
            res = align_and_tune(mag, fgrid, tgrid, chords,
                                 scales=tuple(args.scale_range),
                                 offsets=tuple(args.offset_range))
            if res is None:
                print("    ! refit failed -- take skipped\n")
                continue

        fit, tune = res['fit'], res['tune']
        cents = tune['cents'] if args.tuning_mode != 'none' else 0.0
        print("    time  : scale=%.3f offset=%+.3f r=%.3f%s"
              % (fit['scale'], fit['offset'], fit['r'],
                 "  !SEARCH EDGE" if fit['at_edge'] else ""))
        print("    tuning: %+.1f cents  prominence=%.3f%s"
              % (tune['cents'], tune['prominence'],
                 "  !SEARCH EDGE" if tune['at_edge'] else ""))
        if args.librosa_check:
            try:
                lt = librosa_tuning_cents(wav)
                print("    tuning cross-check (librosa): %+.1f c  diff %+.1f"
                      % (lt, lt - tune['cents']))
            except Exception as e:
                print("    tuning cross-check failed: %s" % e)

        # ---- sustained spans, with the tuning correction on the LABELS ----
        spans = chord_spans(chords, fit['scale'], fit['offset'], T,
                            trim_s=trim, cents=cents)

        # ---- label the TIMELINE, then window the timeline --------------------
        #
        # Not per chord. A window is allowed to straddle a chord boundary,
        # because the neighbouring chord's frames have known labels too, and
        # because at inference the model sees exactly that -- continuous audio
        # in which a frame near a boundary has the next chord inside its
        # receptive field. Windowing within chords would instead force the
        # model to learn from zero-padded context it never meets in use.
        #
        # Every frame gets one of three states:
        #   SUPERVISED, chord   -- inside a trimmed sustain: that chord's pitches
        #   MASKED              -- attack, release, reverb tail: the score cannot
        #                          distinguish a staggered entry from a decaying
        #                          voice from a quiet one, so nothing is asserted
        #   SUPERVISED, silence -- deep inside a rest, past the reverb margin
        #
        # The masked frames are not wasted: they still feed the receptive field
        # of the supervised frames around them. They simply earn no gradient.
        mask = np.zeros(T, dtype=np.float32)
        state = np.zeros(T, dtype=np.int8)          # 0 mask, 1 chord, 2 silence
        pts_t, pts_f = [], []
        for ci, (i0, i1, freqs) in enumerate(spans):
            if i1 <= i0:
                continue
            mask[i0:i1] = 1.0
            state[i0:i1] = 1
            for i in range(i0, i1):
                for f in freqs:
                    pts_t.append(tgrid[i])
                    pts_f.append(f)

        # Deep rest: between one chord's untrimmed release and the next chord's
        # untrimmed attack, minus a margin for the reverb tail. Without these
        # negatives every supervised frame would contain sounding voices and
        # nothing in the loss would ever say "no voice here".
        rest_margin = int(round(args.rest_margin_ms / 1000.0 * SR / HOP))
        n_rest = 0
        if not args.no_rest_negatives:
            bounds = []
            for (t0, t1, _f) in chords:
                bounds.append((int(np.floor((fit['scale'] * t0 + fit['offset']) * SR / HOP)),
                               int(np.ceil((fit['scale'] * t1 + fit['offset']) * SR / HOP))))
            for k in range(len(bounds) + 1):
                lo = 0 if k == 0 else bounds[k - 1][1] + rest_margin
                hi = T if k == len(bounds) else bounds[k][0] - rest_margin
                lo, hi = max(lo, 0), min(hi, T)
                if hi - lo >= args.min_rest_frames:
                    mask[lo:hi] = 1.0
                    state[lo:hi] = 2
                    n_rest += hi - lo

        target = utils.create_annotation_target(
            fgrid, tgrid, np.array(pts_t), np.array(pts_f)).astype(np.float32) \
            if pts_t else np.zeros((len(fgrid), T), dtype=np.float32)
        target[:, state != 1] = 0.0        # silence frames are genuinely zero

        n_chord = int((state == 1).sum())
        print("    timeline: %d supervised frames (%d chord, %d silence), "
              "%d masked (%.0f%%)"
              % (int(mask.sum()), n_chord, n_rest, T - int(mask.sum()),
                 100.0 * (T - mask.sum()) / max(1, T)))

        # Plot BEFORE the gates, and label the picture with the verdict: a take
        # that was rejected is exactly the one worth looking at, and a picture
        # only of the takes that passed cannot show why the others were not.
        reason = ''
        if fit['r'] < args.min_align_r:
            reason = 'REJECTED: align r %.3f < %.2f' % (fit['r'], args.min_align_r)
        elif (args.tuning_mode != 'none'
              and tune['prominence'] < args.min_tuning_prominence):
            reason = ('REJECTED: tuning prominence %.3f < %.2f'
                      % (tune['prominence'], args.min_tuning_prominence))
        elif n_chord == 0:
            reason = 'REJECTED: no supervised chord frames'

        if not args.no_plots:
            title = ("%s [%s]  scale=%.3f offset=%+.3f r=%.3f | tuning %+.1f c "
                     "(prom %.3f) | trim %.0f ms  %s"
                     % (take, split if not reason else 'skipped', fit['scale'],
                        fit['offset'], fit['r'], tune['cents'], tune['prominence'],
                        args.trim_ms, reason))
            plot_take_overview(res['energy'], chords, spans, fit, tune, cents,
                               title, os.path.join(plot_dir, '%s_overview.png' % take),
                               args.plot_harmonics, state=state)
            plot_chord_details(res['energy'], chords, spans, fit, cents, title,
                               os.path.join(plot_dir, '%s_chords.png' % take),
                               args.plot_harmonics)
            plot_tuning_evidence(res['energy'], fgrid, chords, spans, tune, cents,
                                 title, os.path.join(plot_dir, '%s_tuning.png' % take))
            print("    plots : %s_{overview,chords,tuning}.png" % take)

        if reason:
            print("    ! %s -- take skipped (see plot)\n" % reason)
            continue

        # ---- slide windows across the whole file ----
        min_sup = (args.min_supervised if args.min_supervised is not None
                   else max(1, int(round(0.25 * args.win))))
        fw, kept_chords = 0, set()
        for s in range(0, max(1, T - args.win + 1), hop):
            m = mask[s:s + args.win]
            if m.sum() < min_sup:
                continue
            st = state[s:s + args.win]
            in_win = sorted({ci + 1 for ci, (i0, i1, _f) in enumerate(spans)
                             if i1 > i0 and i0 < s + args.win and i1 > s})
            kept_chords.update(in_win)
            np.savez_compressed(
                os.path.join(out_dirs[split], '%s_w%05d.npz' % (take, fw)),
                mag=np.transpose(mag[:, :, s:s + args.win], (1, 2, 0)),
                dph=np.transpose(dph[:, :, s:s + args.win], (1, 2, 0)),
                tgt=target[:, s:s + args.win],
                mask=m.copy(),
                song=song, take=take, cents=cents, start=s,
                n_supervised=int(m.sum()), n_chord=int((st == 1).sum()),
                n_silence=int((st == 2).sum()),
                chords=np.array(in_win, dtype=np.int32))
            fw += 1
        n_written[split] += fw
        manifest.append(dict(
            split=split, song=song, take=take, frames=T, windows=fw,
            chords_total=len(chords), chords_covered=len(kept_chords),
            sup_chord=n_chord, sup_silence=n_rest,
            masked=T - int(mask.sum()),
            cents=round(cents, 1), tuning_prominence=round(tune['prominence'], 3),
            scale=fit['scale'], offset=round(fit['offset'], 3),
            align_r=round(fit['r'], 3)))
        print("    %d windows (>= %d supervised frames each), covering %d/%d chords\n"
              % (fw, min_sup, len(kept_chords), len(chords)))

    if not manifest:
        raise SystemExit("Nothing written. Loosen --min_align_r / "
                         "--min_tuning_prominence, or check --measures.")

    # The manifest describes the OUTPUT DIRECTORY, not this invocation. Takes are
    # usually added a few at a time, so rows for takes this run did not touch are
    # carried over; rows for takes it did touch are replaced, which keeps a
    # re-run idempotent instead of duplicating them.
    mpath = os.path.join(args.out, 'manifest.csv')
    fields = list(manifest[0].keys())
    carried = []
    if os.path.exists(mpath):
        try:
            with open(mpath) as fh:
                old = list(csv.DictReader(fh))
        except (IOError, csv.Error):
            old = []
        if old and set(old[0].keys()) != set(fields):
            print("! existing manifest has different columns (older script "
                  "version) -- it is being replaced, not merged")
        else:
            carried = [r for r in old if r.get('take') not in processed_takes]
    with open(mpath, 'w') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(carried)
        w.writerows(manifest)
    if carried:
        print("manifest: %d row(s) carried over from previous runs" % len(carried))
    # DONE markers so finetune.prepare() reuses these instead of re-slicing
    for d in out_dirs.values():
        open(os.path.join(d, 'DONE'), 'w').close()

    # Report the DIRECTORY, not just this run -- the manifest now spans both, and
    # a summary describing only the current invocation would contradict it. The
    # window counts are read off disk, so they cannot drift from reality.
    print("=" * 66)
    print("this run: %d take(s), %d window(s)"
          % (len(processed_takes), sum(n_written.values())))
    all_rows = carried + [{k: str(v) for k, v in m.items()} for m in manifest]
    on_disk = {s: len(glob.glob(os.path.join(out_dirs[s], '*_w*.npz')))
               for s in ('train', 'valid')}
    takes = {s: len({r['take'] for r in all_rows if r['split'] == s})
             for s in ('train', 'valid')}
    secs = sum(on_disk.values()) * args.win * HOP / SR
    print("in %s:" % args.out)
    print("  train %4d windows (%d takes)   valid %4d windows (%d takes)"
          % (on_disk['train'], takes['train'], on_disk['valid'], takes['valid']))
    print("  %.1f s of labelled sustained real audio" % secs)
    cents_all = [float(r['cents']) for r in all_rows]
    print("  tuning applied: %+.1f .. %+.1f cents" % (min(cents_all), max(cents_all)))
    print("manifest: %s (%d rows)" % (mpath, len(all_rows)))
    if takes['valid'] == 0 and args.valid_take_frac > 0 and not args.valid_takes:
        print("! no validation takes -- raise --valid_take_frac")
    elif takes['valid'] == 0:
        print("(training material only, as requested -- no validation takes)")


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--audio', nargs='+', required=True,
                   help="wav takes, globs allowed (e.g. 'rec/*.wav')")
    p.add_argument('--midi', nargs='+', required=True,
                   help="scores, globs allowed. A take pairs with the score whose "
                        "stem is the longest prefix of the take's stem")
    p.add_argument('--out', required=True, help='output directory')
    p.add_argument('--measures', default='1-2',
                   help='measure range of blocked chords to use (default 1-2)')
    p.add_argument('--beats_per_measure', type=int, default=4)
    p.add_argument('--bpm', type=float, default=None,
                   help="score tempo; default is the MIDI's own tempo event. The "
                        "performance tempo is fitted separately, so this only has "
                        "to be roughly right.")
    p.add_argument('--max_chords', type=int, default=None)
    p.add_argument('--merge_repeats', action='store_true',
                   help='join re-articulated chords (same pitches, immediately '
                        'repeated) into one span. OFF by default: the re-attack '
                        'is a transient the trim is meant to exclude, and merging '
                        'hides it in the middle of a window where no trim can '
                        'reach it. Turn on only to rescue spans too short to use.')

    p.add_argument('--trim_ms', type=float, default=25.0,
                   help='ms removed from each end of every chord. Attack and '
                        'release are where alignment error and ensemble '
                        'raggedness both concentrate, and a masked voice is hard '
                        'to hear throughout its note rather than only at onset, '
                        'so little is lost (default 50)')
    p.add_argument('--win', type=int, default=50, help='window frames (matches finetune.py)')
    p.add_argument('--win_hop', type=int, default=None,
                   help='window stride (default --win, no overlap)')
    p.add_argument('--min_supervised', type=int, default=None,
                   help='a window is written only if at least this many of its '
                        'frames are supervised (default 25%% of --win). Windows '
                        'may straddle chord boundaries -- that is the point, it '
                        'is what the model sees at inference -- so this only '
                        'discards windows that are almost entirely mask.')
    p.add_argument('--rest_margin_ms', type=float, default=250.0,
                   help='how long after a chord releases before silence is '
                        'trusted as silence (default 250). Covers the reverb '
                        'tail: labelling a ringing chord as silent would teach '
                        'the model to suppress it.')
    p.add_argument('--min_rest_frames', type=int, default=6,
                   help='ignore rests shorter than this many frames once the '
                        'reverb margin is removed (default 6)')
    p.add_argument('--no_rest_negatives', action='store_true',
                   help='do not supervise the rests. NOT recommended: without '
                        'them every supervised frame contains sounding voices, '
                        'so nothing in the loss ever says "no voice here" -- the '
                        'same missing counterweight that let --pos_weight 4 '
                        'inflate salience by 55%%.')

    p.add_argument('--tuning_mode', choices=['per_take', 'none'], default='per_take',
                   help="'per_take' (default) measures the choir's offset and "
                        "shifts the LABEL frequencies onto it. 'none' trusts A440 "
                        "-- only safe if you have vr=%erified the choir is within "
                        "~10 cents, since sigma is 20 cents.")
    p.add_argument('--min_tuning_prominence', type=float, default=0.05,
                   help='skip a take whose tuning correlation peak is flatter than '
                        'this. A flat curve means the offset is UNKNOWN, which is '
                        'not the same as zero (default 0.05; a real recording '
                        'measured 0.10, synthetic renders 0.22)')
    p.add_argument('--librosa_check', action='store_true',
                   help='also report librosa.estimate_tuning, an independent '
                        'score-free estimate with different failure modes')

    p.add_argument('--scale', type=float, default=None,
                   help='fix the tempo warp instead of fitting it')
    p.add_argument('--offset', type=float, default=0.0, help='used with --scale')
    p.add_argument('--scale_range', nargs=3, type=float,
                   default=(0.80, 1.2501, 0.005), metavar=('LO', 'HI', 'STEP'))
    p.add_argument('--offset_range', nargs=3, type=float,
                   default=(-1.0, 2.001, 0.01), metavar=('LO', 'HI', 'STEP'))
    p.add_argument('--min_align_r', type=float, default=0.30,
                   help='skip a take whose score/energy correlation is below this. '
                        'Note r ~ 0.5 is a GOOD fit for this sparse-mask metric -- '
                        'a synthetic render of the same MIDI scores 0.53 '
                        '(default 0.30)')

    p.add_argument('--plot_dir', default=None,
                   help='where verification plots go (default <out>/plots). Two '
                        'per take: an overview with every chord marked, and a '
                        'zoomed panel per chord. Plots are written for SKIPPED '
                        'takes too, annotated with why -- those are the ones '
                        'worth looking at.')
    p.add_argument('--no_plots', action='store_true', help='skip the plots')
    p.add_argument('--plot_harmonics', type=int, default=2,
                   help='how many harmonics above the highest label to show, '
                        'setting the frequency range of the plots (default 2)')

    p.add_argument('--valid_take_frac', type=float, default=0.25,
                   help='fraction of TAKES held out (default 0.25). Splitting by '
                        'take rather than by chord is deliberate: chords from a '
                        'performance already seen measure memorisation.')
    p.add_argument('--valid_songs', nargs='+', default=None,
                   help='hold out these songs entirely instead of sampling takes')
    p.add_argument('--valid_takes', nargs='+', default=None, metavar='TAKE',
                   help='hold out these takes by name (the wav stem, e.g. '
                        'late_take03). Takes precedence over --valid_songs and '
                        '--valid_take_frac. Use this when every take is the same '
                        'song, where --valid_songs cannot split anything.')
    p.add_argument('--seed', type=int, default=0)
    main(p.parse_args())
