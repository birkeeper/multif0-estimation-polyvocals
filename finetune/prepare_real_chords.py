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
import fnmatch
import argparse

import numpy as np

import matplotlib
matplotlib.use('Agg')                 # headless: these are written, never shown
import matplotlib.pyplot as plt
import librosa.display

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils
from tuning import (align_and_tune, alignment_quality, chord_spans,
                    estimate_tuning_cents, librosa_tuning_cents, SR, HOP)
from compare_voice_salience import parse_midi, midi_tempo_bpm, track_name, hz, note_name


# --------------------------------------------------------------------------
# Score -> chords
# --------------------------------------------------------------------------
GM_DRUM_CHANNEL = 9        # MIDI channel 10, 1-based


def broadband_db(wav_path, n_frames):
    """Broadband level per frame, in dB, on the CQT hop grid.

    Taken from the WAVEFORM, not by averaging CQT bins. A "20 dB drop" is only
    meaningful on linear power, and the HCQT here is already log-scaled and
    normalised per file, so averaging its bins would not yield a level at all.
    """
    import librosa
    y, _sr = librosa.load(wav_path, sr=int(SR))
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=int(HOP),
                              center=True)[0]
    db = 20.0 * np.log10(np.maximum(rms, 1e-10))
    if len(db) < n_frames:
        db = np.pad(db, (0, n_frames - len(db)), mode='edge')
    return db[:n_frames]


def read_blocked_chords(midi_path, m_from, m_to, bpm=None, beats_per_measure=4,
                        onset_tol=0.06, merge_repeats=False):
    """Chords as the interval in which ALL of the chord's notes sound TOGETHER:
    t0 = max(onset), t1 = min(offset).

    Percussion channels are dropped. On channel 10 (index 9) the GM convention
    is that the note number selects a drum, not a pitch, so a kick drum would
    enter the chord as a low note and a hi-hat as a high one -- inventing voices
    that were never sung, and dragging every chord's span to the intersection of
    the real voices with the drum pattern.

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

    notes, skipped = [], []
    for ti, ev in enumerate(tracks):
        pitched = [(_t, k, s, d) for _t, k, s, d in ev
                   if k == 'midi' and (s & 0x0F) != GM_DRUM_CHANNEL]
        if not any((s & 0xF0) == 0x90 and d[1] > 0 for _t, _k, s, d in pitched):
            # No pitched notes. Report it only if the track had drum notes, so a
            # dropped percussion part is visible rather than silently missing.
            if any(k == 'midi' and (s & 0xF0) == 0x90 and d[1] > 0
                   for _t, k, s, d in ev):
                skipped.append(track_name(ev, 'trk%d' % ti))
            continue                                  # conductor / empty / drums
        name = track_name(ev, 'trk%d' % ti)
        pending = {}
        for tick, kind, status, data in pitched:
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
    if skipped:
        print("    percussion dropped: %s" % ', '.join(skipped))
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


def plot_take_overview(energy, chords, spans, fit, cents, title, out_png,
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
                         span_cents=100.0, step=2.0, per_cents=None):
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
        # Whole-take curve, shown for context only: the offsets actually applied
        # are per chord and appear on the right-hand panel.
        axc.axvline(tune['cents'], color='crimson', lw=1.5,
                    label='whole-take %+.1f c (context)' % tune['cents'])
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
    if per_cents:
        for k, c in enumerate(per_cents):
            axe.axvline(c, color='crimson', lw=1.2, alpha=0.85,
                        label='applied, per chord' if k == 0 else None)
        axe.axvline(float(np.median(per_cents)), color='crimson', ls='--', lw=2.0,
                    label='median %+.1f c' % float(np.median(per_cents)))
    else:
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


def plot_chord_details(energy, chords, spans, fit, cents, title, out_png, n_harm=2,
                       per_cents=None):
    """One zoomed panel per chord. Corrected labels in green, uncorrected in
    red: if the green lines sit on ridges and the red ones do not, the tuning
    correction is doing its job. If neither does, the alignment is wrong."""
    n = len(chords)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 5.0), squeeze=False)
    for ci, ax in enumerate(axes[0]):
        t0, t1, nominal = chords[ci]
        i0, i1, cfr = spans[ci]
        a0, a1 = fit['scale'] * t0 + fit['offset'], fit['scale'] * t1 + fit['offset']
        s0, s1 = i0 * HOP / SR, i1 * HOP / SR
        _specshow(ax, energy)
        ax.axvspan(s0, s1, color='lime', alpha=0.12)
        ax.axvline(a0, color='cyan', ls='--', lw=1.0)
        ax.axvline(a1, color='cyan', ls=':', lw=1.0)
        c_ci = per_cents[ci] if per_cents is not None else cents
        if abs(c_ci) > 1e-6:
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


AUDIO_EXT = ('.wav', '.flac', '.ogg')


def expand_midi(patterns):
    """Globs, plus directories (which contribute every .mid inside them)."""
    out = []
    for pat in patterns:
        pat = os.path.expanduser(pat)
        for hit in (sorted(glob.glob(pat)) or []):
            if os.path.isdir(hit):
                out.extend(sorted(glob.glob(os.path.join(hit, '*.mid'))))
            else:
                out.append(hit)
    return sorted(set(out))


def discover_takes(midis, audio_dir=None, exclude=()):
    """Every audio file whose name begins with some score's name.

    Takes of one song are conventionally `<song>_takeNN.wav` beside `<song>.mid`,
    which is the same prefix rule `pair_audio_with_midi()` uses to decide which
    score a take belongs to -- so discovery and pairing cannot disagree. Matching
    is case-insensitive because `Parijs.mid` and `parijs_take01.wav` are the same
    song, and the directory is listed rather than globbed because glob is
    case-sensitive on Linux.
    """
    stems = [os.path.splitext(os.path.basename(m))[0].lower() for m in midis]
    dirs = ([os.path.expanduser(audio_dir)] if audio_dir
            else sorted({os.path.dirname(m) or '.' for m in midis}))
    found = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            base, ext = os.path.splitext(f)
            if ext.lower() not in AUDIO_EXT:
                continue
            if not any(base.lower().startswith(s) for s in stems):
                continue
            if any(fnmatch.fnmatch(f.lower(), pat.lower()) for pat in exclude):
                print("  - excluded %s" % f)
                continue
            found.append(os.path.join(d, f))
    return found


def main(args):
    midis = expand_midi(args.midi)
    if not midis:
        raise SystemExit("--midi matched no files")

    if args.audio:
        audio = sorted(sum([glob.glob(os.path.expanduser(p)) for p in args.audio], []))
        if not audio:
            raise SystemExit("--audio matched no files")
    else:
        audio = discover_takes(midis, args.audio_dir, args.exclude or ())
        if not audio:
            raise SystemExit(
                "No takes found next to %s. Takes are discovered by name: a file "
                "is a take of <song>.mid when it starts with '<song>' and ends in "
                "%s. Either rename them, or list them explicitly with --audio."
                % (', '.join(os.path.basename(m) for m in midis),
                   '/'.join(AUDIO_EXT)))
        print("discovered %d take(s) for %d score(s):" % (len(audio), len(midis)))
        for a in audio:
            print("    %s" % os.path.basename(a))

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

        fit = res['fit']
        print("    time  : scale=%.3f offset=%+.3f r=%.3f%s"
              % (fit['scale'], fit['offset'], fit['r'],
                 "  !SEARCH EDGE" if fit['at_edge'] else ""))

        # ---- tuning, PER CHORD --------------------------------------------
        #
        # A choir drifts within a passage, so one offset for the take is the
        # median of that drift and is wrong in opposite directions at its two
        # ends. Each chord is measured on its own span instead.
        #
        # The alignment above needs no tuning correction: its mask is +-80
        # cents wide, which already absorbs a choir 20-40 cents out.
        #
        # A chord whose correlation peak is flat has an UNKNOWN offset, which is
        # not the same as zero, so it is dropped rather than labelled at the
        # nominal pitch -- shifting a ridge onto the wrong bin is worse than
        # having one less chord.
        raw_spans = chord_spans(chords, fit['scale'], fit['offset'], T,
                                trim_s=trim, cents=0.0)
        per_cents, bad_tuning = [], []
        for ci, sp in enumerate(raw_spans):
            if args.tuning_mode == 'none' or sp[1] <= sp[0]:
                per_cents.append(0.0)
                continue
            est = estimate_tuning_cents(res['energy'], fgrid, [sp],
                                        search=(-args.tuning_search,
                                                args.tuning_search, 1.0))
            if est is None or est['prominence'] < args.min_tuning_prominence:
                per_cents.append(0.0)
                bad_tuning.append(ci)
            else:
                per_cents.append(est['cents'])
        good = [c for ci, c in enumerate(per_cents) if ci not in bad_tuning]
        if args.tuning_mode != 'none':
            print("    tuning: per chord %s%s"
                  % (' '.join('%+.0f' % c for c in per_cents),
                     '   (drift %.0f c)' % (max(good) - min(good)) if len(good) > 1 else ''))
            if bad_tuning:
                print("      chord(s) %s: tuning peak too flat, offset unknown "
                      "-- dropped" % ', '.join(str(c + 1) for c in bad_tuning))
        if args.librosa_check and good:
            try:
                lt = librosa_tuning_cents(wav)
                print("    tuning cross-check (librosa): %+.1f c  vs per-chord "
                      "median %+.1f" % (lt, float(np.median(good))))
            except Exception as e:
                print("    tuning cross-check failed: %s" % e)

        # ---- sustained spans, with each chord's own correction on the LABELS
        spans = chord_spans(chords, fit['scale'], fit['offset'], T,
                            trim_s=trim, cents=per_cents)
        for ci in bad_tuning:
            spans[ci] = (spans[ci][0], spans[ci][0], spans[ci][2])   # unusable
        cents = float(np.median(good)) if good else 0.0    # reporting only

        # The fit's own r is computed over the whole image and is therefore
        # penalised by COVERAGE, not just by misalignment: a score claiming
        # 0.26 s out of every 1.6 s -- what a fixed MIDI gate time produces --
        # leaves most of the energy at mask=0 however well it is aligned.
        # Gate on the restricted measure, which asks only whether the energy is
        # at the named pitches at the instants the score names them.
        q = alignment_quality(res['energy'], fgrid, spans)
        align_r = fit['r'] if q is None else q
        print("    align : q=%s (fit r=%.3f over the whole file)"
              % ('n/a' if q is None else '%.3f' % q, fit['r']))

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

        # Silence, decided from the audio, ONLY in the gap between a chord's
        # note-off and the next chord's note-on.
        #
        # A fixed reverb margin was a guess; the recording can be asked instead.
        # Broadband level rather than energy at the chord's pitches, because
        # "silence" asserts that NOTHING is sounding, and only a broadband
        # measure can rule out what the score did not predict -- an early entry,
        # a breath, a stray voice. From the waveform, since a 20 dB drop is only
        # meaningful on linear power and the HCQT is already log-scaled and
        # per-file normalised.
        #
        # The chord label itself is never extended past the score's note-off:
        # past that point a held note and a reverb tail are indistinguishable,
        # so the frames stay masked unless the level says they are silent.
        n_rest = 0
        if not args.no_rest_negatives:
            bb = broadband_db(wav, T)
            for ci, (t0, t1, _f) in enumerate(chords):
                i0, i1 = spans[ci][0], spans[ci][1]
                if i1 <= i0:
                    continue
                sustain = float(np.median(bb[i0:i1]))
                gap0 = int(np.ceil((fit['scale'] * t1 + fit['offset']) * SR / HOP))
                gap1 = (int(np.floor((fit['scale'] * chords[ci + 1][0]
                                      + fit['offset']) * SR / HOP))
                        if ci + 1 < len(chords) else T)
                gap0, gap1 = max(gap0, 0), min(gap1, T)
                if gap1 - gap0 < args.min_rest_frames:
                    continue
                quiet = bb[gap0:gap1] < sustain - args.silence_db
                # only runs long enough to be a rest rather than a dip
                run = 0
                for k in range(len(quiet) + 1):
                    if k < len(quiet) and quiet[k]:
                        run += 1
                        continue
                    if run >= args.min_rest_frames:
                        lo = gap0 + k - run
                        state[lo:gap0 + k] = 2
                        mask[lo:gap0 + k] = 1.0
                        n_rest += run
                    run = 0

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
        if align_r < args.min_align_r:
            reason = 'REJECTED: align q %.3f < %.2f' % (align_r, args.min_align_r)
        elif n_chord == 0:
            reason = 'REJECTED: no supervised chord frames'

        if not args.no_plots:
            title = ("%s [%s]  scale=%.3f offset=%+.3f q=%.3f | tuning median "
                     "%+.1f c (%d chord(s) dropped) | trim %.0f ms  %s"
                     % (take, split if not reason else 'skipped', fit['scale'],
                        fit['offset'], align_r, cents, len(bad_tuning),
                        args.trim_ms, reason))
            plot_take_overview(res['energy'], chords, spans, fit, cents,
                               title, os.path.join(plot_dir, '%s_overview.png' % take),
                               args.plot_harmonics, state=state)
            plot_chord_details(res['energy'], chords, spans, fit, cents, title,
                               os.path.join(plot_dir, '%s_chords.png' % take),
                               args.plot_harmonics, per_cents=per_cents)
            plot_tuning_evidence(res['energy'], fgrid, chords, spans,
                                 res['tune'], cents, title,
                                 os.path.join(plot_dir, '%s_tuning.png' % take),
                                 per_cents=[c for ci, c in enumerate(per_cents)
                                            if ci not in bad_tuning])
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
            cents=round(cents, 1),
            cents_per_chord=' '.join('%+.0f' % c for c in per_cents),
            chords_no_tuning=len(bad_tuning),
            scale=fit['scale'], offset=round(fit['offset'], 3),
            align_q=round(align_r, 3), align_r=round(fit['r'], 3)))
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
    p.add_argument('--audio', nargs='+', default=None,
                   help="takes to use, globs allowed. OPTIONAL: if omitted, every "
                        "audio file whose name starts with a score's name is "
                        "discovered automatically, so '<song>_take01.wav' .. "
                        "'<song>_take20.wav' beside '<song>.mid' need not be listed.")
    p.add_argument('--audio_dir', default=None,
                   help='where to look for takes during discovery (default: the '
                        'directory each score sits in)')
    p.add_argument('--exclude', nargs='+', default=None, metavar='PATTERN',
                   help="drop discovered files matching these glob patterns, e.g. "
                        "'*synthetisch*' to keep a soundfont render out of a set "
                        "meant to be real recordings. Case-insensitive.")
    p.add_argument('--midi', nargs='+', required=True,
                   help="scores: files, globs, or a DIRECTORY (every .mid inside "
                        "it). A take pairs with the score whose stem is the "
                        "longest prefix of the take's stem")
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
    p.add_argument('--silence_db', type=float, default=30.0,
                   help="a frame in the gap between a chord's note-off and the "
                        "next note-on counts as silence only when the BROADBAND "
                        "level has fallen at least this far below that chord's "
                        "own sustained level (default 30). This replaces a fixed "
                        "reverb margin: the recording is measured instead of the "
                        "room being guessed at. The chord label is never extended "
                        "past the score's note-off -- frames in the gap are "
                        "either silent by this test or masked.")
    p.add_argument('--min_rest_frames', type=int, default=6,
                   help='ignore rests shorter than this many frames once the '
                        'reverb margin is removed (default 6)')
    p.add_argument('--no_rest_negatives', action='store_true',
                   help='do not supervise the rests. NOT recommended: without '
                        'them every supervised frame contains sounding voices, '
                        'so nothing in the loss ever says "no voice here" -- the '
                        'same missing counterweight that let --pos_weight 4 '
                        'inflate salience by 55%%.')

    p.add_argument('--tuning_mode', choices=['per_chord', 'none'], default='per_chord',
                   help="'per_chord' (default) measures the choir's offset "
                        "separately for each chord and shifts that chord's LABEL "
                        "frequencies onto it, so drift within a take is followed. "
                        "'none' trusts A440 -- only safe if the choir is verified "
                        "within ~10 cents, since sigma is 20 cents.")
    p.add_argument('--tuning_search', type=float, default=100.0,
                   help='cents searched either side of nominal when measuring a '
                        "chord's tuning (default 100). Wider costs time and lets "
                        'the peak land on a neighbouring semitone; narrower can '
                        'miss a badly flat choir.')
    p.add_argument('--min_tuning_prominence', type=float, default=0.05,
                   help="drop a CHORD whose tuning correlation peak is flatter "
                        'than this. A flat curve means the offset is UNKNOWN, '
                        'which is not the same as zero, and labelling it at the '
                        'nominal pitch could put the ridge on the wrong bin '
                        '(default 0.05; a real recording measured 0.10, synthetic '
                        'renders 0.22)')
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
