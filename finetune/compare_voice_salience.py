"""
Compare how evenly two (or more) models represent the individual voices of a
recorded ensemble.

The question this answers: does model B lift the *quiet* voices relative to the
loud ones, compared with model A?

Method
------
Given the score as MIDI (one track per voice), every voice's F0 is known at
every instant. So for each chord and each voice we simply READ the salience at
that voice's F0 -- no thresholding, no peak picking. A voice the model "missed"
is not missing data; it is just a low number, and it widens the spread on its
own. That side-steps the whole question of what to do with undetected voices,
and makes the measure independent of the detection threshold.

Two analyses are produced.

1. WHOLE-MAP COMPARISON (no score, no alignment)
   The models processed the same audio, so their maps line up bin for bin:
   mean salience, bins above threshold, correlation with the baseline, and the
   change as a function of the baseline's own activation level.

   Read this first. If one model's salience is a scaled version of another's it
   is not a different detector, only a differently calibrated one -- and any
   later comparison at a shared threshold would be comparing two operating
   points rather than two models. The reported 'equivalent thr' is the setting
   at which each model is as selective as the baseline is at --thresh.

2. PER-VOICE EVENNESS (needs the score)
   Per chord, for every voice: absolute salience and salience relative to the
   loudest voice of that chord, summarised by
     * spread  = max - min of the relative values (smaller = voices more even)
     * min/max = quietest voice relative to loudest (larger = quiet voice
                 better held) -- the quantity a quiet-voice fine-tune targets.

   Salience indicates pitch PRESENCE, so the ideal output is uniformly high
   across every sounding voice however loudly each was actually sung; spread
   near 0 is the target, whatever the ensemble balance.

Two practical corrections are applied:
    * tempo  -- a live performance does not run at the MIDI tempo, so a linear
                time warp (scale, offset) is fitted by correlating the salience
                against a mask built from the score. Override with --scale /
                --offset; check the reported correlation, as a weak fit is the
                usual reason a voice reads near zero.
    * tuning -- singers are not exactly at A440, so each voice is read as the
                maximum within +-TOL cents of its nominal pitch (--tol).

Inputs are the .npz salience maps written by predict_on_audio.py --save_salience.

Example
-------
    python finetune/compare_voice_salience.py \
        --salience Parijs_model3_exp3multif0_salience.npz \
                   Parijs_model3_exp3multif0_finetuned_AdaBN_salience.npz \
        --midi "Kenny B - Parijs.mid" --measures 1-2
"""

from __future__ import print_function

import os
import struct
import argparse

import numpy as np


# --------------------------------------------------------------------------
# Minimal Standard MIDI File reader (no external dependency)
# --------------------------------------------------------------------------
def _vlq(buf, i):
    val = 0
    while True:
        c = buf[i]
        i += 1
        val = (val << 7) | (c & 0x7F)
        if not c & 0x80:
            return val, i


def parse_midi(path):
    """Return (ticks_per_quarter, [track, ...]) where each track is a list of
    (abs_tick, kind, status, data) with kind in {'meta', 'midi'}."""
    b = open(path, 'rb').read()
    if b[:4] != b'MThd':
        raise ValueError("%s is not a Standard MIDI File" % path)
    _fmt, ntrk, div = struct.unpack('>HHH', b[8:14])
    i, tracks = 14, []
    for _ in range(ntrk):
        ln = struct.unpack('>I', b[i+4:i+8])[0]
        body = b[i+8:i+8+ln]
        i += 8 + ln
        events, tick, j, running = [], 0, 0, None
        while j < len(body):
            d, j = _vlq(body, j)
            tick += d
            status = body[j]
            if status & 0x80:
                running = status
                j += 1
            else:
                status = running
            if status == 0xFF:                       # meta
                mtype = body[j]; j += 1
                ln2, j = _vlq(body, j)
                events.append((tick, 'meta', mtype, body[j:j+ln2]))
                j += ln2
            elif status in (0xF0, 0xF7):             # sysex - skip
                ln2, j = _vlq(body, j)
                j += ln2
            else:
                n = 1 if (status & 0xF0) in (0xC0, 0xD0) else 2
                events.append((tick, 'midi', status, body[j:j+n]))
                j += n
        tracks.append(events)
    return div, tracks


def midi_tempo_bpm(tracks, default=120.0):
    for ev in tracks:
        for _t, kind, a, data in ev:
            if kind == 'meta' and a == 0x51:
                return 60e6 / struct.unpack('>I', b'\0' + data)[0]
    return default


def track_name(events, fallback):
    for _t, kind, a, data in events:
        if kind == 'meta' and a == 0x03:
            return data.decode('latin1').strip()
    return fallback


# --------------------------------------------------------------------------
# Score -> chords
# --------------------------------------------------------------------------
def note_name(m):
    return "%s%d" % (['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G',
                      'G#', 'A', 'A#', 'B'][m % 12], m // 12 - 1)


def hz(m):
    return 440.0 * 2.0 ** ((m - 69) / 12.0)


def read_score(midi_path, m_from, m_to, bpm=None, beats_per_measure=4):
    """Return (chords, total_seconds) in MIDI time. Each chord is
    (t0, t1, [(voice, midi_note), ...]); consecutive chords with an identical
    pitch set are merged (repeated notes are one harmony)."""
    div, tracks = parse_midi(midi_path)
    if bpm is None:
        bpm = midi_tempo_bpm(tracks)
    spb = 60.0 / bpm
    tpm = div * beats_per_measure
    start_tick = (m_from - 1) * tpm
    end_tick = m_to * tpm

    notes = []
    for ti, ev in enumerate(tracks):
        has_notes = any(k == 'midi' and (s & 0xF0) == 0x90 and d[1] > 0
                        for _t, k, s, d in ev)
        if not has_notes:
            continue                                   # conductor / empty track
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
                    if on < end_tick and tick > start_tick:
                        notes.append((name, data[0],
                                      max(on, start_tick), min(tick, end_tick)))
    if not notes:
        raise SystemExit("No notes found in measures %d-%d." % (m_from, m_to))

    onsets = sorted({n[2] for n in notes})
    raw = []
    for k, on in enumerate(onsets):
        off = onsets[k+1] if k + 1 < len(onsets) else end_tick
        sounding = [(v, p) for v, p, o, f in notes if o <= on < f]
        if sounding:
            raw.append([on, off, sorted(set(sounding), key=lambda x: -x[1])])
    merged = []
    for c in raw:
        if merged and {p for _v, p in merged[-1][2]} == {p for _v, p in c[2]}:
            merged[-1][1] = c[1]
        else:
            merged.append(c)
    chords = [((a - start_tick) / div * spb, (b - start_tick) / div * spb, v)
              for a, b, v in merged]
    return chords, (end_tick - start_tick) / div * spb


# --------------------------------------------------------------------------
# Salience reading
# --------------------------------------------------------------------------
def load_salience(path):
    d = np.load(path, allow_pickle=True)
    return (d['salience'].astype(np.float32), d['freq_grid'], d['time_grid'])


def names_from_filenames(paths):
    """Derive a short comparison name per path from the part of its filename
    that differs between paths (predict_on_audio.py names salience maps
    '<audio-stem>_<model>_<weights>_salience.npz', so the shared audio stem
    and the shared '_salience' suffix carry no comparison information)."""
    stems = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    if len(stems) < 2:
        return stems

    split = [s.split('_') for s in stems]
    n_pre = 0
    while n_pre < min(len(s) for s in split) and len({s[n_pre] for s in split}) == 1:
        n_pre += 1
    n_suf = 0
    while (n_suf < min(len(s) for s in split) - n_pre
           and len({s[-1 - n_suf] for s in split}) == 1):
        n_suf += 1

    names = ['_'.join(s[n_pre:len(s) - n_suf]) or s[n_pre - 1] for s in split]
    if len(set(names)) != len(names):
        return stems
    return names


def voice_salience(sal, fgrid, tgrid, f0, t0, t1, scale, offset, tol_cents):
    """Mean over the note's frames of the peak salience within +-tol_cents of
    f0. Returns None if the note falls outside the analysed audio."""
    lo = np.searchsorted(fgrid, f0 * 2.0 ** (-tol_cents / 1200.0))
    hi = np.searchsorted(fgrid, f0 * 2.0 ** (tol_cents / 1200.0))
    if hi <= lo:
        hi = lo + 1
    a0, a1 = scale * t0 + offset, scale * t1 + offset
    k = (tgrid >= a0) & (tgrid <= a1)
    if k.sum() < 3:
        return None
    return float(sal[lo:hi, k].max(axis=0).mean())


def score_mask(chords, fgrid, tgrid, scale, offset, tol_cents):
    """Boolean (freq, time) mask of where the score says energy should be."""
    m = np.zeros((len(fgrid), len(tgrid)), dtype=bool)
    for t0, t1, voices in chords:
        k = (tgrid >= scale * t0 + offset) & (tgrid <= scale * t1 + offset)
        if not k.any():
            continue
        for _v, p in voices:
            f0 = hz(p)
            lo = np.searchsorted(fgrid, f0 * 2.0 ** (-tol_cents / 1200.0))
            hi = np.searchsorted(fgrid, f0 * 2.0 ** (tol_cents / 1200.0))
            m[lo:max(hi, lo + 1), k] = True
    return m


def report_suppression(models, base, thresh):
    """Whole-map comparison against the baseline. Needs no score and no
    alignment: the models processed the same audio, so their maps line up bin
    for bin.

    This matters before any accuracy comparison. If one model's salience is
    simply a scaled version of another's, it is not a different detector, only a
    differently calibrated one -- and comparing them at a shared threshold then
    compares two operating points rather than two models. The 'equivalent thr'
    column is the threshold at which each model becomes as selective as the
    baseline is at `thresh`."""
    names = list(models)
    S = {n: models[n][0] for n in names}
    ref_count = int((S[base] > thresh).sum())

    print("\n" + "=" * 62)
    print("WHOLE-MAP COMPARISON vs %s  (no score or alignment involved)" % base)
    print("  %-10s %9s %11s %8s %12s"
          % ('model', 'mean', 'bins>%.2f' % thresh, 'r vs base', 'equiv thr'))
    for n in names:
        r = np.corrcoef(S[base].ravel().astype(np.float64),
                        S[n].ravel().astype(np.float64))[0, 1]
        if n == base:
            eq = '%.2f' % thresh
        else:
            grid = np.arange(0.02, 1.00, 0.01)
            counts = np.array([(S[n] > t).sum() for t in grid])
            eq = '%.2f' % grid[int(np.argmin(np.abs(counts - ref_count)))]
        rel = '' if n == base else ' (%+.0f%%)' % (100 * (S[n].mean() / S[base].mean() - 1))
        print("  %-10s %9.4f %11d %8.3f %12s%s"
              % (n, S[n].mean(), (S[n] > thresh).sum(), r, eq, rel))

    others = [n for n in names if n != base]
    if not others:
        return
    print("\n  Change as a function of the baseline's own activation level.")
    print("  A constant column is a pure rescaling; a column that varies with")
    print("  level means the models genuinely rank time-frequency bins differently.")
    edges = [0.0, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.001]
    print("  %-14s %9s" % ('baseline bin', 'n bins')
          + "".join(" %12s" % ('d ' + n) for n in others))
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (S[base] >= lo) & (S[base] < hi)
        if m.sum() < 50:
            continue
        print("  %.2f - %-8.2f %9d" % (lo, hi, m.sum())
              + "".join(" %+12.4f" % (S[n][m] - S[base][m]).mean() for n in others))


def peak_pick(sal, fgrid, thresh):
    """Per frame, the frequencies of local maxima along the frequency axis that
    exceed `thresh`. Replicates utils_train.pitch_activations_to_mf0 -- importing
    it would pull in TensorFlow, matplotlib and pandas for three lines of scipy --
    so results match what predict_on_audio.py writes to CSV."""
    import scipy.signal
    m = np.zeros(sal.shape, dtype=sal.dtype)
    pk = scipy.signal.argrelmax(sal, axis=0)
    m[pk] = sal[pk]
    fi, ti = np.where(m >= thresh)
    out = [[] for _ in range(sal.shape[1])]
    for f, t in zip(fi, ti):
        out[t].append(float(fgrid[f]))
    return out


def frame_ground_truth(chords, tgrid, scale, offset):
    """Distinct sounding frequencies per audio frame. Unisons collapse to one
    entry: a multi-F0 estimate cannot represent two voices on the same pitch, so
    counting them twice would charge the model for an impossible miss."""
    gt = [[] for _ in range(len(tgrid))]
    for t0, t1, voices in chords:
        k = np.where((tgrid >= scale * t0 + offset) & (tgrid <= scale * t1 + offset))[0]
        fs = sorted({round(hz(p), 4) for _v, p in voices})
        for i in k:
            gt[i] = fs
    return gt


def detection_scores(est, gt, tol_cents):
    """Precision / recall / F of a peak-picked estimate against the score."""
    tp = fp = fn = 0
    for got, ref in zip(est, gt):
        used = []
        for a in ref:
            hit = [g for g in got if g not in used
                   and abs(1200 * np.log2(g / a)) <= tol_cents]
            if hit:
                used.append(hit[0]); tp += 1
            else:
                fn += 1
        fp += len(got) - len(used)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def report_detection(models, chords, scale, offset, tol_cents, thresh, base):
    """Accuracy against the score, swept over thresholds.

    Swept rather than reported at one setting because §the whole-map comparison
    may already have shown the models sit at different operating points; a
    single shared threshold would then measure calibration, not accuracy. The
    matched-count row is the fair head-to-head."""
    names = list(models)
    fgrid, tgrid = models[base][1], models[base][2]
    gt = frame_ground_truth(chords, tgrid, scale, offset)
    n_gt = sum(len(x) for x in gt)
    if not n_gt:
        print("\n(detection accuracy skipped: the score covers none of the audio)")
        return

    grid = np.arange(0.15, 0.81, 0.05)
    picked = {n: {t: peak_pick(models[n][0], fgrid, t) for t in grid} for n in names}
    res = {n: {t: detection_scores(picked[n][t], gt, tol_cents) for t in grid}
           for n in names}

    print("\n" + "=" * 62)
    print("DETECTION ACCURACY vs the score  (peak-picked, +-%.0fc match, %d "
          "reference pitches)" % (tol_cents, n_gt))
    print("  %-5s" % 'thr' + "".join(" | %-21s" % n for n in names))
    print("  %-5s" % '' + "".join(" |     P      R      F " for _ in names))
    for t in grid:
        print("  %.2f " % t
              + "".join(" | %.3f  %.3f  %.3f" % res[n][t] for n in names))
    print("  %-5s" % 'best'
          + "".join(" | %.3f @thr %.2f    " % (max(res[n][t][2] for t in grid),
                    max(grid, key=lambda t: res[n][t][2])) for n in names))

    # fair head-to-head: same number of detections
    ref_n = sum(len(f) for f in peak_pick(models[base][0], fgrid, thresh))
    print("\n  Matched operating point -- each model at the threshold giving the")
    print("  same number of detections as %s at %.2f (%d peaks):" % (base, thresh, ref_n))
    for n in names:
        if n == base:
            eq = thresh
        else:
            fine = np.arange(0.05, 0.95, 0.01)
            counts = [sum(len(f) for f in peak_pick(models[n][0], fgrid, t)) for t in fine]
            eq = float(fine[int(np.argmin(np.abs(np.array(counts) - ref_n)))])
        est = peak_pick(models[n][0], fgrid, eq)
        p, r, f1 = detection_scores(est, gt, tol_cents)
        print("    %-10s @%.2f  P=%.3f R=%.3f F=%.3f  (%d peaks)"
              % (n, eq, p, r, f1, sum(len(x) for x in est)))


def fit_warp(models, chords, tol_cents, scales, offsets):
    """Choose (scale, offset) by correlating the salience map with a mask built
    from the score. Correlation -- unlike the mean salience under the mask --
    cannot be inflated by squeezing the score onto the loud part of the audio,
    because shrinking the mask is penalised by the frames it then leaves
    unexplained. Averaged over models so both are aligned identically."""
    sal = np.mean([m[0] for m in models.values()], axis=0)
    fgrid, tgrid = list(models.values())[0][1], list(models.values())[0][2]
    flat = sal.ravel().astype(np.float64)
    best = None
    for sc in scales:
        for off in offsets:
            m = score_mask(chords, fgrid, tgrid, sc, off, tol_cents).ravel()
            if m.sum() < 10 or m.all():
                continue
            c = np.corrcoef(flat, m.astype(np.float64))[0, 1]
            if best is None or c > best[0]:
                best = (c, sc, off)
    if best is None:
        raise SystemExit("Could not align the score to the audio.")
    q, sc, off = best
    for val, lo, hi, what in ((sc, scales[0], scales[-1], 'scale'),
                              (off, offsets[0], offsets[-1], 'offset')):
        if abs(val - lo) < 1e-9 or abs(val - hi) < 1e-9:
            print("  ! warning: fitted %s hit the edge of its search range "
                  "(%.3f); the alignment is probably wrong -- pass --scale/--offset."
                  % (what, val))
    return sc, off, q


# --------------------------------------------------------------------------
def main(args):
    models = {}
    names = names_from_filenames(args.salience)
    for name, path in zip(names, args.salience):
        models[name] = load_salience(path)
        print("loaded %-10s %s  %s  %.2f s"
              % (name, os.path.basename(path), models[name][0].shape,
                 models[name][2][-1]))

    shapes = {models[n][0].shape for n in models}
    if len(shapes) == 1:
        report_suppression(models, list(models)[0], args.thresh)
    else:
        print("\n(whole-map comparison skipped: maps differ in shape %s -- they are "
              "not from the same audio)" % sorted(shapes))

    m_from, m_to = (int(x) for x in args.measures.split('-')) \
        if '-' in args.measures else (int(args.measures), int(args.measures))
    chords, dur = read_score(args.midi, m_from, m_to, args.bpm)
    print("\nscore: measures %d-%d, %d chord(s), %.2f s at %s BPM"
          % (m_from, m_to, len(chords), dur,
             args.bpm if args.bpm else "the MIDI's own tempo"))

    if args.scale is not None:
        scale, offset = args.scale, args.offset
        print("warp : scale=%.3f offset=%+.3f (given)" % (scale, offset))
    else:
        scale, offset, q = fit_warp(models, chords, args.tol,
                                    np.arange(0.70, 1.31, 0.01),
                                    np.arange(-0.50, 0.51, 0.02))
        print("warp : scale=%.3f offset=%+.3f (fitted; performance %+.0f%% vs score "
              "tempo, score/salience correlation r=%.3f)"
              % (scale, offset, (1.0 / scale - 1) * 100, q))

    names = list(models)
    if not args.no_detection:
        report_detection(models, chords, scale, offset, args.tol, args.thresh, names[0])

    print("\n" + "=" * 62)
    print("PER-VOICE EVENNESS")
    summary = {n: [] for n in names}

    for ci, (t0, t1, voices) in enumerate(chords, 1):
        rows = []
        for v, p in voices:
            vals = {n: voice_salience(models[n][0], models[n][1], models[n][2],
                                      hz(p), t0, t1, scale, offset, args.tol)
                    for n in names}
            if any(x is None for x in vals.values()):
                continue
            rows.append((v, p, vals))
        if not rows:
            print("\nchord %d: outside the analysed audio, skipped" % ci)
            continue

        print("\n=== chord %d   score %.2f-%.2f s  ->  audio %.2f-%.2f s ==="
              % (ci, t0, t1, scale * t0 + offset, scale * t1 + offset))
        w = max(len(v) for v, _p, _x in rows)
        head = "  %-*s %-5s |" % (w, 'voice', 'pitch')
        print(head + "".join("  %9s" % n for n in names)
              + "  |" + "".join(" %9s" % ('rel ' + n) for n in names))
        peak = {n: max(r[2][n] for r in rows) or 1e-9 for n in names}
        for v, p, vals in rows:
            print("  %-*s %-5s |" % (w, v, note_name(p))
                  + "".join("  %9.3f" % vals[n] for n in names)
                  + "  |" + "".join(" %9.2f" % (vals[n] / peak[n]) for n in names))
        for n in names:
            rel = [r[2][n] / peak[n] for r in rows]
            summary[n].append((max(rel) - min(rel), min(rel)))
        print("  %-*s %-5s |" % (w, 'spread', '')
              + "".join("  %9.2f" % summary[n][-1][0] for n in names)
              + "  |  (max-min of relative; smaller = voices more even)")
        print("  %-*s %-5s |" % (w, 'min/max', '')
              + "".join("  %9.2f" % summary[n][-1][1] for n in names)
              + "  |  (quietest vs loudest; larger = quiet voice better held)")

    print("\n" + "=" * 62)
    print("MEAN OVER %d CHORDS" % len(summary[names[0]]))
    print("  %-10s %10s %10s" % ('model', 'spread', 'min/max'))
    for n in names:
        sp = np.mean([s for s, _ in summary[n]])
        mm = np.mean([m for _, m in summary[n]])
        print("  %-10s %10.3f %10.3f" % (n, sp, mm))
    base = names[0]
    for n in names[1:]:
        d_sp = np.mean([s for s, _ in summary[n]]) - np.mean([s for s, _ in summary[base]])
        d_mm = np.mean([m for _, m in summary[n]]) - np.mean([m for _, m in summary[base]])
        won = sum(1 for a, b in zip(summary[n], summary[base]) if a[1] > b[1])
        print("\n  %s vs %s: spread %+.3f, min/max %+.3f  -> %s"
              % (n, base, d_sp, d_mm,
                 "voices MORE even" if d_sp < 0 and d_mm > 0 else
                 "voices LESS even" if d_sp > 0 and d_mm < 0 else "mixed"))
        print("  quiet voice better held in %d of %d chords"
              % (won, len(summary[base])))


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--salience', nargs='+', required=True, metavar='PATH',
                   help='salience maps to compare, e.g. '
                        'Parijs_model3_exp3multif0_salience.npz '
                        'Parijs_model3_exp3multif0_finetuned_AdaBN_salience.npz. '
                        'Comparison names are derived from the part of each '
                        'filename that differs between them. The first is the '
                        'baseline for the final comparison.')
    p.add_argument('--midi', required=True, help='score, one track per voice')
    p.add_argument('--measures', default='1-2',
                   help='measure range the recording covers, e.g. 1-2 (1-based)')
    p.add_argument('--bpm', type=float, default=None,
                   help="score tempo; default is the MIDI's own tempo event")
    p.add_argument('--tol', type=float, default=80.0,
                   help='cents window around each nominal pitch, absorbing choir '
                        'tuning. Keep below half the smallest interval in the '
                        'chords (default 80).')
    p.add_argument('--thresh', type=float, default=0.5,
                   help='reference threshold for the whole-map comparison: bins are '
                        'counted above it, and each model\'s equivalent threshold is '
                        'the one making it as selective as the baseline here '
                        '(default 0.5, the models\' own default)')
    p.add_argument('--no_detection', action='store_true',
                   help='skip the detection-accuracy sweep (the slowest part)')
    p.add_argument('--scale', type=float, default=None,
                   help='fix the tempo warp instead of fitting it')
    p.add_argument('--offset', type=float, default=0.0,
                   help='time offset in seconds, used with --scale')
    main(p.parse_args())
