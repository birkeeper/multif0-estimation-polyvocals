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

Per chord we report, for every voice:
    * absolute salience
    * salience relative to the loudest voice of that chord
and summarise with
    * spread    = max - min of the relative values  (smaller = voices more even)
    * min/max   = quietest voice relative to loudest (larger = quiet voice better
                  represented) -- this is the quantity a quiet-voice fine-tune
                  is meant to improve.

Two practical corrections are applied:
    * tempo  -- a live performance does not run at the MIDI tempo, so a linear
                time warp (scale, offset) is fitted by maximising the salience
                found at the expected F0s. Override with --scale / --offset.
    * tuning -- singers are not exactly at A440, so each voice is read as the
                maximum within +-TOL cents of its nominal pitch (--tol).

Inputs are the .npz salience maps written by predict_on_audio.py --save_salience.

Example
-------
    python compare_voice_salience.py \
        --salience model3=Parijs_model3_salience.npz \
                   adabn=Parijs_model3_adabn_salience.npz \
        --midi "Kenny B - Parijs.mid" --measures 1-2 --bpm 90
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
    for spec in args.salience:
        if '=' not in spec:
            raise SystemExit("--salience expects name=path.npz, got %r" % spec)
        name, path = spec.split('=', 1)
        models[name] = load_salience(path)
        print("loaded %-10s %s  %s  %.2f s"
              % (name, os.path.basename(path), models[name][0].shape,
                 models[name][2][-1]))

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
    p.add_argument('--salience', nargs='+', required=True, metavar='NAME=PATH',
                   help='salience maps to compare, e.g. model3=a.npz adabn=b.npz. '
                        'The first is the baseline for the final comparison.')
    p.add_argument('--midi', required=True, help='score, one track per voice')
    p.add_argument('--measures', default='1-2',
                   help='measure range the recording covers, e.g. 1-2 (1-based)')
    p.add_argument('--bpm', type=float, default=None,
                   help="score tempo; default is the MIDI's own tempo event")
    p.add_argument('--tol', type=float, default=80.0,
                   help='cents window around each nominal pitch, absorbing choir '
                        'tuning. Keep below half the smallest interval in the '
                        'chords (default 80).')
    p.add_argument('--scale', type=float, default=None,
                   help='fix the tempo warp instead of fitting it')
    p.add_argument('--offset', type=float, default=0.0,
                   help='time offset in seconds, used with --scale')
    main(p.parse_args())
