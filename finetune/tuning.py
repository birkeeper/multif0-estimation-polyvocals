"""Measure how far a choir's actual pitch sits from the nominal score pitch.

Why it has to be measured
-------------------------
`utils.create_annotation_target` blurs the target with sigma = 1 frequency bin,
and the grid is 60 bins/octave, so sigma = 20 cents. A choir singing 30 cents
flat therefore puts the target ridge 1.5 sigma away from the energy it is meant
to mark -- and because every note is off in the same direction, the model learns
a frequency bias instead of averaging the error out.

    off by 20 c  = 1.0 sigma -> target at the true pitch ~0.61 of peak
    off by 30 c  = 1.5 sigma -> ~0.32
    off by 50 c  = 2.5 sigma -> ~0.04

Method: shift-correlation against the score
-------------------------------------------
The obvious approach -- search around each nominal pitch for a spectral peak and
average the deviations -- fails on exactly this material. The search window is
centred on the nominal pitch, so a masked voice with no real peak returns
something close to noise, which averages toward zero: the estimator reports "in
tune" precisely when it has no information. It is also quantised to the 20-cent
bin spacing.

Instead the whole score is shifted as one rigid body. For each candidate shift
the score mask is rebuilt with every label frequency multiplied by 2^(d/1200),
and correlated against the CQT energy. The maximum over d is the choir's offset.
This uses all voices in all chords at once, so a buried voice contributes
nothing rather than contributing noise, and interpolating the correlation peak
gives sub-bin resolution.

The mask must be NARROW here (~+-22 cents, about one bin either side). A wide
mask is right for fitting time alignment but makes the correlation flat in d,
which is the one thing this measurement cannot tolerate.

Trusting the result
-------------------
`prominence` (peak minus median of the correlation curve) says whether there was
any tuning information at all. A flat curve means the answer is unknown, which
is different from zero, and callers should skip rather than "correct" by a
number that means nothing. `librosa.estimate_tuning` provides an independent
check: it uses spectral peaks and equal temperament, needs no score, and has
different failure modes, so agreement between the two is meaningful.

Don't chase precision beyond ~10 cents. A section has real intrinsic spread --
vibrato plus singer-to-singer variation, roughly +-20 cents -- so the "pitch of
the tenor part" is a distribution whose width the sigma = 20 cent blur already
absorbs. Only its centre needs correcting.
"""
from __future__ import print_function

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils

DB_FLOOR = -80.0
SR, HOP = 22050.0, 256.0


# --------------------------------------------------------------------------
def energy_image(mag_hft):
    """h=1 CQT magnitude as (F, T) mapped from dB onto [0, 1]."""
    return (mag_hft[0] - DB_FLOOR) / (-DB_FLOOR)


def spans_mask(fgrid, n_frames, spans, cents=0.0, tol_cents=22.0):
    """Boolean (F, T) mask marking, for each span, a narrow band around every
    one of its frequencies, shifted bodily by `cents`."""
    m = np.zeros((len(fgrid), n_frames), dtype=bool)
    shift = 2.0 ** (cents / 1200.0)
    lo_f = 2.0 ** (-tol_cents / 1200.0)
    hi_f = 2.0 ** (tol_cents / 1200.0)
    for i0, i1, freqs in spans:
        if i1 <= i0:
            continue
        for f in freqs:
            f = f * shift
            lo = np.searchsorted(fgrid, f * lo_f)
            hi = np.searchsorted(fgrid, f * hi_f)
            if lo >= len(fgrid):
                continue
            m[lo:max(hi, lo + 1), i0:i1] = True
    return m


def _parabolic(x, y, k):
    """Sub-step refinement of a peak at index k of the sampled curve y(x)."""
    if k <= 0 or k >= len(y) - 1:
        return x[k]
    a, b, c = y[k - 1], y[k], y[k + 1]
    den = a - 2 * b + c
    if abs(den) < 1e-12:
        return x[k]
    return x[k] + 0.5 * (a - c) / den * (x[1] - x[0])


def estimate_tuning_cents(energy, fgrid, spans, search=(-100.0, 100.0, 1.0),
                          tol_cents=22.0):
    """Cents by which the AUDIO sits above the nominal score pitch.

    A return of -30 means the choir sang 30 cents flat, so label frequencies
    should be multiplied by 2**(-30/1200) to land on the actual energy.
    """
    frames = np.zeros(energy.shape[1], dtype=bool)
    for i0, i1, _f in spans:
        frames[max(0, i0):max(0, i1)] = True
    if frames.sum() < 5:
        return None
    e = energy[:, frames].ravel().astype(np.float64)

    # re-index the spans onto the compacted frame axis
    newidx = np.cumsum(frames) - 1
    compact = [(int(newidx[i0]), int(newidx[i1 - 1]) + 1, f)
               for i0, i1, f in spans if i1 > i0 and frames[i0]]

    grid = np.arange(search[0], search[1] + 1e-9, search[2])
    corr = np.empty(len(grid))
    for i, d in enumerate(grid):
        m = spans_mask(fgrid, int(frames.sum()), compact, d, tol_cents).ravel()
        if m.all() or not m.any():
            corr[i] = np.nan
            continue
        corr[i] = np.corrcoef(e, m.astype(np.float64))[0, 1]
    if np.all(np.isnan(corr)):
        return None

    k = int(np.nanargmax(corr))
    cents = _parabolic(grid, corr, k)
    prominence = float(np.nanmax(corr) - np.nanmedian(corr))
    at_edge = k == 0 or k == len(grid) - 1
    return dict(cents=float(cents), prominence=prominence, peak_corr=float(corr[k]),
                at_edge=at_edge, grid=grid, curve=corr)


def librosa_tuning_cents(wav_path):
    """Score-free cross-check. Returns cents in (-50, 50]; wraps beyond that."""
    import librosa
    y, sr = librosa.load(wav_path, sr=int(SR))
    return float(100.0 * librosa.estimate_tuning(y=y, sr=sr))


# --------------------------------------------------------------------------
def fit_time(energy, fgrid, tgrid, chords, scales, offsets, cents=0.0,
             tol_cents=80.0):
    """Fit (scale, offset) mapping score seconds onto audio seconds.

    Correlation rather than mean-energy-under-the-mask: the latter is maximised
    by shrinking the score onto the loudest instant, whereas correlation is
    penalised by the frames a too-small mask leaves unexplained."""
    flat = energy.ravel().astype(np.float64)
    shift = 2.0 ** (cents / 1200.0)
    best = None
    for sc in scales:
        for off in offsets:
            spans = []
            for t0, t1, freqs in chords:
                i0 = int(np.ceil((sc * t0 + off) * SR / HOP))
                i1 = int(np.floor((sc * t1 + off) * SR / HOP))
                spans.append((max(i0, 0), min(i1, len(tgrid)),
                              [f * shift for f in freqs]))
            m = spans_mask(fgrid, len(tgrid), spans, 0.0, tol_cents).ravel()
            if m.sum() < 10 or m.all():
                continue
            c = np.corrcoef(flat, m.astype(np.float64))[0, 1]
            if best is None or c > best[0]:
                best = (c, sc, off)
    if best is None:
        return None
    r, sc, off = best
    return dict(r=float(r), scale=float(sc), offset=float(off),
                at_edge=(abs(sc - scales[0]) < 1e-9 or abs(sc - scales[-1]) < 1e-9 or
                         abs(off - offsets[0]) < 1e-9 or abs(off - offsets[-1]) < 1e-9))


def chord_spans(chords, scale, offset, n_frames, trim_s=0.0, cents=0.0):
    """Score chords -> (i0, i1, freqs) on the audio frame grid, with `trim_s`
    removed from each end and label frequencies shifted by `cents`."""
    shift = 2.0 ** (cents / 1200.0)
    out = []
    for t0, t1, freqs in chords:
        i0 = int(np.ceil((scale * t0 + offset + trim_s) * SR / HOP))
        i1 = int(np.floor((scale * t1 + offset - trim_s) * SR / HOP))
        out.append((max(i0, 0), min(i1, n_frames), [f * shift for f in freqs]))
    return out


def align_and_tune(mag_hft, fgrid, tgrid, chords, rounds=2,
                   scales=(0.80, 1.2501, 0.005), offsets=(-1.0, 2.001, 0.01),
                   search=(-100.0, 100.0, 1.0), tol_time=80.0, tol_tune=22.0):
    """Alternate time alignment and tuning estimation.

    They interact: a bad time fit smears the tuning peak, and a bad tuning
    assumption weakens the time fit. Two rounds is enough in practice -- the
    second is a refinement, and if it moves much, the first was unreliable.
    """
    energy = energy_image(mag_hft)
    sc_grid = np.arange(*scales)
    off_grid = np.arange(*offsets)
    cents, hist = 0.0, []
    fit = tune = None
    for _ in range(rounds):
        fit = fit_time(energy, fgrid, tgrid, chords, sc_grid, off_grid,
                       cents=cents, tol_cents=tol_time)
        if fit is None:
            return None
        spans = chord_spans(chords, fit['scale'], fit['offset'], len(tgrid))
        tune = estimate_tuning_cents(energy, fgrid, spans, search, tol_tune)
        if tune is None:
            return None
        cents = tune['cents']
        hist.append((fit['r'], cents))
    return dict(energy=energy, fit=fit, tune=tune, cents=cents, history=hist)
