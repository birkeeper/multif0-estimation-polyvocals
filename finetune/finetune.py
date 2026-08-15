"""
Fine-tune model3 to *unlearn* the "quiet voice -> low salience" bias, using the
synthetic SSATBB chords produced by generate_chords.py (rendered to wav by the
PWA).

Method (see ../research/finetune_conversation.md):
  * A soft/quiet voice is an input-amplitude/SNR domain shift whose evidence is
    attenuated at the input BatchNorm and the early/mid harmonic layers
    (conv1..harm2) -- i.e. BEFORE the decision head. Head-only fine-tuning cannot
    recover information the front end already discarded, so it is deliberately
    NOT offered. --strategy adapts where the loss actually happens:
      - bn   : AdaBN recalibration -- freeze all conv/dense weights, adapt only
               BatchNorm (gamma/beta + running stats recalibrate to the new
               amplitude distribution). Cheapest, but see the warning below: it
               adapts exactly the statistics that carry the domain shift.
      - conv : the inverse, and the DEFAULT -- adapt the conv/dense weights so the
               early harmonic detectors themselves learn to keep a quiet voice
               above threshold, while BatchNorm is frozen (Keras holds a
               trainable=False BN in inference mode, so gamma/beta and the running
               statistics all stay put). The normalisation therefore remains
               fitted to the real recordings model3 was trained on. Pair with
               --l2sp to anchor the kernels to their pretrained values (L2-SP,
               Xuhong et al. 2018) so normal-balance performance is not erased.
      - full : everything at once. Note this does NOT avoid the drift below:
               --l2sp anchors only variables named '*kernel*', and BatchNorm's
               running statistics are not trainable variables at all, so nothing
               anchors them.

    WARNING -- the training data is synthetic and the target domain is real
    recordings. Adapting BatchNorm recalibrates the model to the synthetic
    amplitude distribution; on real audio the salience map is then compressed
    downward (confident activations pulled down hardest), which destroys
    detections while the synthetic validation still reports an improvement. This
    happened: see models/exp3multif0_finetuned_AdaBN.md section 4. Use
    --real_audio to watch for it, and prefer 'conv'.
  * --distill_dir adds a SECOND loss term on real, unannotated audio: the
    pre-trained model is run over it once and the student is penalised for moving
    away from that output. Every strategy above changed the real-audio
    CALIBRATION rather than the separation -- -36% (conv), -22% (bn), +55% (bn
    with pos_weight 4) -- because the loss only ever saw soundfont renders, so
    recalibrating to them was the cheapest way to reduce it. This term is the
    same quantity drift_stats() already reports, moved out of the accept/reject
    decision and into the gradient. It is --l2sp measured on OUTPUTS instead of
    weights, which is both the quantity that matters and the only one that works
    under --strategy bn (no variable is named 'kernel' there, so --l2sp anchors
    nothing). Must be disjoint from --real_audio, which stays held out.
  * --pos_weight upweights the loss on annotated (voice) time-frequency bins --
    "reweight near the soft voice's F0" -- countering the sparse-positive target
    so quiet-voice bins are not drowned by the empty background.
  * Each rendered file is featurised once (whole-file CQT, so no edge artifacts),
    sliced into chord segments using the annotation's silent gaps, and each chord
    is cut into fixed WINDOWS (~50 frames). Windows are cached to disk and
    streamed one/few at a time (low memory). Fixed-length windows are required at
    TRAINING time -- not for the fully-convolutional forward pass, but because the
    `distribution` layer's (360,1) kernel makes its backprop-filter memory scale
    with T (a whole chord OOMs; ~50 frames keeps it ~1.6 GB). Windowing within
    chords also skips the inter-chord silence.
  * Checkpoint selection VALIDATES BOTH SIDES on the matched pair (balanced vs.
    one-voice-quiet). Because all 6 voices sound in every chord and only one is
    attenuated, the raw victim recall is diluted 6x; selection therefore ranks
    epochs on the UNDILUTED quiet-voice recall
        R_quiet = 6*R_victim - 5*R_balanced
    guarded so the win is not bought elsewhere: recall and precision on both
    sides must not regress past --bal_tol vs. the pre-training baseline, and the
    quiet-voice gap must not widen past --gap_tol. Small LR, few epochs.
    This runs at every epoch boundary and, with --eval_every, part-way through
    an epoch as well -- the interesting movement is over well inside epoch 1.
  * Progress and per-checkpoint results are teed to <out>.log, so a long run's
    numbers survive the terminal.

Layout expected (from generate_chords.py + PWA render):
    <train_dir>/train_XXXX.wav      + train_XXXX.f0.csv
    <valid_dir>/valid_XXXX_balanced.wav    + .f0.csv
    <valid_dir>/valid_XXXX_victim.wav      + .f0.csv   (same notes, one quiet
                                                        voice per chord)
"""

from __future__ import print_function

import os
import re
import sys
import csv
import json
import glob
import argparse
import datetime

import numpy as np
import tensorflow as tf

# allow running from anywhere: the repo modules live one dir up
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import models
import utils
import utils_train

tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)

CHUNK_LEN = 2000          # time frames per model.predict call (for eval)

# Audio containers the distillation pool accepts. librosa.load() goes through
# soundfile/libsndfile, which resamples to 22050 and downmixes to mono for all
# of these. MP3 is deliberately absent: soundfile 0.9 does not support it and
# librosa's audioread fallback is deprecated -- transcode to FLAC first.
AUDIO_EXT = ('*.wav', '*.flac', '*.ogg')

# Frames to drop from each end of a window before comparing student and teacher.
# The teacher salience is computed whole-file (no boundary artifacts), but the
# student sees a --win-frame window that is zero-padded at both ends, so its
# output near the edges cannot match. Summing the time receptive field:
# conv1..conv4 (5,5) = +-2 each, harm1/harm2 (70,3) = +-1 each, conv7/conv8
# (3,3) = +-1 each, distribution (360,1) = 0  ->  about +-12 frames.
DISTILL_EDGE_CROP = 12


# --------------------------------------------------------------------------
# Logging: mirror stdout to a log file so a long run's results survive the
# terminal scrollback (or a lost ssh session).
# --------------------------------------------------------------------------
class Tee(object):
    """Duplicate stdout to a log file, line-buffered so nothing is lost if the
    run is killed. In-place progress updates (the '\\r batch i/n' line) go to
    the terminal only, keeping the log line-oriented."""

    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(path, 'w', buffering=1)

    def write(self, s):
        self.terminal.write(s)
        if not s.startswith('\r'):
            self.log.write(s)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def setup_logging(out_weights, args):
    """Start teeing stdout to a NEW log per run, next to the output weights and
    stamped with the start time:

        <out minus .weights.h5>_YYYYmmdd-HHMMSS.log

    One file per run rather than one appended file, so runs can be compared
    side by side and a re-run can never be mistaken for a continuation of the
    previous one. Returns the log path."""
    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    path = "%s_%s.log" % (out_weights[:-len('.weights.h5')], stamp)
    sys.stdout = Tee(path)
    print("=" * 78)
    print("run started %s" % datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("args: %s" % json.dumps(vars(args), sort_keys=True))
    print("=" * 78)
    return path


# --------------------------------------------------------------------------
# Annotation / feature helpers
# --------------------------------------------------------------------------
def load_ragged_f0(path):
    """Load a tab-delimited ragged multi-F0 CSV (time \t f1 \t f2 ...).
    Returns (times[np], list_of_freq_arrays)."""
    times, freqs = [], []
    with open(path) as fh:
        for row in csv.reader(fh, delimiter='\t'):
            if not row:
                continue
            times.append(float(row[0]))
            freqs.append(np.array([float(x) for x in row[1:] if x != ''], dtype=float))
    return np.array(times), freqs


def f0_to_points(times, freqs):
    """Flatten ragged (time, [f...]) into the (times, freqs) point arrays that
    utils.create_annotation_target expects."""
    ts, fs = [], []
    for t, fr in zip(times, freqs):
        for f in fr:
            if f > 0:
                ts.append(t)
                fs.append(f)
    return np.array(ts), np.array(fs)


def featurize(pump, wav_path):
    """Whole-file HCQT mag + phase-diff as (H, F, T) arrays.

    NB orientation: pumpp emits (time, freq, harmonic); we transpose (2,1,0) to
    (H, F, T) -- matching the TRAINING code (utils_train.patch_generator), which
    is the convention model3's weights were fit on. This is intentionally NOT the
    transpose predict_on_audio.get_single_test_prediction uses; that path applies
    a (1,2,0) meant for a (channel,freq,time) layout to pumpp's (time,freq,channel)
    output, which is a known mismatch. Staying with the training convention keeps
    fine-tuning and evaluation consistent with how the model was trained."""
    feats = utils.compute_pump_features_segmented(pump, wav_path)
    mag = feats['dphase/mag'][0]        # (T, F, H)
    dph = feats['dphase/dphase'][0]     # (T, F, H)
    mag = np.transpose(mag, (2, 1, 0))  # (H, F, T)
    dph = np.transpose(dph, (2, 1, 0))
    return mag.astype(np.float32), dph.astype(np.float32)


def build_target(n_frames, f0_csv):
    """Blurred binary salience target (F, T) on the feature time grid."""
    freq_grid = utils.get_freq_grid()
    time_grid = utils.get_time_grid(n_frames)
    times, freqs = load_ragged_f0(f0_csv)
    pts_t, pts_f = f0_to_points(times, freqs)
    if len(pts_t) == 0:
        return np.zeros((len(freq_grid), n_frames), dtype=np.float32)
    return utils.create_annotation_target(freq_grid, time_grid, pts_t, pts_f).astype(np.float32)


def segment_active(active, gap_frames=8, min_frames=12):
    """Return [(t0, t1), ...] for runs of True in `active`, merging gaps
    shorter than gap_frames and dropping runs shorter than min_frames."""
    segs = []
    t = 0
    T = len(active)
    while t < T:
        if not active[t]:
            t += 1
            continue
        t0 = t
        gap = 0
        while t < T and (active[t] or gap < gap_frames):
            if active[t]:
                gap = 0
                last = t
            else:
                gap += 1
            t += 1
        t1 = last + 1
        if t1 - t0 >= min_frames:
            segs.append((t0, t1))
    return segs


def segment_chords(target, gap_frames=8, min_frames=12):
    """Return [(t0, t1), ...] for runs of active (non-silent) frames, merging
    gaps shorter than gap_frames and dropping runs shorter than min_frames.
    Uses the silent gaps the generator places between chords."""
    return segment_active(target.sum(axis=0) > 1e-3, gap_frames, min_frames)


# --------------------------------------------------------------------------
# Prepare: featurize + slice chords + cache to disk (one npz per chord)
# --------------------------------------------------------------------------
def is_prebuilt_cache(d):
    """True if `d` is itself a finished window cache (DONE marker + *.npz
    sitting directly in it), as prepare_real_chords.py writes -- as opposed to
    a raw source dir (train_*.wav/.f0.csv, or valid_*_balanced/victim.wav)
    that still needs featurizing."""
    return (os.path.exists(os.path.join(d, 'DONE'))
            and bool(glob.glob(os.path.join(d, '*.npz'))))


def prepare(pump, train_dir, cache_dir, win=50, hop=None, recompute=False):
    """Featurize each file, slice chords, cut chords into fixed `win`-frame
    windows (stride `hop`, default `win` = no overlap), cache one npz per
    window. Fixed length bounds the distribution-layer backprop memory and lets
    windows be batched.

    Window files are named <stem>_c<chord>_w<window>.npz so a subset of chords
    can be selected later (see --TEST) without reading the files."""
    hop = hop or win
    # Windows built directly by prepare_real_chords.py (or any other prebuilt
    # cache) sit as DONE + *.npz in train_dir itself -- no _cache subfolder,
    # and no .f0.csv to reslice from. Use them as-is when present.
    prebuilt = is_prebuilt_cache(train_dir)
    if prebuilt:
        wins = sorted(glob.glob(os.path.join(train_dir, '*.npz')))
        print("Using %d prebuilt windows in %s" % (len(wins), train_dir))
        return wins

    os.makedirs(cache_dir, exist_ok=True)
    done_marker = os.path.join(cache_dir, 'DONE')
    if os.path.exists(done_marker) and not recompute:
        wins = sorted(glob.glob(os.path.join(cache_dir, '*.npz')))
        print("Using %d cached windows in %s" % (len(wins), cache_dir))
        return wins

    f0_files = sorted(glob.glob(os.path.join(train_dir, '*.f0.csv')))
    n_win = 0
    for f0 in f0_files:
        base = f0[:-len('.f0.csv')]
        wav = base + '.wav'
        if not os.path.exists(wav):
            print("  ! missing wav for %s -- render it in the PWA; skipping" % os.path.basename(base))
            continue
        mag, dph = featurize(pump, wav)          # (H, F, T)
        T = mag.shape[2]
        target = build_target(T, f0)             # (F, T)
        segs = segment_chords(target, min_frames=win)
        stem = os.path.basename(base)
        fw = 0
        for ci, (t0, t1) in enumerate(segs):
            for w, s in enumerate(range(t0, t1 - win + 1, hop)):
                # store as (F, win, H) for direct model input; target (F, win)
                mseg = np.transpose(mag[:, :, s:s+win], (1, 2, 0))
                dseg = np.transpose(dph[:, :, s:s+win], (1, 2, 0))
                tseg = target[:, s:s+win]
                np.savez_compressed(
                    os.path.join(cache_dir, '%s_c%04d_w%04d.npz' % (stem, ci, w)),
                    mag=mseg, dph=dseg, tgt=tseg)
                fw += 1
                n_win += 1
        print("  %s -> %d chords, %d windows" % (stem, len(segs), fw))
    open(done_marker, 'w').close()
    wins = sorted(glob.glob(os.path.join(cache_dir, '*.npz')))
    print("Cached %d windows (win=%d, hop=%d) from %d files." % (n_win, win, hop, len(f0_files)))
    return wins


_CHORD_RE = re.compile(r'_c(\d+)_w\d+\.npz$')


def select_first_chords(win_files, n_chords):
    """Keep only windows belonging to the first `n_chords` chords OF EACH FILE.
    prepare()'s synthetic windows carry the chord index in the filename;
    prepare_real_chords.py's windows have no such suffix (a sliding window can
    straddle several chords) but store the chord indices it covers in the
    npz's `chords` field instead, 1-based."""
    kept = []
    for p in win_files:
        m = _CHORD_RE.search(os.path.basename(p))
        if m is not None:
            if int(m.group(1)) < n_chords:
                kept.append(p)
            continue
        with np.load(p) as d:
            chords = d['chords'] if 'chords' in d.files else None
        if chords is not None and len(chords) and int(chords.min()) <= n_chords:
            kept.append(p)
    return kept


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def set_trainable(model, strategy):
    """strategy in {'bn', 'conv', 'full'}. 'head' is intentionally absent: a quiet
    voice is lost before the head, so adapting only the head cannot recover it."""
    if strategy == 'full':
        for layer in model.layers:
            layer.trainable = True
    elif strategy == 'bn':
        # AdaBN: only BatchNorm adapts (weights frozen). trainable=True keeps BN
        # in training mode so its running stats recalibrate to the new amplitudes.
        for layer in model.layers:
            layer.trainable = isinstance(layer, tf.keras.layers.BatchNormalization)
    elif strategy == 'conv':
        # The inverse of 'bn': adapt the conv/dense weights, freeze BatchNorm.
        #
        # Keras special-cases BatchNormalization -- trainable=False also puts the
        # layer in INFERENCE mode, even when the model is called with
        # training=True -- so gamma/beta AND the running mean/variance are all
        # held fixed. The normalisation therefore stays fitted to the real
        # recordings model3 was trained on, instead of drifting to the synthetic
        # amplitude distribution. That drift is what 'bn' does by design, and
        # what 'full' also does incidentally: --l2sp anchors only variables whose
        # name contains 'kernel', and the running statistics are not trainable
        # variables at all, so nothing can anchor them.
        #
        # Pair with --l2sp, which does apply here, and verify with --real_audio.
        for layer in model.layers:
            layer.trainable = not isinstance(layer, tf.keras.layers.BatchNormalization)
    else:
        raise ValueError("unknown strategy %r (use 'bn', 'conv' or 'full')" % strategy)


def build_model(weights_path, strategy):
    model = models.build_model3()
    model.load_weights(weights_path)
    set_trainable(model, strategy)
    tr = [l.name for l in model.layers if l.trainable]
    print("Strategy '%s': %d trainable layers %s"
          % (strategy, len(tr), tr if len(tr) <= 12 else '(%d layers)' % len(tr)))
    return model


def make_bkld(pos_weight=1.0):
    """bkld (Brian's KL divergence) loss, optionally upweighting positive
    (annotated voice) target bins by pos_weight.

    Accepts an optional per-frame `mask` of shape (batch, T): 1 where the label
    is known, 0 where it is not. Real recordings need this. Inside a chord's
    attack, release or reverb tail the score cannot distinguish a staggered
    entry from a decaying voice from a quiet one, so no target is defensible
    there -- but those frames are still wanted as INPUT, because they feed the
    receptive field of the frames that are labelled. Masking lets them do that
    without contributing gradient.

    The mask normalises by its own sum, not by the element count, so a batch of
    mostly-masked windows is not silently scaled down relative to a full one.
    """
    eps = 1e-7
    pw = float(pos_weight)

    def loss(y_true, y_pred, mask=None):
        y_true = tf.clip_by_value(y_true, eps, 1.0 - eps)
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        per = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        if pw != 1.0:
            w = 1.0 + (pw - 1.0) * tf.cast(y_true > 0.5, per.dtype)
            per = per * w
        if mask is None:
            return tf.reduce_mean(per)
        # per is (batch, F, T); the mask is per FRAME, so it broadcasts over F
        m = tf.cast(mask, per.dtype)[:, tf.newaxis, :]
        denom = tf.reduce_sum(m) * tf.cast(tf.shape(per)[1], per.dtype)
        return tf.reduce_sum(per * m) / tf.maximum(denom, eps)

    return loss


def make_distill_loss(gamma=0.0, crop=DISTILL_EDGE_CROP):
    """Real-audio anchor: how far the student's salience has moved from the
    teacher's on the same audio.

    SYMMETRIC by design. An earlier reading of the runs suggested penalising only
    downward deviation, since the failure always looked like compression -- but
    pos_weight=4 then produced +55% INFLATION, so the failure mode is calibration
    drift in either direction and a one-sided penalty would let half of it through.

    `gamma` weights the penalty by the teacher's own confidence, t**gamma:
      gamma=0  uniform -- pin everything equally. Also pins the quiet voices the
               teacher misses, which is exactly what we want to change, so this is
               the conservative baseline, not the ideal.
      gamma>0  pin hard where the teacher reads high, weakly where it reads near
               zero: "keep what you already know, stay free where you were
               unsure". Lets a quiet voice rise without licensing a general
               inflation of the background.

    Both variants are normalised by the mean weight, so the term's scale (and
    hence a given --distill_lambda) does not shift when gamma changes."""
    eps = 1e-7
    g = float(gamma)

    def loss(t_true, y_pred):
        if crop:
            t_true = t_true[:, :, crop:-crop]
            y_pred = y_pred[:, :, crop:-crop]
        t = tf.clip_by_value(t_true, eps, 1.0 - eps)
        p = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        per = -(t * tf.math.log(p) + (1.0 - t) * tf.math.log(1.0 - p))
        if g > 0.0:
            w = tf.pow(t, g)
            return tf.reduce_sum(w * per) / (tf.reduce_sum(w) + eps)
        return tf.reduce_mean(per)

    return loss


def predict_salience(model, mag, dph):
    """Full (F, T) salience for (H, F, T) inputs, chunked over time."""
    x1 = np.transpose(mag, (1, 2, 0))[np.newaxis]   # (1, F, T, H)
    x2 = np.transpose(dph, (1, 2, 0))[np.newaxis]
    T = x1.shape[2]
    out = []
    for t in range(0, T, CHUNK_LEN):
        out.append(model.predict([x1[:, :, t:t+CHUNK_LEN, :],
                                   x2[:, :, t:t+CHUNK_LEN, :]], verbose=0)[0])
    return np.hstack(out)                            # (F, T)


# --------------------------------------------------------------------------
# Validation: invariance gap on matched pairs
# --------------------------------------------------------------------------
def crop_frames_for_chords(tgt, n_chords):
    """Frame index just past the end of the first `n_chords` chords of a cached
    validation target, using the same silent-gap segmentation as training.
    Returns None if the file has no more than n_chords chords (use it whole)."""
    segs = segment_chords(tgt)
    if not segs or n_chords >= len(segs):
        return None
    return segs[n_chords - 1][1]


def prepare_valid(pump, valid_dir, cache_dir, recompute=False):
    """Featurize each validation file ONCE, in full, and cache mag/dph/tgt to
    disk, one npz per file -- same disk-cache pattern (DONE marker, skip unless
    --recompute) as prepare() uses for the training windows.

    The cache is always the complete set; --TEST subsets it at evaluation time
    rather than changing what is cached, so a smoke-test run and a full run
    share one cache."""
    os.makedirs(cache_dir, exist_ok=True)
    done_marker = os.path.join(cache_dir, 'DONE')
    if os.path.exists(done_marker) and not recompute:
        n = len(glob.glob(os.path.join(cache_dir, '*.npz')))
        print("Using cached features for %d validation files in %s" % (n, cache_dir))
        return

    wav_files = sorted(glob.glob(os.path.join(valid_dir, 'valid_*.wav')))
    for wav in wav_files:
        f0 = wav[:-4] + '.f0.csv'
        if not os.path.exists(f0):
            continue
        mag, dph = featurize(pump, wav)
        tgt = build_target(mag.shape[2], f0)
        stem = os.path.basename(wav)[:-len('.wav')]
        np.savez_compressed(os.path.join(cache_dir, '%s.npz' % stem), mag=mag, dph=dph, tgt=tgt)
        print("  cached %s (%d frames)" % (stem, mag.shape[2]))
    open(done_marker, 'w').close()
    print("Cached features for %d validation files in %s" % (len(wav_files), cache_dir))


# --------------------------------------------------------------------------
# Distillation pool: real audio, no annotation.
#
# Every run so far changed the model's real-audio CALIBRATION rather than its
# ability to separate voices: -36% (conv), -22% (bn, pos_weight 1), -8% (bn at
# lr 1e-5) and +55% (bn, pos_weight 4). Different strategies and learning rates,
# opposite signs, one cause -- the loss is measured only on soundfont renders, so
# recalibrating to them is the cheapest way to reduce it, and nothing opposes it.
#
# The fix is a second loss term on real audio. There is no annotation, but none
# is needed: the target is what the PRE-TRAINED model itself predicts for that
# audio, so the term says "do not change here". That is the same reference
# `drift_stats()` already compares against every checkpoint -- this moves it from
# the accept/reject decision into the gradient, where the optimiser can steer
# around the failure instead of stumbling into it and being rejected.
#
# Equivalently: --l2sp anchors WEIGHTS to their pretrained values; this anchors
# OUTPUTS on the input distribution that actually matters. Weight-space distance
# is a poor proxy (and under --strategy bn it anchors nothing at all -- no
# variable is named 'kernel', hence the "L2-SP anchoring 0 kernels" in the log).
# --------------------------------------------------------------------------
_QUIET_MAX = 0.10         # teacher peak below this -> window counts as 'quiet'

# <stem>_w<NNNN>[_q].npz -- stems may contain spaces and underscores, so anchor
# on the _wNNNN suffix rather than splitting on '_'.
_WIN_RE = re.compile(r'^(.*)_w\d{4}(?:_q)?\.npz$')


def list_audio(d):
    """All accepted audio containers in a directory (see AUDIO_EXT)."""
    out = []
    for ext in AUDIO_EXT:
        out.extend(glob.glob(os.path.join(d, ext)))
    return sorted(out)


def audio_stem(path):
    return os.path.basename(path).rsplit('.', 1)[0]


def _file_sig(path):
    st = os.stat(path)
    return [st.st_size, int(st.st_mtime)]


def _cache_stamp(weights, win, hop):
    """Identity of the cached teacher targets. The targets are only valid for the
    weights that produced them, so a cache built from different --weights must be
    rebuilt rather than silently reused."""
    st = os.stat(weights)
    return dict(weights=os.path.abspath(weights), size=st.st_size,
                mtime=int(st.st_mtime), win=win, hop=hop)


def prepare_distill(pump, model, weights, distill_dir, cache_dir, win=50, hop=None,
                    recompute=False, exclude=()):
    """Featurize each real recording, run the CURRENT (still pre-trained) model
    over it whole-file to get the teacher salience, and cache one npz per window
    holding mag/dph/tsal.

    Called before training starts, so `model` still holds the untouched weights
    -- the same trick `base_real` uses. No second model is ever in memory: the
    teacher is frozen by definition, so its outputs are computed once and become
    just another cached target array alongside mag/dph.

    The teacher runs on the WHOLE file and is sliced afterwards, so its targets
    carry no window-boundary artifacts (see DISTILL_EDGE_CROP).

    Windows whose teacher peak is below _QUIET_MAX are tagged '_q' in the
    filename so their share can be capped at load time without reading the npz --
    same filename-encoding trick prepare() uses for chord indices. They are kept
    rather than dropped: a window whose correct answer is "stay near zero" is the
    cheapest available constraint against the upward inflation that pos_weight=4
    produced.

    `exclude` is the set of --real_audio stems. Overlap is a hard error: training
    on the drift-screen files would turn the only held-out signal in the log into
    a training metric."""
    hop = hop or win
    if win <= 2 * DISTILL_EDGE_CROP:
        raise SystemExit(
            "--win %d leaves nothing to distill on: %d frames are cropped from "
            "each end (receptive field), so --win must exceed %d."
            % (win, DISTILL_EDGE_CROP, 2 * DISTILL_EDGE_CROP))
    os.makedirs(cache_dir, exist_ok=True)
    stamp_path = os.path.join(cache_dir, 'DONE')
    stamp = _cache_stamp(weights, win, hop)

    files = list_audio(distill_dir)
    if not files:
        raise SystemExit(
            "--distill_dir %s contains no audio (looked for %s). Put the real "
            "recordings there, or drop --distill_dir to train without the "
            "real-audio anchor." % (distill_dir, ', '.join(AUDIO_EXT)))

    # Overlap check. Exact-basename equality is NOT enough: a guard excerpt is
    # typically CUT from a longer take, so the pool holds 'late.flac' while the
    # screen holds 'late_dada.wav' -- different names, same audio. Training on the
    # parent recording silently turns the drift screen into a training metric,
    # which reports near-zero drift by construction. Comparing stems as prefixes
    # catches that naming pattern; it cannot detect an unrelated filename holding
    # the same audio, so the per-file lines printed below are also there to be
    # audited.
    def stem(p):
        return os.path.basename(p).rsplit('.', 1)[0].lower()

    clash = sorted({"%s <-> %s" % (os.path.basename(f), os.path.basename(e))
                    for f in files for e in exclude
                    if stem(f) == stem(e) or stem(f).startswith(stem(e))
                    or stem(e).startswith(stem(f))})
    if clash:
        raise SystemExit(
            "--distill_dir and --real_audio overlap: %s\n"
            "The --real_audio excerpts are the HELD-OUT drift screen. If the pool "
            "contains that audio (or the longer take it was cut from), the REAL(n) "
            "line stops being a measurement -- the model is trained to preserve "
            "exactly what is then used to check preservation. Move the file out of "
            "the pool. If the names merely look alike and the audio is unrelated, "
            "rename to something that does not share a prefix."
            % '; '.join(clash))

    if os.path.exists(stamp_path) and not recompute:
        try:
            old = json.load(open(stamp_path))
        except ValueError:
            old = None
        if old == stamp:
            print("Using cached distillation targets in %s" % cache_dir)
            return
        print("Distillation cache was built from different settings "
              "(%s) -- rebuilding." % ('weights/window changed' if old else 'no stamp'))

    for p in glob.glob(os.path.join(cache_dir, '*.npz')):
        os.remove(p)

    n_win = n_quiet = 0
    for path in files:
        mag, dph = featurize(pump, path)              # (H, F, T)
        tsal = predict_salience(model, mag, dph)      # (F, T), teacher, whole-file
        stem = os.path.basename(path).rsplit('.', 1)[0]
        T = mag.shape[2]
        fw = 0
        for w, s in enumerate(range(0, T - win + 1, hop)):
            tseg = tsal[:, s:s+win]
            quiet = float(tseg.max()) < _QUIET_MAX
            np.savez_compressed(
                os.path.join(cache_dir, '%s_w%04d%s.npz'
                             % (stem, w, '_q' if quiet else '')),
                mag=np.transpose(mag[:, :, s:s+win], (1, 2, 0)),
                dph=np.transpose(dph[:, :, s:s+win], (1, 2, 0)),
                tsal=tseg)
            fw += 1
            n_win += 1
            n_quiet += quiet
        print("  %s -> %d frames (%.1f s), %d windows%s"
              % (stem, T, T * 256.0 / 22050, fw,
                 '  ! shorter than --win, contributes nothing' if fw == 0 else ''))
    json.dump(stamp, open(stamp_path, 'w'))
    print("Cached %d distillation windows (%d quiet) from %d files, win=%d hop=%d."
          % (n_win, n_quiet, len(files), win, hop))


def load_distill_windows(cache_dir, quiet_cap=0.33, rng=None):
    """Cached window paths, with the share of 'quiet' windows (tagged '_q' by
    prepare_distill) capped at `quiet_cap` of the pool. Quiet windows are
    valuable but must not crowd out windows containing actual singing."""
    wins = sorted(glob.glob(os.path.join(cache_dir, '*.npz')))
    quiet = [p for p in wins if '_q.npz' in p]
    loud = [p for p in wins if '_q.npz' not in p]
    if not loud or quiet_cap >= 1.0:
        return wins
    # keep at most quiet_cap of the final pool
    allowed = int(quiet_cap * len(loud) / max(1e-9, 1.0 - quiet_cap))
    if len(quiet) > allowed:
        idx = (rng or np.random.RandomState(0)).permutation(len(quiet))[:allowed]
        quiet = [quiet[i] for i in sorted(idx)]
        print("  capped quiet windows to %d (%.0f%% of pool)"
              % (len(quiet), 100.0 * len(quiet) / (len(quiet) + len(loud))))
    return sorted(loud + quiet)


# --------------------------------------------------------------------------
# Quiet-voice recall, estimated by undiluting the aggregate victim recall.
#
# All N_VOICES sound in every chord and exactly one is attenuated, so the quiet
# voice contributes only 1/N of the victim file's reference pitches:
#
#     R_victim = ((N-1)/N) * R_loud + (1/N) * R_quiet
#
# Taking R_loud ~= R_balanced (the loud voices sing the same notes in both files
# of a matched pair) and solving for R_quiet:
#
#     R_quiet = N * R_victim - (N-1) * R_balanced
#
# The aggregate gap therefore understates the quiet-voice deficit by a factor N.
# Only recall can be recovered this way: precision is normalised by the ESTIMATE,
# whose pitches cannot be attributed to individual voices.
# --------------------------------------------------------------------------
N_VOICES = 6          # SSATBB -- must match generate_chords.VOICES


def undilute_quiet_recall(recall_balanced, recall_victim, n_voices=N_VOICES):
    return n_voices * recall_victim - (n_voices - 1) * recall_balanced


def _load_valid_features(cache_dir, wav):
    stem = os.path.basename(wav)[:-len('.wav')]
    d = np.load(os.path.join(cache_dir, '%s.npz' % stem))
    return d['mag'], d['dph'], d['tgt']


def _eval_file(model, cache_dir, wav, f0_csv, thresh, loss_fn, n_chords=None):
    """Return (recall, precision, val_loss) for one file. val_loss is the same
    (bkld) loss used in training, computed on the inference-mode prediction vs.
    the target. mag/dph/tgt come from the on-disk validation cache, which always
    holds the WHOLE file; `n_chords` subsets it to the first N chords here."""
    import mir_eval
    mag, dph, tgt = _load_valid_features(cache_dir, wav)
    n_full = tgt.shape[1]

    cut = crop_frames_for_chords(tgt, n_chords) if n_chords is not None else None
    if cut is not None:
        mag, dph, tgt = mag[:, :, :cut], dph[:, :, :cut], tgt[:, :cut]

    sal = predict_salience(model, mag, dph)          # (F, T), BN in inference mode
    loss = float(loss_fn(tf.constant(tgt[np.newaxis]), tf.constant(sal[np.newaxis])))
    est_t, est_f = utils_train.pitch_activations_to_mf0(sal, thresh)

    ref_t, ref_f = load_ragged_f0(f0_csv)
    if cut is not None:
        max_time = utils.get_time_grid(n_full)[cut - 1]
        keep = ref_t <= max_time
        ref_t, ref_f = ref_t[keep], [f for f, k in zip(ref_f, keep) if k]

    m = mir_eval.multipitch.evaluate(ref_t, ref_f, np.array(est_t), est_f)
    return m['Recall'], m['Precision'], loss


def evaluate_invariance(model, valid_dir, cache_dir, thresh, loss_fn, n_chords=None):
    """For each matched pair, score balanced vs victim (same notes).

    Returns mean recall/precision/loss for both sides, the aggregate invariance
    gap, and the undiluted quiet-voice figures `recall_quiet` / `gap_quiet`
    estimated via undilute_quiet_recall() -- the quantity the fine-tuning
    actually targets. `n_chords` caps each file to its first N chords."""
    bal_files = sorted(glob.glob(os.path.join(valid_dir, 'valid_*_balanced.wav')))
    rb, rv, pb, pv, lb, lv = [], [], [], [], [], []
    for bwav in bal_files:
        idx = os.path.basename(bwav).split('_')[1]
        vic = glob.glob(os.path.join(valid_dir, 'valid_%s_victim*.wav' % idx))
        if not vic:
            continue
        recall_b, prec_b, loss_b = _eval_file(model, cache_dir, bwav,
                                              bwav[:-4] + '.f0.csv', thresh, loss_fn, n_chords)
        recall_v, prec_v, loss_v = _eval_file(model, cache_dir, vic[0],
                                              vic[0][:-4] + '.f0.csv', thresh, loss_fn, n_chords)
        rb.append(recall_b); rv.append(recall_v); pb.append(prec_b); pv.append(prec_v)
        lb.append(loss_b); lv.append(loss_v)
    if not rb:
        return None
    recall_balanced, recall_victim = np.mean(rb), np.mean(rv)
    recall_quiet = undilute_quiet_recall(recall_balanced, recall_victim)
    return dict(recall_balanced=recall_balanced, recall_victim=recall_victim,
                gap=recall_balanced - recall_victim,
                recall_quiet=recall_quiet,
                gap_quiet=recall_balanced - recall_quiet,
                precision_balanced=np.mean(pb), precision_victim=np.mean(pv),
                loss_balanced=np.mean(lb), loss_victim=np.mean(lv))


def evaluate_real(model, valid_dir, thresh, loss_fn, n_chords=None):
    """Validate directly on real-audio windows (mag/dph/tgt/mask), as cached by
    prepare_real_chords.py in valid_dir -- one npz per window, already fixed
    length, no whole-file cache and no wav to re-featurize.

    Real single-take recordings have no matched balanced/victim pair (there is
    no quiet-one-voice render of the same performance), so there is no
    invariance gap to measure here. This instead reports plain recall/
    precision/loss over the SUPERVISED frames only (mask == 1): the score-
    derived attack/release/reverb margins that prepare_real_chords.py masked
    out have no defensible target, and scoring them would just add noise to
    the comparison against baseline."""
    import mir_eval
    wins = sorted(glob.glob(os.path.join(valid_dir, '*.npz')))
    if n_chords is not None:
        wins = select_first_chords(wins, n_chords)
    if not wins:
        return None
    recalls, precisions, losses = [], [], []
    for p in wins:
        with np.load(p) as d:
            mag, dph, tgt, mask = d['mag'], d['dph'], d['tgt'], d['mask']
        if mask.sum() == 0:
            continue
        sal = model.predict([mag[np.newaxis], dph[np.newaxis]], verbose=0)[0]
        loss = float(loss_fn(tf.constant(tgt[np.newaxis]), tf.constant(sal[np.newaxis]),
                             tf.constant(mask[np.newaxis].astype(np.float32))))
        losses.append(loss)

        times = utils.get_time_grid(mask.shape[0])
        freq_grid = utils.get_freq_grid()
        _, est_freqs = utils_train.pitch_activations_to_mf0(sal, thresh)
        ref_freqs = [freq_grid[tgt[:, t] > 0.5] for t in range(mask.shape[0])]
        idx = np.where(mask > 0.5)[0]
        m = mir_eval.multipitch.evaluate(times[idx], [ref_freqs[i] for i in idx],
                                         times[idx], [est_freqs[i] for i in idx])
        recalls.append(m['Recall']); precisions.append(m['Precision'])
    if not recalls:
        return None
    return dict(recall=np.mean(recalls), precision=np.mean(precisions),
                loss=np.mean(losses))


def guard_failures_real(inv, baseline, args):
    """Real-audio counterpart of guard_failures(): no balanced/victim sides or
    invariance gap to guard here, just recall/precision vs. the pre-training
    baseline, plus the same real-audio drift screen (only present when
    --real_audio was given, same as guard_failures())."""
    if baseline is None:
        return []
    failed = []
    for key in ('recall', 'precision'):
        if inv[key] < baseline[key] - args.bal_tol:
            failed.append("%s %.3f<%.3f" % (key, inv[key], baseline[key] - args.bal_tol))
    if 'real_rel_mean' in inv:
        if inv['real_worst'] < -args.drift_tol:
            failed.append("real mean %+.0f%%" % (100 * inv['real_worst']))
        if inv['real_d_high'] < -args.drift_high_tol:
            failed.append("real d@high %.3f" % inv['real_d_high'])
    return failed


def format_metrics_real(m):
    return ("val_loss=%.4f  recall=%.3f  precision=%.3f"
            % (m['loss'], m['recall'], m['precision']))


# --------------------------------------------------------------------------
# Real-audio drift screen (no annotation required)
# --------------------------------------------------------------------------
# Training on synthetic audio can silently recalibrate the model to the
# synthetic amplitude distribution. When that happens the salience map on REAL
# audio is compressed downward -- confident activations pulled down hardest --
# which destroys detections while the synthetic validation above still reports
# an improvement. 
#
# Catching it needs no MIDI, no alignment, no tuning correction and no ground
# truth: run the current weights over a few seconds of real audio and compare
# the salience map with the one the PRE-TRAINED model produced for the same
# file. Both come from the same audio, so they line up bin for bin.
def drift_stats(base_sals, sals):
    """Compare candidate salience maps against the pre-training ones, per real
    excerpt, and average. Several short excerpts are much better than one long
    one here: drift shows up as a consistent shift across independent material,
    and a single file can mislead.

    `real_rel_mean` is the overall level change. `real_d_high` is the change
    where the baseline reads 0.80-0.90; if it is much more negative than the
    overall mean change, the model is compressing its confident activations
    downward -- the signature of having adapted to the synthetic distribution.
    `real_worst` is the least favourable per-file mean change, so one bad
    excerpt cannot be averaged away."""
    per = []
    for b, s in zip(base_sals, sals):
        hi = (b >= 0.80) & (b < 0.90)
        per.append((s.mean() / b.mean() - 1.0 if b.mean() else 0.0,
                    float((s[hi] - b[hi]).mean()) if hi.sum() >= 50 else 0.0,
                    float(np.corrcoef(b.ravel().astype(np.float64),
                                      s.ravel().astype(np.float64))[0, 1])))
    a = np.mean(per, axis=0)
    return dict(real_rel_mean=float(a[0]), real_d_high=float(a[1]),
                real_r=float(a[2]), real_worst=float(min(p[0] for p in per)),
                real_n=len(per))


def format_drift(m):
    s = ("REAL(%d) mean %+.1f%%  d@high %+.3f  r=%.2f"
         % (m['real_n'], 100 * m['real_rel_mean'], m['real_d_high'], m['real_r']))
    if m['real_n'] > 1:
        s += "  worst %+.1f%%" % (100 * m['real_worst'])
    return s


def format_train_loss(pairs, distilling):
    """Mean (synthetic, distillation) training loss over a list of per-step pairs.
    The two terms are reported SEPARATELY, never summed: the whole point is to
    watch the trade-off between fitting the synthetic chords and holding the
    real-audio calibration, and a single total hides it."""
    a = np.mean(pairs, axis=0)
    if not distilling:
        return "train_loss=%.4f" % a[0]
    return "train_loss=%.4f distill=%.4f" % (a[0], a[1])


def format_metrics(m):
    return ("val_loss bal=%.4f victim=%.4f  recall bal=%.3f victim=%.3f  GAP=%.3f  "
            "prec bal=%.3f victim=%.3f  | quiet recall=%.3f gap=%.3f"
            % (m['loss_balanced'], m['loss_victim'], m['recall_balanced'],
               m['recall_victim'], m['gap'], m['precision_balanced'],
               m['precision_victim'], m['recall_quiet'], m['gap_quiet']))


def guard_failures(inv, baseline, args):
    """Names of the guards this epoch violates vs. the pre-training baseline.
    Empty list = the epoch is eligible to be saved.

    Recall AND precision are guarded on BOTH sides: raising quiet-voice recall
    by flooding the salience map with false positives would otherwise look like
    progress. The gap guard additionally refuses epochs that improve the quiet
    voice by less than they improve the loud ones."""
    if baseline is None:
        return []
    failed = []
    for key, tol in (('recall_balanced', args.bal_tol),
                     ('precision_balanced', args.bal_tol),
                     ('recall_victim', args.bal_tol),
                     ('precision_victim', args.bal_tol)):
        if inv[key] < baseline[key] - tol:
            failed.append("%s %.3f<%.3f" % (key, inv[key], baseline[key] - tol))
    if inv['gap_quiet'] > baseline['gap_quiet'] + args.gap_tol:
        failed.append("gap_quiet %.3f>%.3f"
                      % (inv['gap_quiet'], baseline['gap_quiet'] + args.gap_tol))
    # Real-audio drift: a synthetic win bought by wrecking the real-audio
    # calibration is not a win. Only checked when --real_audio was given.
    if 'real_rel_mean' in inv:
        if inv['real_worst'] < -args.drift_tol:
            failed.append("real mean %+.0f%%" % (100 * inv['real_worst']))
        if inv['real_d_high'] < -args.drift_high_tol:
            failed.append("real d@high %.3f" % inv['real_d_high'])
    return failed


# --------------------------------------------------------------------------
# Train
# --------------------------------------------------------------------------
def train(args):
    # Keras 3 requires weight files to end in .weights.h5
    if not args.out.endswith('.weights.h5'):
        args.out = (args.out[:-3] if args.out.endswith('.h5') else args.out) + '.weights.h5'
        print("Adjusted output weights path to %s (Keras 3 requirement)" % args.out)

    # Tag the output with what produced it: the adaptation strategy, plus TEST
    # for a smoke run (whose weights, trained on a handful of chords, are not a
    # usable model). Keeps strategies and smoke tests from overwriting each
    # other's results; the log filename follows suit.
    suffix = '_' + args.strategy + ('_TEST' if args.TEST is not None else '')
    args.out = args.out[:-len('.weights.h5')] + suffix + '.weights.h5'

    log_path = setup_logging(args.out, args)
    print("Output weights: %s" % args.out)
    if args.TEST is not None:
        print("--TEST run: output marked TEST -- not a usable model")
    print("Logging to %s" % log_path)

    pump = utils.create_pump_object()

    # The caches always hold the COMPLETE set; --TEST subsets them below, so a
    # smoke run and a full run share one cache.
    cache_dir = os.path.join(args.train_dir, '_cache')
    win_files = prepare(pump, args.train_dir, cache_dir,
                        win=args.win, hop=args.win_hop, recompute=args.recompute)
    if not win_files:
        raise SystemExit("No windows to train on. Render the MIDIs to wav first.")
    if args.TEST is not None:
        n_all = len(win_files)
        win_files = select_first_chords(win_files, args.TEST)
        if not win_files:
            raise SystemExit("--TEST %d selected no windows." % args.TEST)
        print("--TEST: first %d chord(s) per file -> %d of %d windows"
              % (args.TEST, len(win_files), n_all))

    # Real-audio validation windows (prepare_real_chords.py) are already fully
    # featurized, fixed-length npz sitting directly in valid_dir -- there is no
    # whole-file cache to build and no balanced/victim pair to match, so that
    # whole prepare_valid() step is skipped for them.
    valid_cache_dir = None
    real_valid = args.valid_dir and is_prebuilt_cache(args.valid_dir)
    if args.valid_dir and not real_valid:
        valid_cache_dir = os.path.join(args.valid_dir, '_cache')
        prepare_valid(pump, args.valid_dir, valid_cache_dir, recompute=args.recompute)
    if args.valid_dir and args.TEST is not None:
        print("--TEST: validating on the first %d chord(s) of each file" % args.TEST)

    # Real-audio drift screen: featurise once, up front. A few seconds is plenty
    # -- this is a distribution check, not an accuracy measurement.
    real_feat = []
    for path in (args.real_audio or []):
        path = os.path.expanduser(path)
        rm, rd = featurize(pump, path)
        real_feat.append((rm, rd))
        print("Real-audio drift screen: %s (%d frames, %.1f s)"
              % (os.path.basename(path), rm.shape[2], rm.shape[2] * 256.0 / 22050))

    model = build_model(args.weights, args.strategy)
    opt = tf.keras.optimizers.Adam(learning_rate=args.lr)
    loss_fn = make_bkld(args.pos_weight)
    # pos_weight is deliberately NOT applied to the anchor: its 'y_true > 0.5'
    # test would upweight bins wherever the TEACHER happened to be confident,
    # which is unrelated to the annotated-voice reweighting it exists for.
    distill_fn = make_distill_loss(args.distill_gamma)

    # L2-SP: snapshot pretrained conv/dense kernels so we can penalise deviation
    # from them (anchors 'full' fine-tuning against forgetting).
    anchors = []
    if args.l2sp > 0:
        anchors = [(v, tf.constant(v.numpy()))
                   for v in model.trainable_variables if 'kernel' in v.name]
        print("L2-SP anchoring %d kernels (lambda=%g)" % (len(anchors), args.l2sp))

    @tf.function
    def train_step(x1, x2, y, rx1=None, rx2=None, rt=None, ymask=None):
        with tf.GradientTape() as tape:
            loss_s = loss_fn(y, model([x1, x2], training=True), ymask)
            loss = loss_s
            loss_r = tf.constant(0.0)
            if rx1 is not None:
                # Separate forward pass, NOT one concatenated batch: BatchNorm in
                # training mode normalises by batch statistics, and a mixed
                # synthetic/real batch would give a blended statistic matching
                # neither domain. Under --strategy bn this second pass is doing
                # double duty -- it also puts real audio into the running-mean/
                # variance updates, which is what stops them converging onto the
                # soundfont distribution in the first place.
                loss_r = distill_fn(rt, model([rx1, rx2], training=True))
                loss = loss + args.distill_lambda * loss_r
            if anchors:
                loss = loss + args.l2sp * tf.add_n(
                    [tf.reduce_sum(tf.square(v - v0)) for v, v0 in anchors])
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss_s, loss_r

    # Pre-training baseline so we can require the other metrics not to regress.
    if real_valid:
        baseline = evaluate_real(model, args.valid_dir, args.thresh, loss_fn,
                                 n_chords=args.TEST)
    elif args.valid_dir:
        baseline = evaluate_invariance(model, args.valid_dir, valid_cache_dir, args.thresh,
                                       loss_fn, n_chords=args.TEST)
    else:
        baseline = None
    if baseline is not None:
        print("baseline     | " + (format_metrics_real(baseline) if real_valid
                                   else format_metrics(baseline)))

    # Pre-training salience on the real excerpt: the reference every epoch is
    # compared against. Taken from the untouched weights, so it is the model's
    # own real-audio behaviour before any synthetic data was seen.
    base_real = [predict_salience(model, m, d) for m, d in real_feat]

    # Distillation pool. Built HERE, after the baseline and base_real and before
    # the first optimizer step, so `model` still holds the pretrained weights and
    # can act as its own teacher -- no second copy in memory.
    rng = np.random.RandomState(args.seed)
    distill_files, next_real = [], None
    if args.distill_dir:
        distill_cache = os.path.join(args.distill_dir, '_cache')
        prepare_distill(pump, model, args.weights, args.distill_dir, distill_cache,
                        win=args.win, hop=args.win_hop, recompute=args.recompute,
                        exclude=[os.path.expanduser(p) for p in (args.real_audio or [])])
        # Its OWN RandomState, deliberately not `rng`. If the distillation pool
        # drew from the same generator it would shift every subsequent synthetic
        # epoch shuffle, so a --distill_lambda 0 ablation would not see the same
        # synthetic ordering as a --distill_lambda 1 run and the comparison would
        # be confounded by data order.
        drng = np.random.RandomState(args.seed + 1)
        distill_files = load_distill_windows(distill_cache, args.distill_quiet_cap, drng)
        if not distill_files:
            raise SystemExit("Distillation cache is empty: %s" % distill_cache)
        print("Distilling on %d real windows (%.1f min), lambda=%g gamma=%g, "
              "edge crop %d frames"
              % (len(distill_files), len(distill_files) * args.win * 256.0 / 22050 / 60,
                 args.distill_lambda, args.distill_gamma, DISTILL_EDGE_CROP))

        def make_cycler(files, batch, r):
            """Draw `batch` window paths per call, reshuffling when the pool is
            exhausted. The real pool is cycled independently of the synthetic one,
            so the two need not be the same size; epoch length stays defined by
            the synthetic set."""
            order, pos = [list(r.permutation(len(files)))], [0]

            def nxt():
                out = []
                while len(out) < batch:
                    if pos[0] >= len(order[0]):
                        order[0] = list(r.permutation(len(files)))
                        pos[0] = 0
                    out.append(files[order[0][pos[0]]])
                    pos[0] += 1
                return out
            return nxt

        next_real = make_cycler(distill_files, args.batch_size, drng)

    bs = args.batch_size
    # Seed the ranking with the PRE-TRAINING quiet-voice recall, so a checkpoint
    # has to beat the base model to be saved -- not merely survive the guards.
    # The guards are tolerances (--bal_tol etc.), so a checkpoint slightly worse
    # than baseline on the objective still passes them; starting from None meant
    # the first such checkpoint was written to --out unconditionally and the run
    # could ship a model worse than the one it started from.
    # Real-audio validation has no invariance gap to target, so it ranks
    # checkpoints on plain recall instead of the undiluted quiet-voice figure.
    rank_key = 'recall' if real_valid else 'recall_quiet'
    state = dict(best_rank=baseline[rank_key] if baseline else None,
                 saved=None)
    history = []

    def assess(tag, label):
        """Checkpoint + evaluate + guard the CURRENT weights, and update the
        best-so-far. Called at the end of every epoch and, when --eval_every is
        set, part-way through one as well.

        Sub-epoch evaluation exists because the useful movement and the
        real-audio drift do not happen on the same timescale: both recorded runs
        show the synthetic metrics converged and the salience already collapsed
        by the end of epoch 1, so epoch granularity cannot show where the two
        separate. `tag` names the checkpoint file, `label` opens the log line."""
        msg = label

        # Keep every checkpoint so it can be chosen AFTER the run, from real
        # audio, instead of only by the synthetic in-loop metric. ~5 MB each.
        if args.save_every_epoch:
            ck_path = args.out[:-len('.weights.h5')] + '_%s.weights.h5' % tag
            model.save_weights(ck_path)
            msg += "  [-> %s]" % os.path.basename(ck_path)

        drift = drift_stats(base_real,
                            [predict_salience(model, m, d) for m, d in real_feat]) \
            if base_real else {}
        if drift:
            print("  " + format_drift(drift))

        if args.valid_dir:
            if real_valid:
                inv = evaluate_real(model, args.valid_dir, args.thresh, loss_fn,
                                    n_chords=args.TEST)
            else:
                inv = evaluate_invariance(model, args.valid_dir, valid_cache_dir, args.thresh,
                                          loss_fn, n_chords=args.TEST)
            if inv is not None:
                inv.update(drift)
                history.append(dict(tag=tag, **inv))
                msg += "  | " + (format_metrics_real(inv) if real_valid
                                 else format_metrics(inv))
                if drift:
                    msg += "  | " + format_drift(drift)
                # Rank checkpoints on the UNDILUTED quiet-voice recall for synthetic
                # validation (the actual objective there), or plain recall for real
                # validation, and refuse any that pays for it by regressing recall
                # or precision (both sides, for synthetic), or by widening the gap.
                failed = (guard_failures_real(inv, baseline, args) if real_valid
                         else guard_failures(inv, baseline, args))
                if failed:
                    msg += "  [rejected: %s]" % ", ".join(failed)
                elif state['best_rank'] is None or inv[rank_key] > state['best_rank']:
                    improved = (baseline is None or
                                inv[rank_key] - baseline[rank_key])
                    state['best_rank'] = inv[rank_key]
                    state['saved'] = tag
                    model.save_weights(args.out)
                    msg += ("  [saved best%s]" % ('' if baseline is None
                                                  else ', %+.3f vs baseline' % improved))
        else:
            model.save_weights(args.out)
        print(msg)

    step = 0
    for epoch in range(args.epochs):
        order = rng.permutation(len(win_files))
        losses = []
        since_eval = []
        n_batches = (len(order) + bs - 1) // bs
        for bi, b in enumerate(range(0, len(order), bs)):
            batch = [np.load(win_files[i]) for i in order[b:b+bs]]
            x1 = tf.convert_to_tensor(np.stack([d['mag'] for d in batch]), tf.float32)
            x2 = tf.convert_to_tensor(np.stack([d['dph'] for d in batch]), tf.float32)
            y = tf.convert_to_tensor(np.stack([d['tgt'] for d in batch]), tf.float32)
            # Windows built by prepare_real_chords.py carry a per-frame mask;
            # the synthetic ones from prepare() do not, and are fully labelled.
            ymask = None
            if all('mask' in d for d in batch):
                ymask = tf.convert_to_tensor(
                    np.stack([d['mask'] for d in batch]), tf.float32)
            if next_real is not None:
                rb = [np.load(p) for p in next_real()]
                ls, lr = train_step(
                    x1, x2, y,
                    tf.convert_to_tensor(np.stack([d['mag'] for d in rb]), tf.float32),
                    tf.convert_to_tensor(np.stack([d['dph'] for d in rb]), tf.float32),
                    tf.convert_to_tensor(np.stack([d['tsal'] for d in rb]), tf.float32),
                    ymask)
            else:
                ls, lr = train_step(x1, x2, y, ymask=ymask)
            losses.append((float(ls), float(lr)))
            since_eval.append(losses[-1])
            step += 1
            print("\r  batch %d/%d  loss=%.4f%s"
                  % (bi + 1, n_batches, losses[-1][0],
                     '  distill=%.4f' % losses[-1][1] if next_real else ''),
                  end='', flush=True)
            # Mid-epoch checkpoint. Skipped on the final batch of an epoch,
            # where it would duplicate the end-of-epoch one below.
            if args.eval_every and step % args.eval_every == 0 and bi + 1 < n_batches:
                print()
                assess('e%02d_s%06d' % (epoch + 1, step),
                       "  step %d (epoch %d, batch %d/%d)  %s"
                       % (step, epoch + 1, bi + 1, n_batches,
                          format_train_loss(since_eval, next_real is not None)))
                since_eval = []
        print()

        assess('e%02d' % (epoch + 1),
               "epoch %d/%d  %s"
               % (epoch + 1, args.epochs,
                  format_train_loss(losses, next_real is not None)))
    rank_label = 'recall' if real_valid else 'quiet-voice recall'
    if not args.valid_dir:
        print("Saved final weights to %s" % args.out)
    elif state['saved'] is None:
        # Deliberately do NOT fall back to the final-epoch weights: nothing beat
        # the pre-trained model on the objective, so writing anything to --out
        # would ship a regression under a name that implies an improvement. The
        # per-checkpoint files are still on disk if the run is worth salvaging.
        print("No checkpoint beat the baseline %s (%.3f) within "
              "the guards; %s NOT written -- use %s instead."
              % (rank_label, state['best_rank'], args.out, os.path.basename(args.weights)))
    else:
        print("Best (highest %s, guards satisfied) weights saved "
              "to %s -- from %s, %s %.3f vs baseline %.3f"
              % (rank_label, args.out, state['saved'], rank_label, state['best_rank'],
                 baseline[rank_key]))

    if history:
        print("\ncheckpoint summary")
        w = max(len(h['tag']) for h in history)
        fmt = format_metrics_real if real_valid else format_metrics
        for h in history:
            print("  %-*s | %s" % (w, h['tag'], fmt(h)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--train_dir', required=True,
                   help='dir with train_*.wav + .f0.csv, OR a prebuilt window '
                        'cache (DONE + *.npz directly in the dir) such as '
                        'prepare_real_chords.py writes')
    p.add_argument('--valid_dir', default=None,
                   help='dir with the matched valid_*_balanced/victim pairs, OR '
                        'a prebuilt real-audio window cache (DONE + *.npz), same '
                        'as --train_dir')
    p.add_argument('--weights', default='./models/exp3multif0.h5',
                   help='model3 weights to fine-tune from')
    p.add_argument('--out', default='./models/exp3multif0_finetuned.weights.h5',
                   help='where to write fine-tuned weights (must end .weights.h5)')
    p.add_argument('--recompute', action='store_true',
                   help='rebuild the feature caches (<train_dir>/_cache and <valid_dir>/_cache)')
    p.add_argument('--TEST', type=int, default=None,
                   help='smoke-test on the first N chords of every file, for both '
                        'training and validation. The caches are still built in full; '
                        'only their use is subset, so a TEST run and a full run share '
                        'one cache. Chords are randomly generated, so the first N are '
                        'representative.')

    p.add_argument('--no_save_every_epoch', dest='save_every_epoch',
                   action='store_false',
                   help='by default every checkpoint is also written to '
                        '<out>_eNN[_sNNNNNN].weights.h5 (~5 MB each), so it can be '
                        'selected afterwards from real audio rather than only by the '
                        'synthetic in-loop metric -- screen them with '
                        'finetune/screen_checkpoints.py. This switch turns that off.')
    p.add_argument('--eval_every', type=int, default=0, metavar='STEPS',
                   help='also checkpoint, validate and guard every STEPS optimizer '
                        'steps within an epoch (0 = at epoch boundaries only). Both '
                        'recorded runs converged on the synthetic metrics AND lost '
                        '~36%% of their real-audio salience inside epoch 1, so epoch '
                        'granularity cannot show where the quiet-voice gain and the '
                        'drift separate. Each evaluation costs a full pass over the '
                        'validation pair plus the --real_audio excerpts, so set this '
                        'to a fraction of an epoch (e.g. 1/10th of the batch count), '
                        'not to a handful of steps.')

    p.add_argument('--win', type=int, default=50, help='training window length (frames)')
    p.add_argument('--win_hop', type=int, default=None,
                   help='window stride in frames (default: --win, i.e. no overlap)')
    p.add_argument('--batch_size', type=int, default=10,
                   help='windows per step. Keep small on CPU: the (360,1) distribution '
                        'layer backprop scales with batch*win (batch=1 ~1.6GB at win=50).')
    p.add_argument('--epochs', type=int, default=6)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--thresh', type=float, default=0.5, help='peak threshold for eval')
    p.add_argument('--seed', type=int, default=0)

    p.add_argument('--strategy', choices=['bn', 'conv', 'full'], default='conv',
                   help="which weights adapt. 'conv' (default) = conv/dense weights, "
                        "BatchNorm frozen -- the normalisation stays fitted to real "
                        "audio, so it cannot drift to the synthetic distribution; "
                        "pair with --l2sp. 'bn' = AdaBN recalibration only (cheap, "
                        "but adapts exactly the statistics that cause that drift). "
                        "'full' = everything, which drifts too since --l2sp cannot "
                        "anchor BatchNorm's running statistics.")
    p.add_argument('--l2sp', type=float, default=1e-3,
                   help='L2-SP anchor strength for full fine-tuning (0 disables). '
                        'Penalises deviation of conv kernels from pretrained values.')
    p.add_argument('--distill_dir', default=None, metavar='DIR',
                   help='directory of REAL recordings (%s), no annotation needed, '
                        'used as a second loss term: the pre-trained model is run '
                        'over them once and the student is penalised for moving '
                        'away from its output. This is --l2sp measured on OUTPUTS '
                        'instead of weights, and it is the same quantity the '
                        'REAL(n) drift line reports -- moved from the accept/reject '
                        'decision into the gradient. Must be DISJOINT from '
                        '--real_audio, which stays the held-out screen. Diversity '
                        'matters more than duration: 8 recordings x 3 min beats one '
                        '25 min file.' % ', '.join(e[1:] for e in AUDIO_EXT))
    p.add_argument('--distill_lambda', type=float, default=1.0,
                   help='weight of the real-audio anchor relative to the synthetic '
                        'loss (default 1.0; 0 disables the term but STILL passes '
                        'real audio through the network, which under --strategy bn '
                        'alone stops the BatchNorm running statistics converging '
                        'onto the synthetic distribution -- a useful ablation)')
    p.add_argument('--distill_gamma', type=float, default=0.0,
                   help='confidence-weight the anchor by teacher_salience**gamma. '
                        '0 (default) = uniform, which also pins the quiet voices '
                        'the teacher misses. >0 pins hard where the teacher is '
                        'confident and leaves the student free where it read near '
                        'zero: "keep what you know, stay free where you were '
                        'unsure". Try 1.0 once a uniform run has given a baseline.')
    p.add_argument('--distill_quiet_cap', type=float, default=0.33,
                   help='max share of the distillation pool made up of near-silent '
                        'windows (teacher peak < %.2f). These are kept rather than '
                        'dropped -- a window whose correct answer is "stay near '
                        'zero" is the cheapest constraint against upward salience '
                        'inflation -- but must not crowd out actual singing.'
                        % _QUIET_MAX)
    p.add_argument('--pos_weight', type=float, default=1.0,
                   help='loss upweight on annotated (voice) target bins (1.0 = off)')
    p.add_argument('--bal_tol', type=float, default=0.03,
                   help='max allowed regression, vs the pre-training baseline, of '
                        'recall and precision on BOTH the balanced and victim sides '
                        'when selecting the best epoch')
    p.add_argument('--real_audio', nargs='+', default=None, metavar='WAV',
                   help='one or more REAL recordings (no annotation needed). Each '
                        'epoch their salience maps are compared with the ones the '
                        'pre-trained weights produced for the same files, catching '
                        'the case where fine-tuning recalibrates the model to the '
                        'synthetic distribution -- which the synthetic validation '
                        'above cannot see. Several short excerpts beat one long one: '
                        'a consistent shift across independent material is the '
                        'signal. Check the line after epoch 1 and abort if the '
                        'salience has already collapsed.')
    p.add_argument('--drift_tol', type=float, default=0.10,
                   help='reject an epoch if mean salience on the WORST --real_audio '
                        'file falls more than this fraction below the pre-trained '
                        'model (default 0.10 = 10%%). Worst rather than mean, so one '
                        'bad excerpt cannot be averaged away.')
    p.add_argument('--drift_high_tol', type=float, default=0.15,
                   help='reject an epoch whose salience on --real_audio drops more '
                        'than this where the pre-trained model reads 0.80-0.90, '
                        'averaged over files. This is the signature of downward '
                        'compression (default 0.15).')
    p.add_argument('--gap_tol', type=float, default=0.0,
                   help='max allowed widening of the undiluted quiet-voice gap '
                        '(recall_balanced - recall_quiet) vs baseline. 0 = the gap '
                        'must not get worse than it already was.')

    train(p.parse_args())
