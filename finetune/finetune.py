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
               amplitude distribution). Cheapest; try first. Note BN pools its
               statistics over the whole window, so a narrow quiet-voice sub-band
               is diluted -- bn alone may be insufficient (hence full below).
      - full : fine-tune all layers so the early harmonic detectors themselves
               learn to keep a quiet voice above threshold. Pair with --l2sp to
               anchor weights to their pretrained values (L2-SP, Xuhong et al.
               2018) so normal-balance performance is not erased.
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
  * Epoch selection VALIDATES BOTH SIDES on the matched pair (balanced vs.
    one-voice-quiet). Because all 6 voices sound in every chord and only one is
    attenuated, the raw victim recall is diluted 6x; selection therefore ranks
    epochs on the UNDILUTED quiet-voice recall
        R_quiet = 6*R_victim - 5*R_balanced
    guarded so the win is not bought elsewhere: recall and precision on both
    sides must not regress past --bal_tol vs. the pre-training baseline, and the
    quiet-voice gap must not widen past --gap_tol. Small LR, few epochs.
  * Progress and per-epoch results are teed to <out>.log, so a long run's
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
        self.log = open(path, 'a', buffering=1)

    def write(self, s):
        self.terminal.write(s)
        if not s.startswith('\r'):
            self.log.write(s)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def setup_logging(out_weights, args):
    """Start teeing stdout to <out_weights minus .weights.h5>.log (appended, so
    successive runs accumulate). Returns the log path."""
    path = out_weights[:-len('.weights.h5')] + '.log'
    sys.stdout = Tee(path)
    print("\n" + "=" * 78)
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
def prepare(pump, train_dir, cache_dir, win=50, hop=None, recompute=False):
    """Featurize each file, slice chords, cut chords into fixed `win`-frame
    windows (stride `hop`, default `win` = no overlap), cache one npz per
    window. Fixed length bounds the distribution-layer backprop memory and lets
    windows be batched.

    Window files are named <stem>_c<chord>_w<window>.npz so a subset of chords
    can be selected later (see --TEST) without reading the files."""
    hop = hop or win
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
    """Keep only windows belonging to the first `n_chords` chords OF EACH FILE
    (chord index is encoded in the filename by prepare())."""
    kept = []
    for p in win_files:
        m = _CHORD_RE.search(os.path.basename(p))
        if m is not None and int(m.group(1)) < n_chords:
            kept.append(p)
    return kept


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def set_trainable(model, strategy):
    """strategy in {'bn', 'full'}. 'head' is intentionally absent: a quiet voice
    is lost before the head, so adapting only the head cannot recover it."""
    if strategy == 'full':
        for layer in model.layers:
            layer.trainable = True
    elif strategy == 'bn':
        # AdaBN: only BatchNorm adapts (weights frozen). trainable=True keeps BN
        # in training mode so its running stats recalibrate to the new amplitudes.
        for layer in model.layers:
            layer.trainable = isinstance(layer, tf.keras.layers.BatchNormalization)
    else:
        raise ValueError("unknown strategy %r (use 'bn' or 'full')" % strategy)


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
    (annotated voice) target bins by pos_weight."""
    eps = 1e-7
    pw = float(pos_weight)

    def loss(y_true, y_pred):
        y_true = tf.clip_by_value(y_true, eps, 1.0 - eps)
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        per = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        if pw != 1.0:
            w = 1.0 + (pw - 1.0) * tf.cast(y_true > 0.5, per.dtype)
            per = per * w
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

    valid_cache_dir = None
    if args.valid_dir:
        valid_cache_dir = os.path.join(args.valid_dir, '_cache')
        prepare_valid(pump, args.valid_dir, valid_cache_dir, recompute=args.recompute)
        if args.TEST is not None:
            print("--TEST: validating on the first %d chord(s) of each file" % args.TEST)

    model = build_model(args.weights, args.strategy)
    opt = tf.keras.optimizers.Adam(learning_rate=args.lr)
    loss_fn = make_bkld(args.pos_weight)

    # L2-SP: snapshot pretrained conv/dense kernels so we can penalise deviation
    # from them (anchors 'full' fine-tuning against forgetting).
    anchors = []
    if args.l2sp > 0:
        anchors = [(v, tf.constant(v.numpy()))
                   for v in model.trainable_variables if 'kernel' in v.name]
        print("L2-SP anchoring %d kernels (lambda=%g)" % (len(anchors), args.l2sp))

    @tf.function
    def train_step(x1, x2, y):
        with tf.GradientTape() as tape:
            pred = model([x1, x2], training=True)
            loss = loss_fn(y, pred)
            if anchors:
                loss = loss + args.l2sp * tf.add_n(
                    [tf.reduce_sum(tf.square(v - v0)) for v, v0 in anchors])
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    # Pre-training baseline so we can require the other metrics not to regress.
    baseline = evaluate_invariance(model, args.valid_dir, valid_cache_dir, args.thresh,
                                   loss_fn, n_chords=args.TEST) if args.valid_dir else None
    if baseline is not None:
        print("baseline     | " + format_metrics(baseline))

    rng = np.random.RandomState(args.seed)
    bs = args.batch_size
    best_quiet = None
    history = []
    for epoch in range(args.epochs):
        order = rng.permutation(len(win_files))
        losses = []
        n_batches = (len(order) + bs - 1) // bs
        for bi, b in enumerate(range(0, len(order), bs)):
            batch = [np.load(win_files[i]) for i in order[b:b+bs]]
            x1 = tf.convert_to_tensor(np.stack([d['mag'] for d in batch]), tf.float32)
            x2 = tf.convert_to_tensor(np.stack([d['dph'] for d in batch]), tf.float32)
            y = tf.convert_to_tensor(np.stack([d['tgt'] for d in batch]), tf.float32)
            losses.append(float(train_step(x1, x2, y)))
            print("\r  batch %d/%d  loss=%.4f" % (bi + 1, n_batches, losses[-1]),
                  end='', flush=True)
        print()

        msg = "epoch %d/%d  train_loss=%.4f" % (epoch + 1, args.epochs, float(np.mean(losses)))
        if args.valid_dir:
            inv = evaluate_invariance(model, args.valid_dir, valid_cache_dir, args.thresh,
                                      loss_fn, n_chords=args.TEST)
            if inv is not None:
                history.append(dict(epoch=epoch + 1, **inv))
                msg += "  | " + format_metrics(inv)
                # Rank epochs on the UNDILUTED quiet-voice recall (the actual
                # objective), and refuse any epoch that pays for it by regressing
                # recall or precision on either side, or by widening the gap.
                failed = guard_failures(inv, baseline, args)
                if failed:
                    msg += "  [rejected: %s]" % ", ".join(failed)
                elif best_quiet is None or inv['recall_quiet'] > best_quiet:
                    best_quiet = inv['recall_quiet']
                    model.save_weights(args.out)
                    msg += "  [saved best]"
        else:
            model.save_weights(args.out)
        print(msg)

    if not args.valid_dir:
        print("Saved final weights to %s" % args.out)
    elif best_quiet is None:
        model.save_weights(args.out)
        print("No epoch improved quiet-voice recall within the guards; "
              "saved final-epoch weights to %s" % args.out)
    else:
        print("Best (highest quiet-voice recall, guards satisfied) weights saved to %s"
              % args.out)

    if history:
        print("\nepoch summary")
        for h in history:
            print("  %2d | %s" % (h['epoch'], format_metrics(h)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--train_dir', required=True, help='dir with train_*.wav + .f0.csv')
    p.add_argument('--valid_dir', default=None, help='dir with the matched valid pairs')
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

    p.add_argument('--strategy', choices=['bn', 'full'], default='full',
                   help="which weights adapt: 'bn' = AdaBN recalibration only "
                        "(cheap, try first); 'full' = all layers (pair with --l2sp)")
    p.add_argument('--l2sp', type=float, default=1e-3,
                   help='L2-SP anchor strength for full fine-tuning (0 disables). '
                        'Penalises deviation of conv kernels from pretrained values.')
    p.add_argument('--pos_weight', type=float, default=1.0,
                   help='loss upweight on annotated (voice) target bins (1.0 = off)')
    p.add_argument('--bal_tol', type=float, default=0.03,
                   help='max allowed regression, vs the pre-training baseline, of '
                        'recall and precision on BOTH the balanced and victim sides '
                        'when selecting the best epoch')
    p.add_argument('--gap_tol', type=float, default=0.0,
                   help='max allowed widening of the undiluted quiet-voice gap '
                        '(recall_balanced - recall_quiet) vs baseline. 0 = the gap '
                        'must not get worse than it already was.')

    train(p.parse_args())
