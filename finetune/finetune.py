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
    one-voice-quiet): keep the epoch with the highest quiet-voice recall among
    those whose balanced recall/precision do not regress past --bal_tol vs. the
    pre-training baseline. Small LR, few epochs.

Layout expected (from generate_chords.py + PWA render):
    <train_dir>/train_XXXX.wav      + train_XXXX.f0.csv
    <valid_dir>/valid_XXXX_balanced.wav    + .f0.csv
    <valid_dir>/valid_XXXX_victim.wav      + .f0.csv   (same notes, one quiet
                                                        voice per chord)
"""

from __future__ import print_function

import os
import sys
import csv
import glob
import argparse

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


def segment_chords(target, gap_frames=8, min_frames=12):
    """Return [(t0, t1), ...] for runs of active (non-silent) frames, merging
    gaps shorter than gap_frames and dropping runs shorter than min_frames.
    Uses the silent gaps the generator places between chords."""
    active = target.sum(axis=0) > 1e-3
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


# --------------------------------------------------------------------------
# Prepare: featurize + slice chords + cache to disk (one npz per chord)
# --------------------------------------------------------------------------
def prepare(pump, train_dir, cache_dir, win=50, hop=None, recompute=False):
    """Featurize each file, slice chords, cut chords into fixed `win`-frame
    windows (stride `hop`, default win//2 for overlap), cache one npz per
    window. Fixed length bounds the distribution-layer backprop memory and lets
    windows be batched."""
    hop = hop or max(1, win // 2)
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
        for (t0, t1) in segs:
            for s in range(t0, t1 - win + 1, hop):
                # store as (F, win, H) for direct model input; target (F, win)
                mseg = np.transpose(mag[:, :, s:s+win], (1, 2, 0))
                dseg = np.transpose(dph[:, :, s:s+win], (1, 2, 0))
                tseg = target[:, s:s+win]
                np.savez_compressed(os.path.join(cache_dir, '%s_w%04d.npz' % (stem, fw)),
                                    mag=mseg, dph=dseg, tgt=tseg)
                fw += 1
                n_win += 1
        print("  %s -> %d chords, %d windows" % (stem, len(segs), fw))
    open(done_marker, 'w').close()
    wins = sorted(glob.glob(os.path.join(cache_dir, '*.npz')))
    print("Cached %d windows (win=%d, hop=%d) from %d files." % (n_win, win, hop, len(f0_files)))
    return wins


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
def _eval_file(pump, model, wav, f0_csv, thresh, loss_fn):
    """Return (recall, precision, val_loss) for one file. val_loss is the same
    (bkld) loss used in training, computed on the inference-mode full-file
    prediction vs. the target."""
    import mir_eval
    mag, dph = featurize(pump, wav)
    sal = predict_salience(model, mag, dph)          # (F, T), BN in inference mode
    tgt = build_target(sal.shape[1], f0_csv)         # (F, T)
    loss = float(loss_fn(tf.constant(tgt[np.newaxis]), tf.constant(sal[np.newaxis])))
    est_t, est_f = utils_train.pitch_activations_to_mf0(sal, thresh)
    ref_t, ref_f = load_ragged_f0(f0_csv)
    m = mir_eval.multipitch.evaluate(ref_t, ref_f, np.array(est_t), est_f)
    return m['Recall'], m['Precision'], loss


def evaluate_invariance(pump, model, valid_dir, thresh, loss_fn):
    """For each matched pair, recall on balanced vs victim (same notes).
    Returns dict with mean recalls, the gap, balanced precision, and the mean
    validation loss (over both balanced and victim files)."""
    bal_files = sorted(glob.glob(os.path.join(valid_dir, 'valid_*_balanced.wav')))
    rb, rv, pb, lb, lv = [], [], [], [], []
    for bwav in bal_files:
        idx = os.path.basename(bwav).split('_')[1]
        vic = glob.glob(os.path.join(valid_dir, 'valid_%s_victim*.wav' % idx))
        if not vic:
            continue
        recall_b, prec_b, loss_b = _eval_file(pump, model, bwav, bwav[:-4] + '.f0.csv', thresh, loss_fn)
        recall_v, _, loss_v = _eval_file(pump, model, vic[0], vic[0][:-4] + '.f0.csv', thresh, loss_fn)
        rb.append(recall_b); rv.append(recall_v); pb.append(prec_b)
        lb.append(loss_b); lv.append(loss_v)
    if not rb:
        return None
    return dict(recall_balanced=np.mean(rb), recall_victim=np.mean(rv),
                gap=np.mean(rb) - np.mean(rv), precision_balanced=np.mean(pb),
                loss_balanced=np.mean(lb), loss_victim=np.mean(lv))


# --------------------------------------------------------------------------
# Train
# --------------------------------------------------------------------------
def train(args):
    # Keras 3 requires weight files to end in .weights.h5
    if not args.out.endswith('.weights.h5'):
        args.out = (args.out[:-3] if args.out.endswith('.h5') else args.out) + '.weights.h5'
        print("Adjusted output weights path to %s (Keras 3 requirement)" % args.out)

    pump = utils.create_pump_object()

    cache_dir = args.cache or os.path.join(args.train_dir, '_cache')
    win_files = prepare(pump, args.train_dir, cache_dir,
                        win=args.win, hop=args.win_hop, recompute=args.recompute)
    if not win_files:
        raise SystemExit("No windows to train on. Render the MIDIs to wav first.")

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

    # Pre-training baseline so we can require the balanced case not to regress.
    baseline = evaluate_invariance(pump, model, args.valid_dir, args.thresh, loss_fn) if args.valid_dir else None
    if baseline is not None:
        print("baseline    | val_loss bal=%.4f victim=%.4f  recall bal=%.3f victim=%.3f  GAP=%.3f  prec_bal=%.3f"
              % (baseline['loss_balanced'], baseline['loss_victim'],
                 baseline['recall_balanced'], baseline['recall_victim'],
                 baseline['gap'], baseline['precision_balanced']))

    rng = np.random.RandomState(args.seed)
    bs = args.batch_size
    best_victim = None
    for epoch in range(args.epochs):
        order = rng.permutation(len(win_files))
        losses = []
        for b in range(0, len(order), bs):
            batch = [np.load(win_files[i]) for i in order[b:b+bs]]
            x1 = tf.convert_to_tensor(np.stack([d['mag'] for d in batch]), tf.float32)
            x2 = tf.convert_to_tensor(np.stack([d['dph'] for d in batch]), tf.float32)
            y = tf.convert_to_tensor(np.stack([d['tgt'] for d in batch]), tf.float32)
            losses.append(float(train_step(x1, x2, y)))

        msg = "epoch %d/%d  train_loss=%.4f" % (epoch + 1, args.epochs, float(np.mean(losses)))
        if args.valid_dir:
            inv = evaluate_invariance(pump, model, args.valid_dir, args.thresh, loss_fn)
            if inv is not None:
                msg += ("  | val_loss bal=%.4f victim=%.4f  recall bal=%.3f victim=%.3f  GAP=%.3f  prec_bal=%.3f"
                        % (inv['loss_balanced'], inv['loss_victim'],
                           inv['recall_balanced'], inv['recall_victim'],
                           inv['gap'], inv['precision_balanced']))
                # keep the epoch with highest quiet-voice recall, PROVIDED the
                # balanced side did not regress beyond bal_tol vs. baseline.
                ok = baseline is None or (
                    inv['recall_balanced'] >= baseline['recall_balanced'] - args.bal_tol and
                    inv['precision_balanced'] >= baseline['precision_balanced'] - args.bal_tol)
                if ok and (best_victim is None or inv['recall_victim'] > best_victim):
                    best_victim = inv['recall_victim']
                    model.save_weights(args.out)
                    msg += "  [saved best]"
        else:
            model.save_weights(args.out)
        print(msg)

    if not args.valid_dir:
        print("Saved final weights to %s" % args.out)
    elif best_victim is None:
        model.save_weights(args.out)
        print("No epoch improved quiet-voice recall within the balanced guard; "
              "saved final-epoch weights to %s" % args.out)
    else:
        print("Best (highest quiet-voice recall, balanced preserved) weights saved to %s" % args.out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--train_dir', required=True, help='dir with train_*.wav + .f0.csv')
    p.add_argument('--valid_dir', default=None, help='dir with the matched valid pairs')
    p.add_argument('--weights', default='./models/exp3multif0.h5',
                   help='model3 weights to fine-tune from')
    p.add_argument('--out', default='./models/exp3multif0_finetuned.weights.h5',
                   help='where to write fine-tuned weights (must end .weights.h5)')
    p.add_argument('--cache', default=None, help='chord-segment cache dir (default <train_dir>/_cache)')
    p.add_argument('--recompute', action='store_true', help='rebuild the feature cache')

    p.add_argument('--win', type=int, default=50, help='training window length (frames)')
    p.add_argument('--win_hop', type=int, default=None, help='window stride (default win//2)')
    p.add_argument('--batch_size', type=int, default=1,
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
                   help='max allowed regression of balanced recall/precision vs baseline '
                        'when selecting the best epoch')

    train(p.parse_args())
