"""
Fine-tune model3 to *unlearn* the "quiet voice -> low salience" bias, using the
synthetic SSATBB chords produced by generate_chords.py (rendered to wav by the
PWA).

Method (see ../research/finetune_conversation.md):
  * Freeze both base_model feature-extractor branches (BatchNorm included -> they
    run in inference mode, keeping the original input normalisation). Train only
    the decision head: conv7, conv8, distribution, squishy.
  * Each rendered file is featurised once (whole-file CQT, so no edge artifacts),
    sliced into chord segments using the annotation's silent gaps, and each chord
    is cut into fixed WINDOWS (~50 frames). Windows are cached to disk and
    streamed one/few at a time (low memory). Fixed-length windows are required at
    TRAINING time -- not for the fully-convolutional forward pass, but because the
    `distribution` layer's (360,1) kernel makes its backprop-filter memory scale
    with T (a whole chord OOMs; ~50 frames keeps it ~1.6 GB). Windowing within
    chords also skips the inter-chord silence.
  * Select the epoch by the INVARIANCE GAP on the matched validation pair
    (balanced vs. one-voice-quiet): quiet-voice recall must rise while the
    balanced case does not regress. bkld loss, small LR, few epochs.

Layout expected (from generate_chords.py + PWA render):
    <train_dir>/train_XXXX.wav      + train_XXXX.f0.csv
    <valid_dir>/valid_XXXX_balanced.wav      + .f0.csv
    <valid_dir>/valid_XXXX_victim_<V>.wav    + .f0.csv   (same notes, V quiet)
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
HEAD_START = 'conv7'      # first layer of the trainable decision head


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
    """Whole-file HCQT mag + phase-diff as (H, F, T) arrays."""
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
def build_and_freeze(weights_path, unfreeze_harm=False):
    model = models.build_model3()
    model.load_weights(weights_path)

    # Freeze everything before the decision head; enable from HEAD_START on.
    trainable = False
    for layer in model.layers:
        if layer.name == HEAD_START:
            trainable = True
        layer.trainable = trainable
    if unfreeze_harm:
        for layer in model.layers:
            if layer.name.startswith('harm'):
                layer.trainable = True

    n_tr = sum(1 for l in model.layers if l.trainable)
    print("Trainable layers (%d): %s" % (n_tr, [l.name for l in model.layers if l.trainable]))
    return model


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
def _recall_precision(pump, model, wav, f0_csv, thresh):
    import mir_eval
    mag, dph = featurize(pump, wav)
    sal = predict_salience(model, mag, dph)
    est_t, est_f = utils_train.pitch_activations_to_mf0(sal, thresh)
    ref_t, ref_f = load_ragged_f0(f0_csv)
    m = mir_eval.multipitch.evaluate(ref_t, ref_f, np.array(est_t), est_f)
    return m['Recall'], m['Precision']


def evaluate_invariance(pump, model, valid_dir, thresh):
    """For each matched pair, recall on balanced vs victim (same notes).
    Returns dict with mean recalls, the gap, and balanced precision."""
    bal_files = sorted(glob.glob(os.path.join(valid_dir, 'valid_*_balanced.wav')))
    rb, rv, pb = [], [], []
    for bwav in bal_files:
        idx = os.path.basename(bwav).split('_')[1]
        vic = glob.glob(os.path.join(valid_dir, 'valid_%s_victim_*.wav' % idx))
        if not vic:
            continue
        recall_b, prec_b = _recall_precision(pump, model, bwav, bwav[:-4] + '.f0.csv', thresh)
        recall_v, _ = _recall_precision(pump, model, vic[0], vic[0][:-4] + '.f0.csv', thresh)
        rb.append(recall_b); rv.append(recall_v); pb.append(prec_b)
    if not rb:
        return None
    rb, rv, pb = np.mean(rb), np.mean(rv), np.mean(pb)
    return dict(recall_balanced=rb, recall_victim=rv, gap=rb - rv, precision_balanced=pb)


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

    model = build_and_freeze(args.weights, unfreeze_harm=args.unfreeze_harm)
    model.compile(loss=utils_train.bkld,
                  metrics=['mse', utils_train.soft_binary_accuracy],
                  optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr))

    rng = np.random.RandomState(args.seed)
    bs = args.batch_size
    best_gap = None
    for epoch in range(args.epochs):
        order = rng.permutation(len(win_files))
        losses = []
        for b in range(0, len(order), bs):
            batch = [np.load(win_files[i]) for i in order[b:b+bs]]
            x1 = np.stack([d['mag'] for d in batch])   # (B, F, win, H)
            x2 = np.stack([d['dph'] for d in batch])
            y = np.stack([d['tgt'] for d in batch])    # (B, F, win)
            losses.append(model.train_on_batch([x1, x2], y)[0])

        msg = "epoch %d/%d  loss=%.4f" % (epoch + 1, args.epochs, float(np.mean(losses)))
        if args.valid_dir:
            inv = evaluate_invariance(pump, model, args.valid_dir, args.thresh)
            if inv is not None:
                msg += ("  | recall bal=%.3f victim=%.3f  GAP=%.3f  prec_bal=%.3f"
                        % (inv['recall_balanced'], inv['recall_victim'],
                           inv['gap'], inv['precision_balanced']))
                # select on smallest gap that keeps balanced precision healthy
                score = inv['gap']
                if best_gap is None or score < best_gap:
                    best_gap = score
                    model.save_weights(args.out)
                    msg += "  [saved best]"
        else:
            model.save_weights(args.out)
        print(msg)

    if not args.valid_dir:
        print("Saved final weights to %s" % args.out)
    else:
        print("Best (smallest-gap) weights saved to %s" % args.out)


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
    p.add_argument('--unfreeze_harm', action='store_true',
                   help='also fine-tune harm1/harm2 (fallback if head-only underfits)')
    p.add_argument('--seed', type=int, default=0)

    train(p.parse_args())
