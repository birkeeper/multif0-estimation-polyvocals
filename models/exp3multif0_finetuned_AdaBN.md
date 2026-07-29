# exp3multif0_finetuned.weights.h5

Fine-tuned variant of `models/exp3multif0.h5` (model3, `models.build_model3()`),
adapted by **AdaBN BatchNorm recalibration** on synthetic SSATBB choir chords in
which single voices are attenuated.

Goal: reduce the *"quiet voice → low salience"* bias, i.e. raise multi-F0 recall
on a voice sung softly relative to the rest of the ensemble, without regressing
performance on balanced ensembles.

| | |
|---|---|
| Base weights | `models/exp3multif0.h5` |
| Architecture | model3, unchanged (weights-only file, Keras 3 format) |
| Adapted parameters | BatchNormalization only (γ, β + running mean/variance) |
| Frozen parameters | all conv and dense kernels/biases |
| Training data | synthetic, 5 scenes / ~75 min, generated + PWA-rendered (below) |
| Date produced | 2026-07-29 |

---

## 1. Method

A voice sung quietly is an **input-amplitude / SNR domain shift**. Its evidence is
attenuated at the input BatchNorm and in the early/mid harmonic layers
(`conv1`..`harm2`) — i.e. *before* the decision head — so adapting only the head
cannot recover it.

This model uses `--strategy bn`, the cheapest of the two offered adaptations:
**AdaBN recalibration**. Every `BatchNormalization` layer is set trainable and all
other layers frozen (`set_trainable()` in `finetune/finetune.py`). Two things then
change during training:

1. γ and β receive gradients and are updated by Adam.
2. Because the BN layers run in training mode, their **running mean/variance
   recalibrate** to the new amplitude distribution.

No convolutional filter is modified. Verified in code: `train_step` computes
`tape.gradient(loss, model.trainable_variables)`, and with `strategy='bn'` that
variable list contains only BN parameters.

Loss is `bkld` (binary KL divergence) on the blurred binary salience target.

> **Note on L2-SP.** The run used the default `--l2sp 1e-3`, but the anchor list is
> built by filtering `model.trainable_variables` for names containing `kernel`.
> Under `--strategy bn` no kernel is trainable, so the anchor set is empty and the
> L2-SP penalty contributed **nothing**. The flag was inert for this model. (It is
> only meaningful with `--strategy full`.)

> **Note on `--pos_weight`.** Left at the default `1.0`, i.e. **disabled** — no
> upweighting of annotated voice bins.

---

## 2. Procedure

### 2.1 Dataset generation

`finetune/generate_chords.py` was invoked **with no arguments**. Equivalent
explicit call:

```bash
python finetune/generate_chords.py \
    --out ./finetune/data \
    --train_scenes 5 \
    --valid_scenes 1 \
    --chords_per_scene 120 \
    --seed 0 \
    --step_units eighth quarter \
    --gap_unit quarter \
    --valid_victim_db -12.0
```

This produced, per scene, a `.mid` (format 1, one track per voice), an aligned
ground-truth `.f0.csv` on the pump frame grid (hop 256 @ 22050 Hz, tab-delimited
ragged multi-F0), and a `.notes.csv` debug table.

Structure of the generated material:

- **6 voices (SSATBB)** — S1, S2, A, T, B1, B2. Every chord uses **all six**.
- Voices enter one at a time, sustain together (plateau), then release one at a
  time, so each chord sweeps cardinality 1→…→6→…→1. Segment lengths are drawn per
  chord from {eighth, quarter} at 80 BPM; chords are separated by a quarter-note
  silence.
- Velocity is **constant** everywhere; loudness is carried solely by **CC7**
  (main volume), so there is no velocity→timbre confound. Vowel (doo/da) is set
  by Program Change.
- **train/** — 5 scenes × 120 chords. Per-chord balance randomized:
  35 % balanced (all 0 dB), 45 % single-victim (one voice at −6/−12/−18 dB, quiet
  end oversampled), 20 % random-all (every voice drawn from {0, −6, −12, −18} dB).
- **valid/** — 1 scene → one **matched pair** of 120 chords:
  `valid_0000_balanced` (all voices 0 dB) and `valid_0000_victim` (identical
  notes, exactly **one voice per chord at −12 dB**, the victim voice cycling
  chord-to-chord so all six are exercised). Quiet-but-sounding voices remain
  present in the annotation — that is the entire training signal.

### 2.2 Rendering to audio

Each `.mid` was rendered to **mono 22050 Hz wav** using the PWA:

<https://birkeeper.github.io/Choir-practice-midi-player/recorder/recorder.html>

Soundfont: `Choir_practice.sf2` (bank 1; programs 1–4 = doo, 11–14 = da).
The PWA's synthesis engine (SpessaSynth_core) derives attenuation from CC7 as
`attenuation_dB = -40·log10(cc7·128/16385)`; `generate_chords.py` inverts exactly
this curve, so the dB levels in `notes.csv` are the levels actually rendered.

Resulting audio:

| file | duration |
|---|---|
| `train_0000.wav` … `train_0004.wav` | 14.69 / 15.07 / 15.14 / 14.93 / 15.55 min |
| `valid_0000_balanced.wav` | 15.21 min |
| `valid_0000_victim.wav` | 15.21 min |
| **total** | **105.8 min** |

### 2.3 Fine-tuning

`finetune/finetune.py` was invoked with `--train_dir`, `--valid_dir` and
`--strategy bn`; everything else was left at its default. Equivalent explicit
call:

```bash
python finetune/finetune.py \
    --train_dir ./finetune/data/train \
    --valid_dir ./finetune/data/valid \
    --strategy bn \
    --weights ./models/exp3multif0.h5 \
    --out ./models/exp3multif0_finetuned.weights.h5 \
    --win 50 \
    --win_hop 25 \
    --batch_size 1 \
    --epochs 6 \
    --lr 1e-4 \
    --thresh 0.5 \
    --seed 0 \
    --l2sp 1e-3 \
    --pos_weight 1.0 \
    --bal_tol 0.03
```

(`--win_hop` defaults to `None`, which `prepare()` resolves to `win // 2` = 25.
`--recompute` and `--TEST` were not passed. `--l2sp` was inert, see §1.)

Data pipeline:

- Each training wav is featurised **once** over the whole file (HCQT magnitude +
  phase-difference, so there are no segment edge artifacts), oriented `(H, F, T)`
  to match the convention model3's weights were originally fit on.
- The file is sliced into chord segments using the annotation's silent gaps, and
  each chord is cut into fixed **50-frame windows with stride 25** (50 % overlap).
- Fixed-length windows are required at training time because the `distribution`
  layer's (360, 1) kernel makes backprop-filter memory scale with T (~1.6 GB at
  batch 1 × 50 frames). Windowing within chords also skips inter-chord silence.
- **13,145 windows** were cached to `finetune/data/train/_cache` (one `.npz` per
  window, `mag`/`dph`/`tgt`). At `--batch_size 1` this is 13,145 optimizer steps
  per epoch.
- Validation features are cached to `finetune/data/valid/_cache`, one `.npz` per
  file, and reused across epochs.

Epoch selection rule: after each epoch, evaluate both sides of the matched pair.
Keep the epoch with the **highest victim recall**, subject to balanced recall
*and* balanced precision not regressing more than `--bal_tol` (0.03) below the
pre-training baseline. The saved file is that best checkpoint, not necessarily the
final epoch.

---

## 3. Results

The checkpoint in `exp3multif0_finetuned.weights.h5` is the one saved after
**epoch 6** (the final epoch, which was also the best under the selection rule).

The per-epoch log from the original run was not retained, so the figures below
were **re-measured after the fact**: both the baseline and the stored weights
were scored through `finetune.py`'s own `evaluate_invariance()` path, over the
complete validation matched pair (120 chord pairs, one quiet voice per chord at
−12 dB), `mir_eval.multipitch` at threshold 0.5. Both columns therefore come from
one consistent code path. Only the final checkpoint could be recovered this way —
there is no epoch-by-epoch trajectory.

### Measured

| metric | baseline (`exp3multif0.h5`) | fine-tuned (epoch 6) | Δ |
|---|---|---|---|
| Recall, balanced | 0.8095 | 0.8834 | **+0.0740** |
| Recall, victim | 0.7826 | 0.8650 | **+0.0825** |
| Precision, balanced | 0.8779 | 0.9557 | **+0.0779** |
| Precision, victim | 0.8780 | 0.9550 | **+0.0771** |
| Invariance gap (R_bal − R_vic) | 0.0269 | 0.0184 | −0.0085 |
| val loss, balanced | 0.0401 | 0.0281 | −0.0121 |
| val loss, victim | 0.0437 | 0.0294 | −0.0142 |

### Derived

F-measure, computed from the above:

| | baseline | fine-tuned | Δ |
|---|---|---|---|
| F, balanced | 0.8423 | 0.9182 | +0.0759 |
| F, victim | 0.8275 | 0.9078 | +0.0803 |

**Recall and precision rose together on both sides**, and the validation loss
fell on both. This rules out a pure threshold/operating-point shift (which trades
recall against precision) and indicates the salience map genuinely separated
better.

### Interpreting the gap — dilution by voice count

The aggregate victim recall is **diluted 6×**: all six voices sound in every
chord and only one is attenuated, so the quiet voice contributes just 1/6 of the
reference pitches. With `R_victim = (5/6)·R_loud + (1/6)·R_quiet` and assuming
`R_loud ≈ R_balanced`, solving for the quiet voice gives
`R_quiet = 6·R_victim − 5·R_balanced`:

| | R_balanced | R_victim | R_quiet (implied) | quiet deficit |
|---|---|---|---|---|
| baseline | 0.8095 | 0.7826 | **0.6480** | 0.1615 |
| fine-tuned | 0.8834 | 0.8650 | **0.7729** | 0.1105 |

So the quiet voice gained an estimated **+0.1249**, against +0.0740 for the
normal voices — it improved substantially *more*, which is the intended
direction. The small movement in the raw gap (0.0269 → 0.0184) understates this
by the same factor of 6: the underlying per-voice deficit went 0.1615 → 0.1105,
**about 32 % of it removed**.

These `R_quiet` figures are **inferred, not measured**. The evaluation scores all
six voices jointly; it does not isolate the victim voice. The estimate also
assumes the five loud voices in the victim file perform as they do in the
balanced file, which slightly understates `R_quiet` if attenuating one voice
unmasks the others.

---

## 4. Limitations

- **Synthetic audio only.** All training and validation material is
  soundfont-rendered from generated MIDI. The model has not been evaluated on real
  polyvocal recordings.
- **Part of the gain is domain adaptation, not de-biasing.** AdaBN corrects the
  statistics mismatch between model3's original real-recording training data and
  these PWA renders. Because BN running statistics have been moved toward the
  synthetic distribution, performance on *real* audio may be unchanged or worse.
  Validate on real material before adopting this checkpoint for real input.
- **One validation scene.** 120 chord pairs is a reasonable frame-level sample,
  but it is a single scene, single seed, single soundfont, and a single victim
  level (−12 dB). No error bars.
- **No per-epoch record.** The run's console output was not captured, so only the
  final checkpoint could be re-measured (§3). Whether the quiet-voice recall was
  still climbing at epoch 6, or had plateaued earlier, is unknown.
- **Victim precision is reported but not guarded.** The epoch-selection rule
  constrains balanced recall and balanced precision only; victim precision can
  degrade without blocking a checkpoint from being saved. (It did not here — it
  rose.)
- **The quiet-voice metric is diluted.** As shown in §3, selection ranks epochs on
  a number where 5/6 of the signal comes from voices the fine-tuning does not
  target. An epoch that improves loud voices while regressing the quiet one could
  in principle win.
- **AdaBN cannot restore discarded information.** BN pools statistics over the
  whole window, so a narrow quiet-voice sub-band is diluted in the statistics it
  recalibrates against. `--strategy full` (with a functioning L2-SP anchor) is the
  intended route to making the early harmonic detectors themselves keep a quiet
  voice above threshold.

---

