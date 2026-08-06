# exp3multif0_finetuned_distill_conv.weights.h5

Fine-tuned variant of `models/exp3multif0.h5` (model3, `models.build_model3()`),
adapted by **convolutional fine-tuning with a real-audio distillation anchor** on
synthetic SSATBB choir chords in which single voices are attenuated.

Goal: reduce the *"quiet voice → low salience"* bias, i.e. raise multi-F0 recall
on a voice sung softly relative to the rest of the ensemble, without regressing
performance on balanced ensembles **and without altering the model's behaviour on
real recordings**.

| | |
|---|---|
| Base weights | `models/exp3multif0.h5` |
| Architecture | model3, unchanged (weights-only file, Keras 3 format) |
| Adapted parameters | all conv/dense kernels and biases (992,512 + head) |
| Frozen parameters | every `BatchNormalization` layer (γ, β **and** running mean/variance) |
| Training data | synthetic, 5 scenes / ~75 min, generated + PWA-rendered (§2.1–2.2) |
| Anchor data | 7 real a cappella recordings, 26.4 min, unannotated (§2.3) |
| Date produced | 2026-08-05 |
| Log | `models/exp3multif0_finetuned_distill_conv_20260805-182553.log` |
| Status | candidate validated on two guard excerpts — calibration preserved (`r ≈ 0.996`), detection accuracy not harmed, **+0.025 to +0.031 voice evenness across all 4 excerpt × checkpoint cells** (§4.3.7). Untested on any recording free of selection contact |
| Recommended checkpoint | `e01` or `e02` — indistinguishable on real audio (§4.3.5); **not** the `e02_s001122` currently in `--out` (§3.5) |

> **Note on the filename.** `train()` appends the strategy to `--out`, so the
> `_conv` suffix is generated, not typed. **This model is `--strategy conv`** — no
> BatchNorm parameter was modified.

---

## 1. Method

A voice sung quietly is an **input-amplitude / SNR domain shift**. Its evidence is
attenuated in the early/mid harmonic layers (`conv1`..`harm2`) — *before* the
decision head — so adapting only the head cannot recover it. Adapting only
BatchNorm cannot either: γ and β are two parameters per channel, applied
identically at every time-frequency position, so they express gain and bias but
cannot create a detector that responds to a faint harmonic series. Deciding
whether a weak harmonic stack is a voice is kernel work, and the (70,3) kernels of
`harm1`/`harm2` are where it happens.

This model therefore uses `--strategy conv`: every conv/dense kernel and bias is
trainable, and every `BatchNormalization` layer is frozen. Keras special-cases BN
— `trainable=False` also puts the layer in **inference mode** even when the model
is called with `training=True` — so γ, β *and* the running mean/variance all stay
at their pretrained values. The normalisation remains fitted to the real
recordings model3 was originally trained on and cannot drift onto the soundfont
distribution. Verified in the log: `Strategy 'conv': 20 trainable layers`.

### 1.1 The loss

Three terms:

```
L =        bkld_pw4( y_synth , f(x_synth) )        # synthetic, annotated
  + 3.0 ·  anchor_γ1( t_real , f(x_real) )         # real, unannotated
  + 1e-3 · Σ ‖W − W₀‖²                             # L2-SP on 16 kernels
```

**Synthetic term.** `bkld` (binary KL divergence) against the blurred binary
salience target, with `--pos_weight 4.0`: bins where the blurred target exceeds
0.5 — a 3-bin ridge per voice per frame, roughly 3 % of all bins — have their loss
multiplied by 4. Without this, ~97 % of every gradient concerns empty bins, and
globally suppressing weak activations is a cheaper loss reduction than raising a
quiet voice.

**Anchor term — the distinguishing feature of this model.** The synthetic loss
alone cannot distinguish "learned something about quiet voices" from
"recalibrated to the soundfont", and the second is cheaper. The anchor makes that
route expensive. A frozen copy of the pretrained model (the *teacher*) is run over
26 minutes of real a cappella audio; the student is penalised for moving away from
that output:

```python
w   = t**gamma                                   # gamma = 1.0
per = -(t·log(p) + (1−t)·log(1−p))               # per-bin, t = teacher, p = student
L_r = Σ(w·per) / Σ(w)                            # normalised by mean weight
```

No annotation is required — the target is what the model itself predicted before
training, so the term encodes *"do not change here"*. This is the same quantity
`drift_stats()` reports as the `REAL(n)` log line, moved out of the accept/reject
decision and into the gradient. It is equivalently `--l2sp` measured on **outputs**
instead of weights, on the input distribution that actually matters.

`gamma=1.0` weights the penalty by the teacher's own confidence: bins the teacher
reads high are pinned hard, bins it reads near zero are left almost free. The
intent is *"keep what you already know, stay free where you were unsure"*, so a
quiet voice the teacher missed can rise without the confident detections moving.
The trade-off is stated in §5.

The anchor is **symmetric** — it penalises deviation in both directions. This is
deliberate: calibration drift is not reliably one-signed, and a one-sided penalty
would leave upward inflation unopposed.

**Loss balance.** `--distill_lambda 3.0` rather than 1.0, because `--pos_weight 4`
multiplies the synthetic positive-bin gradient fourfold and `conv` has ~1000×
more freedom to drift than BatchNorm-only adaptation.

> **Note on the two forward passes.** Synthetic and real batches go through the
> model in *separate* calls, not one concatenated batch. Under a strategy with
> trainable BN this would matter because a mixed batch yields a blended batch
> statistic matching neither domain; here BN is frozen, so the separation is
> merely tidy. It also means the "real audio also corrects the BN running
> statistics" mechanism does **not** apply to this model — the entire anchor
> effect is the gradient term.

> **Note on L2-SP.** Unlike BatchNorm-only adaptation, `--l2sp 1e-3` is
> *functional* here: the anchor list filters `model.trainable_variables` for names
> containing `kernel`, and the log confirms `L2-SP anchoring 16 kernels`. Its
> contribution was not isolated by ablation.

> **Note on the edge crop.** The teacher is evaluated whole-file, so its targets
> carry no boundary artifacts, but the student sees a 50-frame window that is
> zero-padded at both ends. Summing the time receptive field (`conv1..conv4` ±2
> each, `harm1`/`harm2` ±1 each, `conv7`/`conv8` ±1 each) gives ≈ ±12 frames, so
> the anchor is computed on the middle **26 of 50 frames** only.

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

### 2.3 The distillation pool

Seven real a cappella rehearsal recordings in `finetune/data/distill/`, FLAC,
**unannotated**. `librosa.load` resamples to 22050 Hz and downmixes to mono, so
the container and source rate are irrelevant.

| file | duration | windows |
|---|---|---|
| `My Love_opname.flac` | 243.0 s | 418 |
| `Ren Lenny_opname.flac` | 179.8 s | 309 |
| `Sweet child of mine_opname.flac` | 202.4 s | 348 |
| `That don't impress me much_opname.flac` | 212.4 s | 365 |
| `heroes_opname.flac` | 217.2 s | 374 |
| `multicolor_opname.flac` | 244.7 s | 421 |
| `sing_opname.flac` | 287.1 s | 494 |
| **total** | **26.4 min** | **2,729** |

Each file was featurised whole, the frozen teacher run over it whole (so its
targets have no window-boundary artifacts), then cut into non-overlapping 50-frame
windows cached as one `.npz` per window holding `mag`/`dph`/`tsal`. Because the
teacher never changes, this is computed once; no second model is held in memory
during training.

18 of the 2,729 windows (0.7 %) have a teacher peak below 0.10 and are tagged
"quiet"; `--distill_quiet_cap 0.33` therefore never engaged. These are continuous
rehearsal recordings with almost no silence.

**Disjointness from the drift screen.** An eighth recording, `late.flac`, was
initially in the pool and was **removed before this run** — it is the full take
from which the `--real_audio` guard excerpt `late_dada.wav` was cut. Training on
it would have made the `REAL(2)` line report near-zero drift by construction. It
now sits in `finetune/data/distill_excluded/`. `prepare_distill()` checks stems as
prefixes in both directions and refuses to start on overlap; that check cannot
detect an unrelated filename holding the same audio, which is why the per-file
list above is printed to the log for auditing.

### 2.4 Fine-tuning

```bash
python finetune/finetune.py \
    --train_dir ./finetune/data/train/ \
    --valid_dir ./finetune/data/valid/ \
    --distill_dir ./finetune/data/distill \
    --real_audio ./finetune/data/valid/late_dada.wav \
                 ./finetune/data/valid/Parijs_dedetdoe.wav \
    --strategy conv \
    --weights ./models/exp3multif0.h5 \
    --out ./models/exp3multif0_finetuned_distill_conv.weights.h5 \
    --win 50 \
    --batch_size 10 \
    --epochs 6 \
    --eval_every 66 \
    --lr 1e-4 \
    --pos_weight 4.0 \
    --distill_lambda 3.0 \
    --distill_gamma 1.0 \
    --distill_quiet_cap 0.33 \
    --l2sp 1e-3 \
    --thresh 0.5 \
    --seed 0 \
    --bal_tol 0.03 \
    --gap_tol 0.0 \
    --drift_tol 0.10 \
    --drift_high_tol 0.15
```

`--win_hop` was not passed; `prepare()` resolves it to `win`, i.e. **no overlap**.
`--recompute` and `--TEST` were not passed.

Data pipeline:

- Each training wav is featurised **once** over the whole file (HCQT magnitude +
  phase-difference, so there are no segment edge artifacts), oriented `(H, F, T)`
  to match the convention model3's weights were originally fit on.
- The file is sliced into chord segments using the annotation's silent gaps, and
  each chord is cut into fixed **50-frame windows, stride 50**.
- Fixed-length windows are required at training time because the `distribution`
  layer's (360, 1) kernel makes backprop-filter memory scale with T. Windowing
  within chords also skips inter-chord silence.
- **6,650 windows** cached to `finetune/data/train/_cache`. At `--batch_size 10`
  this is **665 optimizer steps per epoch**.
- Validation features are cached to `finetune/data/valid/_cache`, one `.npz` per
  file, and reused.

**Checkpoint selection.** With `--eval_every 66` the full validation pair, the
drift screen and the guards run **ten times per epoch** as well as at each epoch
boundary. Every checkpoint is written to `<out>_eNN[_sNNNNNN].weights.h5`. A
checkpoint is saved to `--out` only if it has the highest undiluted quiet-voice
recall so far *and* passes all guards:

| guard | tolerance | direction |
|---|---|---|
| `recall_balanced`, `precision_balanced`, `recall_victim`, `precision_victim` | `--bal_tol 0.03` below baseline | rejects regression |
| `gap_quiet` | `--gap_tol 0.0` above baseline | gap must not widen |
| `real_worst` | `--drift_tol 0.10` | **negative drift only** |
| `real_d_high` | `--drift_high_tol 0.15` | **negative only** |

The ranking is seeded with the *baseline* quiet-voice recall, so a checkpoint must
beat `exp3multif0.h5` to be written, not merely survive the guards. If nothing
does, `--out` is not written at all.

---

## 3. Results

All figures in this section come from the run's own log — the per-checkpoint
trajectory was captured, so nothing here is re-measured after the fact. Scoring is
`evaluate_invariance()` over the complete validation matched pair (120 chord
pairs, one quiet voice per chord at −12 dB), `mir_eval.multipitch` at threshold
0.5.

### 3.1 Trajectory

| step | R bal | R vic | P bal | quiet recall | gap_quiet | real drift |
|---|---|---|---|---|---|---|
| baseline | 0.809 | 0.783 | 0.878 | 0.648 | 0.161 | — |
| 66 | 0.860 | 0.831 | 0.877 | 0.687 | 0.173 | +6.7 % |
| 132 | 0.874 | 0.846 | 0.868 | 0.706 | 0.167 | +5.4 % |
| 198 | 0.883 | 0.855 | 0.860 | 0.719 | 0.163 | +6.3 % |
| 264 | 0.883 | 0.856 | 0.866 | 0.723 | **0.160** | +5.4 % |
| 330 | 0.889 | 0.864 | 0.858 | 0.739 | 0.149 | +5.9 % |
| 396 | 0.890 | 0.866 | 0.858 | 0.748 | 0.142 | +6.2 % |
| 528 | 0.894 | 0.871 | 0.852 | 0.756 | 0.139 | +5.3 % |
| 660 | 0.893 | 0.872 | 0.864 | 0.763 | 0.130 | +6.2 % |
| **epoch 1** | 0.894 | 0.873 | 0.862 | **0.764** | **0.130** | +5.8 % |
| 1122 | 0.898 | 0.877 | 0.851 | 0.771 | 0.127 | +6.5 % |
| **epoch 2** | 0.898 | 0.876 | 0.857 | 0.767 | 0.131 | +4.7 % |
| 1452 (e3) | 0.895 | 0.873 | 0.864 | 0.763 | 0.132 | +4.3 % |

`gap_quiet` first drops below the 0.161 baseline at step 264, which is the first
checkpoint written to `--out`. Ten consecutive checkpoints were then saved,
through the end of epoch 1.

**The run converged at the end of epoch 1.** Quiet recall over the fourteen
checkpoints from that point onward:

```
0.764  0.760  0.762  0.760  0.765  0.764  0.763  0.771  0.760  0.771  0.770  0.767  0.764  0.763
mean 0.765,  sd 0.004,  no trend across 800 steps
```

Precision oscillates 0.850–0.865 and balanced recall 0.892–0.899 over the same
span, likewise without direction. Epochs 2 and 3 added nothing.

### 3.2 Measured, at the epoch-1 checkpoint

| metric | baseline (`exp3multif0.h5`) | epoch 1 | Δ |
|---|---|---|---|
| Recall, balanced | 0.809 | 0.894 | **+0.085** |
| Recall, victim | 0.783 | 0.873 | **+0.090** |
| Precision, balanced | 0.878 | 0.862 | −0.016 |
| Precision, victim | 0.878 | 0.863 | −0.015 |
| Invariance gap (R_bal − R_vic) | 0.027 | 0.022 | −0.005 |
| val loss, balanced | 0.1149 | 0.0817 | −0.0332 |
| val loss, victim | 0.1284 | 0.0862 | −0.0422 |

Validation losses are **not comparable to runs with a different `--pos_weight`**:
the weighting rescales the loss without renormalising, so absolute values only
mean something within this run. The baseline row was scored with the same
`loss_fn`, so the comparison above is internally consistent.

### 3.3 Derived

F-measure, computed from §3.2:

| | baseline | epoch 1 | Δ |
|---|---|---|---|
| F, balanced | 0.8421 | **0.8777** | **+0.0356** |
| F, victim | 0.8278 | **0.8680** | **+0.0402** |

Recall rose 8.5 points while precision gave up 1.6, so F rose on both sides. That
is the test that separates a genuine improvement from an operating-point shift: a
pure threshold move trades recall against precision and leaves F flat. Validation
loss also fell on both sides.

### 3.4 Interpreting the gap — dilution by voice count

The aggregate victim recall is **diluted 6×**: all six voices sound in every
chord and only one is attenuated, so the quiet voice contributes just 1/6 of the
reference pitches. With `R_victim = (5/6)·R_loud + (1/6)·R_quiet` and assuming
`R_loud ≈ R_balanced`, solving for the quiet voice gives
`R_quiet = 6·R_victim − 5·R_balanced` (`undilute_quiet_recall()`):

| | R_balanced | R_victim | R_quiet (implied) | quiet deficit |
|---|---|---|---|---|
| baseline | 0.809 | 0.783 | **0.648** | 0.161 |
| epoch 1 | 0.894 | 0.873 | **0.764** | 0.130 |
| epoch 2 | 0.898 | 0.876 | **0.767** | 0.131 |

The quiet voice gained **+0.116**, against **+0.085** for the normal voices — it
improved *more*, which is the intended direction and is why the deficit narrowed:
0.161 → 0.130, **about 19 % of it removed**. The raw aggregate gap (0.027 → 0.022)
understates this by the same factor of 6.

These `R_quiet` figures are **inferred, not measured**. The evaluation scores all
six voices jointly; it does not isolate the victim voice. The estimate also
assumes the five loud voices in the victim file perform as they do in the
balanced file, which slightly understates `R_quiet` if attenuating one voice
unmasks the others.

### 3.5 The file in `--out` is a selection artifact

`--out` holds **`e02_s001122`**, reported as `+0.123 vs baseline`, because 0.771
was the highest quiet recall observed. Given the plateau in §3.1 (mean 0.765,
sd 0.004) and ~24 checkpoints, the expected maximum of the noise sits 1.5–2 sd
above the mean — which is 0.771 almost exactly, and two separate checkpoints tie
at that value. The guards are absolute thresholds, not significance tests, so
nothing prevents the rule from selecting a noise peak.

| candidate | quiet recall | P bal | F bal | real drift |
|---|---|---|---|---|
| `e01` | 0.764 | **0.862** | **0.8777** | +5.8 % |
| `e02` | 0.767 | 0.857 | 0.8770 | **+4.7 %** |
| `e02_s001122` (in `--out`) | 0.771 | 0.851 | 0.8739 | +6.5 % |

`e01` is better on precision, F and drift, and its quiet recall is within one sd.
**The honest headline figure is quiet recall ≈ 0.765, +0.117 over baseline**, and
`e01`/`e02` are the checkpoints worth carrying forward.

> **Confirmed on real audio (§4.3.5).** `e01` and `e02` were later compared
> directly on a real excerpt and are indistinguishable — `r = 0.999`, every
> detection and evenness metric within 0.004. The 460 optimizer steps separating
> them changed nothing measurable, which is what a plateau predicts.

---

## 4. Behaviour on real recordings

§3 is entirely soundfont renders scored at a threshold of 0.5. §4.1–4.2 document
what the run itself measured (a distribution check). §4.3 reports the real-audio
accuracy and evenness evaluation, which has now been performed **on one excerpt**;
§4.4 states what remains.

### 4.1 The drift screen

Two held-out real excerpts — `late_dada.wav` (7.8 s) and `Parijs_dedetdoe.wav`
(4.8 s), neither in the distillation pool — were featurised once, and at every
checkpoint their salience maps were compared with the ones the *pretrained* weights
produced for the same audio (`drift_stats()`). No annotation, alignment or tuning
correction is involved: both maps come from the same audio and line up bin for
bin.

Across all **24 checkpoints** spanning 1,452 optimizer steps:

| statistic | range | at epoch 1 |
|---|---|---|
| mean salience change | **+4.3 % … +6.7 %** | +5.8 % |
| `d@high` (change where baseline reads 0.80–0.90) | **−0.003 … −0.009** | −0.008 |
| map correlation `r` | **0.99 … 1.00** | 1.00 |
| worst single excerpt | +3.9 % … +6.2 % | +5.7 % |

The anchor term stayed flat at 0.519–0.524 throughout.

Three readings:

- **The change is a small uniform lift, not a compression.** `d@high` is within
  0.01 of zero at every checkpoint, so the confident activations are essentially
  untouched; the ~6 % is spread across the range.
  → **Revised by §4.3.1.** The level-resolved breakdown shows it is in fact a mild
  compression crossing zero at ≈ 0.45, and that ~71 % of the "+6 %" is the
  near-empty background bins rising by +0.0006 each. `drift_stats()` reports only
  `d@high`, so it could not see either.
- **`r = 1.00` means the map *shape* is preserved.** Voice-relative comparisons
  and threshold sweeps on this model should therefore behave like model3's, which
  is the property the anchor was built to protect.
- **It equilibrated rather than merely slowing.** The series oscillates inside a
  band from step 66 onward instead of trending, over three epochs of training on
  material from a different domain.

### 4.2 What this does and does not establish

It establishes that the model's real-audio salience is within ~6 % of the base
model's and correlated at r ≈ 1.00 on two short excerpts. Because a shared
threshold therefore compares near-identical operating points, a like-for-like
comparison on real audio is *possible* for this model — which is the precondition
for the evaluation below, not a substitute for it.

It does **not** establish detection accuracy, per-voice evenness, or that the
+0.116 quiet-voice gain in §3.4 exists on real singers. The drift screen is
deliberately blind to those: it asks only "did the output move", and its answer
here is "barely". A model could pass it perfectly by learning nothing at all.

§4.3 measures those three directly on one excerpt. Its verdict on this
section's precondition is favourable — the equivalent threshold came out at
exactly 0.50 — but its verdict on the +0.116 is that it did **not** appear as
detection accuracy, and appeared only weakly as evenness.

### 4.3 Real-audio evaluation

§4.3.1–4.3.4 cover `e01` vs baseline on `late_dada`; §4.3.5 compares the two
candidate checkpoints; §4.3.6 repeats the evaluation on `Parijs_dedetdoe`;
**§4.3.7 is the summary across both excerpts and the section to read first.**

#### 4.3.0 `e01` on `late_dada`

Date: 2026-08-06. Checkpoint `exp3multif0_finetuned_distill_conv_e01.weights.h5`
— the one §3.5 recommends — against the baseline `exp3multif0.h5`.

```bash
python predict_on_audio.py --model model3 --save_salience \
    --model_weights ./models/exp3multif0_finetuned_distill_conv_e01.weights.h5 \
    --audiofile late_dada.wav        # and once without --model_weights

python finetune/compare_voice_salience.py \
    --salience late_dada_model3_exp3multif0_salience.npz \
               late_dada_model3_exp3multif0_finetuned_distill_conv_e01.weights_salience.npz \
    --midi "Late Night Talking - Full score - Late Night Talking.mid" \
    --measures 1-4
```

Excerpt: 7.78 s, 4 voices (Soprano / Mezzo / Tenor / Bass), 6 chords, 2,659
reference pitches. Warp fitted at `scale=0.860 offset=+0.280` (performance +16 %
vs score tempo), score/salience correlation **r = 0.416**. Pitch tolerance
±80 cents.

**Note on excerpt choice.** `late_dada.wav` is one of the two `--real_audio` guard
excerpts. It carried **no gradient** — `late.flac`, the take it was cut from, was
deliberately removed from the distillation pool (§2.3) — but it *did* enter the
checkpoint accept/reject decision via `drift_stats()`. So this is held out from
training but not from selection. §4.4 covers the fully-independent excerpt.

#### 4.3.1 Whole-map

| | baseline | `e01` |
|---|---|---|
| mean salience | 0.0130 | 0.0138 (**+6 %**) |
| bins > 0.50 | 2,052 | 2,064 (+0.6 %) |
| correlation `r` vs baseline | — | **0.995** |
| threshold matching baseline's selectivity at 0.50 | 0.50 | **0.50** |

The two models sit at the **same operating point** — equivalent threshold 0.50,
+0.6 % on bins above it. Every comparison below is therefore like-for-like, which
is the precondition §4.2 required.

Change as a function of the baseline's own activation level:

| baseline bin | n bins | Δ `e01` |
|---|---|---|
| 0.00 – 0.05 | 231,321 | +0.0006 |
| 0.05 – 0.10 | 2,752 | **+0.0112** |
| 0.10 – 0.20 | 2,428 | **+0.0116** |
| 0.20 – 0.30 | 1,289 | +0.0077 |
| 0.30 – 0.40 | 909 | +0.0032 |
| 0.40 – 0.50 | 808 | −0.0007 |
| 0.50 – 0.60 | 753 | −0.0037 |
| 0.60 – 0.70 | 620 | −0.0069 |
| 0.70 – 0.80 | 367 | −0.0098 |
| 0.80 – 0.90 | 231 | −0.0143 |
| 0.90 – 1.00 | 82 | −0.0141 |

This **revises §4.1's reading**. The change is not a uniform lift: on this excerpt
it is a mild *compression*, crossing zero at ≈ 0.45 — weak and mid activations
rise, confident ones fall. The direction is the intended one (evidence for a faint
voice is worth more than another decibel on an already-certain detection), and the
magnitude is small — ≤ 0.014 everywhere, hence `r = 0.995`. §4.1 could not see the
crossover because `drift_stats()` reports only `d@high`, which is the last two
rows.

> **Qualified by §4.3.6.** On the second excerpt the lift stays positive up to
> ≈ 0.85, so the crossover point is not a property of the model. What replicates
> across both is the rise of the 0.05–0.30 bins by +0.011 to +0.012; the high-end
> behaviour is excerpt-dependent (§4.3.7).

The headline "+6 %" is also **not** what it sounds like. Decomposing the +0.0008
mean shift by row, **71 % of it comes from the 0.00–0.05 bin alone** — 95.8 % of
all bins, creeping up by +0.0006 each. It is background, not signal, and 6 % of a
mean of 0.013 is a very small absolute number. The same caution applies to the
+4–7 % figures throughout §4.1.

#### 4.3.2 Detection accuracy

Peak-picked, ±80 cent match against the score.

| | baseline | `e01` |
|---|---|---|
| P / R / F @ 0.50 (default) | 0.889 / 0.299 / **0.448** | 0.886 / 0.300 / **0.448** |
| P / R / F @ matched count (894 peaks) | 0.889 / 0.299 / **0.448** | 0.886 / 0.300 / **0.448** (901 peaks) |
| best F over the sweep | **0.546** @ 0.15 | **0.542** @ 0.15 |

**Flat.** All three columns agree within ±0.004, which is smaller than the sd of
the synthetic metric (§3.1) and far smaller than anything that would matter. The
per-threshold sweep is flat too: across 0.15–0.80 the largest F difference in
either direction is 0.007. **The +0.116 synthetic quiet-voice gain of §3.4 did not
show up as detection accuracy on this excerpt.**

Note also that **best F sits at 0.15, the bottom of the swept grid** — for both
models. The true optimum is below it. This is direct support for §5's suspicion
that 0.5 is the wrong operating point for real audio, and it is worth noting that
recall at the default 0.50 is only ~0.30 for either model.

#### 4.3.3 Per-voice evenness — the targeted property

Salience read at each voice's known F0; relative values are per-chord, against the
loudest voice.

| | baseline | `e01` | Δ |
|---|---|---|---|
| spread (max−min of relative), mean over 6 chords | 0.738 | **0.708** | **−0.030** |
| min/max (quietest vs loudest), mean over 6 chords | 0.262 | **0.292** | **+0.030** |

Both move the intended way: **voices more even**. The effect is entirely the Bass
line, and only where the Bass is actually present:

| chord | Bass pitch | baseline rel | `e01` rel |
|---|---|---|---|
| 1 | G3 | 0.30 | **0.35** |
| 2 | C3 | 0.08 | **0.11** |
| 3 | F3 | 0.89 | **0.97** |
| 4 | F2 | 0.00 | 0.00 |
| 5 | A♯2 | 0.00 | 0.01 |
| 6 | D3 | 0.45 | **0.53** |

Soprano, Mezzo and Tenor move by ≤ 0.03 in either direction — as expected, since
they are the loud voices the anchor pins. In chords 4 and 5 the Bass reads
**absolutely zero salience** (0.000 / 0.001 raw) at F2 (87 Hz) and A♯2 (116 Hz);
that is not a quiet voice the model under-reads, it is no energy at all at those
pitches, and no amount of low-end sensitivity will recover it. Those two chords
contribute a spread of exactly 1.00 to both models and dilute the mean.

Excluding them, the four chords where the Bass sounds give spread 0.63 → 0.56 and
min/max 0.35 → 0.42 — roughly **double** the aggregate effect, and the honest
figure for "chords where the target property is measurable".

#### 4.3.4 Reading

> Written on `late_dada` alone. **Superseded by §4.3.7**, which reconciles it with
> the second excerpt: the evenness result replicates exactly, the detection
> result is flat-to-slightly-positive rather than flat, and the compression
> reading below turns out to be excerpt-specific.

The three measurements say different things and should not be averaged:

- **Calibration is preserved** (§4.3.1). `r = 0.995`, identical equivalent
  threshold. The anchor did its job; this is the clearest positive result, and it
  is what the AdaBN sibling failed.
- **Detection accuracy is unchanged** (§4.3.2). Not better, not worse. The
  synthetic recall gain did not transfer as F on this excerpt.
- **Voice evenness improved slightly** (§4.3.3), concentrated exactly where the
  method predicts — the quietest voice, in chords where it is audible at all.
- **All three replicate on `e02`** (§4.3.5), an independently-selected checkpoint
  460 steps later. That rules out the pattern being one checkpoint's noise.

So the model is **not harmful and mildly helpful on the property it targets**,
which is a materially better outcome than the AdaBN variant (worse on both real
recordings), but it is a small effect on one short excerpt. The gap between
+0.116 synthetic quiet recall and +0.030 real evenness / +0.000 real F is the
substantive finding here, and §5 lists the candidate explanations —
soundfont-specific gains, the anchor pinning toward a teacher that itself misses
quiet voices, and a synthetic victim level (−12 dB) that may not resemble real
ensemble imbalance.

#### 4.3.5 `e01` vs `e02` — the two candidate checkpoints

**On `late_dada`.** Same excerpt, same procedure (warp fitted identically at
`scale=0.860 offset=+0.280`, `r = 0.417`), `e02` as the baseline column:

| | `e02` | `e01` |
|---|---|---|
| mean salience | 0.0136 | 0.0138 (+1 %) |
| bins > 0.50 | 2,065 | 2,064 |
| correlation `r` | — | **0.999** |
| equivalent threshold | 0.50 | **0.50** |
| P / R / F @ 0.50 | 0.889 / 0.300 / 0.449 | 0.886 / 0.300 / 0.448 |
| P / R / F @ matched count (901 peaks) | 0.888 / 0.301 / **0.449** | 0.886 / 0.300 / **0.448** |
| best F | **0.546** @ 0.15 | 0.542 @ 0.15 |
| spread (mean) | 0.712 | **0.708** |
| min/max (mean) | 0.288 | **0.292** |

**The two checkpoints are the same model for practical purposes.** `r = 0.999`,
identical equivalent threshold, and every difference — detection F, spread,
min/max — is ≤ 0.004, which is exactly the sd of the synthetic quiet-recall
plateau (§3.1). The level-resolved deltas peak at +0.0045 and are ≈ 0 above 0.50.
`e02` is a hair better on detection (F 0.546 vs 0.542, precision +0.002 to +0.008
across the sweep), `e01` a hair better on evenness; neither difference is
meaningful.

> The tool reports "quiet voice better held in **6 of 6** chords" for `e01`. That
> is a sign test on differences of ≤ 0.01 — three of the six chords tie to two
> decimals (0.00/0.00, 0.53/0.53) and are decided at floating-point precision. It
> is not evidence of anything.

This **empirically confirms §3.1's convergence claim**: the run had plateaued by
the end of epoch 1, and 460 further optimizer steps produced no change detectable
on real audio. It also confirms §3.5 — the choice between these checkpoints, and
the `+0.123` figure that put `e02_s001122` in `--out`, is selection over noise.

The more useful consequence is a **replication of §4.3.1–4.3.3 against the
baseline**, since `e02` was selected independently of `e01`:

| | baseline | `e02` | `e01` |
|---|---|---|---|
| mean salience | 0.0130 | 0.0136 | 0.0138 |
| best F | 0.546 | 0.546 | 0.542 |
| F @ 0.50 | 0.448 | 0.449 | 0.448 |
| spread (mean) | 0.738 | **0.712** | **0.708** |
| min/max (mean) | 0.262 | **0.288** | **0.292** |

Both checkpoints land in the same place: **detection F flat to ±0.004, evenness
improved by +0.026 to +0.030.** The §4.3.4 reading is therefore not an artifact of
which checkpoint was picked.

**On `Parijs_dedetdoe`.** The same comparison, repeated on the second excerpt
(warp `scale=0.840 offset=+0.280`, `r = 0.507`):

| | `e02` | `e01` |
|---|---|---|
| mean salience | 0.0250 | 0.0253 (+1 %) |
| bins > 0.50 | 2,845 | 2,863 |
| correlation `r` | — | **1.000** |
| equivalent threshold | 0.50 | **0.50** |
| P / R / F @ 0.50 | 0.841 / 0.552 / 0.666 | 0.842 / 0.555 / **0.669** |
| P / R / F @ matched count (1,137 peaks) | 0.841 / 0.552 / 0.666 | 0.842 / 0.555 / **0.669** |
| best F | 0.705 @ 0.15 | 0.706 @ 0.20 |
| spread (mean) | 0.520 | **0.514** |
| min/max (mean) | 0.480 | **0.486** |

Identical conclusion, and slightly cleaner: `r = 1.000` to three decimals, every
metric within 0.006, level-resolved deltas peaking at +0.0046 and ≈ 0 above 0.80.
Here `e01` is marginally ahead on *both* detection and evenness rather than
splitting them — which, given the margins, is itself just noise resolving
differently. The sign test flips too: "better held in **2 of 4** chords" here
against "**6 of 6**" on `late_dada`, on differences of the same ≤ 0.01 size,
confirming that line carries no information.

`e02` against the baseline on this excerpt replicates §4.3.6 as well:

| | baseline | `e02` | `e01` |
|---|---|---|---|
| mean salience | 0.0240 | 0.0250 | 0.0253 |
| F @ 0.50 | 0.664 | 0.666 | 0.669 |
| F @ 0.70 | 0.583 | **0.596** | **0.599** |
| F @ 0.75 | 0.541 | **0.560** | **0.563** |
| best F | 0.708 | 0.705 | 0.706 |
| spread (mean) | 0.545 | **0.520** | **0.514** |
| min/max (mean) | 0.455 | **0.480** | **0.486** |

The high-threshold recall gain of §4.3.6 is present in `e02` too (+0.013 at 0.70,
+0.019 at 0.75), so it is not an `e01` artifact either.

**Summary of the 2 × 2.** Across both checkpoints and both excerpts, the evenness
gain over baseline is +0.025, +0.026, +0.030, +0.031 — a tight cluster. The
checkpoint choice moves it by ≤ 0.006; the excerpt choice by ≤ 0.001. `e01` and
`e02` can be treated as interchangeable, and the result is robust to which one is
shipped.

#### 4.3.6 Second excerpt: `Parijs_dedetdoe`

`e01` vs baseline on the other `--real_audio` guard excerpt: 4.82 s, **5 voices**
(Sopraan / Mezzo / Alt / Tenor / Bas), 4 chords, 1,732 reference pitches, scored
against `Kenny B - Parijs.mid` measures 1–2. Warp fitted at `scale=0.840
offset=+0.280`, **r = 0.505** — a better alignment than `late_dada`'s 0.416.

This is the **more informative of the two excerpts**: every voice is audible in
every chord (no zero readings anywhere), so all four chords bear on the targeted
property, versus four of six on `late_dada`. Baseline detection is also far
healthier here — F ≈ 0.66 at 0.50 and recall 0.55, against 0.45 and 0.30.

**Whole-map**

| | baseline | `e01` |
|---|---|---|
| mean salience | 0.0240 | 0.0253 (**+6 %**) |
| bins > 0.50 | 2,807 | 2,863 (+2 %) |
| correlation `r` | — | **0.996** |
| equivalent threshold | 0.50 | **0.51** |

| baseline bin | n bins | Δ `e01` |
|---|---|---|
| 0.00 – 0.05 | 139,254 | +0.0008 |
| 0.05 – 0.10 | 2,566 | **+0.0119** |
| 0.10 – 0.20 | 2,475 | **+0.0120** |
| 0.20 – 0.30 | 1,148 | **+0.0116** |
| 0.30 – 0.40 | 787 | +0.0059 |
| 0.40 – 0.50 | 723 | +0.0096 |
| 0.50 – 0.60 | 831 | +0.0041 |
| 0.60 – 0.70 | 757 | +0.0042 |
| 0.70 – 0.80 | 467 | +0.0014 |
| 0.80 – 0.90 | 390 | −0.0008 |
| 0.90 – 1.00 | 362 | −0.0032 |

The `+6 %` mean and `r ≈ 1.00` replicate `late_dada` exactly. The *shape* does
not: here the lift stays positive up to ≈ 0.85, whereas on `late_dada` it crossed
zero at ≈ 0.45. **The level-dependence is excerpt-specific** — see §4.3.7. As
before, the mean shift is dominated by background: 55 % of the +0.0013 comes from
the 0.00–0.05 bin (93 % of all bins).

**Detection accuracy**

| | baseline | `e01` |
|---|---|---|
| P / R / F @ 0.50 | 0.842 / 0.548 / 0.664 | 0.842 / 0.555 / **0.669** |
| P / R / F @ matched count (1,128 peaks) | 0.842 / 0.548 / 0.664 | 0.844 / 0.551 / **0.666** (@0.51) |
| best F | **0.708** @ 0.15 | 0.706 @ 0.20 |

Flat at the default and at matched count, as on `late_dada` — but with a pattern
absent there. **At high thresholds `e01` is consistently better at equal
precision:**

| thr | baseline F | `e01` F | Δ |
|---|---|---|---|
| 0.65 | 0.610 | 0.617 | +0.007 |
| 0.70 | 0.583 | 0.599 | **+0.016** |
| 0.75 | 0.541 | 0.563 | **+0.022** |
| 0.80 | 0.503 | 0.519 | +0.016 |

Precision is within 0.002 across these rows, so this is recall gained, not an
operating-point trade. That is the signature the method predicts: activations
just under a high threshold are pushed over it, while confident detections and
the noise floor stay put. It does not show up at 0.15–0.50 because there the
threshold is already below the lifted mass.

**Per-voice evenness**

| | baseline | `e01` | Δ |
|---|---|---|---|
| spread (mean over 4 chords) | 0.545 | **0.514** | **−0.031** |
| min/max (mean over 4 chords) | 0.455 | **0.486** | **+0.031** |

Near-identical in magnitude to `late_dada` (−0.030 / +0.030). Again concentrated
in the quietest voices, and again the loud voices barely move:

| chord | quietest voice | baseline rel | `e01` rel |
|---|---|---|---|
| 1 | Bas F♯3 | 0.38 | 0.38 |
| 2 | Bas A3 | 0.69 | 0.68 |
| 3 | Bas D3 | 0.29 | **0.32** |
| 4 | Bas A2 | 0.47 | **0.57** |

Chord 4 carries most of it — Bas A2 rises 0.377 → 0.457 raw and Tenor B3
0.432 → 0.489, dropping that chord's spread from 0.53 to 0.43. Chord 2, where the
ensemble is already even (spread 0.31), moves −0.01 the wrong way. So the
improvement appears **where there is an imbalance to correct and not otherwise**,
which is the desired behaviour rather than a blanket gain.

#### 4.3.7 Cross-excerpt summary

The full design is **2 checkpoints × 2 excerpts**, all against the baseline:

| | `late_dada` `e01` | `late_dada` `e02` | `Parijs` `e01` | `Parijs` `e02` |
|---|---|---|---|---|
| Δ spread | **−0.030** | **−0.026** | **−0.031** | **−0.025** |
| Δ min/max | **+0.030** | **+0.026** | **+0.031** | **+0.025** |
| Δ F @ 0.50 | 0.000 | +0.001 | +0.005 | +0.002 |
| Δ F @ 0.70–0.80 | −0.002 to −0.007 | ≈ 0 | **+0.016 to +0.022** | **+0.013 to +0.019** |
| map `r` vs baseline | 0.995 | ~0.995 | 0.996 | ~0.996 |

Per excerpt, with `e01` as the representative checkpoint:

| | `late_dada` | `Parijs_dedetdoe` |
|---|---|---|
| voices / chords used | 4 / 6 (2 unusable) | 5 / 4 (all usable) |
| warp fit `r` | 0.416 | 0.505 |
| baseline F @ 0.50 | 0.448 | 0.664 |
| mean salience | +6 % | +6 % |
| map correlation `r` | 0.995 | 0.996 |
| equivalent threshold | 0.50 | 0.51 |
| Δ F @ 0.50 | 0.000 | **+0.005** |
| Δ F @ matched count | 0.000 | **+0.002** |
| Δ best F | −0.004 | −0.002 |
| Δ F @ 0.70–0.80 | −0.002 to −0.007 | **+0.016 to +0.022** |
| Δ spread | **−0.030** | **−0.031** |
| Δ min/max | **+0.030** | **+0.031** |

What replicates:

- **Calibration preservation.** `r ≈ 0.996`, equivalent threshold 0.50–0.51, mean
  +6 % on both. This is now well established, and it is the property the anchor
  was built for.
- **Voice evenness.** All four cells fall in **+0.025 to +0.031**. Two recordings,
  different ensembles, different voice counts, two independently-selected
  checkpoints, same magnitude. This is the strongest evidence in the report that
  the fine-tune does what it was meant to on real audio.
- **Lift of weak-to-mid activations.** The 0.05–0.30 bins rise by +0.011 to
  +0.012 on both excerpts — the single most consistent number in the comparison.
- **Checkpoint-invariance.** `e01` and `e02` agree to ≤ 0.006 on every metric on
  both excerpts (§4.3.5), so none of the above depends on which was shipped.

What does not:

- **High-end behaviour.** `late_dada` falls −0.014 at 0.80–0.90; `Parijs` falls
  −0.0008. The "mild compression" of §4.3.1 is one excerpt's shape, not the
  model's. What both share is the lift below 0.30; above it they diverge.
- **Detection accuracy.** Flat on `late_dada`, slightly positive on `Parijs`
  (+0.005 at 0.50, +0.016 to +0.022 at 0.70–0.80, replicated by `e02` at +0.013
  to +0.019). The `Parijs` result is the more trustworthy — better alignment, no
  dead voices, baseline F 0.66 vs 0.45 — and the high-threshold gain now holds for
  both checkpoints, which makes it more than a single artifact. But it is still
  one 4.8 s excerpt, and it vanishes at the thresholds where F actually peaks.
  **The defensible statement remains: detection accuracy is not harmed, and may be
  marginally helped at high thresholds.**

This supersedes §4.3.4, which was written on `late_dada` and `e01` alone.

### 4.4 What remains

- **A fully independent excerpt**, in neither the distillation pool nor the guard
  set. **This is now the only substantial gap.** Both excerpts scored so far are
  `--real_audio` guard excerpts: no gradient touched them, but both fed the
  checkpoint accept/reject decision, so they are held out from training and not
  from selection.
- **A threshold sweep extended below 0.15**, since best F sits at or beside the
  grid edge on both excerpts (0.15, 0.15, 0.20). The deployment operating point is
  still unknown, and every comparison here was made at thresholds that are
  probably too high — which matters, because §4.3.6 finds the model's clearest
  accuracy gain at 0.70–0.80 and nothing at 0.15–0.50.
- **Longer excerpts.** 7.8 s and 4.8 s, 10 chords in total. The evenness effect
  replicated at +0.030 on both, but neither carries error bars.

**Settled by §4.3.5:** `e02` is indistinguishable from `e01` on **both** excerpts
(`r = 0.999` and `1.000`, all metrics within 0.006). Either can be shipped; the
choice between them is not worth further evaluation, and `e02_s001122` in `--out`
should be replaced by either one.

**Settled by §4.3.6:** the `late_dada` dead-bass problem — the second excerpt has
all five voices audible in all four chords.

This model is now a **candidate validated on two guard excerpts**: shown not to
regress, shown to preserve calibration (`r ≈ 0.996` on both), and shown a
consistent **+0.025 to +0.031** gain on the targeted property across the full
2 × 2 of recordings and checkpoints. What is missing is a recording with no
selection contact.

---

## 5. Limitations

- **Real-audio evaluation covers two short guard excerpts, both with selection
  contact.** 7.8 s + 4.8 s, 10 chords. Neither carried gradient, but both fed the
  checkpoint accept/reject guard, so both are held out from training and not from
  selection. No fully independent recording has been scored (§4.4). This is the
  largest remaining gap.
- **The synthetic gain transferred only partially.** +0.116 inferred quiet-voice
  recall on soundfont renders (§3.4) became **+0.025 to +0.031 voice evenness**
  across all four excerpt × checkpoint cells and **+0.000 to +0.005 detection F**
  at the default threshold (§4.3.7). Candidate explanations —
  not distinguished by any measurement here — are that the gain is
  soundfont-specific; that the anchor pins the student toward a teacher which
  itself misses quiet voices (see below); that a −12 dB synthetic victim does not
  resemble real ensemble imbalance; or that §3.4's inferred `R_quiet` overstates
  the effect. The consistency of the evenness figure across two excerpts argues
  for a real but small effect rather than noise.
- **The alignments are moderate at best.** `compare_voice_salience.py` fitted the
  tempo warp at score/salience `r = 0.416` (`late_dada`) and `r = 0.505`
  (`Parijs`). The script's own docstring names a weak fit as the usual reason a
  voice reads near zero, so the two zero-Bass chords in §4.3.3 are not
  conclusively absent energy rather than misalignment — though F2/A♯2 in a
  recording whose other voices are S/M/T makes genuine absence the likelier
  reading, and `Parijs` at the better fit has no zero readings at all.
- **The level-dependence of the change does not replicate.** §4.3.1 and §4.3.6
  disagree on where the salience shift crosses zero (≈ 0.45 vs ≈ 0.85). Only the
  lift of the 0.05–0.30 bins is common to both. Any account of *how* this model
  differs from the baseline should rest on that, not on the compression story.
- **All training and validation material is synthetic**, soundfont-rendered from
  generated MIDI, from one soundfont and one seed.
- **One validation scene.** 120 chord pairs is a reasonable frame-level sample,
  but it is a single scene, a single victim level (−12 dB), and there are no error
  bars. §3.1 shows the run-to-run noise floor on quiet recall is sd ≈ 0.004, which
  is the scale at which §3.4's +0.116 should be read (large relative to noise) and
  §3.5's +0.123 should not (inside it).
- **Selection over 24 checkpoints biases the reported best**, as quantified in
  §3.5. The guards are absolute tolerances and do not correct for this. The
  practical consequence turned out to be nil — §4.3.5 shows the top checkpoints
  are indistinguishable on real audio, so picking a noise peak cost nothing here —
  but that is luck, not a property of the selection rule.
- **Positive drift is not guarded.** `guard_failures()` rejects only
  `real_worst < −drift_tol` and `real_d_high < −drift_high_tol`. This run's +4–7 %
  upward lift passed unchecked; it was read off the log manually. A run that
  inflated real-audio salience severely would not be rejected.
- **`gamma=1.0` leaves the low range of the anchor nearly unconstrained.** That is
  the intent — a quiet voice the teacher missed must be free to rise — but "quiet
  voice" and "background" both look like low-teacher-salience bins to the loss, so
  this setting provides little protection against inflation of the weak end. The
  ~6 % lift in §4.1 is consistent with that, and the 18 near-silent windows in the
  pool that would have counterweighted it carry weight ≈ 0 under `gamma=1`.
  `gamma=0` (uniform) and `gamma=0.5` were not run for comparison.
- **The anchor pins toward a teacher that itself misses quiet voices.** Confidence
  weighting mitigates this but does not remove it; a quiet voice audible in the
  distillation pool is one the anchor is partly instructing the student to keep
  missing.
- **The anchor pool is narrow in provenance.** Seven recordings, apparently one
  ensemble and recording setup. The anchor *defines* "real audio" for the model, so
  its generalisation beyond that setup is untested — and the drift screen shares
  the same limitation at n = 2 excerpts, 12.6 s total.
- **The anchor sees 26 of 50 frames per window.** The ±12-frame receptive-field
  crop discards roughly half of each window; the effective anchored duration is
  ~13.7 min of the 26.4 min pool.
- **Threshold 0.5 is very likely the wrong operating point.** All of §3 is scored
  there. §4.3.2 now shows directly that on real audio both models peak at F ≈ 0.54
  at **0.15, the bottom of the swept grid** — the true optimum is lower still, and
  recall at 0.50 is only ~0.30. The synthetic ranking of checkpoints is still
  informative, but the absolute recall and precision figures in §3 do not describe
  deployment, and the real-audio comparison in §4.3 has not been repeated at a
  sensible operating point.
- **Contributions are not separated.** `--pos_weight 4`, the anchor at λ=3, γ=1,
  and a functioning L2-SP all changed at once relative to plain `conv`
  fine-tuning. No ablation isolates them, so the attribution in §1 is mechanistic
  reasoning, not measurement.
- **BatchNorm is frozen by design, so no normalisation adaptation occurs.** If part
  of the quiet-voice deficit lives in the input normalisation, this strategy cannot
  address it.
- **The quiet-voice metric is inferred and diluted.** §3.4's `R_quiet` is solved
  for, not measured, and selection ranks checkpoints on a number where 5/6 of the
  signal comes from voices the fine-tuning does not target.
- **The run was stopped during epoch 3 of 6.** §3.1 justifies this — the metric had
  been flat for 800 steps — but whether a much longer run behaves differently is
  unknown.
