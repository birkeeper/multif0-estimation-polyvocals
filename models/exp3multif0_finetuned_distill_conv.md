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

---

## 4. Behaviour on real recordings

**The real-audio evaluation that decides whether this model is useful has not been
performed.** §3 is entirely soundfont renders scored at a threshold of 0.5. This
section documents only what the run itself measured.

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

### 4.3 The outstanding evaluation

To settle it, the procedure from the sibling report on the AdaBN variant should be
repeated on `e01` and `e02`:

- salience maps via `predict_on_audio.py --save_salience`, analysed with
  `finetune/compare_voice_salience.py` (peak-picked, ±80 cent match), against each
  excerpt's own MIDI;
- detection F **at the default 0.50, at a matched detection count, and at each
  model's own best threshold** — the three columns that separate calibration from
  accuracy;
- per-chord **evenness across voices** (spread and min/max relative to the loudest
  voice), which is the property actually targeted.

Until that exists, this model should be treated as **an unvalidated candidate**.
A synthetic gain of this shape is not self-evidently transferable: the sibling
report documents a variant that posted a *larger* synthetic improvement and was
then worse on both real recordings. The difference in this run's favour is §4.1 —
that model's salience was compressed ~20 % with `r = 0.66–0.85`, this one's is
lifted ~6 % with `r ≈ 1.00` — but that is a reason to expect transfer, not
evidence of it.

---

## 5. Limitations

- **No real-audio accuracy evaluation.** §4 is a distribution check, not a
  measurement of detection accuracy or voice evenness. This is the single largest
  gap in the report and the reason the model is not recommended for use yet.
- **All training and validation material is synthetic**, soundfont-rendered from
  generated MIDI, from one soundfont and one seed.
- **One validation scene.** 120 chord pairs is a reasonable frame-level sample,
  but it is a single scene, a single victim level (−12 dB), and there are no error
  bars. §3.1 shows the run-to-run noise floor on quiet recall is sd ≈ 0.004, which
  is the scale at which §3.4's +0.116 should be read (large relative to noise) and
  §3.5's +0.123 should not (inside it).
- **Selection over 24 checkpoints biases the reported best**, as quantified in
  §3.5. The guards are absolute tolerances and do not correct for this.
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
- **Threshold 0.5 may be the wrong operating point.** All of §3 is scored there,
  and there is independent evidence that 0.5 is far too high for real audio. The
  synthetic ranking of checkpoints is still informative, but the absolute recall
  and precision figures are unlikely to describe deployment.
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
