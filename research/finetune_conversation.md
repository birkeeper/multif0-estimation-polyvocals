# Conversation: unlearning mix-level bias in model3 (verbatim)

Date: 2026-07-20
Repo: multif0-estimation-polyvocals (branch `modern-libraries`)

---

## User

Model3 seems to have learned that when a voice is less prominent in the mix, its chance of being a F0 is also less. I want to unlearn that behaviour. However, I do not have the original dataset. I do have a custom sound font (`~/Documents/git/Choir-practice-midi-player/soundfonts/Choir_practice.sf2`) with SATB doo and da samples over their respective ranges. I also have an PWA application (`~/Documents/git/Choir-practice-midi-player/midi_player.html`) where these soundfont instruments (voices) can be assigned to each track in a loaded midi file and played. Given this. How would you approach the unlearning?

## Assistant

Good framing — "unlearn salience ∝ mix level" is exactly right, and your soundfont + MIDI setup is everything needed to manufacture the training data the original authors had (they used synthesized/multitracked choirs too). Here's how I'd approach it.

### Core idea

Fine-tune model3 on synthetic SATB mixes where **per-stem gain is randomized** but the **annotation target is gain-independent** (a voice is either singing or not). The model currently maps "quiet voice → low salience"; the fine-tuning data explicitly punishes that by presenting the same ground truth at −0, −6, −12, −20 dB stem levels. The correlation it learned only survives if the data supports it — remove the correlation from the data and gradient descent removes it from the model.

### Pipeline

**1. Render stems, not mixes.** Bypass the PWA for data generation — batch-render instead with `fluidsynth` from the CLI (it isn't installed yet, but it's one `apt install fluidsynth` away and reads your `Choir_practice.sf2` directly):

- Take a corpus of SATB MIDI files (you presumably have a folder of choir practice MIDIs; public-domain chorales — e.g. Bach chorale MIDI sets — are a good bulk source too).
- Split each MIDI into per-voice tracks, render **each voice to its own wav** with the doo/da instruments at full velocity.
- Keep the MIDI note on/off + pitch per track — that *is* your F0 annotation, perfectly aligned, for free. (One caveat: sung/sampled vowels aren't perfectly flat in pitch, but soundfont samples are close enough to the MIDI nominal pitch for the 60-bins/octave grid, especially with the σ=1-bin Gaussian blur the target already applies.)

**2. Mix with randomized stem gains.** For each training example, draw a random gain per stem — e.g. uniform in [−24, 0] dB, occasionally dropping a voice entirely (gain −∞, and remove it from the annotation) so silence is also learned. Sum to a mix. Each MIDI can yield many mixes with different gain draws — this is your augmentation multiplier and the *active ingredient* of the unlearning.

**3. Build inputs/targets with the existing code.** The repo already has everything:
- `utils.create_pump_object()` / `compute_pump_features` → HCQT+dphase input
- `utils.create_annotation_target()` → blurred binary target from (times, freqs) — feed it the MIDI-derived annotations, **identical across all gain draws** of the same piece
- `utils_train.patch_generator` / `keras_generator` → training patches

**4. Fine-tune, don't train from scratch.** Load `exp3multif0.h5`, keep `bkld` loss, use a small LR (~1e-4 or less), and fine-tune for a few epochs. Two guardrails:

- **Hold-out test both ways:** your balanced-tenor and quiet-tenor synthetic renders from earlier are a perfect probe pair — success = the quiet-tenor salience rises toward the balanced one, *without* the balanced case degrading.
- **Watch for catastrophic forgetting / domain shift:** your soundfont is not real singing (no vibrato, no consonants, static timbre). Fine-tuning only on it risks overfitting to soundfont timbre and getting worse on real recordings. Mitigations, in order of value: (a) also randomize which vowel (doo vs da), tempo, transposition ±2 semitones, and add light reverb/noise; (b) freeze the early conv layers and only fine-tune the later ones; (c) mix in some real data if you ever have any, even unannotated pieces via pitch-shifted self-training is possible but that's a later refinement.

**5. Optional but cheap and powerful:** since your final target is *unbalanced* mixes, make the gain distribution match the deployment condition — e.g. always pick one "victim" voice at −10 to −20 dB rather than mild jitter on all four. Curriculum-style: start mild, increase imbalance.

### Practical notes

- The current `requirements`/venv lacks `mido`/`pretty_midi` and `fluidsynth` — the data-gen script needs those (data generation can even live in a separate venv; only steps 3–4 need the TF environment).
- Training here is CPU-viable for *fine-tuning* scale (a few epochs on a few hundred 30–60 s mixes), though a GPU would obviously help. Your machine showed no CUDA driver earlier — worth knowing this will be slow but feasible.
- The repo has the training utilities but no ready-made fine-tune entry point, so a small `finetune.py` (load weights → compile → fit on the new generator) needs writing.

### Suggested first milestone

Smallest end-to-end loop: 10 MIDIs × 20 gain-draws → ~200 mixes → fine-tune a copy of model3 for 1–2 epochs → evaluate on your quiet-tenor probe. If the salience gap closes even partially, scale up the corpus and augmentation.

Want me to start building this — a `fluidsynth`-based stem-render + random-mix + annotation script, then the fine-tuning script? If so, point me at a folder of SATB MIDI files you'd like to use as the corpus.

---

## User

I was thinking more along the line of generating a midi with a large variety of SSATBB chords with randomised offset of the voices so they do not all start at the same time and you also get less complicated chords. Also the mixing of the voices is randomised using fixed levels and a chord is either doo or da, or mixed. This midi is converted to wav.

## Assistant

That's a good design — arguably better than a chorale corpus for this purpose. Random chords with staggered onsets give you dense, uniform coverage of the (pitch × voicing × balance) space, instead of the highly correlated harmony a real repertoire gives you. It's the same philosophy as training on noise-like stimuli: you want the model to learn "a voice is a voice at any level," not the statistics of Bach. A few refinements I'd suggest, then the build plan.

### Refinements to your scheme

**1. Do the level randomization *outside* the MIDI, per stem — not via velocity.** This is the one place your plan has a trap: if you encode loudness as MIDI velocity, most soundfonts (very possibly your doo/da samples too) map velocity to *timbre and attack*, not just gain — and the annotation-relevant thing you want to randomize is pure mix level. Worse, the model could learn "quiet = soft-velocity timbre" and you've reintroduced a confound. So: render each voice (or each voice-group) to a **separate wav stem** at constant velocity — fluidsynth can render one MIDI channel at a time, or you generate one MIDI per voice — and apply your fixed gain levels in numpy when summing. That also means one rendering pass yields *many* mixes (re-draw gains per training example), which is your augmentation multiplier for free. If you'd rather keep it inside MIDI, use CC7/CC11 (channel volume/expression) rather than velocity — but stems + numpy gains is simpler and more controllable.

**2. Fixed level set: good idea.** Something like {0, −6, −12, −18} dB, drawn per voice per chord (or per phrase). Discrete levels give you clean coverage and easy stratified evaluation ("recall at −18 dB stems"). Include the corner cases: all-equal (don't unlearn the balanced case) and one-voice-solo.

**3. Chord/voicing generation — a few things to include deliberately:**
- **Vary the voice count** per chord (2–6), not always full SSATBB — the model must not learn "there are always six F0s."
- **Unisons and octaves** between voices — the hardest and most choir-typical case; the annotation then has coinciding F0s, which the target generation handles naturally.
- **Semitone/tone clashes** occasionally — close seconds are where peak-picking struggles; good training signal.
- Keep each voice within its realistic range (S: C4–A5, A: F3–D5, T: C3–A4, B: E2–E4 roughly), random within that — no need for the chords to be musically sensible; "less complicated chords" isn't even necessary, random is fine and better for coverage.
- **Random onset *and* offset stagger** (your point) plus occasional rests per voice — so voices enter/exit independently and the target has per-voice silence. Stagger also prevents the model keying on synchronized onsets.

**4. Annotations come from the generator, not the MIDI file.** Since you're generating the chords programmatically, emit the (time, freq, voice) list directly from the generator (converting MIDI note → Hz), using the *actual* staggered on/off times. That sidesteps MIDI parsing entirely and is exact. One correction to apply: the sounding pitch of a sampled voice can sit a few cents off nominal — the 60-bins/octave grid (20 cents/bin) plus the σ=1-bin target blur absorbs that comfortably.

**5. Doo/da per chord, or mixed — agreed,** and make it per-voice-per-chord sometimes (soprano "doo" against tenor "da"), so timbre and voice identity are decorrelated too.

**6. Length/structure:** many short "scenes" (say 5–10 s of chords with one gain-draw) concatenated is better than one long uniform file — the training patch sampler (`patch_generator`, 50-frame ≈ 0.6 s patches) then sees diverse balance conditions across patches.

### Resulting pipeline

1. **`generate_chords.py`** — emits per-voice MIDI (constant velocity) + exact annotation (times/freqs) as CSV/JSON, with randomized chords, voicing, onsets/offsets, rests, doo/da program per note.
2. **Render stems** with fluidsynth CLI against `Choir_practice.sf2` (one call per voice-MIDI, `-F stem.wav`, then resampled to 22050).
3. **`make_mixes.py`** — draws fixed-level gains per voice per scene, sums stems, writes mix wavs + copies annotations (targets identical across gain draws of the same scene).
4. **Feature/target extraction** with the existing `utils.compute_pump_features` + `utils.create_annotation_target` → the `inputs/…_input.npy` / `outputs/…_output.npy` layout `patch_generator` expects.
5. **`finetune.py`** — load `exp3multif0.h5`, `bkld` loss, low LR, fit on the new generator; evaluate on your existing balanced/quiet-tenor probe pair after each epoch.

Prereqs to install: `fluidsynth` (system), `mido` or direct MIDI writing (pip) — everything else is already in the repo's environment.

One honest caveat to keep in view: this dataset is pure isolated-chord soundfont audio, so the fine-tuned model may drift toward that domain (I'd freeze the early conv layers and keep the LR low, and check the balanced probe doesn't regress — if it does, we reduce epochs or mix in the original-style balanced data more heavily).

Shall I start with step 1, the chord/MIDI/annotation generator? I'd put the new scripts in a `finetune/` folder in the repo. And do you want doo/da mapped as two instruments (bank/program numbers) in `Choir_practice.sf2` — if so, which program numbers are they?

---

## User

My PWA can be rewritten so that that it can convert an 7 stem (SSATBB), or 14 if we have the doos and das as separate tracks, to wave.

## Assistant

That works, and it has one real advantage: your PWA already embodies the correct mapping of `Choir_practice.sf2` programs to voices/vowels, so we don't have to reverse-engineer bank/program numbers — the renderer is already "ground truth" for how the choir should sound. Rendering in the browser is fully viable via `OfflineAudioContext` (faster than realtime, deterministic, no audio hardware involved).

If we go that route, here's what the PWA rewrite needs to guarantee for the stems to be usable as training data:

**Rendering requirements**
1. **One wav per voice-track (the 14-track layout is better).** Keeping doo and da as separate tracks per voice means gain and vowel stay independently controllable after rendering — 14 mono stems per scene, silent where that voice/vowel isn't singing. All stems must be **sample-aligned and equal length** (render each track over the full scene duration, including leading silence — don't trim).
2. **No master processing** — no reverb, compression, limiter, or per-track panning; fixed unity gain; constant velocity. Any loudness shaping must stay *out* of the render so we can apply the fixed-level mixing in numpy afterwards. (If the synth applies a velocity→filter curve, constant velocity sidesteps it.)
3. **Sample rate:** render at 44100 (or directly at 22050 if the synth allows — `OfflineAudioContext` accepts arbitrary rates); the pipeline resamples to 22050 anyway, so this is soft.
4. **Batch mode:** ability to load N MIDI files (or one long generated MIDI) and export all stems without manual clicking per file — e.g. a "render all to folder" using the File System Access API, with a filename convention like `scene0007_S1_doo.wav`.
5. **Determinism:** if the synth has any random element (round-robin samples, LFO phase), fine — it doesn't affect annotations — but the note timing must exactly follow the MIDI, no humanization added at render time.

**Division of labor then becomes:**
- **My side (this repo, Python):** `generate_chords.py` produces the multi-track MIDI files *plus* the exact annotation CSVs (times/freqs per voice) — annotations never touch the audio path, so PWA rendering quirks can't corrupt ground truth. Then `make_mixes.py` (random fixed-level gains over the 14 stems, sum, write mixes), feature/target extraction with the existing pump code, and `finetune.py`.
- **Your side (PWA):** MIDI in → 14 aligned stems out, per the requirements above.

One thing to settle so the two halves meet: **track↔instrument convention in the generated MIDI**. Simplest is 12 fixed tracks/channels — S1,S2,A,T,B1,B2 × {doo,da} — where the generator expresses vowel choice by placing notes on the doo-track or da-track of that voice, and the PWA just renders each track with its assigned instrument. (That's 12; you said 7/14 — if you have two distinct alto or another split in mind, tell me the exact voice list.) Note the annotation is per *voice*, merging its doo+da tracks.

Also confirm: SSATBB = S1, S2, A, T, B1, B2 (six voices, one alto)? You wrote "7 stem" — if there's a seventh (second alto? a hum vowel?), give me the full track list and I'll match the generator to it.

Shall I start on `generate_chords.py` + the annotation writer, with the MIDI track convention above (adjusted to your answer), while you rework the PWA's export? If you want, I can also look at `midi_player.html` and sketch the `OfflineAudioContext` render-to-wav changes for it.
---

## User

Why is mixing done afterwards? The PWA can mix it based on per track volume settings. Volume can even be adjusted before each chord in MIDI. MIDIs containing chords can be generated and the PWA converts them to mono 22050 Hz wavs

## Assistant

You're right, and my post-mixing recommendation doesn't hold up under what you're describing. Let me be precise about where my earlier objection actually applied and why it doesn't apply here.

### What my objection was really about

My "mix afterward" argument had one legitimate core — **don't use MIDI velocity for level** — because in most soundfonts velocity drives sample selection / filter cutoff / attack, so a quiet voice would also be a *different timbre*, and the model could learn "quiet-timbre = less likely F0." That reintroduces exactly the confound you're trying to remove.

But **per-track volume is not velocity.** In a WebAudio PWA, a track volume control is a `GainNode` — a pure linear amplitude scaling applied *after* synthesis. It changes level and nothing else. CC7 (channel volume) behaves the same way in essentially every synth. So the timbre confound I was guarding against simply doesn't exist in your approach. My objection was aimed at the wrong mechanism.

### The other reason evaporates too

The only real thing post-mixing bought was the "render once, redraw gains many times" augmentation multiplier — valuable *if rendering is expensive*. But `OfflineAudioContext` renders faster than realtime and your whole pipeline is automated MIDI-in → wav-out. Re-generating a MIDI with different per-track volumes and re-rendering costs essentially nothing. So the multiplier isn't worth a separate stem-storage + numpy-mixing stage. Your one-artifact pipeline (MIDI → mono 22050 wav) is simpler and just as capable.

And critically — recall what we established earlier: the model is **globally gain-invariant** (the pump does `amplitude_to_db(..., ref=np.max)` per harmonic). Only the *relative* balance between voices matters. That means:
- The PWA can apply any master gain to avoid clipping on the sum — it's irrelevant to training.
- What matters is only the *relative* per-track volumes, which the PWA controls exactly.

So mixing in the PWA is not just acceptable, it's clean.

### Two things to get right in the PWA path

1. **Clipping on the summed mix:** summing 6 voices at unity can exceed [−1, 1]. Because of gain-invariance you can freely apply a master attenuation (or normalize the final mono buffer) before WAV encoding — that won't affect the model. Just make sure the encode step doesn't hard-clip. Rendering/encoding in float and normalizing the peak is the safe move.
2. **Confirm volume ≠ velocity in your synth path.** Set velocity constant across everything and vary *only* the mixer gain (or CC7). As long as that's true, no timbre confound.

### One thing that stays on the generator side regardless

The **annotation** still comes from the generator, not the audio — the notes and their staggered on/off times define ground truth. The one rule the generator must apply: a voice that is **fully muted** (gain −∞ / not sounding) is *removed* from the annotation, but a voice that's merely *quiet* (−18 dB) **stays in** the annotation — that quiet-but-present case is the entire training signal for the unlearning. So the generator needs to know each voice's level to decide inclusion, which is fine since it's producing both the MIDI (with the volume settings) and the annotation.

So, revised division of labor:
- **Generator (Python, this repo):** chords, staggered timing, doo/da assignment, per-chord/per-track volume settings baked into the MIDI (via CC7 or track volume), **and** the matching annotation CSV (fully-muted voices excluded, quiet voices included).
- **PWA:** MIDI → mono 22050 Hz wav, mixed per its track/volume settings, no timbre-affecting level control, no clipping.
- **This repo:** feature/target extraction (existing pump code) → `finetune.py`.

Net: you dropped a whole stage and I have no good counterargument. Want me to write `generate_chords.py` to emit MIDI-with-volume + annotation in this single-mix convention? I just need the volume mechanism your PWA will read — per-track mixer volume, or CC7 events in the MIDI — and the exact voice list (is it 6 voices S1 S2 A T B1 B2, or is there a 7th?).

---

## User

midicontroller 7 (mainVolume) is settable per channel

## Assistant

Good — CC7 per channel is exactly the right hook. That locks the level mechanism: the generator writes a CC7 event on each voice's channel before each chord, and the PWA reads CC7 to set that channel's gain when it renders the mono mix. No velocity involved, no timbre confound, and because the model is gain-invariant only the *relative* CC7 values across channels carry meaning.

Here's the convention I'd lock in, with the two decisions that are still open:

### Locked
- **Channel = voice.** CC7 per channel = that voice's balance for the upcoming chord. Values chosen from your fixed set (e.g. 127 = 0 dB, and the dB steps mapped to CC7's 0–127 curve — I'll do the dB→CC7 mapping in the generator so the levels are perceptually the dB steps you want, not raw CC7 units).
- **Constant velocity everywhere.** Level lives only in CC7.
- **Annotation from the generator, per voice.** Fully-muted voice (CC7 = 0 / not sounding) → excluded; quiet-but-present voice (e.g. −18 dB) → **included**. That inclusion is the whole training signal.

### Two things I need from you

**1. The exact voice list.** You've said SSATBB and also "7 stem." Six voices would be S1, S2, A, T, B1, B2. What's the 7th — a second alto (SSATTB? SSAATB?), or something else? Tell me the list and I'll map channels to it.

**2. How the PWA picks doo vs da, since a chord can be "doo, da, or mixed" (i.e. vowel varies per voice per chord).** Two clean options:

- **(a) One channel per voice + Program Change events.** 6–7 channels total; the generator emits a MIDI Program Change on a voice's channel to switch it between the doo and da instruments per chord. This needs the **program (and bank) numbers** of the doo/da instruments in `Choir_practice.sf2`, and the PWA must honor in-file program changes.
- **(b) One channel per voice-vowel (the 12/14-channel layout).** The generator puts each note on the voice's doo-channel *or* da-channel; the PWA just statically assigns the 12/14 soundfont instruments to those channels once in its UI — no program numbers needed, and it matches how your PWA already assigns instruments to tracks. CC7 goes on whichever channel is active for that voice.

I recommend **(b)** — it needs no program-number bookkeeping, survives the PWA's existing static-assignment model, and the only cost is more channels. (Heads-up: MIDI channel 10 is the percussion channel by convention; if 14 channels are needed I'll route around it or you confirm your PWA treats channel 10 as melodic.)

If you give me the voice list and pick (a) or (b), I'll write `generate_chords.py` — randomized chords with staggered onsets/offsets, per-voice rests, doo/da assignment, CC7 balance draws from your fixed dB set (including the all-equal and single-victim cases), emitting the multichannel MIDI **plus** the aligned annotation CSV. Or, if you'd rather I just proceed, I'll default to (b) with S1 S2 A T B1 B2 and you can correct.

---

## User

What is the best method to retrain model3 given the chords?

## Assistant

Grounded in the actual model3 architecture ([models.py:251](models.py#L251)): two parallel 6-conv branches (`base_model`, mag + phase) → concat → `conv7`/`conv8` → `distribution` (360×1) → `squishy` (1×1 sigmoid), BatchNorm throughout, `bkld` loss, fully convolutional. That architecture shapes the answer, because the right method here is **not** "retrain," it's a **targeted debias** that removes the level→salience coupling while protecting everything the model already does well.

### 1. Freeze the feature extractors, fine-tune only the decision head

The bias "quiet voice → low salience" almost certainly lives in the **later layers** that integrate harmonic evidence into a confidence, not in the early 5×5 conv feature extractors (which just detect local time-frequency structure and are level-general). So:

- **Freeze** both `base_model` branches (`conv1a…harm2a`, `conv1b…harm2b`) — the front end that already generalizes to real audio.
- **Train** `conv7`, `conv8`, `distribution`, `squishy` — where the evidence→decision mapping is.

Why this is the *best* method and not just the cheapest:
- Far fewer parameters move → needs much less data and can't overfit the soundfont timbre as easily.
- It **protects against domain shift/forgetting**: the frozen front end keeps expecting the original HCQT distribution, so the model doesn't drift toward "soundfont-only" competence. This directly mitigates the biggest risk of training on synthetic doo/da audio.
- It concentrates learning exactly where the bias is.

Fallback: if the quiet-voice probe doesn't improve enough with the head alone, unfreeze `harm1`/`harm2` (the harmonic-integration layers) next — but only then.

### 2. Handle BatchNorm explicitly (the classic fine-tuning trap here)

The model is BN-heavy, including a BN right on the input. In TF2/Keras, setting `layer.trainable = False` on a BN layer also puts it in **inference mode** (stops running-stat updates) — which is what you want for the frozen branches: they keep the original dataset's normalization statistics, keeping the input domain aligned. Set `trainable=False` *before* `compile()`. Do **not** leave the frozen BN layers updating their running stats on soundfont data — that silently shifts the model off the real-audio domain even though the weights are "frozen."

### 3. Data/batch composition is half the method

The debias comes from the *distribution the head sees*, so:
- **Include balanced chords, not only unbalanced ones.** If you only show quiet-victim scenes, you'll overcorrect and blow up false positives on balanced mixes. Mix the all-equal case in (~30–40%).
- **Oversample the quiet levels** (−12/−18 dB, and the "present-but-quiet" voice) so the corrective signal is dense.
- **Keep the target identical across level variants** of the same scene — that identity *is* the supervision that says "level shouldn't change the label."

### 4. Strongest variant: explicit invariance (consistency) loss

Because you generate the data, you can render the **same scene at two balances** and put both in the same batch. Then add, on top of `bkld` to the shared target, a consistency term:

```
L = bkld(pred_A, target) + bkld(pred_B, target) + λ · mean( (pred_A − pred_B)² )
```

where A = balanced, B = quiet-victim, same notes. This *directly* optimizes "output invariant to relative level" rather than hoping the data average teaches it. It's the most surgical possible attack on this specific bias. It needs a small custom training loop (the stock generator yields single examples), so start with plain supervised fine-tuning (§1–3) and add this only if the plain version leaves residual level-dependence.

### 5. Optimizer / schedule

- Adam, **low LR** — 1e-4 (head-only can tolerate up to ~3e-4; stay conservative to preserve behavior).
- **Few epochs** — this is fine-tuning; expect single-digit epochs. Overtraining on synthetic data is the main way to make real-audio performance worse.
- Keep `bkld` loss unchanged (the target is still a blurred binary salience).

### 6. Validation = the invariance gap, not accuracy

Don't select on overall accuracy — select on the thing you're fixing. Held-out matched pairs (balanced vs. quiet-victim, same notes), tracked every epoch:
- **quiet-voice recall must rise**, and
- **balanced-case precision/recall must not drop.**

Stop at the epoch that maximizes the first without regressing the second. Your existing balanced/quiet-tenor synthetic renders are exactly this probe. If you have *any* real annotated recording, keep it as a forgetting canary even if it's just one file.

---

**Summary of the recipe:** freeze `base_model` branches (BN in inference mode) → fine-tune `conv7/conv8/distribution/squishy` with Adam@1e-4, `bkld`, few epochs → data = balanced + quiet-oversampled with identical targets across levels → select on the balanced-vs-quiet invariance gap → escalate to unfreezing `harm1/harm2` or adding the consistency loss only if needed.
