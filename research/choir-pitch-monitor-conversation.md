# Conversation: Real-time Choir Pitch Monitoring Application

*Conversation with Claude Code (Claude Fable 5), 2026-07-11. Saved for continuation on another computer.*

---

## User

How would you structure an application that can real-time show if a choir and which voices (soprano, mezo, alto, tenor, bass) is/are off pitch? What algorithm would you use to determine if a voice (multiple singers) is off pitch? What programming language would you use? (Needs to run on a Windows laptop with audio equipment attached) What libraries would you use?

## Claude

Nice problem — real-time polyphonic pitch monitoring for a choir. Here's how I'd structure it.

### The core difficulty first

A choir is a **polyphonic, multi-singer-per-part** signal. If you record the whole choir with one microphone, separating "which section is off" is a genuinely hard source-separation problem — five parts, each sung by multiple voices with vibrato, overlapping harmonics (an alto's note is often literally a harmonic of the bass note). So the single most important architectural decision is:

**Use one microphone per section (5 close mics), not one room mic.** This turns an unsolved research problem into five parallel *monophonic-ish* pitch trackers, which is very tractable. Section mics (e.g. cardioid mics on stands per section, into a multi-channel USB audio interface like a Focusrite Scarlett 18i20 or Behringer UMC1820) give you channel-per-voice-part.

### Architecture

```
[5 mics] → [USB audio interface, 5 channels @ 44.1 kHz]
                     │
        Audio capture thread (callback-based, ~46 ms blocks)
                     │
     Per-channel DSP pipeline (5 parallel workers):
       1. High-pass filter (remove rumble)
       2. Voice-activity detection (RMS/spectral gate — is the section singing?)
       3. Pitch estimation → f0 per frame
       4. Median smoothing over ~5 frames (kill octave-error blips)
                     │
        Reference comparison:
       score-following (MusicXML/MIDI of the piece) OR
       nearest equal-tempered note relative to a set tuning/key
                     │
       Deviation in cents + confidence per section
                     │
        UI thread: real-time dashboard
       (per-section pitch trace, cents-deviation meter,
        red/green "off pitch" indicator with hysteresis)
```

Key structural points:

- **Decouple audio from UI**: audio callback pushes frames into a queue; DSP workers consume; UI polls results at ~20–30 fps. Never do heavy work in the audio callback.
- **Hysteresis on the "off pitch" verdict**: only flag a section if deviation exceeds a threshold (e.g. ±25–30 cents) for several consecutive frames (~200 ms). Singers scoop into notes and use vibrato (±50 cents is normal vibrato!) — you must average through vibrato (track the mean f0 over ~300 ms), or you'll flag everyone constantly.
- **Two comparison modes**:
  - *Score-following mode*: load the piece (MusicXML via `music21`), track position (simplest: conductor taps tempo, or a click track; fancier: online DTW alignment), compare each section's f0 to its expected note. This tells you off-pitch *relative to the score*.
  - *Free mode*: snap to nearest semitone of a chosen reference (A=440 or drifted reference) and show deviation. Also lets you detect the classic choir failure of *collective pitch drift* — track the ensemble's average deviation over time.

### Pitch algorithm

For monophonic-per-channel tracking, in order of recommendation:

1. **pYIN** (probabilistic YIN) — the practical gold standard for singing voice. Robust to breathiness and vibrato, gives a confidence value you can use to gate the display. Available in `librosa.pyin`.
2. **YIN** — cheaper, sim—

*(interrupted by user)*

## User

Single microphone

## Claude

With a **single microphone**, the problem changes fundamentally: you now need **multi-pitch estimation (multi-f0)** from a polyphonic mixture — and honest expectation-setting matters, so let me give you the realistic version.

### What's feasible with one mic

- **Detecting the set of pitches currently sounding** (e.g. "there's an F3, C4, A4, F5 sounding"): feasible with modern deep-learning multi-f0 models.
- **Assigning each detected pitch to a section**: not solvable from timbre alone (choir voices blend deliberately), but very solvable **score-informed** — you know soprano should be on A4 and tenor on C4, so you match detected pitches to expected notes by proximity.
- **Detecting that a section is off**: works when the section is off *together* or the section's pitch is audibly displaced. What you **cannot** reliably do with one mic is spot one wobbly singer inside an otherwise-correct section, or distinguish "altos are 30 cents flat" when their note coincides with a strong harmonic of the bass. Also: if two sections sing in unison/octaves, they partially merge into one detection.

So the honest product framing: *"per-section deviation estimate, score-informed, with confidence indicators"* — not a perfect per-voice detector.

### Architecture

```
[1 mic] → audio capture (callback, 44.1 kHz, ~92 ms hop)
              │
   Multi-f0 estimation (neural net, GPU optional)
   → list of (f0, salience) per frame, typically 1–6 pitches
              │
   Score follower (MusicXML of the piece → expected note per section per beat;
   position via click track / tap tempo / online DTW)
              │
   Assignment: match each detected f0 to the nearest expected section note
   (Hungarian algorithm / greedy within ±100 cents, weighted by salience)
              │
   Per-section: deviation in cents + confidence
   (confidence low when: expected note not detected, sections in unison,
    note coincides with another section's harmonic)
              │
   Smoothing + hysteresis (~300 ms, threshold ±25–35 cents)
              │
   UI: piano-roll overlay of expected vs detected pitch per section,
   per-section meter, red/amber/green with confidence dimming
```

The **score-informed assignment step is what makes this work at all** — without the score, you can't say which pitch belongs to which section. So the app must load the piece (MusicXML) and know roughly where you are in it. Simplest reliable position tracking: the conductor/operator uses a click track or foot-taps the beat; online DTW alignment is a later upgrade.

### Algorithm for multi-f0

Classical methods (autocorrelation, YIN) are monophonic — useless here. Use a trained neural multi-pitch model:

1. **Basic Pitch (Spotify)** — open source, lightweight (runs real-time on CPU via ONNX/TFLite), outputs multi-f0 with cent-level resolution. Easiest to integrate; good first choice.
2. **Cuesta & Gómez's vocal-ensemble multi-f0 models** (research code on GitHub, trained specifically on **choir/vocal quartet** data — the *Choral Singing Dataset*, ESMUC quartets). Best accuracy for exactly this use case, more integration work.
3. **Deep Salience (Bittner et al.)** — the classic multi-f0 salience representation; heavier, mostly superseded by the above.

Recommendation: prototype with Basic Pitch, evaluate against recordings of the actual choir, and swap in the Cuesta model if section resolution is insufficient.

### Language & libraries

**Python** — every relevant model has a Python implementation, and it's fine performance-wise since the heavy lifting is in C/ONNX under the hood. Windows-friendly throughout.

| Purpose | Library |
|---|---|
| Audio capture | `sounddevice` (PortAudio, WASAPI on Windows, callback-based) |
| Multi-f0 model | `basic-pitch` (ONNX) or Cuesta model in PyTorch |
| Inference runtime | `onnxruntime` (CPU is enough for Basic Pitch) |
| DSP / smoothing | `numpy`, `scipy` |
| Score parsing | `music21` (MusicXML → expected notes per part) |
| Assignment | `scipy.optimize.linear_sum_assignment` |
| UI | `PyQt6` + `pyqtgraph` (fast real-time plotting; Dear PyGui is a good alternative) |

Threading: audio callback → `queue` → inference worker thread → UI timer at ~20 fps. Expect ~100–200 ms end-to-end latency, which is fine for a monitoring dashboard (it's not a performance effect that needs <10 ms).

### Practical tips

- Vibrato is ±30–50 cents at ~6 Hz — compare the **~300 ms mean** f0, not instantaneous frames, or everything flags red.
- Track **ensemble drift** separately: average deviation of all matched pitches over time. Choirs drifting flat together is the most common real-world issue, and it's the one thing a single mic detects *well*.
- Dim/grey the indicator for a section whenever confidence is low (unison passages, expected pitch masked by harmonics) — showing "unknown" honestly beats false accusations, which choir members will notice immediately.
- If the single-mic constraint is about setup effort rather than hardware: even a single **stereo** mic or two mics (left/right of choir) adds spatial cues that meaningfully help assignment. Worth keeping as an option.

## User

Want to know difference from nearest tuned note for all voices. In the end, I want to see which parts are vulnerable to get off tune (lower)

## Claude

That simplifies things nicely — you don't need score-following at all. Snap every detected pitch to the nearest equal-tempered note, log the signed deviation in cents, and analyze over time. This is a **rehearsal analytics** tool more than a live alarm, and single-mic multi-f0 is well suited to it.

### Pipeline

```
[1 mic] → multi-f0 (Basic Pitch / Cuesta model)
   → per frame: list of (f0, salience)
   → for each f0: nearest note = round(12·log2(f0/440)) semitones from A4
     deviation = 1200·log2(f0 / f_nearest)  → signed cents, range −50…+50
   → assign to voice part by pitch register (see below)
   → log (time, part, note, deviation, salience) to disk
   → live view + post-rehearsal report
```

### Two things you must handle

**1. Assigning pitches to parts without a score.** Voice ranges overlap heavily (alto and tenor share nearly an octave). Practical options, in increasing quality:

- **Register bands with overlap-flagging**: bass ≲ E3, tenor ~C3–G4, alto ~F3–D5, soprano ≳ B3. Unambiguous frames get assigned; ambiguous ones get logged as "alto-or-tenor". Over a whole rehearsal you still get useful per-part statistics because much of the material sits in unambiguous register.
- **Voice ordering**: in homophonic passages, sort simultaneous f0s top-to-bottom → soprano, mezzo, alto, tenor, bass. Works when all five are singing; breaks on crossings and rests.
- Combine both: order when you detect ~5 pitches, register bands otherwise.

If per-part attribution turns out too noisy, the fallback is loading a MusicXML of the piece purely for *pitch-to-part matching* (not time alignment — just "which part contains this pitch class near this register"), which is much less work than full score-following.

**2. The drift trap — the crucial one for this goal.** If the choir sinks *gradually*, nearest-note deviation goes 0 → −20 → −40 → then **wraps to +45** as the nearest note flips to the semitone below. Naïve nearest-note comparison makes a slowly flattening choir look like it "recovered." Since sagging pitch is exactly what you want to measure, do both of these:

- **Unwrap the deviation**: track continuity — if deviation jumps by ~100 cents between frames, the note snapped; add ±100 to keep a continuous drift curve. Maintain a **cumulative ensemble drift** signal (median deviation of all detected pitches, unwrapped over time). This directly plots "the choir has sunk 60 cents since the start."
- **Per-part deviation relative to ensemble drift**: `part_deviation − ensemble_drift` tells you *who is dragging*. If ensemble drift is −40 cents and the tenors are consistently at −55 while sopranos hold −25, the tenors are your vulnerability — they're leading the descent. This decomposition (who's flat relative to the group vs. group flat relative to A440) is exactly the "which parts are vulnerable" answer.

### Output views

- **Live**: per-part meter (signed cents, green ±15 / amber ±30 / red beyond), plus one big "ensemble drift" trace.
- **Post-rehearsal report** (the real payoff):
  - Histogram of deviations per part — a part whose histogram is centered at −18 cents is chronically flat even if rarely "red."
  - Drift-onset analysis: cumulative drift curve with per-part contribution overlaid → shows *which part starts sinking first* before the others follow.
  - Optionally break down per part × pitch (e.g., "tenors go flat specifically above E4" — descending after high passages is a classic mechanism).

### Stack (unchanged)

Python on Windows: `sounddevice` for capture, `basic-pitch`/ONNX (or the Cuesta vocal-ensemble model — extra worthwhile here, it's trained on choirs) for multi-f0, `numpy`/`scipy` for the cents math and unwrapping, `pyqtgraph`/PyQt6 for the live view, and a simple Parquet/CSV log + `matplotlib` for the report. All real-time on a normal laptop CPU.

Validation tip: before pointing it at the choir, test with a synthesized four-part chorale (render MIDI with a slow global pitch bend applied to one part) — you'll know your unwrapping and part-assignment logic works before dealing with real acoustic mess.

## User

Maybe not show per part, but per note sung. Not all parts may sing all of the time

## Claude

That's actually a cleaner design — it drops the hardest, least reliable step (part assignment) entirely. You just report on **what's sounding**: every detected pitch becomes "nearest note + signed cents deviation", keyed by the note itself (e.g. F♯3, C4, A4).

### What changes

**Live view** becomes a dynamic list of currently-sounding notes:

```
 A4   ──────●──   −8 cents        (salience 0.9)
 F4   ────●────   −2 cents        (0.7)
 C4   ───────●─  −22 cents  ⚠     (0.8)
 F3   ─────●───   −6 cents        (0.6)
```

Notes appear and disappear as the music moves — no assumption about who is singing or how many parts are active. A piano-roll trace (detected pitch curves against the equal-tempered grid lines) works well as the second view: sagging notes are visible as curves drooping below their grid line.

- Only show a note once it's been stable for ~150–250 ms (kills onset scoops and octave-error flickers).
- Weight/dim by salience so weak spurious detections don't draw attention.

**Analytics** become per-note instead of per-part:

- **Deviation histogram per note**: "C4 is sung at −20 cents on average, A4 is fine."
- **Note × time heatmap** of mean deviation across the rehearsal — flat-prone notes light up as a row of blue/red.
- The **ensemble drift** signal stays exactly as before (median unwrapped deviation of everything sounding) — still the headline "the choir sank 40 cents in this passage" number, and per-note deviations should be shown *relative to it* so a globally-sunk choir doesn't paint every note red.

And the nice part: **per-note statistics still answer the original question**. Since you (or the conductor) know the repertoire, "C4 and D4 are chronically flat" translates immediately to "that's the tenors' sustained line in bar 30" — the human does the part-mapping for free, and does it more reliably than register heuristics would.

### Pipeline simplification

The whole assignment/Hungarian-matching stage disappears:

```
mic → multi-f0 → per f0: nearest note + signed cents
    → stability gate (~200 ms)
    → log (time, note, cents, salience)
    → live meters + per-note report
```

One remaining subtlety: keep the **unwrapping per tracked note contour**, not per frame — track each sounding note as a continuous contour (a pitch trajectory that stays within ~±80 cents of where it started), so if a held C4 sinks past the quarter-tone boundary it's logged as "C4 at −60 cents", not as a new B3 at +40. Basic Pitch already gives you note-event grouping that helps with this; otherwise a simple contour tracker (nearest-continuation matching frame to frame) does it.

Everything else — Python, `sounddevice`, Basic Pitch or the Cuesta choir model, `pyqtgraph` live view, logged CSV/Parquet plus a post-rehearsal report — stays the same.

## User

I want to be able to scroll back in time so see and hear where it went wrong

## Claude

Good addition — that turns it from a live meter into a reviewable session recorder. The key architectural change: **record everything to disk continuously, and make time the primary axis of the UI.**

### What to store

Everything is keyed by one master clock (sample position of the audio stream — don't use wall-clock, it drifts from the audio):

1. **Raw audio** — append to a WAV file as it's captured (44.1 kHz mono is ~5 MB/min, so a 2-hour rehearsal is ~600 MB; fine. Optionally FLAC-compress afterwards).
2. **Analysis log** — per frame: `(sample_pos, note, cents, salience)` plus the derived contours and ensemble-drift value. Parquet or SQLite; a full rehearsal is a few MB.

Because analysis is stored, scrolling back is instant — you never re-run the model on playback, you just re-render logged data. (Bonus: you can improve the analysis code later and re-run it over old recordings.)

### UI structure

Think "DAW-style transport":

```
┌─────────────────────────────────────────────────────┐
│  Piano-roll: detected note contours, colored by      │
│  cents deviation (blue = flat, red = sharp),         │
│  equal-tempered gridlines                            │
│  ────────────────▌ (playhead)                        │
├─────────────────────────────────────────────────────┤
│  Ensemble drift curve (same time axis)               │
├─────────────────────────────────────────────────────┤
│  Overview strip: full session minimap, ⚠ markers     │
│  where deviation exceeded threshold — click to jump  │
├─────────────────────────────────────────────────────┤
│  ⏵ ⏸ │ LIVE ● │ ◀◀ ▶▶ │ loop A–B │ speed 0.75×      │
└─────────────────────────────────────────────────────┘
```

- **LIVE mode**: view follows the incoming audio (recording never stops).
- **Scrub mode**: drag/scroll anywhere in the timeline while recording continues in the background; hit ⏵ to hear that spot. A "back to live" button returns you.
- **Trouble markers**: whenever a note's deviation crossed threshold for >N ms, drop a marker in the overview strip — "where it went wrong" becomes click-the-warning rather than hunting.
- **A–B loop**: select the bad passage and loop it — the natural rehearsal workflow ("listen, this is where the tenors sink").
- Half-speed playback (via `librosa.effects.time_stretch` or just resampled playback) helps hearing exactly who drifts.

### Implementation notes

- **Playback while recording**: `sounddevice` handles simultaneous input and output streams fine on Windows (WASAPI). Playback reads from the already-written WAV via `soundfile` — seek to any sample position, stream from there. Keep the last ~30 s also in a RAM ring buffer so scrubbing near "now" doesn't race the file writer.
- **Playhead sync**: drive the piano-roll cursor from the playback stream's sample position, so what you *see* highlighted is exactly what you *hear*.
- `pyqtgraph` handles the scrollable/zoomable piano-roll well (set the x-axis to seconds, use `setXRange` to follow the playhead). The minimap is a downsampled second plot with a click handler.
- Store sessions as a folder: `session_2026-07-11/audio.wav + analysis.parquet + markers.json` — trivially archivable and comparable across rehearsals ("is the drift on this piece improving week to week?").

Nothing in the analysis pipeline changes — you're adding a persistence layer and a transport/timeline UI on top of it. This is the point where the app is worth structuring as three clean modules: **capture+analyze** (writes the session), **session store**, and **viewer** (which works identically live or on an old session file — you get an offline "load a rehearsal recording" feature for free, which is also the easiest way to develop and test the UI without a choir in the room).

## User

Why python? Is that fast enough?

## Claude

Yes — comfortably, because in this app Python is the *glue*, not the compute. Where the cycles actually go:

| Stage | Actual workload | Who does the work |
|---|---|---|
| Audio capture | Copy 4k samples/callback | PortAudio (C), via `sounddevice` |
| Multi-f0 inference | ~99% of the CPU budget | ONNX Runtime (C++, SIMD, multithreaded) |
| Cents math, unwrapping, contours | A few thousand floats per second | NumPy (C) |
| Logging | A few KB/s to disk | trivial |
| UI | Drawing plots at ~20 fps | pyqtgraph on Qt (C++) |

The Python interpreter itself executes maybe a few thousand lines of orchestration logic per second. Rule of thumb: Python is slow at *tight per-element loops*, but this pipeline is all *batch operations on arrays* handed to native code.

Concrete numbers for the heavy stage: Basic Pitch's ONNX model runs at roughly 10–30× real-time on a normal laptop CPU — i.e., analyzing one second of audio takes ~30–100 ms. You need 1× real-time with headroom; you have an order of magnitude. Audio I/O latency is handled entirely inside PortAudio's callback thread and is unaffected by Python's speed. And crucially, this is a **monitoring** app: the feedback loop is a human looking at a screen, so 100–300 ms end-to-end latency is completely fine. This isn't a guitar-effects pedal where 10 ms matters.

The one real Python hazard is the **GIL blocking the audio callback** if you do heavy Python work at the wrong moment. Standard mitigation is the architecture described: the callback only copies the buffer into a queue (microseconds), and inference runs in a worker — and ONNX Runtime *releases the GIL* during inference, so the app stays responsive.

Why Python beats the alternatives *for this project specifically*:

- **The models live there.** Basic Pitch, the Cuesta choir models, librosa — all Python. In C++/Rust/C# you'd export models to ONNX (fine) but rewrite all the pre/post-processing (spectrogram parameters, note-event grouping) yourself and hope it matches — a classic source of subtle wrongness.
- **Iteration speed.** Most of this project is tuning thresholds, smoothing, and visualizations against real choir recordings. That experimentation is 5× faster in Python.

When *would* I switch? If this became a polished product for distribution (installer size, no Python runtime), C# + ONNX Runtime + NAudio is the natural Windows choice; or Rust/C++ for sub-10 ms latency effects processing someday. For a proof-of-concept-to-internal-tool on your own laptop, neither reason applies.

Cheap test if skeptical: a 30-line script that captures mic audio and runs Basic Pitch on 1-second chunks, printing processing time per chunk — you'll see the real-time factor on the actual laptop before committing.

## User

Is a rust application an option? Single executable?

## Claude

Yes, Rust is a genuine option, and it delivers the single-executable goal well. The trade: you get a distributable, fast, robust binary; you pay by re-implementing the model's pre/post-processing yourself and iterating more slowly.

### Rust stack

| Purpose | Crate |
|---|---|
| Audio capture/playback | `cpal` (WASAPI on Windows, callback-based, solid) |
| Neural inference | `ort` (ONNX Runtime bindings) or `tract` (pure-Rust inference) |
| DSP (spectrograms, filters) | `rustfft` / `realfft` + hand-rolled |
| UI + real-time plots | `egui`/`eframe` — immediate-mode GUI, excellent for a live-updating piano-roll dashboard |
| WAV read/write | `hound`; FLAC via `claxon`/`symphonia` |
| Session log | `parquet` crate or `rusqlite` (SQLite statically linked) |

### Single-executable question

- With **`tract`** (pure Rust, no C++ dependency): genuinely one `.exe`, model weights embedded via `include_bytes!`, zero DLLs, zero installer. Basic Pitch-sized models run fine in tract.
- With **`ort`**: faster inference, but ONNX Runtime is a native library — static-link or embed-and-extract the DLL; still shippable as one file with build plumbing.
- `egui`, `cpal`, `hound`, `rusqlite` all compile in statically. Realistic final size: 20–50 MB self-contained exe. Compare: a PyInstaller one-file build of the Python version works too, but it's ~500 MB+, slow to start, and periodically trips antivirus — if "hand a colleague one exe" is a real requirement, Rust wins clearly.

### The real cost — and it isn't the language

The model exports to ONNX trivially. What doesn't come along: Basic Pitch's specific input transform (a CQT-style spectrogram with exact hop/window/normalization) and its post-processing (turning frame activations into note events with onset/offset logic). You must port that to Rust and verify it matches the reference implementation, or accuracy silently degrades. That's the risky, fiddly part — budget real time, and validate by running the same WAV through both implementations and diffing the output. (Worth a search before starting: C++ ports of Basic Pitch exist, e.g. from the NeuralNote project — good porting references.)

Second cost: iteration speed — threshold tuning, smoothing, UI experiments against real recordings (the bulk of the project) is slower in Rust than Python.

### Recommendation: sequence it

1. **Prototype in Python** — validate that single-mic multi-f0 on *your* choir in *your* room is good enough at all, and tune the detection/smoothing/drift logic. This is the project's actual risk, and Python retires it fastest.
2. **Port to Rust** once the pipeline is proven and parameters are frozen — at that point it's a well-specified translation job, not research, and NeuralNote's C++ code plus the Python reference make the DSP port checkable.

If deployment matters from day one and you'd rather go straight to Rust, that's defensible — just do the model-plumbing spike first (mic → spectrogram → ONNX → pitches printed to console, diffed against Python output) before investing in the UI, since that's where the project would fail if it fails.

## User

What is the ONNX model?

## Claude

ONNX (Open Neural Network Exchange) is a **standard file format for trained neural networks** — think "PDF for models." It solves the portability problem: models get *trained* in frameworks like TensorFlow or PyTorch — huge Python-centric ecosystems you don't want to ship inside an application. Exporting to ONNX freezes the trained network — its architecture (the graph of layers/operations) and its learned weights — into a single `.onnx` file that any compliant runtime can execute, no TensorFlow/PyTorch required.

Two separate pieces, easy to conflate:

- **The ONNX model** — a data file, e.g. `basic_pitch.onnx` (~10 MB for Basic Pitch). Pure data: graph + weights. This is what you'd embed in a Rust executable.
- **An ONNX runtime** — the engine that loads that file and executes it. Microsoft's **ONNX Runtime** is the dominant one (C++, with bindings for Python, Rust via `ort`, C#, etc.); **tract** is a pure-Rust alternative. The runtime takes your input array, walks the graph doing the matrix math, and returns the output arrays.

In this app, concretely:

```
audio chunk (f32 samples)
  → your code: compute the spectrogram (input transform)
  → ONNX runtime: run basic_pitch.onnx on it
  → output: matrices of pitch activations (which f0s are sounding, per frame)
  → your code: turn activations into note events, cents deviations
```

So "the ONNX model" = **Basic Pitch's trained multi-pitch network, exported to this portable format**. Spotify ships it in several formats (TensorFlow, TFLite, CoreML, ONNX) so integrators can pick their runtime — the ONNX one is what you'd use from Rust or C#.

Key implication: only the network is in the file. The code *before* it (exact spectrogram computation) and *after* it (activation → note events) is ordinary application code in Basic Pitch's Python package — that's the part you must faithfully re-implement when porting to Rust; the network itself transfers for free.

## User

Are there pretrained multi-pitch models available

## Claude

Yes — several, ready to download. The landscape, filtered for this use case:

### Directly usable

**Basic Pitch** (Spotify) — the pragmatic default.
- Trained on a broad mix of instruments *and vocals*; outputs multi-f0 with cent-resolution pitch-bend information, plus note-event grouping.
- Small (~10 MB), documented, actively used, ships in ONNX/TFLite/CoreML/TF formats. `pip install basic-pitch`, works in minutes.
- GitHub: `spotify/basic-pitch`.

**Cuesta & Gómez vocal-ensemble models** — the best match for choirs specifically.
- Research models trained on **vocal quartets and choir recordings** (Choral Singing Dataset, ESMUC/Dagstuhl quartet sets), built exactly for "multiple simultaneous singing voices," including handling of unisons and voice proximity that general models struggle with.
- GitHub: `helenacuesta/multif0-estimation-polyvocals` (and her follow-up voice-assignment work). Pretrained Keras/TensorFlow weights included; more integration effort than Basic Pitch. Paper: Cuesta et al., ISMIR 2020.

**Deep Salience** (Bittner et al., ISMIR 2017) — the classic multi-f0 salience model, pretrained weights at `rabitt/ismir2017-deepsalience`. Historically important, still works, largely superseded by the two above.

### Adjacent, worth knowing about

- **MT3 / Google's music transcription models** — multi-instrument transcription to MIDI notes. Outputs quantized notes rather than continuous f0, so you lose the cents-deviation signal that is the whole product. Not a fit.
- **Onsets & Frames / MAESTRO models** — piano-only. Not a fit.
- **CREPE, PESTO, SPICE** — excellent pretrained pitch trackers, but **monophonic**. Only relevant for the mic-per-section design, where CREPE would be a top choice.

### What I'd do

Prototype with **Basic Pitch** (lowest friction, decent vocal performance), and record a few minutes of the actual choir as a benchmark clip. If Basic Pitch's output on unisons/close harmony is too smeared, evaluate the **Cuesta model** on the same clip — it exists precisely because general-purpose models underperform on ensemble singing. Both are free for this use (Apache-2.0 for Basic Pitch; check the Cuesta weights license if this becomes a distributed product).

Caveat for calibration: all of these are happiest with 2–6 clearly distinct simultaneous pitches. A 40-voice choir in 5 parts still presents as ~5 pitch streams (good), but heavy vibrato spread and reverberant rooms blur the streams — hence the benchmark-clip-from-the-actual-rehearsal-room test before building anything around the model.

## User

Are there newer models based on the work of "Multiple F0 Estimation in Vocal Ensembles using Convolutional Neural Networks"?

## Claude

*(after a web search, July 2026)*

Short answer: the *direct* Cuesta lineage continued through about 2022 and then largely stopped; the newer (2024–2026) advances are in general-purpose, instrument-agnostic multi-pitch estimation rather than choir-specific models.

### Direct follow-ups (same authors/lineage)

- **Voice Assignment in Vocal Quartets Using Deep Learning Models Based on Pitch Salience** (Cuesta & Gómez, TISMIR 2022) — the natural sequel: takes the multi-f0 salience output and assigns each pitch to S/A/T/B with a neural model, trained partly on a synthetic dataset built from ~5,400 CPDL choral scores. Relevant for per-part attribution from a single mic. https://transactions.ismir.net/articles/10.5334/tismir.121
- **A Deep-Learning Based Framework for Source Separation, Analysis, and Synthesis of Choral Ensembles** (Frontiers in Signal Processing, 2022) — same research circle; combines multi-f0, separation, and analysis of choir recordings. https://www.frontiersin.org/journals/signal-processing/articles/10.3389/frsip.2022.808594/full
- After Cuesta's PhD, no 2024–2025 successor model from that group was found — the 2020 CNN + 2022 voice-assignment models appear to still be the state of the art *specifically trained on vocal ensembles*. The bottleneck is data: annotated multitrack choir datasets (Choral Singing Dataset, Dagstuhl ChoirSet, ESMUC) are tiny.

### Newer general multi-pitch work (2024–2026)

- **Timbre-Trap** (Cwitkowitz & Duan, ICASSP 2024) — instrument-agnostic transcription/MPE framework combining a transcription objective with self-supervised reconstruction; designed for low-resource settings (exactly the choir-data-scarcity problem). Code public. https://arxiv.org/pdf/2309.15717
- **Toward Fully Self-Supervised Multi-Pitch Estimation** (2024) — SS-MPE, trained without pitch annotations using timbre-invariance and pitch-equivariance objectives. Sidesteps missing choir annotations, but research code, less accurate than supervised models today. https://arxiv.org/html/2402.15569v1
- **A Lightweight Slot-Attention Framework for Multi-Instrument Multi-Pitch Estimation** (arXiv, 2026) — separates the mixture into unordered per-source pitch maps with a timbre encoder; conceptually attractive for "which stream is which voice," but new and unproven on singing. https://arxiv.org/html/2606.01460

### Implication for the project

Nothing changes: **Basic Pitch remains the practical engine**, and the **Cuesta 2020 model is still the choir-specialized benchmark** to compare on your own recordings. Timbre-Trap is the one newer codebase worth throwing the benchmark clip at if both disappoint. The frontier work (self-supervised, slot-attention) is worth revisiting in a year, not building on today.

Other sources: [Multiple F0 Estimation in Vocal Ensembles using CNNs (arXiv 2020)](https://arxiv.org/abs/2009.04172), [Dagstuhl ChoirSet (TISMIR)](https://transactions.ismir.net/articles/10.5334/tismir.48)

---

## Summary of decisions so far

1. **Single microphone**, multi-f0 neural model (Basic Pitch first; Cuesta 2020 choir model as accuracy benchmark; Timbre-Trap as backup).
2. **No score-following, no part assignment** — report **per detected note**: nearest equal-tempered note + signed cents deviation.
3. Handle **gradual drift** via contour-based unwrapping + a cumulative **ensemble drift** signal; goal is finding which notes/passages tend to go flat.
4. **Continuous recording + DAW-style timeline**: scroll back, click warning markers, hear (A–B loop, half-speed) the moment it went wrong. Sessions stored as folder (WAV + Parquet analysis log + markers).
5. **Language**: prototype in Python (`sounddevice`, `basic-pitch`/onnxruntime, numpy/scipy, PyQt6 + pyqtgraph); optionally port to Rust later (`cpal`, `tract`/`ort`, `egui`) for a single ~20–50 MB self-contained .exe. Porting risk is Basic Pitch's pre/post-processing, not the network (see NeuralNote's C++ port as reference).
6. Validation plan: benchmark clip recorded from the actual choir in the actual room; synthesized four-part chorale with deliberate pitch-bend to test drift logic.
