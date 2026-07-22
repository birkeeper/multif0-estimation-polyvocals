"""
Generate MIDI files (+ aligned F0 annotations) for fine-tuning model3 to
*unlearn* the "quiet voice -> low salience" bias.

Design and rationale: see ../research/finetune_conversation.md

What this produces
------------------
For each "scene" (a few seconds of randomly generated SSATBB chords):

  * ``<scene>.mid``     -- a Standard MIDI File, format 1, one track per voice.
                          Per-voice level is encoded with CC7 (main volume) and
                          the vowel (doo/da) with a Program Change, both set
                          before each chord. Velocity is CONSTANT everywhere, so
                          loudness lives only in CC7 (no velocity->timbre
                          confound). The PWA renders this to a mono 22050 Hz wav.
  * ``<scene>.f0.csv``  -- the exact ground-truth multi-F0 annotation, sampled
                          on the pump's frame grid (hop 256 @ 22050 Hz), in the
                          repo's tab-delimited ragged format
                          (``time \t f1 \t f2 ...``). A voice that is *quiet but
                          sounding* (e.g. -18 dB) is INCLUDED; a voice that is
                          simply not singing is absent. Quiet-but-present is the
                          entire training signal.
  * ``<scene>.notes.csv`` -- per-note debug table (voice, vowel, midi, onset,
                          offset, level_db, cc7).

Scene structure (staircase chords):
  Every chord uses ALL voices. They enter one at a time (entry staircase),
  sustain together (plateau), then release one at a time (release staircase),
  so a single chord sweeps cardinality 1->..->6->..->1 -- each intermediate
  count dwelling for one step. Entry/release/plateau segments are each a 1/16
  or 1/8 note (80 BPM); chords are separated by a 1/2-note silence. This gives
  every cardinality a sustained dwell in far less audio than random overlap.

Two splits:
  * train/  -- per-chord randomized balance (balanced + single-victim +
               random-all), quiet levels oversampled.
  * valid/  -- MATCHED PAIRS: the same notes rendered once balanced and once
               with a single victim voice quiet. This is the probe for the
               "invariance gap" (quiet-voice recall must rise, balanced must not
               regress) used to select the fine-tuning epoch.

IMPORTANT -- CC7 -> dB mapping. The dB levels below assume the PWA interprets
CC7 as a LINEAR amplitude gain (gain = cc/127), giving dB = 20*log10(cc/127).
If your PWA uses a different volume curve, adjust ``cc7_from_db`` so the intended
relative dB between voices is what actually gets rendered. Only the *relative*
levels matter (the model is globally gain-invariant).

The soundfont (Choir_practice.sf2) bank/program + ranges are baked in below.
"""

from __future__ import print_function

import os
import csv
import math
import struct
import argparse
import random


# --------------------------------------------------------------------------
# Feature grid (must match utils.get_hcqt_params in the parent repo)
# --------------------------------------------------------------------------
SR = 22050
HOP = 256
FRAME_RATE = SR / float(HOP)          # ~86.13 frames/sec

# --------------------------------------------------------------------------
# MIDI timing
# --------------------------------------------------------------------------
PPQ = 480                              # ticks per quarter note
BPM = 80
TEMPO_US_PER_QN = int(round(60000000.0 / BPM))   # 80 BPM -> 1 qn = 0.75 s
TICKS_PER_SEC = PPQ * 1e6 / TEMPO_US_PER_QN       # = 640
VELOCITY = 100                         # constant; loudness is CC7 only

# Note lengths (seconds) at the configured tempo.
QUARTER = 60.0 / BPM                    # 0.75 s @ 80 BPM
NOTE_LEN = {
    'sixteenth': QUARTER / 4.0,        # 0.1875 s
    'eighth':    QUARTER / 2.0,        # 0.375 s
    'quarter':   QUARTER,              # 0.75 s
    'half':      QUARTER * 2.0,        # 1.5 s
}

# --------------------------------------------------------------------------
# Voice definitions (SSATBB). Ranges + bank/program from Choir_practice.sf2.
# Range is identical for doo and da of a given voice type.
# The soprano instrument covers soprano+mezzo, bass covers baritone+bass.
# --------------------------------------------------------------------------
def n(name):
    """Note name (e.g. 'F3', 'C#4') -> MIDI number, C4 = 60."""
    names = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    name = name.strip()
    letter = name[0].upper()
    i = 1
    semis = names[letter]
    if i < len(name) and name[i] in ('#', 'b'):
        semis += 1 if name[i] == '#' else -1
        i += 1
    octave = int(name[i:])
    return semis + 12 * (octave + 1)


VOICES = [
    # name, instrument, lo, hi, doo program, da program (all bank 0)
    dict(name='S1', inst='soprano', lo=n('F3'), hi=n('A5'), doo=1, da=11),   # soprano
    dict(name='S2', inst='soprano', lo=n('F3'), hi=n('A5'), doo=1, da=11),   # mezzo
    dict(name='A',  inst='alto',    lo=n('C3'), hi=n('F5'), doo=2, da=12),   # alto
    dict(name='T',  inst='tenor',   lo=n('A2'), hi=n('A4'), doo=3, da=13),   # tenor
    dict(name='B1', inst='bass',    lo=n('C2'), hi=n('E4'), doo=4, da=14),   # baritone
    dict(name='B2', inst='bass',    lo=n('C2'), hi=n('E4'), doo=4, da=14),   # bass
]
N_VOICES = len(VOICES)

# --------------------------------------------------------------------------
# Level set (dB) and CC7 mapping
# --------------------------------------------------------------------------
LEVELS_DB = [0.0, -6.0, -12.0, -18.0]
# oversample the quiet end when picking a "victim" level (the corrective signal)
VICTIM_DB_CHOICES = [-6.0, -12.0, -12.0, -18.0, -18.0]


def cc7_from_db(db):
    """Map a target dB (<=0) to a CC7 value, assuming the PWA treats CC7 as a
    linear amplitude gain (gain = cc/127). Adjust if your PWA differs."""
    amp = 10.0 ** (db / 20.0)
    return max(1, min(127, int(round(127 * amp))))


def midi_to_freq(m):
    return 440.0 * (2.0 ** ((m - 69) / 12.0))


# ==========================================================================
# Minimal Standard MIDI File writer (no external deps)
# ==========================================================================
def _vlq(value):
    """Variable-length quantity encoding of a non-negative int."""
    out = bytearray([value & 0x7F])
    value >>= 7
    while value:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    return bytes(reversed(out))


def _sec_to_tick(sec):
    return int(round(sec * TICKS_PER_SEC))


class MidiTrack(object):
    """Collects (abs_tick, order, event_bytes) then serialises to an MTrk."""
    # ordering at equal tick: note-off < control/program < note-on
    ORDER_NOTEOFF = 0
    ORDER_CTRL = 1
    ORDER_NOTEON = 2

    def __init__(self):
        self.events = []

    def _add(self, tick, order, data):
        self.events.append((tick, order, bytes(data)))

    def name(self, text):
        b = text.encode('latin1')
        self._add(0, -1, b'\xFF\x03' + _vlq(len(b)) + b)

    def tempo(self, us_per_qn):
        self._add(0, -2, b'\xFF\x51\x03' + struct.pack('>I', us_per_qn)[1:])

    def time_signature(self, num=4, den=4):
        self._add(0, -2, bytes([0xFF, 0x58, 0x04, num, int(math.log2(den)), 24, 8]))

    def control(self, tick, ch, controller, value):
        self._add(tick, self.ORDER_CTRL, bytes([0xB0 | ch, controller & 0x7F, value & 0x7F]))

    def program(self, tick, ch, program):
        self._add(tick, self.ORDER_CTRL, bytes([0xC0 | ch, program & 0x7F]))

    def note_on(self, tick, ch, note, vel):
        self._add(tick, self.ORDER_NOTEON, bytes([0x90 | ch, note & 0x7F, vel & 0x7F]))

    def note_off(self, tick, ch, note):
        self._add(tick, self.ORDER_NOTEOFF, bytes([0x80 | ch, note & 0x7F, 0]))

    def serialize(self):
        self.events.sort(key=lambda e: (e[0], e[1]))
        body = bytearray()
        prev = 0
        for tick, _order, data in self.events:
            body += _vlq(tick - prev)
            body += data
            prev = tick
        body += _vlq(0) + b'\xFF\x2F\x00'      # End of Track
        return b'MTrk' + struct.pack('>I', len(body)) + bytes(body)


def write_midi(path, voice_tracks):
    """voice_tracks: list of MidiTrack (voice tracks). A conductor track with
    tempo/time-sig is prepended."""
    conductor = MidiTrack()
    conductor.name('conductor')
    conductor.tempo(TEMPO_US_PER_QN)
    conductor.time_signature(4, 4)

    tracks = [conductor] + voice_tracks
    header = b'MThd' + struct.pack('>IHHH', 6, 1, len(tracks), PPQ)
    with open(path, 'wb') as f:
        f.write(header)
        for t in tracks:
            f.write(t.serialize())


# ==========================================================================
# Scene generation
# ==========================================================================
class Note(object):
    __slots__ = ('voice_idx', 'midi', 'onset', 'offset', 'vowel', 'db', 'cc7')

    def __init__(self, voice_idx, midi, onset, offset, vowel, db):
        self.voice_idx = voice_idx
        self.midi = midi
        self.onset = onset
        self.offset = offset
        self.vowel = vowel            # 'doo' or 'da'
        self.db = db
        self.cc7 = cc7_from_db(db)


def gen_scene(rng, cfg):
    """Generate a scene as a sequence of staircase chords, WITHOUT levels.

    Every chord uses ALL available voices (k = N_VOICES). The voices enter one
    at a time (entry staircase), sustain together (plateau), then release one at
    a time (release staircase), so a single chord sweeps cardinality
    1->2->...->6->...->2->1. Entry/release/plateau segment lengths are each a
    1/16 or 1/8 note (drawn per chord); consecutive chords are separated by a
    1/2-note silent gap. Entry and release orders are independently randomised
    per chord so cardinality is decorrelated from which voices are present.

    Returns (chords, scene_end) where ``chords`` is a list of per-chord Note
    lists (one note per voice per chord) and ``scene_end`` is the scene length
    in seconds (used to size the annotation grid)."""
    chords = []
    t = 0.0
    gap = NOTE_LEN[cfg['gap_unit']]
    for _ in range(cfg['chords_per_scene']):
        s_e = NOTE_LEN[rng.choice(cfg['step_units'])]   # entry step
        s_r = NOTE_LEN[rng.choice(cfg['step_units'])]   # release step
        plateau = NOTE_LEN[rng.choice(cfg['step_units'])]

        entry_order = list(range(N_VOICES));   rng.shuffle(entry_order)
        release_order = list(range(N_VOICES)); rng.shuffle(release_order)

        onset = {vi: t + pos * s_e for pos, vi in enumerate(entry_order)}
        plateau_start = t + (N_VOICES - 1) * s_e        # last voice in -> full chord
        plateau_end = plateau_start + plateau
        offset = {vi: plateau_end + (pos + 1) * s_r
                  for pos, vi in enumerate(release_order)}

        # chord-wide vowel policy: all-doo / all-da / mixed-per-voice
        policy = rng.choices(['doo', 'da', 'mixed'], weights=[0.35, 0.35, 0.30])[0]

        chord = []
        for vi in range(N_VOICES):
            v = VOICES[vi]
            midi = rng.randint(v['lo'], v['hi'])
            vowel = rng.choice(['doo', 'da']) if policy == 'mixed' else policy
            chord.append(Note(vi, midi, onset[vi], offset[vi], vowel, 0.0))
        chords.append(chord)

        last_off = plateau_end + N_VOICES * s_r         # count returns to 0 here
        t = last_off + gap                              # 1/2-note silence before next chord

    return chords, t


def assign_balance_train(rng, chords):
    """Per-chord balance for training: balanced / single-victim / random-all.
    Victim levels oversample the quiet end (the corrective signal)."""
    for chord in chords:
        scenario = rng.choices(['balanced', 'victim', 'random'],
                               weights=[0.35, 0.45, 0.20])[0]
        if scenario == 'balanced':
            for note in chord:
                note.db = 0.0
        elif scenario == 'victim':
            victim = rng.choice(chord)
            vdb = rng.choice(VICTIM_DB_CHOICES)
            for note in chord:
                note.db = vdb if note is victim else 0.0
        else:  # random-all
            for note in chord:
                note.db = rng.choice(LEVELS_DB)
        for note in chord:
            note.cc7 = cc7_from_db(note.db)


def assign_balance_valid(chords, victim_voice_idx, victim_db):
    """Two variants sharing identical notes: 'balanced' (all 0 dB) and 'victim'
    (victim voice quiet in every chord). Returns (balanced_notes, victim_notes)
    as flat lists of copies differing only in db/cc7."""
    def clone(db_fn):
        out = []
        for chord in chords:
            for note in chord:
                c = Note(note.voice_idx, note.midi, note.onset, note.offset, note.vowel, 0.0)
                c.db = db_fn(note)
                c.cc7 = cc7_from_db(c.db)
                out.append(c)
        return out

    balanced = clone(lambda note: 0.0)
    victim = clone(lambda note: victim_db if note.voice_idx == victim_voice_idx else 0.0)
    return balanced, victim


# ==========================================================================
# Serialisation: MIDI + annotation
# ==========================================================================
def notes_to_midi(path, notes):
    tracks = []
    for vi, v in enumerate(VOICES):
        tr = MidiTrack()
        tr.name(v['name'])
        ch = vi                       # channels 0..5 (none is the percussion ch 9)
        tr.control(0, ch, 0, 0)       # bank select MSB = 0
        tr.control(0, ch, 32, 0)      # bank select LSB = 0
        seq = sorted([note for note in notes if note.voice_idx == vi], key=lambda x: x.onset)
        last_program = None
        for note in seq:
            on = _sec_to_tick(note.onset)
            off = _sec_to_tick(note.offset)
            program = v['doo'] if note.vowel == 'doo' else v['da']
            if program != last_program:
                tr.program(on, ch, program)
                last_program = program
            tr.control(on, ch, 7, note.cc7)     # CC7 main volume for this note
            tr.note_on(on, ch, note.midi, VELOCITY)
            tr.note_off(off, ch, note.midi)
        tracks.append(tr)
    write_midi(path, tracks)


def notes_to_annotation(path, notes, scene_dur):
    """Sample the sounding (all included -- quiet still counts) pitches on the
    pump frame grid and write the repo's tab-delimited ragged multi-F0 CSV."""
    n_frames = int(math.floor(scene_dur * FRAME_RATE)) + 1
    with open(path, 'w') as fh:
        w = csv.writer(fh, delimiter='\t')
        for k in range(n_frames):
            t = k / FRAME_RATE
            freqs = [midi_to_freq(note.midi) for note in notes
                     if note.onset <= t < note.offset]
            freqs.sort()
            w.writerow([("%.6f" % t)] + [("%.4f" % f) for f in freqs])


def notes_to_debug_csv(path, notes):
    with open(path, 'w') as fh:
        w = csv.writer(fh)
        w.writerow(['voice', 'vowel', 'midi', 'freq', 'onset', 'offset', 'level_db', 'cc7'])
        for note in sorted(notes, key=lambda x: (x.onset, x.voice_idx)):
            w.writerow([VOICES[note.voice_idx]['name'], note.vowel, note.midi,
                        "%.3f" % midi_to_freq(note.midi),
                        "%.3f" % note.onset, "%.3f" % note.offset,
                        "%.1f" % note.db, note.cc7])


# ==========================================================================
# Main
# ==========================================================================
def build_config(args):
    return dict(
        chords_per_scene=args.chords_per_scene,
        step_units=args.step_units,       # segment lengths drawn from these
        gap_unit=args.gap_unit,           # silence between chords
    )


def main(args):
    rng = random.Random(args.seed)
    cfg = build_config(args)

    train_dir = os.path.join(args.out, 'train')
    valid_dir = os.path.join(args.out, 'valid')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)

    manifest = []
    total_sec = 0.0

    # ---- training scenes ----
    for i in range(args.train_scenes):
        chords, scene_end = gen_scene(rng, cfg)
        assign_balance_train(rng, chords)
        notes = [note for chord in chords for note in chord]
        base = os.path.join(train_dir, 'train_%04d' % i)
        notes_to_midi(base + '.mid', notes)
        notes_to_annotation(base + '.f0.csv', notes, scene_end)
        notes_to_debug_csv(base + '.notes.csv', notes)
        manifest.append(['train', 'train_%04d' % i, 'mixed', ''])
        total_sec += scene_end

    # ---- validation matched pairs ----
    for i in range(args.valid_scenes):
        chords, scene_end = gen_scene(rng, cfg)
        victim_idx = i % N_VOICES               # cycle victim across all voices
        balanced, victim = assign_balance_valid(chords, victim_idx, args.valid_victim_db)
        vname = VOICES[victim_idx]['name']

        b_base = os.path.join(valid_dir, 'valid_%04d_balanced' % i)
        notes_to_midi(b_base + '.mid', balanced)
        notes_to_annotation(b_base + '.f0.csv', balanced, scene_end)

        v_base = os.path.join(valid_dir, 'valid_%04d_victim_%s' % (i, vname))
        notes_to_midi(v_base + '.mid', victim)
        notes_to_annotation(v_base + '.f0.csv', victim, scene_end)
        # the victim annotation intentionally still contains the quiet voice

        manifest.append(['valid', 'valid_%04d_balanced' % i, 'balanced', ''])
        manifest.append(['valid', 'valid_%04d_victim_%s' % (i, vname), 'victim',
                         '%s@%.0fdB' % (vname, args.valid_victim_db)])
        total_sec += 2 * scene_end

    with open(os.path.join(args.out, 'manifest.csv'), 'w') as fh:
        w = csv.writer(fh)
        w.writerow(['split', 'scene', 'scenario', 'detail'])
        w.writerows(manifest)

    print("Wrote %d training scenes and %d validation pairs to %s"
          % (args.train_scenes, args.valid_scenes, args.out))
    print("Approx %.1f minutes of audio once rendered." % (total_sec / 60.0))
    print("Next: render each .mid to mono %d Hz wav in the PWA "
          "(constant velocity, CC7-driven volume, no master FX)." % SR)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default='./finetune/data',
                   help='output directory (train/ and valid/ created inside)')
    p.add_argument('--train_scenes', type=int, default=150)
    p.add_argument('--valid_scenes', type=int, default=30,
                   help='number of matched pairs (each yields balanced + victim)')
    p.add_argument('--chords_per_scene', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)

    p.add_argument('--step_units', nargs='+', default=['eighth', 'quarter'],
                   choices=['sixteenth', 'eighth', 'quarter', 'half'],
                   help='segment lengths (entry step / release step / plateau) '
                        'are drawn from these note values')
    p.add_argument('--gap_unit', default='quarter',
                   choices=['sixteenth', 'eighth', 'quarter', 'half'],
                   help='silent gap between chords')

    p.add_argument('--valid_victim_db', type=float, default=-12.0,
                   help='level of the single quiet voice in validation pairs')

    main(p.parse_args())
