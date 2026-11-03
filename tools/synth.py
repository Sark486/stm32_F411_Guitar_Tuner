#!/usr/bin/env python3
"""
Synthetic test signals with exactly known pitch.

Real captures have a reference-tuner offset and a fixed set of notes; synthetic ones
have exact ground truth and one difficulty switched on at a time. Every file is
32 kHz int16 mono, 32768 samples (1.024 s), like a capture, and gets a manifest row
in the same columns as data/captures/manifest.csv, so tools/evaluate.py reads them
unchanged.

Usage:
    python tools/synth.py                 # (re)generate data/synthetic/*.wav + manifest.csv

Signal model (one knob per difficulty):
    harmonics   amplitude per harmonic: plucked-string shape |sin(pi h p)| / h, where p is
                the pluck position as a fraction of the string (p = 1/2 removes the even
                harmonics: the 12th-fret pluck), or an explicit list
    B           inharmonicity: partial h sits at h f0 sqrt(1 + B h^2)
    decay       exponential decay, faster for higher harmonics
    glide       pitch change in cents over the file (real strings fall a few cents)
    mic         2nd-order high-pass at 60 Hz: the INMP441's -3 dB point (datasheet Table 1)
    noise       real room/mic noise taken from a silence capture, at its recorded level
    level       RMS level of the note in dBFS before the noise is added

Bass stand-ins from real audio: a guitar capture resampled 2x plays one octave lower
(E2 -> E1); the mic high-pass is applied again so the new fundamental is weakened the way
a real E1 would be.
"""
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, resample_poly, sosfilt

FS = 32000
N = 32768
REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "synthetic"
NOISE_FILE = REPO_ROOT / "data" / "captures" / "20261025_130307_silence.wav"
CAPTURES = REPO_ROOT / "data" / "captures"

NOTES = {"B0": 30.868, "E1": 41.203, "A1": 55.0, "D2": 73.416, "G2": 97.999,
         "E2": 82.407, "A2": 110.0, "D3": 146.832, "G3": 195.998, "B3": 246.942,
         "E4": 329.628, "E5": 659.255, "A5": 880.0, "E6": 1318.51}


# --------------------------------------------------------------- building blocks

def pluck_amplitudes(n_harm, pluck=0.2):
    h = np.arange(1, n_harm + 1)
    return np.abs(np.sin(np.pi * h * pluck)) / h


def harmonic_tone(f0, amps, B=0.0, decay_s=1.5, glide_cents=0.0, n=N, fs=FS, seed=0):
    """Sum of partials. Each partial's frequency follows the glide; phases are random but
    reproducible. Partials above 15 kHz are dropped (no aliasing at 32 kHz)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    glide = 2.0 ** (glide_cents * t / t[-1] / 1200.0)       # pitch multiplier over time
    x = np.zeros(n)
    for h, a in enumerate(amps, start=1):
        if a == 0:
            continue
        fh = h * f0 * np.sqrt(1.0 + B * h * h)
        if fh > 15000:
            break
        phase = 2 * np.pi * np.cumsum(fh * glide) / fs + rng.uniform(0, 2 * np.pi)
        tau = decay_s / (1.0 + 0.15 * (h - 1)) if decay_s else None
        env = np.exp(-t / tau) if tau else 1.0
        x += a * env * np.sin(phase)
    return x


def mic_highpass(x, fs=FS, corner=60.0):
    return np.asarray(sosfilt(butter(2, corner, btype="highpass", fs=fs, output="sos"), x))


def set_level(x, dbfs):
    rms = np.sqrt(np.mean(x ** 2))
    return x * (10 ** (dbfs / 20.0) * 32768.0 / rms) if rms > 0 else x


def room_noise(n=N, seed=0):
    """Real mic + room noise from a silence capture, randomly offset and looped."""
    _, data = wavfile.read(NOISE_FILE)
    data = data.astype(np.float64) - np.mean(data)
    start = np.random.default_rng(seed).integers(0, len(data))
    return np.resize(np.roll(data, -start), n)


def to_int16(x):
    return np.clip(np.round(x), -32768, 32767).astype(np.int16)


def half_speed(capture_name):
    """Guitar capture played an octave lower: 2x upsampled, first N samples kept."""
    _, data = wavfile.read(CAPTURES / capture_name)
    slow = resample_poly(data.astype(np.float64), 2, 1)[:N]
    return mic_highpass(slow)                                # weaken the new fundamental


# ---------------------------------------------------------------------- the set

def presets():
    """(label, note, ref_cents, float signal at int16 scale, notes text)."""
    out = []

    def tone(label, note, amps, text, level=-45.0, noise=True, mic=True, cents=0.0,
             seed=0, **kw):
        """cents: offset of the starting f0 from the note; the reference is the mean pitch."""
        x = harmonic_tone(NOTES[note] * 2.0 ** (cents / 1200.0), amps, seed=seed, **kw)
        ref_cents = cents + kw.get("glide_cents", 0.0) / 2.0
        if mic:
            x = mic_highpass(x)
        x = set_level(x, level)
        if noise:
            x = x + room_noise(seed=seed)
        out.append((label, note, ref_cents, x, text))

    # bass: E1 is mandatory, B0 nice to have
    for i, note in enumerate(("E1", "A1", "D2", "G2", "B0")):
        tone(f"syn_{note}_bass", note, pluck_amplitudes(30, 0.2), "bass model, pluck 1/5, mic HP",
             seed=10 + i)
    tone("syn_E1_bass_inharm", "E1", pluck_amplitudes(30, 0.2), "bass model, B = 2e-4", B=2e-4, seed=20)
    tone("syn_E1_missing_fund", "E1", np.r_[0.0, pluck_amplitudes(30, 0.2)[1:]],
         "no fundamental at all", seed=21)
    tone("syn_E1_quiet", "E1", pluck_amplitudes(30, 0.2), "bass model at -60 dBFS (low SNR)",
         level=-60.0, seed=22)
    tone("syn_E1_flat45", "E1", pluck_amplitudes(30, 0.2), "45 cents flat (40.1 Hz)", cents=-45.0,
         seed=23)

    # guitar strings
    for i, note in enumerate(("E2", "A2", "D3", "G3", "B3", "E4")):
        tone(f"syn_{note}_guitar", note, pluck_amplitudes(20, 0.15), "guitar model, pluck 0.15",
             seed=30 + i)
    tone("syn_E2_fret12pluck", "E2", pluck_amplitudes(20, 0.5), "pluck at 1/2: no even harmonics",
         seed=40)
    h3 = np.array([0.1, 0.3, 1.0, 0.5, 0.6, 0.2, 0.2, 0.1])
    tone("syn_A2_h3loud", "A2", h3, "3rd harmonic 20 dB above the fundamental", seed=41)
    tone("syn_E2_glide", "E2", pluck_amplitudes(20, 0.15), "falls 6 cents over 1 s",
         glide_cents=-6.0, seed=42)
    tone("syn_E2_inharm", "E2", pluck_amplitudes(20, 0.15), "guitar B = 1e-4", B=1e-4, seed=43)
    tone("syn_E2_noisy", "E2", pluck_amplitudes(20, 0.15), "-58 dBFS: barely above the room",
         level=-58.0, seed=44)

    # top of the range
    for i, note in enumerate(("E5", "A5", "E6")):
        tone(f"syn_{note}_guitar", note, pluck_amplitudes(8, 0.15), "fretted high note", seed=50 + i)
    tone("syn_E6_sine", "E6", [1.0], "pure sine at 1318.5 Hz", seed=53, mic=False)
    tone("syn_E1_sine", "E1", [1.0], "pure sine at 41.2 Hz", seed=54)

    # bass stand-ins from real guitar audio
    for src, note, text in (("20261025_140314_E2_open_amp.wav", "E1", "E2 open (amp) at half speed"),
                            ("20261025_140341_A2_open_amp.wav", "A1", "A2 open (amp) at half speed"),
                            ("20261025_140631_E2_open_unpl.wav", "E1", "E2 open (unplugged) at half speed")):
        out.append((f"halfspeed_{src[16:-4]}", note, 0.0, half_speed(src), text))
    return out


def write_set(out_dir=OUT_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, note, ref_cents, x, text in presets():
        name = f"20261025_000000_{label}.wav"            # same name pattern as captures
        y = to_int16(x)
        wavfile.write(out_dir / name, FS, y)
        rms = np.sqrt(np.mean(y.astype(np.float64) ** 2))
        rows.append({"file": name, "time": "", "label": label, "count": len(y), "rate_hz": FS,
                     "dbfs": round(20 * np.log10(max(rms, 1e-9) / 32768.0), 1),
                     "min": int(y.min()), "max": int(y.max()), "ref_note": note,
                     "ref_cents": ref_cents, "notes": f"synthetic: {text}"})
    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    rows = write_set()
    print(f"wrote {len(rows)} files to {OUT_DIR.relative_to(REPO_ROOT)}")
    for r in rows:
        print(f"  {r['label']:32} {r['ref_note']:3} {r['ref_cents']:+6.1f} c  {r['dbfs']:6.1f} dBFS  {r['notes']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
