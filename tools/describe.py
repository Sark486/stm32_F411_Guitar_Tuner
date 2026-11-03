#!/usr/bin/env python3
"""
Describe captures: harmonic balance, decay, pitch drift and high-frequency energy.

The expected note comes from the manifest's ref_note column, or from the label
prefix (E2_open_amp -> E2). Captures without a note (silence, test tones) are skipped.

This is NOT a pitch detector: it looks for peaks near the frequencies it already
knows to expect, to describe the signal a real detector will face.

Usage:
    python tools/describe.py data/captures/*.wav
    python tools/describe.py data/captures/*.wav --csv data/captures/describe.csv

Columns:
    level      overall level, dBFS
    h2..h5     each harmonic's peak relative to the fundamental's, dB (+ = louder)
    loudest    which harmonic has the strongest peak (h3, h5, ... -> wrong note name
               for a "loudest peak" detector; h2, h4 -> right note, wrong octave)
    decay      level change from the first to the last 128 ms, in dB per second
    drift      pitch change from the first to the second half, cents
               (measured on the loudest harmonic)
    hf         energy above 4 kHz relative to the total, dB (-20 dB = 1 %)
    cents      pitch measured near the expected note vs equal temperament, cents
"""
import argparse
import csv
import math
import re
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

A4_HZ = 440.0
SEMITONES = {"C": -9, "D": -7, "E": -5, "F": -4, "G": -2, "A": 0, "B": 2}
HARMONICS = 5                  # fundamental + 4 overtones
SEARCH_WIDTH = 0.03            # look for each harmonic within +-3 % of its expected frequency
EDGE_MS = 128                  # length of the first/last chunk for the decay estimate
HF_LIMIT_HZ = 4000.0           # Nyquist frequency of 8 kHz audio (32 kHz decimated by 4)
LOW_LIMIT_HZ = 20.0


def note_to_hz(note):
    """'E2', 'F#3', 'Bb1' -> frequency in Hz (equal temperament, A4 = 440 Hz)."""
    m = re.fullmatch(r"([A-G])([#b]?)(-?\d)", note.strip())
    if not m:
        return None
    letter, accidental, octave = m.groups()
    semis = SEMITONES[letter] + {"#": 1, "b": -1, "": 0}[accidental] + 12 * (int(octave) - 4)
    return A4_HZ * 2.0 ** (semis / 12.0)


def load_manifest(folder):
    path = folder / "manifest.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {row["file"]: row for row in csv.DictReader(f)}


def expected_note(wav_path, manifest):
    row = manifest.get(wav_path.name, {})
    if row.get("ref_note"):
        return row["ref_note"], row.get("label", "")
    label = row.get("label") or re.sub(r"^\d{8}_\d{6}_", "", wav_path.stem)
    return label.split("_")[0], label


def amplitude_spectrum(x, rate):
    window = np.hanning(len(x))
    amplitude = 2.0 * np.abs(np.fft.rfft(x * window)) / window.sum() + 1e-12
    return np.fft.rfftfreq(len(x), 1.0 / rate), amplitude


def peak_near(freqs, amplitude, target_hz):
    """Strongest bin within +-SEARCH_WIDTH of target; returns (interpolated Hz, dBFS)."""
    idx = np.where((freqs > target_hz * (1 - SEARCH_WIDTH)) & (freqs < target_hz * (1 + SEARCH_WIDTH)))[0]
    if len(idx) == 0:
        return None
    k = int(idx[np.argmax(amplitude[idx])])
    if 0 < k < len(amplitude) - 1:
        a, b, c = np.log(amplitude[k - 1:k + 2])      # parabolic interpolation on log magnitude
        denom = a - 2 * b + c
        offset = 0.5 * (a - c) / denom if denom != 0 else 0.0
    else:
        offset = 0.0
    bin_hz = freqs[1] - freqs[0]
    return (k + offset) * bin_hz, 20 * math.log10(amplitude[k] / 32768.0)


def rms_dbfs(x):
    return 20 * math.log10(max(math.sqrt(float(np.mean(x ** 2))), 1e-9) / 32768.0)


def describe(wav_path, manifest):
    note, label = expected_note(wav_path, manifest)
    f_expected = note_to_hz(note)
    if f_expected is None:
        return None, f"{wav_path.name}: no expected note (label '{label}'), skipped"

    rate, data = wavfile.read(wav_path)
    if data.ndim > 1:
        data = data[:, 0]
    x = data.astype(np.float64)
    x -= x.mean()
    n = len(x)

    freqs, amp = amplitude_spectrum(x, rate)
    peaks = []
    for h in range(1, HARMONICS + 1):
        if f_expected * h < rate / 2 - 200:
            peaks.append(peak_near(freqs, amp, f_expected * h))
    if not peaks or any(p is None for p in peaks):
        return None, f"{wav_path.name}: expected note out of range, skipped"

    levels = [p[1] for p in peaks if p is not None]
    loudest = int(np.argmax(levels)) + 1

    edge = int(rate * EDGE_MS / 1000)
    decay = (rms_dbfs(x[-edge:]) - rms_dbfs(x[:edge])) / ((n - edge) / rate)

    half = n // 2
    pa = peak_near(*amplitude_spectrum(x[:half], rate), f_expected * loudest)
    pb = peak_near(*amplitude_spectrum(x[half:], rate), f_expected * loudest)
    drift = 1200 * math.log2(pb[0] / pa[0]) if pa and pb else float("nan")

    band = freqs > LOW_LIMIT_HZ
    hf = 10 * math.log10(np.sum(amp[freqs > HF_LIMIT_HZ] ** 2) / np.sum(amp[band] ** 2))

    result = {
        "file": wav_path.name,
        "label": label,
        "note": note,
        "expected_hz": round(f_expected, 2),
        "level_dbfs": round(rms_dbfs(x), 1),
        **{f"h{h}_rel_db": round(levels[h - 1] - levels[0], 1) for h in range(2, len(levels) + 1)},
        "loudest": f"h{loudest}",
        "decay_db_per_s": round(decay, 1),
        "drift_cents": round(drift, 1),
        "hf_db": round(hf, 1),
        "cents": round(1200 * math.log2(peaks[0][0] / f_expected), 1),
    }
    return result, None


def print_table(rows):
    header = (f"{'label':20} {'level':>6} | {'h2':>4} {'h3':>4} {'h4':>4} {'h5':>4} | "
              f"{'loudest':7} | {'decay':>6} | {'drift':>6} | {'hf':>6} | {'cents':>6}")
    print(header)
    print("-" * len(header))
    for r in rows:
        rel = " ".join(f"{r.get(f'h{h}_rel_db', float('nan')):+4.0f}" for h in range(2, 6))
        mark = "  <- wrong note" if r["loudest"] in ("h3", "h5") else ""
        print(f"{r['label'][:20]:20} {r['level_dbfs']:6.1f} | {rel} | {r['loudest']:7} | "
              f"{r['decay_db_per_s']:+6.1f} | {r['drift_cents']:+6.1f} | {r['hf_db']:6.1f} | "
              f"{r['cents']:+6.1f}{mark}")


def main():
    parser = argparse.ArgumentParser(description="Describe guitar captures before writing detectors.")
    parser.add_argument("wav", nargs="+", type=Path, help="WAV file(s) from tools/capture.py")
    parser.add_argument("--csv", type=Path, help="also write the results to this CSV file")
    args = parser.parse_args()

    manifests = {}
    rows = []
    for path in sorted(args.wav):
        folder = path.parent
        if folder not in manifests:
            manifests[folder] = load_manifest(folder)
        try:
            row, message = describe(path, manifests[folder])
        except (OSError, ValueError) as e:
            row, message = None, f"{path.name}: {e}"
        if message:
            print(message, file=sys.stderr)
        if row:
            rows.append(row)

    if not rows:
        print("Nothing to describe.", file=sys.stderr)
        return 1
    print_table(rows)

    if args.csv:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nsaved {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
