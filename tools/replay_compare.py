#!/usr/bin/env python3
"""
Replay benchmark (M5 step 4): run the firmware's pitch module on the recorded WAVs,
compare every result with the Python reference, and report the cycles of every push.

For each WAV:
    PC sends:     u
    board prints: READY <count>
    PC sends:     2 * count bytes, int16 little-endian
    board prints: UPLOAD OK <sum of samples>   or   UPLOAD FAILED
    PC sends:     r
    board prints: REPLAY <pushes> <results>, then P / R lines, an S line and END
                  (line format in Core/Inc/replay.h)

The board's results are checked against tools/pitch_ref.py in float32 (the settings the
golden vectors use), with the tolerances from tests/vectors/vectors.txt and the rules of
tests/test_pitch.c: valid on every row, tau and frequency on valid rows, confidence on
every row. Rows are matched by sample_count, the number of 32 kHz samples pushed so far.

The board needs a TUNER_CAPTURE=ON build. Raw board output and summary.csv go to
data/replay/<tag>/, so a Release and a Debug run sit side by side.

Usage:
    python tools/replay_compare.py --port COM5 --tag release
    python tools/replay_compare.py --port COM5 --tag debug data/captures/*E2*.wav
    python tools/replay_compare.py --list-ports

Close any serial monitor on the same port first.
"""
import argparse
import configparser
import csv
import math
import re
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import serial
from scipy.io import wavfile

from capture import (BAUD_RATE, BITS_PER_BYTE_ON_WIRE, REPO_ROOT, list_ports, open_port,
                     read_text_line)
from pitch_ref import BLOCK, FS_IN, Config, PitchTracker

CPU_HZ = 100_000_000
PUSHES_PER_HOP = 4                                  # 32 ms hop / 8 ms per push
CYCLES_PER_HOP = CPU_HZ * PUSHES_PER_HOP * BLOCK // FS_IN    # 3.2 M: 100 % CPU over one hop
READY_DEADLINE_S = 2.0          # the board answers u at the top of its loop, within milliseconds
REPLAY_DEADLINE_S = 60.0        # 128 pushes; a Debug (-O0) replay takes several seconds
LINES_DEADLINE_S = 10.0         # the P/R lines after the REPLAY header (~4 KB, ~0.4 s)

DEFAULT_GLOB = "data/captures/*.wav"
VECTORS_TXT = REPO_ROOT / "tests" / "vectors" / "vectors.txt"
OUT_ROOT = REPO_ROOT / "data" / "replay"
SUMMARY_FIELDS = [
    "file", "label", "rows", "rows_ok", "valid_mismatch", "max_cents", "max_tau", "max_confidence",
    "plain_min", "plain_max", "result_min", "result_max", "hop_load", "upload_s",
]


class ReplayError(Exception):
    """Something went wrong on the wire or on the board for one file."""


# ---------------------------------------------------------------- arguments

def parse_args():
    parser = argparse.ArgumentParser(description="Replay recorded WAVs on the board and compare.")
    parser.add_argument("wavs", nargs="*", type=Path,
                        help=f"WAV files (default: {DEFAULT_GLOB})")
    parser.add_argument("--port", help="serial port, e.g. COM5 or /dev/cu.SLAB_USBtoUART")
    parser.add_argument("--tag", default="run",
                        help="name of this run, e.g. release or debug: output goes to data/replay/<tag>/")
    parser.add_argument("--list-ports", action="store_true", help="list serial ports and exit")
    args = parser.parse_args()
    if not args.list_ports and not args.port:
        parser.error("--port is required (run with --list-ports to see the available ports)")
    return args


def load_tolerances(path=VECTORS_TXT):
    cp = configparser.ConfigParser()
    cp.read(path, encoding="ascii")
    return {key: float(value) for key, value in cp["tolerances"].items()}


def read_wav(path):
    rate, data = wavfile.read(path)
    if rate != FS_IN or data.dtype != np.int16 or data.ndim != 1:
        raise ReplayError(f"expected mono int16 at {FS_IN} Hz, got {data.dtype} {data.shape} at {rate} Hz")
    return data


# ------------------------------------------------------------------ the wire

def wait_for_line(ser, prefix, deadline_s):
    """Skip other text (dBFS lines) until a line starting with prefix; fail on FAILED."""
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        line = read_text_line(ser)
        if "FAILED" in line:
            raise ReplayError(f"board reported: {line}")
        if line.startswith(prefix):
            return line
    raise ReplayError(f"no {prefix!r} line within {deadline_s:.0f} s "
                      "(wrong port? TUNER_CAPTURE=OFF build? serial monitor still open?)")


def upload(ser, samples):
    """Send one recording into the board's capture buffer; return the transfer time in s."""
    ser.reset_input_buffer()                  # drop dBFS lines received before we asked
    ser.write(b"u")
    parts = wait_for_line(ser, "READY", READY_DEADLINE_S).split()
    if len(parts) != 2 or not parts[1].isdigit():
        raise ReplayError(f"malformed READY line: {' '.join(parts)!r}")
    count = int(parts[1])
    if count != len(samples):
        raise ReplayError(f"board buffer holds {count} samples, the file has {len(samples)}")

    start = time.monotonic()
    payload = samples.astype("<i2").tobytes()
    ser.write(payload)
    transfer_s = len(payload) * BITS_PER_BYTE_ON_WIRE / BAUD_RATE
    parts = wait_for_line(ser, "UPLOAD", 2 * transfer_s + 2.0).split()
    elapsed = time.monotonic() - start

    expected = int(np.sum(samples, dtype=np.int64))
    if len(parts) != 3 or parts[1] != "OK":
        raise ReplayError(f"malformed upload answer: {' '.join(parts)!r}")
    if int(parts[2]) != expected:
        raise ReplayError(f"upload sum {parts[2]} != {expected}: bytes were lost or shifted")
    return elapsed


def replay(ser):
    """Run the replay; return the raw lines from REPLAY to END."""
    ser.reset_input_buffer()
    ser.write(b"r")
    lines = [wait_for_line(ser, "REPLAY", REPLAY_DEADLINE_S)]
    deadline = time.monotonic() + LINES_DEADLINE_S
    while time.monotonic() < deadline:
        line = read_text_line(ser)
        if not line:
            continue
        lines.append(line)
        if line == "END":
            return lines
    raise ReplayError(f"no END within {LINES_DEADLINE_S:.0f} s after the REPLAY line")


def parse_replay(lines):
    """-> (cycles per push, list of result dicts)"""
    head = lines[0].split()
    pushes, n_results = int(head[1]), int(head[2])
    cycles, results = [], []
    for line in lines[1:]:
        f = line.split(",")
        if f[0] == "P":
            cycles.append(int(f[2]))
        elif f[0] == "R":
            results.append({"push": int(f[1]), "sample_count": int(f[2]), "freq_hz": float(f[3]),
                            "confidence": float(f[4]), "tau": float(f[5]), "valid": f[6] == "1"})
    if len(cycles) != pushes or len(results) != n_results:
        raise ReplayError(f"got {len(cycles)} pushes and {len(results)} results, "
                          f"the header says {pushes} and {n_results}")
    return cycles, results


# ----------------------------------------------------------------- compare

def reference(samples):
    """The Python chain in float32, block by block like the firmware; keyed by sample_count."""
    tracker = PitchTracker(replace(Config(), dtype=np.float32))
    rows = {}
    n = len(samples) - len(samples) % BLOCK
    for i in range(0, n, BLOCK):
        for r in tracker.push(samples[i:i + BLOCK]):
            rows[tracker.n_in] = r
    return rows


def compare(board, ref, tol):
    stats = {"rows": len(ref), "rows_ok": 0, "valid_mismatch": 0,
             "max_cents": 0.0, "max_tau": 0.0, "max_confidence": 0.0, "problems": []}
    by_count = {b["sample_count"]: b for b in board}
    extra = set(by_count) - set(ref)
    if extra:
        stats["problems"].append(f"board rows with no reference: sample_count {sorted(extra)}")
    for count, r in sorted(ref.items()):
        b = by_count.get(count)
        if b is None:
            stats["problems"].append(f"sample_count {count}: no board row")
            continue
        ok = True
        if b["valid"] != bool(r.valid):
            stats["valid_mismatch"] += 1
            stats["problems"].append(f"sample_count {count}: valid {b['valid']}, reference {r.valid}")
            ok = False
        d_conf = abs(b["confidence"] - r.confidence)
        stats["max_confidence"] = max(stats["max_confidence"], d_conf)
        if d_conf > tol["confidence_abs"]:
            stats["problems"].append(f"sample_count {count}: confidence off by {d_conf:.3g}")
            ok = False
        if r.valid:      # tau and frequency only on valid rows, as in test_pitch.c
            d_tau = abs(b["tau"] - r.tau)
            d_cents = abs(1200.0 * math.log2(b["freq_hz"] / r.freq_hz))
            stats["max_tau"] = max(stats["max_tau"], d_tau)
            stats["max_cents"] = max(stats["max_cents"], d_cents)
            if d_tau > tol["tau_abs"]:
                stats["problems"].append(f"sample_count {count}: tau off by {d_tau:.3g}")
                ok = False
            if d_cents > tol["freq_cents"]:
                stats["problems"].append(f"sample_count {count}: frequency off by {d_cents:.3g} cents")
                ok = False
        stats["rows_ok"] += ok
    return stats


def timing(cycles, board):
    result_pushes = {b["push"] for b in board}
    plain = [c for k, c in enumerate(cycles) if k not in result_pushes]
    with_result = [c for k, c in enumerate(cycles) if k in result_pushes]
    # Steady state: whole hops after the first result (3 plain pushes + 1 with a result each)
    first = min(result_pushes) if result_pushes else len(cycles)
    hops = (len(cycles) - first - 1) // PUSHES_PER_HOP
    steady = cycles[first + 1: first + 1 + hops * PUSHES_PER_HOP]
    load = sum(steady) / (hops * CYCLES_PER_HOP) if hops else float("nan")
    return {"plain_min": min(plain, default=0), "plain_max": max(plain, default=0),
            "result_min": min(with_result, default=0), "result_max": max(with_result, default=0),
            "hop_load": load}


# --------------------------------------------------------------------- main

def label_of(path):
    return re.sub(r"^\d{8}_\d{6}_", "", path.stem)


def run_file(ser, path, tol, out_dir):
    samples = read_wav(path)
    upload_s = upload(ser, samples)
    lines = replay(ser)
    (out_dir / f"{path.stem}.txt").write_text("\n".join(lines) + "\n", encoding="ascii")
    cycles, board = parse_replay(lines)
    stats = compare(board, reference(samples), tol)
    stats.update(timing(cycles, board))
    stats.update({"file": path.name, "label": label_of(path), "upload_s": round(upload_s, 2)})
    return stats


def main():
    args = parse_args()
    if args.list_ports:
        list_ports()
        return 0

    wavs = args.wavs or sorted(REPO_ROOT.glob(DEFAULT_GLOB))
    if not wavs:
        print(f"No WAV files found ({DEFAULT_GLOB})", file=sys.stderr)
        return 1
    tol = load_tolerances()
    out_dir = OUT_ROOT / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, failures = [], 0
    print(f"{len(wavs)} files, tolerances: freq {tol['freq_cents']} cents, tau {tol['tau_abs']}, "
          f"confidence {tol['confidence_abs']}")
    print(f"{'label':26} {'rows ok':>8} {'max cents':>10}   {'plain push (k cyc)':>18}"
          f"   {'result push (M cyc)':>19}   {'hop load':>8}")
    try:
        with open_port(args.port) as ser:
            for path in wavs:
                try:
                    s = run_file(ser, path, tol, out_dir)
                except ReplayError as e:
                    print(f"{label_of(path):26} ERROR: {e}")
                    failures += 1
                    continue
                rows.append(s)
                bad = s["rows_ok"] != s["rows"] or s["problems"]
                failures += bool(bad)
                print(f"{s['label']:26} {s['rows_ok']:>3}/{s['rows']:<4} {s['max_cents']:>10.5f}"
                      f"   {s['plain_min'] / 1e3:>8.1f} – {s['plain_max'] / 1e3:<7.1f}"
                      f"   {s['result_min'] / 1e6:>8.3f} – {s['result_max'] / 1e6:<8.3f}"
                      f"   {s['hop_load']:>7.0%}" + ("   MISMATCH" if bad else ""))
                for p in s["problems"][:5]:
                    print(f"    {p}")
    except serial.SerialException as e:
        print(f"ERROR opening or using {args.port}: {e}", file=sys.stderr)
        return 1

    with (out_dir / "summary.csv").open("w", newline="", encoding="ascii") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        w.writeheader()
        for s in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in s.items()})

    if rows:
        worst = max(rows, key=lambda s: s["result_max"])
        print(f"\nWorst push: {worst['result_max']:,} cycles = {worst['result_max'] / CPU_HZ * 1e3:.1f} ms"
              f" ({worst['label']}); burst limit 16 ms = {16 * CPU_HZ // 1000:,} cycles")
        print(f"Hop load: {min(s['hop_load'] for s in rows):.0%} – {max(s['hop_load'] for s in rows):.0%}"
              f" of {CYCLES_PER_HOP:,} cycles per 32 ms hop")
    try:
        shown = out_dir.relative_to(REPO_ROOT)
    except ValueError:
        shown = out_dir
    print(f"{len(rows)} files compared, {failures} with errors or mismatches. Output in {shown}/")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
