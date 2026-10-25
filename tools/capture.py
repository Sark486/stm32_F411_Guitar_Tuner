#!/usr/bin/env python3
"""
Trigger one capture on the STM32 tuner board and save it as a WAV file.

Protocol:
    PC sends:     c
    board prints: text lines ("Capture start", "Record result=...", dBFS lines)
    board sends:  DUMP <count> <rate>\\r\\n
                  <count> samples, int16 little-endian (2 * count bytes)
                  END\\r\\n
                  Send result=OK|FAILED\\r\\n

Each successful capture is saved as data/captures/<timestamp>_<label>.wav,
gets one row in data/captures/manifest.csv, and is plotted with tools/plot.py
(PNG saved next to the WAV, plot window opened unless --no-show).

Usage:
    python tools/capture.py --list-ports
    python tools/capture.py --port COM5 --label E2_open_amp --ref-note E2 --ref-cents -1 \
                            --notes "phone tuner app, amp 15 cm"
    python tools/capture.py --port COM5 --label silence --no-plot
    python tools/capture.py --port /dev/cu.SLAB_USBtoUART --label A2_open_unpl --no-show

Close any serial monitor on the same port first.
"""
import argparse
import csv
import math
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import serial
import serial.tools.list_ports
from scipy.io import wavfile

BAUD_RATE = 115200
BITS_PER_BYTE_ON_WIRE = 10      # 8N1: start bit + 8 data bits + stop bit
LINE_TIMEOUT_S = 1.0            # timeout for reading one text line
HEADER_DEADLINE_S = 5.0         # max wait for the DUMP line (the capture itself takes ~1 s)
TRAILER_DEADLINE_S = 2.0        # max wait for the "Send result" line after END
MAX_SAMPLES = 1_000_000         # sanity limit for the count in the DUMP line
FULL_SCALE = 32768.0            # int16 full scale, for dBFS

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "captures"
MANIFEST_NAME = "manifest.csv"
MANIFEST_FIELDS = [
    "file", "time", "label", "count", "rate_hz", "dbfs", "min", "max",
    "ref_note", "ref_cents", "notes",
]


class CaptureError(Exception):
    """Something went wrong on the wire or on the board; nothing is saved."""


# ---------------------------------------------------------------- arguments

def parse_args():
    parser = argparse.ArgumentParser(
        description="Trigger a capture on the board and save it as a WAV file.")
    parser.add_argument("--port",
                        help="serial port, e.g. COM5 or /dev/cu.SLAB_USBtoUART")
    parser.add_argument("--label", default="capture",
                        help="short description used in the file name, e.g. E2_open_bridge")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help=f"where WAVs and {MANIFEST_NAME} go (default: data/captures)")
    parser.add_argument("--ref-note", default="",
                        help="expected note from the reference tuner, e.g. E2 (manifest column ref_note)")
    parser.add_argument("--ref-cents", type=float, default=None,
                        help="reference tuner reading in cents, e.g. -1.5 (manifest column ref_cents)")
    parser.add_argument("--notes", default="",
                        help='free text for the manifest, e.g. "phone app, amp 15 cm, bridge pluck"')
    parser.add_argument("--no-plot", action="store_true",
                        help="don't plot the capture after saving it")
    parser.add_argument("--no-show", action="store_true",
                        help="save the plot PNG but don't open a plot window")
    parser.add_argument("--list-ports", action="store_true",
                        help="list serial ports and exit")
    args = parser.parse_args()
    if not args.list_ports and not args.port:
        parser.error("--port is required (run with --list-ports to see the available ports)")
    return args


def list_ports():
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("No serial ports found.")
    for port in ports:
        print(f"{port.device:28} {port.description}")


# ------------------------------------------------------------ serial helpers

def open_port(name):
    kwargs = {"baudrate": BAUD_RATE, "timeout": LINE_TIMEOUT_S}
    if os.name == "posix":
        # Refuse to share the port with another program (e.g. a serial monitor).
        # Windows ports are always exclusive.
        kwargs["exclusive"] = True
    return serial.Serial(name, **kwargs)


def read_text_line(ser):
    """One line, decoded tolerantly, without the line ending. '' on timeout."""
    return ser.readline().decode("ascii", errors="replace").strip()


def wait_for_dump_header(ser):
    """Skip text until the DUMP line; return (count, rate_hz)."""
    deadline = time.monotonic() + HEADER_DEADLINE_S
    while time.monotonic() < deadline:
        line = read_text_line(ser)
        if not line:
            continue
        if line.startswith("DUMP"):
            parts = line.split()
            if len(parts) != 3:
                raise CaptureError(f"malformed header line: {line!r}")
            try:
                count, rate_hz = int(parts[1]), int(parts[2])
            except ValueError:
                raise CaptureError(f"malformed header line: {line!r}") from None
            if not (0 < count <= MAX_SAMPLES) or rate_hz <= 0:
                raise CaptureError(f"implausible header values: {line!r}")
            return count, rate_hz
        if "FAILED" in line:
            raise CaptureError(f"board reported: {line}")
        if line.startswith(("Capture start", "Record result")):
            print(f"  board: {line}")
        # anything else (dBFS lines, leftovers) is ignored
    raise CaptureError(
        f"no DUMP line within {HEADER_DEADLINE_S:.0f} s "
        "(wrong port? board not running? serial monitor still open?)")


def read_payload(ser, count):
    """Read exactly 2 * count bytes of samples."""
    n_bytes = 2 * count
    transfer_s = n_bytes * BITS_PER_BYTE_ON_WIRE / BAUD_RATE
    deadline = time.monotonic() + 2 * transfer_s + 1.0
    # Do NOT change ser.timeout here: on Windows that reconfigures the port
    # (SetCommState) while the samples are already arriving, and a few bytes
    # get lost. Keep the short line timeout and loop until all bytes are in.
    # read(n) returns fewer than n bytes when its timeout expires, without raising.
    data = bytearray()
    while len(data) < n_bytes and time.monotonic() < deadline:
        data += ser.read(n_bytes - len(data))
    if len(data) != n_bytes:
        hint = ""
        if b"END\r\n" in data or b"Send result" in data:
            hint = " (the board stopped sending early; its END/result lines are inside the data)"
        raise CaptureError(f"expected {n_bytes} bytes of samples, got {len(data)}{hint}")
    return bytes(data)


def expect_end(ser):
    line = read_text_line(ser)
    if line != "END":
        raise CaptureError(
            f"expected END after the samples, got {line!r}: bytes were lost or added")


def check_send_result(ser):
    """The board prints 'Send result=...' after END. Warn if it never comes."""
    deadline = time.monotonic() + TRAILER_DEADLINE_S
    while time.monotonic() < deadline:
        line = read_text_line(ser)
        if line.startswith("Send result"):
            if "OK" not in line:
                raise CaptureError(f"board reported: {line}")
            return
    print("  warning: no 'Send result' line from the board", file=sys.stderr)


# ------------------------------------------------------------- save results

def level_dbfs(samples):
    rms = math.sqrt(float(np.mean(samples.astype(np.float64) ** 2)))
    return 20.0 * math.log10(rms / FULL_SCALE) if rms > 0 else float("-inf")


def save_capture(out_dir, label, rate_hz, samples, ref_note="", ref_cents=None, notes=""):
    out_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    safe_label = re.sub(r"[^A-Za-z0-9_-]+", "_", label).strip("_") or "capture"
    wav_path = out_dir / f"{now:%Y%m%d_%H%M%S}_{safe_label}.wav"
    wavfile.write(wav_path, rate_hz, samples)   # int16 array -> 16-bit PCM WAV

    dbfs = level_dbfs(samples)
    manifest = out_dir / MANIFEST_NAME
    is_new = not manifest.exists()
    with manifest.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({
            "file": wav_path.name,
            "time": now.isoformat(timespec="seconds"),
            "label": label,
            "count": len(samples),
            "rate_hz": rate_hz,
            "dbfs": f"{dbfs:.1f}",
            "min": int(samples.min()),
            "max": int(samples.max()),
            "ref_note": ref_note,
            "ref_cents": "" if ref_cents is None else f"{ref_cents:g}",
            "notes": notes,
        })
    return wav_path, dbfs


# --------------------------------------------------------------------- main

def capture(port_name):
    """Run one capture over the wire; return (rate_hz, samples)."""
    with open_port(port_name) as ser:
        ser.reset_input_buffer()     # drop dBFS lines received before we asked
        ser.write(b"c")
        print(f"Sent 'c' on {port_name}, waiting for the recording...")

        count, rate_hz = wait_for_dump_header(ser)
        transfer_s = 2 * count * BITS_PER_BYTE_ON_WIRE / BAUD_RATE
        print(f"  DUMP: {count} samples at {rate_hz} Hz, "
              f"receiving {2 * count} bytes (~{transfer_s:.1f} s)")

        data = read_payload(ser, count)
        expect_end(ser)
        check_send_result(ser)

    samples = np.frombuffer(data, dtype="<i2")   # little-endian signed 16-bit
    return rate_hz, samples


def main():
    args = parse_args()
    if args.list_ports:
        list_ports()
        return 0

    try:
        rate_hz, samples = capture(args.port)
    except CaptureError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except serial.SerialException as e:
        print(f"ERROR opening or using {args.port}: {e}\n"
              "Is the port name right, and is no serial monitor using it?",
              file=sys.stderr)
        return 1

    wav_path, dbfs = save_capture(args.out_dir, args.label, rate_hz, samples,
                                  args.ref_note, args.ref_cents, args.notes)
    try:
        shown = wav_path.relative_to(REPO_ROOT)
    except ValueError:
        shown = wav_path
    print(f"Saved {shown}")
    print(f"  {len(samples)} samples, {len(samples) / rate_hz:.3f} s, "
          f"min {samples.min()}, max {samples.max()}, level {dbfs:.1f} dBFS")

    if not args.no_plot:
        show_plot(wav_path, args.no_show)
    return 0


def show_plot(wav_path, no_show):
    """Plot the saved WAV with tools/plot.py (same folder as this script)."""
    import matplotlib.pyplot as plt     # imported here so --no-plot needs no matplotlib
    from plot import plot_capture
    if no_show:
        plt.switch_backend("Agg")
    plot_capture(wav_path)
    if not no_show:
        plt.show()


if __name__ == "__main__":
    sys.exit(main())
