#!/usr/bin/env python3
"""
Plot a capture: waveform, spectrum and spectrogram.

Works offline on the WAV files written by tools/capture.py. For each file it
saves <file>.png next to the WAV and prints the strongest spectral peaks.

Usage:
    python tools/plot.py data/captures/20261025_130430_aaaa.wav
    python tools/plot.py data/captures/*.wav --no-show
    python tools/plot.py some.wav --zoom-start 0.3 --zoom-ms 30

How to read it:
    Waveform     samples over time (int16 counts). Broken data shows up here first.
    Spectrum     the whole capture through one FFT: which frequencies, how strong.
                 Bin width = Fs / N (0.98 Hz for 32768 samples at 32 kHz).
                 Hann window against leakage; level in dBFS (full-scale sine = 0 dBFS).
    Spectrogram  many short FFTs side by side: how the frequencies change over time.
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from scipy.io import wavfile
from scipy.signal import find_peaks

FULL_SCALE = 32768.0
DB_FLOOR = -160.0               # instead of log10(0)

# Open-string frequencies (equal temperament, A4 = 440 Hz)
BASS_STRINGS = {"E1": 41.20, "A1": 55.00, "D2": 73.42, "G2": 98.00}
GUITAR_STRINGS = {"E2": 82.41, "A2": 110.00, "D3": 146.83,
                  "G3": 196.00, "B3": 246.94, "E4": 329.63}

# Colours (reference palette: series blue, text inks, recessive grid)
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = "#2a78d6"
MARKER_LINE = "#9a998f"
# Sequential one-hue ramp for the spectrogram: quiet = light, loud = dark blue
SPECTROGRAM_CMAP = LinearSegmentedColormap.from_list(
    "level", [SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


# ------------------------------------------------------------------ analysis

def load_wav(path):
    rate, data = wavfile.read(path)
    if data.ndim > 1:                       # keep the first channel if stereo
        data = data[:, 0]
    if data.dtype != np.int16:
        raise ValueError(f"{path}: expected 16-bit PCM, got {data.dtype}")
    return rate, data.astype(np.float64)


def to_dbfs(amplitude):
    return 20.0 * np.log10(np.maximum(amplitude, 1e-12) / FULL_SCALE).clip(min=DB_FLOOR)


def amplitude_spectrum(x, rate):
    """One-sided amplitude spectrum of x with a Hann window.

    Scaled so that a sine of amplitude A (in counts) shows a peak of about A:
    multiply by 2 for the one-sided spectrum, divide by the window's sum.
    """
    window = np.hanning(len(x))
    spectrum = np.fft.rfft(x * window)
    amplitude = 2.0 * np.abs(spectrum) / window.sum()
    freqs = np.fft.rfftfreq(len(x), d=1.0 / rate)
    return freqs, amplitude


def spectrogram(x, rate, frame_len, hop):
    """Short-time spectra (same scaling as amplitude_spectrum), frames x bins."""
    if len(x) < frame_len:
        frame_len = len(x)
    window = np.hanning(frame_len)
    starts = np.arange(0, len(x) - frame_len + 1, hop)
    frames = np.stack([x[s:s + frame_len] * window for s in starts])
    amplitude = 2.0 * np.abs(np.fft.rfft(frames, axis=1)) / window.sum()
    freqs = np.fft.rfftfreq(frame_len, d=1.0 / rate)
    times = (starts + frame_len / 2) / rate        # centre of each frame
    return times, freqs, amplitude


def strongest_peaks(freqs, level_db, count=8, fmin=20.0):
    """Local maxima standing at least 10 dB above their surroundings."""
    valid = freqs >= fmin
    idx, _ = find_peaks(level_db[valid], prominence=10.0)
    idx = idx[np.argsort(level_db[valid][idx])[::-1][:count]]
    return [(freqs[valid][i], level_db[valid][i]) for i in idx]


# ------------------------------------------------------------------ plotting

def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MARKER_LINE)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    ax.xaxis.label.set_color(TEXT_SECONDARY)
    ax.yaxis.label.set_color(TEXT_SECONDARY)
    ax.title.set_color(TEXT_PRIMARY)


def mark_strings(ax, bottom_fraction=0.03):
    """Thin dashed lines at the open-string frequencies, labelled near the bottom."""
    transform = ax.get_xaxis_transform()            # x in data, y in axes units
    for row, strings in enumerate((BASS_STRINGS, GUITAR_STRINGS)):
        for i, (name, freq) in enumerate(strings.items()):
            ax.axvline(freq, color=MARKER_LINE, linewidth=0.7, linestyle=(0, (3, 3)))
            y = bottom_fraction + 0.07 * ((i % 2) + 2 * row)   # stagger to avoid overlap
            ax.text(freq, y, name, transform=transform, fontsize=7,
                    color=TEXT_SECONDARY, ha="center", va="bottom",
                    bbox=dict(boxstyle="square,pad=0.1", fc=SURFACE, ec="none"))


def plot_capture(path, zoom_start=None, zoom_ms=20.0, frame=4096, fmax=4000.0, range_db=90.0):
    """Plot one WAV, save <file>.png next to it, print the strongest peaks. Returns the figure."""
    rate, x = load_wav(path)
    n = len(x)
    t = np.arange(n) / rate

    freqs, amplitude = amplitude_spectrum(x, rate)
    level_db = to_dbfs(amplitude)
    peaks = strongest_peaks(freqs, level_db)

    frame_len = min(frame, n)
    hop = max(1, frame_len // 8)
    st_times, st_freqs, st_amp = spectrogram(x, rate, frame_len, hop)
    st_db = to_dbfs(st_amp)

    rms = np.sqrt(np.mean(x ** 2))
    overall_dbfs = to_dbfs(np.array([rms]))[0]

    fig = plt.figure(figsize=(12, 10), facecolor=SURFACE)
    grid = GridSpec(3, 2, figure=fig, width_ratios=[2, 1], height_ratios=[1, 1.2, 1.2],
                    hspace=0.45, wspace=0.18, top=0.93)
    fig.suptitle(f"{Path(path).name}   ·   {n} samples, {n / rate:.3f} s at {rate} Hz, "
                 f"level {overall_dbfs:.1f} dBFS, min {int(x.min())}, max {int(x.max())}",
                 color=TEXT_PRIMARY, fontsize=10, x=0.01, ha="left")

    # 1a. Waveform, whole capture
    ax = fig.add_subplot(grid[0, 0])
    ax.plot(t, x, color=SERIES, linewidth=0.5)
    zoom_start = zoom_start if zoom_start is not None else (n / rate) / 2
    zoom_end = min(zoom_start + zoom_ms / 1000.0, n / rate)
    ax.axvspan(zoom_start, zoom_end, color=MARKER_LINE, alpha=0.15, linewidth=0)
    ax.set_title("Waveform", loc="left", fontsize=10)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("sample value (int16 counts)")
    ax.set_xlim(0, n / rate)
    style_axes(ax)

    # 1b. Waveform, zoomed (shaded region on the left)
    ax = fig.add_subplot(grid[0, 1])
    sel = (t >= zoom_start) & (t <= zoom_end)
    ax.plot((t[sel] - zoom_start) * 1000.0, x[sel], color=SERIES, linewidth=1.2)
    ax.set_title(f"Zoom: {zoom_start:.3f} s + {zoom_ms:g} ms", loc="left", fontsize=10)
    ax.set_xlabel("time from zoom start (ms)")
    style_axes(ax)

    # 2. Spectrum of the whole capture
    ax = fig.add_subplot(grid[1, :])
    ax.semilogx(freqs[1:], level_db[1:], color=SERIES, linewidth=0.9)
    for f, db in peaks[:5]:
        ax.plot(f, db, "o", markersize=4, color=SERIES, markeredgecolor=SURFACE, markeredgewidth=1)
        ax.annotate(f"{f:.1f} Hz", (f, db), textcoords="offset points", xytext=(4, 4),
                    fontsize=7, color=TEXT_PRIMARY)
    mark_strings(ax)
    top = level_db[freqs >= 20].max()
    ax.set_xlim(20, rate / 2)
    ax.set_ylim(max(top - range_db, DB_FLOOR) - 15, top + 10)   # room below for string labels
    bin_hz = rate / n
    ax.set_title(f"Spectrum of the whole capture (Hann window, bin width {bin_hz:.2f} Hz)",
                 loc="left", fontsize=10)
    ax.set_xlabel("frequency (Hz, log scale)")
    ax.set_ylabel("level (dBFS)")
    style_axes(ax)

    # 3. Spectrogram
    ax = fig.add_subplot(grid[2, :])
    keep = (st_freqs >= 20) & (st_freqs <= fmax)
    vmax = st_db[:, keep].max()
    mesh = ax.pcolormesh(st_times, st_freqs[keep], st_db[:, keep].T, shading="nearest",
                         cmap=SPECTROGRAM_CMAP, vmin=vmax - range_db, vmax=vmax)
    ax.set_yscale("log")
    ax.set_ylim(20, fmax)
    ax.set_xlim(0, n / rate)
    for freq in list(BASS_STRINGS.values()) + list(GUITAR_STRINGS.values()):
        ax.axhline(freq, color=MARKER_LINE, linewidth=0.5, linestyle=(0, (3, 3)))
    ax.set_title(f"Spectrogram ({frame_len} samples = {1000 * frame_len / rate:.0f} ms per frame, "
                 f"bin width {rate / frame_len:.1f} Hz, hop {1000 * hop / rate:.0f} ms)",
                 loc="left", fontsize=10)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz, log scale)")
    style_axes(ax)
    ax.grid(False)
    cbar = fig.colorbar(mesh, ax=ax, pad=0.01, fraction=0.03)
    cbar.set_label("level (dBFS)", color=TEXT_SECONDARY, fontsize=8)
    cbar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=7)
    cbar.outline.set_visible(False)

    png = Path(path).with_suffix(".png")
    fig.savefig(png, dpi=120, facecolor=SURFACE, bbox_inches="tight")

    print(f"{Path(path).name}: {n} samples at {rate} Hz, level {overall_dbfs:.1f} dBFS, "
          f"min {int(x.min())}, max {int(x.max())}")
    print(f"  spectrum bin width {bin_hz:.3f} Hz; strongest peaks:")
    for f, db in peaks:
        print(f"    {f:8.1f} Hz  {db:6.1f} dBFS")
    print(f"  saved {png}")
    return fig


def parse_args():
    parser = argparse.ArgumentParser(description="Plot waveform, spectrum and spectrogram of WAV captures.")
    parser.add_argument("wav", nargs="+", type=Path, help="WAV file(s) from tools/capture.py")
    parser.add_argument("--zoom-start", type=float, default=None,
                        help="start of the zoomed waveform in seconds (default: middle of the capture)")
    parser.add_argument("--zoom-ms", type=float, default=20.0,
                        help="length of the zoomed waveform in ms (default: 20)")
    parser.add_argument("--frame", type=int, default=4096,
                        help="spectrogram frame length in samples (default: 4096 = 128 ms at 32 kHz)")
    parser.add_argument("--fmax", type=float, default=4000.0,
                        help="top of the spectrogram frequency axis in Hz (default: 4000)")
    parser.add_argument("--range-db", type=float, default=90.0,
                        help="dB range shown below the strongest level (default: 90)")
    parser.add_argument("--no-show", action="store_true", help="only save the PNGs, don't open windows")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.no_show:
        plt.switch_backend("Agg")
    failed = 0
    for path in args.wav:
        try:
            plot_capture(path, zoom_start=args.zoom_start, zoom_ms=args.zoom_ms,
                         frame=args.frame, fmax=args.fmax, range_db=args.range_db)
        except (OSError, ValueError) as e:
            print(f"ERROR: {e}", file=sys.stderr)
            failed += 1
    if not args.no_show and failed < len(args.wav):
        plt.show()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
