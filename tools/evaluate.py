#!/usr/bin/env python3
"""
Run a pitch detector over labelled captures and measure how it does.

For each capture the recording is (optionally) filtered, cut into sliding windows,
and the detector from tools/detectors.py is called on every window. The answers are
compared with the reference: the manifest's ref_note shifted by ref_cents (the tuner
reading). Test tones labelled like 440, 1khz or 82_4hz are understood as frequencies.

Usage:
    python tools/evaluate.py data/captures/*.wav --detector yin          (streaming chain)
    python tools/evaluate.py data/captures/*.wav --detector mpm --decim 2 --no-refine
    python tools/evaluate.py data/captures/*.wav --detector zc
    python tools/evaluate.py data/captures/*.wav --detector zc --highpass 30 --lowpass 400
    python tools/evaluate.py data/captures/*.wav --detector zc --window 2048 --hop 512 --csv

Per capture it prints:
    answered   % of windows where the detector returned a frequency
    note ok    % of windows with the right note name (octave errors count as right:
               the tuner shows note name and cents only)
    cents      median |error| in cents over the "note ok" windows (octave-folded)
    signed     median signed error in cents (bias: + = sharp)
    typical    the most common wrong answer as a ratio to the reference, e.g. x3.0
               (third harmonic) or x0.5 (one octave low)

Detectors yin and mpm are not window functions: they run the whole streaming chain from
tools/pitch_ref.py (block-by-block FIR decimation, 128 ms window, 32 ms hop, confidence
gate). Their window and hop come from pitch_ref.Config, not from --window/--hop, and a
result that fails the confidence gate counts as "no answer".

Captures without a reference (silence) are not skipped: for them "answered" is the
false-alarm rate, which should be 0 %.

Plots (unless --no-plot) go to build/eval/<run name>/: one pitch track per capture
plus overview.png with all captures as small multiples.
"""
import argparse
import csv
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt

from describe import load_manifest, note_to_hz
from detectors import DETECTORS
from pitch_ref import TRACKERS, replace as cfg_replace, track

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO_ROOT / "build" / "eval"
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Colours (same reference palette as plot.py)
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = "#2a78d6"
REFERENCE = "#0b0b0b"
GUIDE = "#9a998f"
BAD = "#e34948"


# ------------------------------------------------------------ reference pitch

def tone_label_hz(label):
    """'440' -> 440, '1khz' -> 1000, '82_4hz' -> 82.4; None if the label is not a frequency."""
    m = re.fullmatch(r"(\d+)(?:_(\d+))?(k?)(?:hz)?", label.lower())
    if not m:
        return None
    whole, frac, kilo = m.groups()
    hz = float(f"{whole}.{frac}" if frac else whole)
    return hz * 1000 if kilo else hz


def reference_hz(wav_path, manifest):
    """Reference frequency for a capture, or (None, reason)."""
    row = manifest.get(wav_path.name, {})
    label = re.sub(r"^\d{8}_\d{6}_", "", wav_path.stem)   # file name, not the manifest label:
                                                              # labels can repeat, file names can't
    tone = tone_label_hz(label)
    if tone:
        return tone, label
    note = row.get("ref_note") or label.split("_")[0]
    hz = note_to_hz(note)
    if hz is None:
        return None, label
    cents = row.get("ref_cents", "")
    if cents not in ("", None):
        hz *= 2.0 ** (float(cents) / 1200.0)
    return hz, label


def note_number(hz):
    """Semitones from A4, rounded (A4 = 0, E2 = -29)."""
    return int(round(12.0 * math.log2(hz / 440.0)))


def note_name(hz):
    n = note_number(hz)
    return f"{NOTE_NAMES[(n + 9) % 12]}{(n + 9) // 12 + 4}"


def folded_cents(f, f_ref):
    """Cents error with whole octaves removed, in [-600, 600)."""
    c = 1200.0 * math.log2(f / f_ref)
    return (c + 600.0) % 1200.0 - 600.0


# ------------------------------------------------------------------ analysis

def prefilter(x, fs, highpass, lowpass):
    """Zero-phase Butterworth filters on the whole recording (no delay, no edge effects per window)."""
    if highpass:
        x = sosfiltfilt(butter(4, highpass, btype="highpass", fs=fs, output="sos"), x)
    if lowpass:
        x = sosfiltfilt(butter(4, lowpass, btype="lowpass", fs=fs, output="sos"), x)
    return x


def tracker_config(args):
    """pitch_ref.Config for --detector yin/mpm with the command-line overrides applied."""
    cfg = TRACKERS[args.detector]
    over = {}
    if args.decim:
        from pitch_ref import config_for_rate
        cfg = config_for_rate(args.decim, method=cfg.method, interp=cfg.interp)
    for name, value in (("yin_threshold", args.threshold), ("mpm_k", args.k),
                        ("interp", args.interp), ("gate", args.gate)):
        if value is not None:
            over[name] = value
    if args.no_refine:
        over["refine"] = False
    if args.float32:
        over["dtype"] = np.float32
    return cfg_replace(cfg, **over)


def detect_all(data, fs, detector_name, args):
    """(times, answers) for one recording. answers[i] is None for 'no answer'."""
    if detector_name in TRACKERS:
        results = track(data, tracker_config(args))
        return ([r.time_s for r in results],
                [r.freq_hz if r.valid else None for r in results])
    detector = DETECTORS[detector_name]
    x = prefilter(data.astype(np.float64), fs, args.highpass, args.lowpass)
    times, answers = [], []
    for start in range(0, len(x) - args.window + 1, args.hop):
        f = detector(x[start:start + args.window].copy(), fs)
        times.append((start + args.window / 2) / fs)
        answers.append(f)
    return times, answers


def score(times, answers, f_ref, fs):
    """Statistics of one recording's answers against the reference (f_ref None = silence)."""
    answers = [f if (f is not None and np.isfinite(f) and f > 0) else None for f in answers]
    total = len(answers)
    found = [f for f in answers if f is not None]
    res = {"times": times, "answers": answers, "fs": fs, "windows": total,
           "answered_pct": 100.0 * len(found) / total if total else 0.0,
           "note_ok_pct": float("nan"), "median_cents": float("nan"),
           "signed_cents": float("nan"), "typical_wrong": ""}
    if f_ref is None:
        return res
    ref_note = note_number(f_ref) % 12
    ok = [f for f in found if note_number(f) % 12 == ref_note]
    wrong = [f for f in found if note_number(f) % 12 != ref_note]
    if wrong:
        ratios = Counter(round(f / f_ref * 2) / 2 for f in wrong)   # nearest 0.5
        ratio, _ = ratios.most_common(1)[0]
        res["typical_wrong"] = f"x{ratio:g}"
    res["note_ok_pct"] = 100.0 * len(ok) / total if total else 0.0
    if ok:
        errs = [folded_cents(f, f_ref) for f in ok]
        res["median_cents"] = float(np.median(np.abs(errs)))
        res["signed_cents"] = float(np.median(errs))
    return res


def run_capture(path, detector_name, f_ref, args):
    fs, data = wavfile.read(path)
    if data.ndim > 1:
        data = data[:, 0]
    times, answers = detect_all(data, fs, detector_name, args)
    return score(times, answers, f_ref, fs)


# ------------------------------------------------------------------ plotting

def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GUIDE)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=7)


def draw_track(ax, res, f_ref, label, small=False):
    if f_ref is None:                       # silence: just show any (false) answers
        t = [t for t, f in zip(res["times"], res["answers"]) if f is not None]
        f = [f for f in res["answers"] if f is not None]
        ax.scatter(t, f, s=9 if small else 16, color=BAD, linewidths=0, zorder=3)
        ax.set_yscale("log")
        ax.set_ylim(30, 3000)
        ax.set_xlim(0, (res["times"][-1] + res["times"][0]) if res["times"] else 1)
        ax.set_title(f"{label}  ·  false answers {res['answered_pct']:.0f}%", loc="left",
                     fontsize=7 if small else 10, color=TEXT_PRIMARY)
        style(ax)
        return
    ref_class = note_number(f_ref) % 12
    t_ok, f_ok, t_bad, f_bad = [], [], [], []
    for t, f in zip(res["times"], res["answers"]):
        if f is None:
            continue
        (t_ok if note_number(f) % 12 == ref_class else t_bad).append(t)
        (f_ok if note_number(f) % 12 == ref_class else f_bad).append(f)
    for mult, name in ((0.5, "x½"), (2, "x2"), (3, "x3"), (4, "x4"), (5, "x5")):
        ax.axhline(f_ref * mult, color=GUIDE, linewidth=0.6, linestyle=(0, (3, 3)))
        if not small:
            ax.text(1.005, f_ref * mult, name, transform=ax.get_yaxis_transform(),
                    fontsize=7, color=TEXT_SECONDARY, va="center")
    ax.axhline(f_ref, color=REFERENCE, linewidth=1.0)
    size = 9 if small else 16
    ax.scatter(t_ok, f_ok, s=size, color=SERIES, linewidths=0, zorder=3, label="right note")
    ax.scatter(t_bad, f_bad, s=size, color=BAD, linewidths=0, zorder=3, label="wrong note")
    ax.set_yscale("log")
    found = f_ok + f_bad
    low = min([f_ref * 0.35] + [f * 0.8 for f in found])
    high = max([f_ref * 7] + [f * 1.25 for f in found])
    ax.set_ylim(low, high)
    ax.set_xlim(0, res["times"][-1] + res["times"][0] if res["times"] else 1)
    title = f"{label}  ·  note ok {res['note_ok_pct']:.0f}%"
    ax.set_title(title, loc="left", fontsize=7 if small else 10, color=TEXT_PRIMARY)
    style(ax)


def save_plots(results, out_dir, run_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, NullFormatter

    out_dir.mkdir(parents=True, exist_ok=True)
    for label, f_ref, res in results:
        fig, ax = plt.subplots(figsize=(9, 4), facecolor=SURFACE)
        draw_track(ax, res, f_ref, label)
        ax.set_xlabel("time (s)", color=TEXT_SECONDARY, fontsize=8)
        ax.set_ylabel("detected frequency (Hz, log)", color=TEXT_SECONDARY, fontsize=8)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(FuncFormatter(
            lambda v, _: f"{v:g}" if f"{v:g}"[0] in "235" else ""))
        ax.tick_params(axis="y", which="both", labelsize=7)
        if f_ref is None:
            fig.savefig(out_dir / f"{label}.png", dpi=110, facecolor=SURFACE, bbox_inches="tight")
            plt.close(fig)
            continue
        fig.text(0.01, -0.04, f"{run_name}   reference {f_ref:.2f} Hz ({note_name(f_ref)}), black line   "
                 f"blue = right note, red = wrong note   answered {res['answered_pct']:.0f}%   "
                 f"median error {res['median_cents']:.1f} c",
                 fontsize=7, color=TEXT_SECONDARY)
        fig.savefig(out_dir / f"{label}.png", dpi=110, facecolor=SURFACE, bbox_inches="tight")
        plt.close(fig)

    cols = 5
    rows = math.ceil(len(results) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.2 * rows), facecolor=SURFACE,
                             squeeze=False)
    for ax in axes.flat[len(results):]:
        ax.set_visible(False)
    for ax, (label, f_ref, res) in zip(axes.flat, results):
        draw_track(ax, res, f_ref, label, small=True)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
    fig.suptitle(f"{run_name}: detected pitch vs reference (black), harmonics dashed; "
                 f"blue = right note, red = wrong note", fontsize=9, color=TEXT_PRIMARY,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / "overview.png", dpi=100, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------- main

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a pitch detector on labelled captures.")
    p.add_argument("wav", nargs="+", type=Path)
    p.add_argument("--detector", default="zc", choices=sorted(DETECTORS) + sorted(TRACKERS))
    p.add_argument("--window", type=int, default=4096, help="window length in samples (default 4096 = 128 ms)")
    p.add_argument("--hop", type=int, default=1024, help="step between windows in samples (default 1024 = 32 ms)")
    p.add_argument("--highpass", type=float, default=None, help="high-pass cutoff in Hz before detecting")
    p.add_argument("--lowpass", type=float, default=None, help="low-pass cutoff in Hz before detecting")
    g = p.add_argument_group("yin / mpm (tools/pitch_ref.py) overrides")
    g.add_argument("--decim", type=int, choices=(1, 2, 4), help="decimation factor (default 4: 8 kHz)")
    g.add_argument("--threshold", type=float, help="YIN absolute threshold (default 0.15)")
    g.add_argument("--k", type=float, help="MPM key-maximum factor (default 0.9)")
    g.add_argument("--interp", choices=("none", "d", "dprime", "nsdf"), help="sub-sample interpolation")
    g.add_argument("--gate", type=float, help="confidence gate (default 0.35)")
    g.add_argument("--no-refine", action="store_true", help="skip the 32 kHz period refinement")
    g.add_argument("--float32", action="store_true", help="run the chain in float32")
    p.add_argument("--name", help="run name for the output folder (default built from the options)")
    p.add_argument("--csv", action="store_true", help="also write results.csv to the output folder")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def run_name_for(args):
    if args.name:
        return args.name
    if args.detector in TRACKERS:
        cfg = tracker_config(args)
        parts = [args.detector, f"fs{cfg.fs // 1000}k", cfg.interp,
                 f"thr{cfg.yin_threshold:g}" if cfg.method == "yin" else f"k{cfg.mpm_k:g}"]
        parts += [] if cfg.refine else ["noref"]
        parts += ["f32"] if args.float32 else []
        return "_".join(parts)
    return "_".join([args.detector, f"w{args.window}", f"h{args.hop}"]
                    + ([f"hp{args.highpass:g}"] if args.highpass else [])
                    + ([f"lp{args.lowpass:g}"] if args.lowpass else []))


def evaluate_files(paths, detector_name, args):
    """[(label, f_ref, result)] for every file; f_ref None for silences."""
    manifests, results = {}, []
    for path in sorted(paths):
        manifest = manifests.setdefault(path.parent, load_manifest(path.parent))
        f_ref, label = reference_hz(path, manifest)
        if f_ref is None and "silence" not in label:
            print(f"skipped {path.name}: no reference (label '{label}')", file=sys.stderr)
            continue
        results.append((label, f_ref, run_capture(path, detector_name, f_ref, args)))
    return results


def summary_rows(results):
    rows = []
    for label, f_ref, res in results:
        rows.append({"label": label, "ref_hz": round(f_ref, 2) if f_ref else "",
                     "answered_pct": round(res["answered_pct"], 1),
                     "note_ok_pct": round(res["note_ok_pct"], 1),
                     "median_cents": round(res["median_cents"], 2),
                     "signed_cents": round(res["signed_cents"], 2),
                     "typical_wrong": res["typical_wrong"]})
    return rows


def print_table(rows, header_line):
    print(header_line + "\n")
    print(f"{'label':24} {'ref Hz':>8} | {'answered':>8} {'note ok':>8} {'cents':>6} {'signed':>7}  typical wrong")
    print("-" * 82)
    for r in rows:
        if r["ref_hz"] == "":
            print(f"{r['label'][:24]:24} {'silence':>8} | {r['answered_pct']:7.0f}%  (false answers)")
            continue
        cents = f"{r['median_cents']:6.1f}" if not math.isnan(r["median_cents"]) else "     -"
        signed = f"{r['signed_cents']:+7.1f}" if not math.isnan(r["signed_cents"]) else "      -"
        print(f"{r['label'][:24]:24} {r['ref_hz']:8.2f} | {r['answered_pct']:7.0f}% {r['note_ok_pct']:7.0f}% "
              f"{cents} {signed}  {r['typical_wrong']}")
    notes = [r for r in rows if r["ref_hz"] != ""]
    silences = [r for r in rows if r["ref_hz"] == ""]
    print("-" * 82)
    if notes:
        total_ok = np.mean([r["note_ok_pct"] for r in notes])
        good = sum(1 for r in notes if r["note_ok_pct"] >= 90)
        print(f"mean note ok {total_ok:.0f}%   captures with >= 90% note ok: {good} of {len(notes)}", end="")
    if silences:
        print(f"   false answers on silence {np.mean([r['answered_pct'] for r in silences]):.0f}%", end="")
    print()


def main():
    args = parse_args()
    run_name = run_name_for(args)
    out_dir = OUT_ROOT / run_name
    results = evaluate_files(args.wav, args.detector, args)
    if not results:
        print("Nothing to evaluate.", file=sys.stderr)
        return 1
    rows = summary_rows(results)

    if args.detector in TRACKERS:
        cfg = tracker_config(args)
        header = (f"Detector '{args.detector}' (pitch_ref chain): {cfg.fs} Hz, window {cfg.window_ms:g} ms, "
                  f"hop {cfg.hop_ms:g} ms, interp {cfg.interp}, refine {'on' if cfg.refine else 'off'}, "
                  f"gate {cfg.gate:g}, {np.dtype(cfg.dtype).name}")
    else:
        header = (f"Detector '{args.detector}', window {args.window} "
                  f"({1000 * args.window / results[0][2]['fs']:.0f} ms), hop {args.hop}, "
                  f"highpass {args.highpass or '-'}, lowpass {args.lowpass or '-'}")
    print_table(rows, header)

    if args.csv:
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if not args.no_plot:
        save_plots(results, out_dir, run_name)
    if args.csv or not args.no_plot:
        try:
            shown = out_dir.relative_to(REPO_ROOT)
        except ValueError:
            shown = out_dir
        print(f"output in {shown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
