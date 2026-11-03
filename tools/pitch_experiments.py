#!/usr/bin/env python3
"""
Pitch-detector experiments: every measurement behind the detector choice, reproducible.

Usage:
    python tools/pitch_experiments.py              # all experiments (about two minutes)
    python tools/pitch_experiments.py filter sweep # only some of them

Experiments (figures and the log results.txt go to build/experiments/, or to --out DIR):
    filter      anti-alias FIR: response, delay, block vs one-shot, aliasing demo, 4 vs 5 kHz stopband
    functions   d(tau), d'(tau) and NSDF plots for typical windows
    sweep       fine error in cents vs frequency on synthetic tones, per sample rate / interpolation
    params      YIN threshold, MPM k and the confidence gate on captures + synthetics
    robust      DC offset, 30 Hz hum, MPM with/without high-pass, f_min 38 vs 29 Hz (B0)
    precision   float32 vs float64 end to end; energy-form d(tau) cancellation; FFT vs direct
    cost        multiply-adds per hop and RAM per configuration
    compare     per capture x detector table (HPS, YIN, MPM) -> build/eval/comparison_pitch.txt
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import freqz, lfilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pitch_ref as pr                                        # noqa: E402
from pitch_ref import Config, replace, track                  # noqa: E402
from synth import harmonic_tone, mic_highpass, pluck_amplitudes, room_noise, set_level  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "build" / "experiments"      # default; change with --out
CAPTURES = REPO_ROOT / "data" / "captures"
SYNTHETIC = REPO_ROOT / "data" / "synthetic"

# reference palette (same as plot.py / evaluate.py), categorical slots in fixed order
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
GUIDE = "#9a998f"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]

LOG = []


def log(text=""):
    print(text)
    LOG.append(text)


def section(title):
    log("")
    log("=" * 78)
    log(title)
    log("=" * 78)


def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GUIDE)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=110, facecolor=SURFACE, bbox_inches="tight")
    log(f"  figure: {OUT / name}")


def cents(f, ref):
    return 1200.0 * math.log2(f / ref)


def read(path):
    _, data = wavfile.read(path)
    return data[:, 0] if data.ndim > 1 else data


def recordings(folder):
    """[(path, ref_hz or None)] using evaluate.py's reference rules."""
    from describe import load_manifest
    from evaluate import reference_hz
    manifest = load_manifest(folder)
    out = []
    for p in sorted(folder.glob("*.wav")):
        f_ref, label = reference_hz(p, manifest)
        if f_ref is None and "silence" not in label:
            continue
        out.append((p, f_ref, label))
    return out


GUITAR_EXCLUDE = ("440", "1khz", "82_4hz", "silence", "bass")


def is_guitar(label):
    return not any(k in label for k in GUITAR_EXCLUDE)


def note_ok(freq, f_ref):
    from evaluate import note_number
    return note_number(freq) % 12 == note_number(f_ref) % 12


def score_set(cfg, files):
    """Per-file note-ok %, |cents| median and false-alarm % over a list of recordings."""
    rows = []
    for path, f_ref, label in files:
        res = track(read(path), cfg)
        valid = [r for r in res if r.valid and np.isfinite(r.freq_hz)]
        if f_ref is None:
            rows.append({"label": label, "false_pct": 100.0 * len(valid) / len(res)})
            continue
        ok = [r for r in valid if note_ok(r.freq_hz, f_ref)]
        errs = [abs((cents(r.freq_hz, f_ref) + 600) % 1200 - 600) for r in ok]
        rows.append({"label": label, "note_ok": 100.0 * len(ok) / len(res),
                     "cents": float(np.median(errs)) if errs else float("nan")})
    return rows


def summarise(rows, only=None):
    notes = [r for r in rows if "note_ok" in r and (only is None or only(r["label"]))]
    sil = [r for r in rows if "false_pct" in r]
    return (float(np.mean([r["note_ok"] for r in notes])) if notes else float("nan"),
            float(np.mean([r["false_pct"] for r in sil])) if sil else float("nan"))


# ======================================================================= filter

def exp_filter():
    section("1. Anti-alias FIR and decimator")
    plt = plt_setup()
    cfg = Config()
    h = pr.fir_taps(cfg)
    n = len(h)
    log(f"Design: Kaiser window method, passband {cfg.fir_pass_hz:g} Hz, stopband {cfg.fir_stop_hz:g} Hz, "
        f"{cfg.fir_atten_db:g} dB")
    log(f"  taps {n} (odd), symmetric: {np.allclose(h, h[::-1])}, sum {h.sum():.6f}")
    log(f"  group delay (N-1)/2 = {(n - 1) // 2} samples = {1000 * (n - 1) / 2 / cfg.fs_in:.2f} ms at 32 kHz")
    a_db, df = cfg.fir_atten_db, (cfg.fir_stop_hz - cfg.fir_pass_hz) / cfg.fs_in
    log(f"  fred harris estimate A/(22*df/Fs) = {a_db / (22 * df):.0f} taps")
    log(f"  cost: {n} MACs per output sample = {n * cfg.fs / 1e6:.2f} M MAC/s "
        f"({n / cfg.decim:.1f} per input sample)")

    w, resp = freqz(h, worN=8192, fs=cfg.fs_in)
    mag = 20 * np.log10(np.maximum(np.abs(resp), 1e-12))
    pass_ripple = np.ptp(mag[w <= cfg.fir_pass_hz])
    stop_att = -mag[w >= cfg.fir_stop_hz].max()
    log(f"  measured: passband ripple {pass_ripple:.4f} dB, stopband attenuation {stop_att:.1f} dB")

    # block-by-block == one-shot
    rng = np.random.default_rng(1)
    x = rng.standard_normal(32768)
    dec = pr.Decimator(h, cfg.decim)
    stream = np.concatenate([dec.process(x[i:i + pr.BLOCK]) for i in range(0, len(x), pr.BLOCK)])
    oneshot = lfilter(h, 1.0, x)[cfg.decim - 1::cfg.decim]
    log(f"  block-by-block vs one-shot lfilter + every 4th: max difference {np.max(np.abs(stream - oneshot)):.2e}")

    # tones through the decimator
    def tone_level(f, taps):
        t = np.arange(32768) / cfg.fs_in
        dec = pr.Decimator(taps, cfg.decim)                  # ONE decimator: state carries over
        y = np.concatenate([dec.process(c) for c in np.split(np.sin(2 * np.pi * f * t), 128)])[200:]
        return 20 * np.log10(np.sqrt(2 * np.mean(y ** 2)) + 1e-15)

    for f in (1000.0, 3000.0, 6000.0):
        log(f"  {f:6.0f} Hz tone in: {tone_level(f, h):7.2f} dB out with the filter, "
            f"{tone_level(f, np.array([1.0])):6.2f} dB without"
            + ("   (6 kHz aliases to 2 kHz)" if f == 6000 else ""))

    # 5 kHz stopband alternative
    h5 = pr.design_fir(cfg.fs_in, 3000.0, 5000.0, 60.0)
    log(f"  alternative stopband 5 kHz: {len(h5)} taps; a 4.5 kHz tone (aliases to 3.5 kHz) comes out at "
        f"{tone_level(4500.0, h5):.1f} dB vs {tone_level(4500.0, h):.1f} dB with the chosen filter")
    files = recordings(CAPTURES) + recordings(SYNTHETIC)
    base = summarise(score_set(cfg, files))[0]
    alt = summarise(score_set(replace(cfg, fir_stop_hz=5000.0), files))[0]
    log(f"  mean note ok, all recordings: stopband 4 kHz {base:.1f} %, 5 kHz {alt:.1f} %")

    fig, ax = plt.subplots(figsize=(8, 3.6), facecolor=SURFACE)
    ax.plot(w, mag, color=SERIES[0], linewidth=1.5)
    ax.axvspan(cfg.fir_pass_hz, cfg.fir_stop_hz, color=GRID, alpha=0.6, linewidth=0)
    ax.axvline(cfg.fs / 2, color=GUIDE, linestyle=(0, (3, 3)), linewidth=0.8)
    ax.axhline(-cfg.fir_atten_db, color=GUIDE, linestyle=(0, (3, 3)), linewidth=0.8)
    ax.text(cfg.fs / 2 + 150, 5, "new Nyquist 4 kHz", fontsize=8, color=TEXT_SECONDARY)
    ax.text(9000, -cfg.fir_atten_db + 3, f"-{cfg.fir_atten_db:g} dB spec", fontsize=8, color=TEXT_SECONDARY)
    ax.set_xlim(0, 16000)
    ax.set_ylim(-110, 10)
    ax.set_xlabel("frequency (Hz)", color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel("gain (dB)", color=TEXT_SECONDARY, fontsize=9)
    ax.set_title(f"Anti-alias FIR: {n} taps, passband to 3 kHz (grey band = transition)", loc="left",
                 fontsize=10, color=TEXT_PRIMARY)
    style(ax)
    save(fig, "filter_response.png")
    plt.close(fig)


# ==================================================================== functions

def exp_functions():
    section("2. What the detector functions look like")
    plt = plt_setup()
    cfg = Config()
    picks = [("E2 open (amp)", CAPTURES / "20261025_140314_E2_open_amp.wav", 82.41),
             ("E2 12th-fret pluck", CAPTURES / "20261025_140923_E2_fret12pluck_amp.wav", 82.41),
             ("E1 bass", CAPTURES / "20261025_194529_E1_bass_amp.wav", 41.20),
             ("silence", CAPTURES / "20261025_130307_silence.wav", None)]
    fig, axes = plt.subplots(len(picks), 2, figsize=(10, 2.1 * len(picks)), facecolor=SURFACE)
    for row, (title, path, f_ref) in enumerate(picks):
        x = read(path).astype(np.float64) / pr.FULL_SCALE
        dec = pr.Decimator(pr.fir_taps(cfg), cfg.decim)
        y = np.concatenate([dec.process(x[i:i + pr.BLOCK]) for i in range(0, 8192, pr.BLOCK)])
        win = y[-cfg.window:]
        est, f_yin = pr.yin(win, cfg, return_functions=True)
        est_m, f_mpm = pr.mpm(win, replace(cfg, method="mpm", interp="nsdf"), return_functions=True)
        tau = np.arange(len(f_yin["dprime"]))
        ax = axes[row, 0]
        ax.plot(tau, f_yin["dprime"], color=SERIES[0], linewidth=1.2)
        ax.axhline(cfg.yin_threshold, color=GUIDE, linestyle=(0, (3, 3)), linewidth=0.8)
        if f_ref:
            for k in (1, 2, 3):
                ax.axvline(k * cfg.fs / f_ref, color=GUIDE, linewidth=0.6)
        ax.set_ylim(0, 2.5)
        ax.set_xlim(0, tau[-1])
        ax.set_title(f"{title}: YIN d'(tau), threshold dashed", loc="left", fontsize=9, color=TEXT_PRIMARY)
        style(ax)
        ax = axes[row, 1]
        ax.plot(tau, f_mpm["nsdf"], color=SERIES[1], linewidth=1.2)
        ax.axhline(0, color=GUIDE, linewidth=0.6)
        if f_ref:
            for k in (1, 2, 3):
                ax.axvline(k * cfg.fs / f_ref, color=GUIDE, linewidth=0.6)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlim(0, tau[-1])
        ax.set_title(f"{title}: MPM NSDF", loc="left", fontsize=9, color=TEXT_PRIMARY)
        style(ax)
        f_y = cfg.fs / est.tau if np.isfinite(est.tau) else float("nan")
        log(f"  {title:20} YIN tau {est.tau:7.2f} ({f_y:7.2f} Hz) d'={est.confidence:.3f} below thr {est.below_threshold}"
            f" | MPM clarity {est_m.confidence:.3f}")
    for ax in axes[-1]:
        ax.set_xlabel("lag tau (samples at 8 kHz); grey lines = 1, 2, 3 periods", color=TEXT_SECONDARY, fontsize=8)
    fig.tight_layout()
    save(fig, "functions.png")
    plt.close(fig)
    log("  On silence the NSDF never crosses zero (room rumble below 40 Hz dominates), so MPM finds no key")
    log("  maxima there; on quiet unplugged notes the same rumble hides the zero crossings MPM needs.")
    log("  YIN's d' has no such requirement.")


# ======================================================================== sweep

SWEEP_F = np.geomspace(30.0, 1400.0, 40)


def folded(c):
    """Cents with whole octaves removed: the tuner shows no octave."""
    return (c + 600.0) % 1200.0 - 600.0


def sweep_errors(cfg, kind, seeds=(0, 1, 2)):
    """Worst octave-folded |cents| over seeds and frames for synthetic tones at SWEEP_F (no
    noise), plus the count of frames that named the wrong note (gross, non-octave errors)."""
    worst, gross, octave, frames = [], 0, 0, 0
    for f in SWEEP_F:
        errs = []
        for s in seeds:
            if kind == "sine":
                x = harmonic_tone(f, [1.0], decay_s=0, seed=s)
            else:
                x = mic_highpass(harmonic_tone(f, pluck_amplitudes(30, 0.15), decay_s=0, seed=s))
            x = set_level(x, -30.0)
            for r in track(np.round(x).astype(np.int16), cfg):
                if not np.isfinite(r.freq_hz):
                    continue
                frames += 1
                c = cents(r.freq_hz, f)
                if abs(folded(c)) > 50:
                    gross += 1
                    continue
                if abs(c) > 600:
                    octave += 1
                errs.append(abs(folded(c)))
        worst.append(max(errs) if errs else float("nan"))
    return np.array(worst), gross, octave, frames


def exp_sweep():
    section("3. Fine error vs frequency (synthetic, no noise, worst case over 3 phases x all frames)")
    plt = plt_setup()
    configs = [("8 kHz, no interpolation", replace(Config(), interp="none", refine=False)),
               ("8 kHz, parabola on d'", replace(Config(), interp="dprime", refine=False)),
               ("8 kHz, parabola on d", replace(Config(), interp="d", refine=False)),
               ("8 kHz + 32 kHz refinement", Config()),
               ("16 kHz, parabola on d", pr.config_for_rate(2, refine=False)),
               ("32 kHz, parabola on d", pr.config_for_rate(1, refine=False))]
    shown = {"8 kHz, parabola on d": SERIES[0], "16 kHz, parabola on d": SERIES[1],
             "8 kHz + 32 kHz refinement": SERIES[2]}           # 32 kHz direct overlaps the last one
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE, sharey=True)
    table, counts = {}, {}
    for col, kind in enumerate(("harmonic", "sine")):
        ax = axes[col]
        for name, cfg in configs:
            e, gross, octave, frames = sweep_errors(cfg, kind)
            table[(name, kind)] = e
            counts[(name, kind)] = (gross, octave, frames)
            if name in shown:
                label = name + (" (= 32 kHz direct)" if "refinement" in name else "")
                ax.plot(SWEEP_F, np.maximum(e, 0.003), color=shown[name], linewidth=1.6, marker="o",
                        markersize=3, label=label)
        ax.axhline(1.0, color=TEXT_SECONDARY, linestyle=(0, (3, 3)), linewidth=0.9)
        ax.text(31, 1.12, "±1 cent target", fontsize=8, color=TEXT_SECONDARY)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(0.003, 60)
        ax.set_xlabel("fundamental (Hz, log)", color=TEXT_SECONDARY, fontsize=9)
        ax.set_title("harmonic-rich tone (30 partials)" if kind == "harmonic" else "pure sine", loc="left",
                     fontsize=10, color=TEXT_PRIMARY)
        style(ax)
    axes[0].set_ylabel("worst |error|, octave-folded (cents, log)", color=TEXT_SECONDARY, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("YIN fine error vs frequency: refinement on the 32 kHz signal meets ±1 cent up to ~330 Hz",
                 x=0.01, ha="left", fontsize=11, color=TEXT_PRIMARY)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "accuracy_sweep.png")
    plt.close(fig)

    bands = [("30-100 Hz", SWEEP_F < 100), ("100-330 Hz", (SWEEP_F >= 100) & (SWEEP_F < 330)),
             ("330-700 Hz", (SWEEP_F >= 330) & (SWEEP_F < 700)), ("700-1400 Hz", SWEEP_F >= 700)]
    log("  worst octave-folded |cents| per band (harmonic-rich / sine); octave and wrong-note frames counted apart")
    log(f"  {'configuration':28}" + "".join(f"{b:>18}" for b, _ in bands))
    for name, _ in configs:
        cells = "".join(f"{np.nanmax(table[(name, 'harmonic')][m]):8.2f} /{np.nanmax(table[(name, 'sine')][m]):7.2f}  "
                        for _, m in bands)
        g, o, n = (sum(v) for v in zip(counts[(name, "harmonic")], counts[(name, "sine")]))
        log(f"  {name:28}{cells} octave {o}/{n}, wrong note {g}/{n}")
    mpm = replace(Config(), method="mpm", interp="nsdf", hp_hz=20.0)
    for kind in ("harmonic", "sine"):
        e, g, o, n = sweep_errors(mpm, kind)
        log(f"  MPM + HP + 32 kHz refinement, {kind}: worst {np.nanmax(e):.2f} cents, octave {o}/{n}, wrong note {g}/{n}")


# ======================================================================= params

def exp_params():
    section("4. Detector parameters (captures + synthetics)")
    plt = plt_setup()
    files = recordings(CAPTURES) + recordings(SYNTHETIC)
    log("  YIN absolute threshold (gate 0.5):")
    thr = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    yin_ok = []
    for t in thr:
        rows = score_set(replace(Config(), yin_threshold=t), files)
        ok, fa = summarise(rows)
        gok, _ = summarise(rows, is_guitar)
        yin_ok.append(ok)
        log(f"    threshold {t:4.2f}: mean note ok {ok:5.1f} %  (guitar captures {gok:5.1f} %)  false answers on silence {fa:4.1f} %")
    log("  MPM k (20 Hz high-pass in front, see robust):")
    ks = [0.8, 0.85, 0.9, 0.95, 1.0]
    mpm_ok = []
    for k in ks:
        rows = score_set(replace(Config(), method="mpm", interp="nsdf", mpm_k=k, hp_hz=20.0), files)
        ok, fa = summarise(rows)
        mpm_ok.append(ok)
        log(f"    k {k:4.2f}: mean note ok {ok:5.1f} %  false answers on silence {fa:4.1f} %")

    # confidence distributions: silence vs notes
    conf_notes, conf_sil = [], []
    for path, f_ref, _ in files:
        res = track(read(path), Config())
        (conf_sil if f_ref is None else conf_notes).extend(r.confidence for r in res)
    conf_notes, conf_sil = np.array(conf_notes), np.array(conf_sil)
    log(f"  YIN d' at the chosen dip: notes median {np.median(conf_notes):.3f}, 95th pct "
        f"{np.percentile(conf_notes, 95):.3f}; silence min {conf_sil.min():.3f}, median {np.median(conf_sil):.3f}")
    for g in (0.2, 0.35, 0.5, 0.6):
        log(f"    gate {g:4.2f}: notes passing {100 * np.mean(conf_notes <= g):5.1f} %, "
            f"silence frames passing {100 * np.mean(conf_sil <= g):5.1f} %")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), facecolor=SURFACE)
    ax = axes[0]
    bins = np.linspace(0, 1.6, 65)
    ax.hist(conf_notes, bins=bins, color=SERIES[0], label="note frames", alpha=0.9)
    ax.hist(conf_sil, bins=bins, color=SERIES[1], label="silence frames", alpha=0.9)
    ax.axvline(0.5, color=TEXT_SECONDARY, linestyle=(0, (3, 3)), linewidth=0.9)
    ax.text(0.52, ax.get_ylim()[1] * 0.5, "gate 0.5", fontsize=8, color=TEXT_SECONDARY)
    ax.set_yscale("log")
    ax.set_xlabel("YIN d' at the chosen dip (0 = perfectly periodic)", color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel("frames (log)", color=TEXT_SECONDARY, fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Confidence separates notes from silence", loc="left", fontsize=10, color=TEXT_PRIMARY)
    style(ax)
    ax = axes[1]
    ax.plot(thr, yin_ok, color=SERIES[0], marker="o", markersize=4, linewidth=1.5, label="YIN vs threshold")
    ax.set_xlabel("YIN absolute threshold", color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel("mean note ok (%)", color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylim(90, 100)
    ax.set_title("Note ok vs YIN threshold (all recordings)", loc="left", fontsize=10, color=TEXT_PRIMARY)
    style(ax)
    fig.tight_layout()
    save(fig, "parameters.png")
    plt.close(fig)


# ======================================================================= robust

def exp_robust():
    section("5. Robustness")
    files = recordings(CAPTURES) + recordings(SYNTHETIC)
    log("  DC offset and hum added to synthetic notes (median |cents|, note ok):")
    for note_hz, name in ((41.203, "E1"), (82.407, "E2")):
        clean = set_level(mic_highpass(harmonic_tone(note_hz, pluck_amplitudes(30, 0.2), seed=3)), -45.0)
        t = np.arange(len(clean)) / 32000
        hum = np.sqrt(2) * np.std(clean) * 10 ** (-10 / 20)          # peak of a hum 10 dB below the note
        variants = {"clean": clean,
                    "DC +2000 counts": clean + 2000.0,
                    "30 Hz hum, -10 dB": clean + hum * np.sin(2 * np.pi * 30 * t),
                    "50 Hz hum, -10 dB": clean + hum * np.sin(2 * np.pi * 50 * t),
                    "30 Hz hum, 0 dB": clean + hum * np.sqrt(10) * np.sin(2 * np.pi * 30 * t)}
        for vname, x in variants.items():
            line = f"    {name} {vname:24}"
            for det, cfg in (("YIN", Config()), ("MPM", replace(Config(), method="mpm", interp="nsdf")),
                             ("MPM+HP20", replace(Config(), method="mpm", interp="nsdf", hp_hz=20.0))):
                # median of octave-folded |cents|: octave answers count as right (tuner shows no octave)
                all_res = track(np.round(x + room_noise(seed=5)).astype(np.int16), cfg)
                ok = [r for r in all_res if r.valid and note_ok(r.freq_hz, note_hz)]
                med = np.median([abs(folded(cents(r.freq_hz, note_hz))) for r in ok]) if ok else float("nan")
                line += f" | {det} {100 * len(ok) / len(all_res):5.1f} % {med:6.2f} c"
            log(line)

    log("  MPM with a high-pass in front (all recordings):")
    for hp in (0.0, 20.0, 30.0):
        for det, cfg in (("YIN", Config()), ("MPM", replace(Config(), method="mpm", interp="nsdf"))):
            rows = score_set(replace(cfg, hp_hz=hp), files)
            ok, fa = summarise(rows)
            e1 = [r["note_ok"] for r in rows if r["label"] == "E1_bass_amp"][0]
            log(f"    high-pass {hp:4.0f} Hz {det}: mean note ok {ok:5.1f} %, E1 capture {e1:5.1f} %, "
                f"false answers on silence {fa:4.1f} %")

    log("  Lowest note searched (f_min):")
    for fmin in (38.0, 29.0):
        cfg = replace(Config(), f_min=fmin)
        rows = score_set(cfg, files)
        ok, fa = summarise(rows)
        b0 = {r["label"]: r["note_ok"] for r in rows if "B0" in r["label"]}
        cost = pr.cost_per_hop(cfg)["total"]
        log(f"    f_min {fmin:4.0f} Hz (tau_max {cfg.tau_max}): mean note ok {ok:5.1f} %, B0 {b0}, "
            f"false answers {fa:4.1f} %, {cost / 1e3:.0f} k MAC per hop")

    log("  Attack: note ok in the first window (128 ms after the capture starts) vs the rest:")
    first, rest = [], []
    for path, f_ref, _ in recordings(CAPTURES):
        if f_ref is None:
            continue
        res = track(read(path), Config())
        first.append(res[0].valid and note_ok(res[0].freq_hz, f_ref))
        rest += [r.valid and note_ok(r.freq_hz, f_ref) for r in res[1:]]
    log(f"    first window {100 * np.mean(first):.0f} %, later windows {100 * np.mean(rest):.0f} % "
        "(captures start after the pluck, so this is the ring, not the attack)")


# ==================================================================== precision

def exp_precision():
    section("6. Precision: float32 vs float64, cancellation, FFT")
    files = recordings(CAPTURES) + recordings(SYNTHETIC)
    worst, worst_label, flips, total = 0.0, "", 0, 0
    for method, cfg in (("yin", Config()), ("mpm", replace(Config(), method="mpm", interp="nsdf", hp_hz=20.0))):
        worst_m = 0.0
        for path, _, label in files:
            x = read(path)
            a = track(x, cfg)
            b = track(x, replace(cfg, dtype=np.float32))
            for ra, rb in zip(a, b):
                total += 1
                if ra.valid != rb.valid:
                    flips += 1
                if ra.valid and rb.valid:
                    d = abs(cents(rb.freq_hz, ra.freq_hz))
                    if d > worst_m:
                        worst_m, worst_label = d, label
        log(f"  {method}: float32 vs float64, worst frequency difference {worst_m:.1e} cents ({worst_label})")
        worst = max(worst, worst_m)
    log(f"  frames whose valid flag differs: {flips} of {total}")

    # direct vs energy-form difference function in float32
    cfg = Config()
    w = cfg.window - cfg.tau_max - 1
    t8 = np.arange(cfg.window) / cfg.fs
    cases = [("E2 capture (d'=0.03)", read(CAPTURES / "20261025_140314_E2_open_amp.wav")[:4096:4] / 32768.0),
             ("E1 bass capture (d'=0.003)", read(CAPTURES / "20261025_194529_E1_bass_amp.wav")[:4096:4] / 32768.0),
             ("pure 246.9 Hz tone (d' ~ 1e-3)", 0.01 * np.sin(2 * np.pi * 246.94 * t8))]
    for name, win in cases:
        win = np.asarray(win[:cfg.window], dtype=np.float64)
        ref = pr.difference_function(win, cfg.tau_max)
        k = int(np.argmin(ref[cfg.tau_min:cfg.tau_max])) + cfg.tau_min
        for scale_name, s_ in (("+-1", 1.0), ("int16 counts", 32768.0)):
            x32 = (win * s_).astype(np.float32)
            direct = pr.difference_function(x32, cfg.tau_max).astype(np.float64) / s_ ** 2
            e0 = np.dot(x32[:w], x32[:w])
            energy = np.array([e0 + np.dot(x32[t:t + w], x32[t:t + w]) - np.float32(2) * np.dot(x32[:w], x32[t:t + w])
                               for t in range(cfg.tau_max + 2)], dtype=np.float32).astype(np.float64) / s_ ** 2
            log(f"  float32 d(tau) at the dip, {name:26} scaled {scale_name:12}: direct rel. error "
                f"{abs(direct[k] - ref[k]) / ref[k]:.1e}, energy form {abs(energy[k] - ref[k]) / ref[k]:.1e}")
    log("  (scaling changes nothing: floating point is scale-invariant until overflow/underflow)")

    # FFT correlation vs direct
    n = cfg.window
    xw = np.asarray(cases[0][1][:n], dtype=np.float64)
    r_direct = np.array([np.dot(xw[:n - t], xw[t:]) for t in range(cfg.tau_max + 2)])
    for nfft, name in ((n, "no padding (circular)"), (2048, "padded to 2048")):
        spec = np.fft.rfft(xw, nfft)
        r_fft = np.fft.irfft(spec * np.conj(spec), nfft)[:cfg.tau_max + 2]
        log(f"  autocorrelation by FFT, {name}: max |error| / r(0) = {np.max(np.abs(r_fft - r_direct)) / r_direct[0]:.1e}")
    log(f"  FFT route: forward + inverse 2048-point real FFT, O(N log N) = about 2 x 1024 x 10 = 20 k complex "
        f"butterflies (4 real MACs each, ~80 k) plus the energy terms, vs {pr.cost_per_hop(cfg)['detector'] // 1000} k MACs direct."
        " Worth it only if the direct loop measures too slow on the target; it also needs the energy form (cancellation above).")


# ========================================================================= cost

def exp_cost():
    section("7. Cost per hop (32 ms) and RAM, float32")
    rows = [("8 kHz YIN + refinement (chosen)", Config()),
            ("8 kHz YIN + refinement, f_min 38 Hz", replace(Config(), f_min=38.0)),
            ("8 kHz YIN, no refinement", replace(Config(), refine=False)),
            ("8 kHz MPM + HP + refinement", replace(Config(), method="mpm", interp="nsdf", hp_hz=20.0)),
            ("16 kHz YIN, no refinement", pr.config_for_rate(2, refine=False)),
            ("32 kHz YIN (no decimation)", pr.config_for_rate(1, refine=False))]
    log(f"  {'configuration':38} {'FIR':>7} {'detector':>9} {'refine':>7} {'total':>8} {'M MAC/s':>8} {'RAM KB':>7}")
    for name, cfg in rows:
        c = pr.cost_per_hop(cfg)
        ram = pr.ram_bytes(cfg)["total"] / 1024
        log(f"  {name:38} {c['fir'] / 1e3:6.0f}k {c['detector'] / 1e3:8.0f}k {c['refine'] / 1e3:6.0f}k "
            f"{c['total'] / 1e3:7.0f}k {c['total'] * 1000 / cfg.hop_ms / 1e6:8.2f} {ram:7.1f}")
    log("  At 100 MHz and ~2 cycles per float MAC (a guess, to be measured on the target), 10 M MAC/s is about 20 % CPU.")


# ====================================================================== compare

def exp_compare():
    section("8. Per-recording comparison: HPS vs YIN vs MPM")
    import evaluate as ev
    from argparse import Namespace
    base = dict(decim=None, threshold=None, k=None, interp=None, gate=None, no_refine=False, float32=False,
                highpass=None, lowpass=None)
    runs = [("HPS 128 ms", "fft_hps", Namespace(detector="fft_hps", window=4096, hop=1024, **base)),
            ("YIN", "yin", Namespace(detector="yin", window=4096, hop=1024, **base)),
            ("MPM+HP20", "mpm", Namespace(detector="mpm", window=4096, hop=1024, **base))]
    lines = []
    for folder in (CAPTURES, SYNTHETIC):
        paths = sorted(folder.glob("*.wav"))
        per = {name: ev.summary_rows(ev.evaluate_files(paths, det, a)) for name, det, a in runs}
        labels = [r["label"] for r in per["YIN"]]
        lines.append(f"{folder.relative_to(REPO_ROOT)}")
        lines.append(f"{'recording':26} {'ref Hz':>8} | " + " | ".join(f"{n:>21}" for n, _, _ in runs))
        lines.append(f"{'':26} {'':>8} | " + " | ".join(f"{'note ok   cents  sgn':>21}" for _ in runs))
        lines.append("-" * 105)
        for i, lab in enumerate(labels):
            r0 = per["YIN"][i]
            ref = "silence" if r0["ref_hz"] == "" else f"{r0['ref_hz']:.2f}"
            cells = []
            for n, _, _ in runs:
                r = per[n][i]
                if r["ref_hz"] == "":
                    cells.append(f"{'false ' + format(r['answered_pct'], '.0f') + '%':>21}")
                else:
                    c = "-" if math.isnan(r["median_cents"]) else f"{r['median_cents']:.1f}"
                    s = "-" if math.isnan(r["signed_cents"]) else f"{r['signed_cents']:+.1f}"
                    cells.append(f"{r['note_ok_pct']:6.0f}% {c:>6} {s:>6}")
            lines.append(f"{lab[:26]:26} {ref:>8} | " + " | ".join(cells))
        for n, _, _ in runs:
            notes = [r for r in per[n] if r["ref_hz"] != ""]
            guitar = [r for r in notes if is_guitar(r["label"])]
            lines.append(f"  {n}: mean note ok {np.mean([r['note_ok_pct'] for r in notes]):.1f} %"
                         + (f", guitar captures {np.mean([r['note_ok_pct'] for r in guitar]):.1f} %" if folder == CAPTURES else ""))
        lines.append("")
    out = REPO_ROOT / "build" / "eval" / "comparison_pitch.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        log("  " + line)
    log(f"  table: {out.relative_to(REPO_ROOT)}")


EXPERIMENTS = {"filter": exp_filter, "functions": exp_functions, "sweep": exp_sweep, "params": exp_params,
               "robust": exp_robust, "precision": exp_precision, "cost": exp_cost, "compare": exp_compare}


def main():
    p = argparse.ArgumentParser(description="Pitch-detector experiments")
    p.add_argument("which", nargs="*", help=f"any of: {', '.join(EXPERIMENTS)} (default: all)")
    p.add_argument("--out", type=Path, help="output folder (default build/experiments)")
    args = p.parse_args()
    if args.out:
        global OUT
        OUT = args.out.resolve()
    unknown = [w for w in args.which if w not in EXPERIMENTS]
    if unknown:
        p.error(f"unknown experiment(s): {', '.join(unknown)}")
    names = args.which or list(EXPERIMENTS)
    if not (SYNTHETIC / "manifest.csv").exists():
        print("data/synthetic is missing: run python tools/synth.py first", file=sys.stderr)
        return 1
    for n in names:
        EXPERIMENTS[n]()
    OUT.mkdir(parents=True, exist_ok=True)
    name = "results.txt" if not args.which else f"results_{'_'.join(names)}.txt"
    (OUT / name).write_text("\n".join(LOG) + "\n", encoding="utf-8")
    print(f"\nsaved {OUT / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
