"""
Reference pitch-detection chain for the tuner.

This file is the "golden reference" the C pitch module is checked against, so it is
written the way the firmware will run, not the way numpy would like:

    int16 block (256 samples, one DMA half, 32 kHz)
        -> scale to +-1.0
        -> anti-alias FIR + keep every DECIM-th sample (streaming, state kept between blocks)
        -> ring of the latest WINDOW_MS of decimated samples
        -> every HOP_MS: pitch detector (YIN or MPM) on that window
        -> optional refinement of the period on the raw 32 kHz samples
        -> Result(time, frequency, confidence, valid)

Every stage is a plain function or a small class with explicit state, so each maps
onto one C function / struct. Nothing here looks ahead in time.

Usage from other scripts:

    from pitch_ref import Config, PitchTracker
    tracker = PitchTracker(Config(method="yin"))
    for block in blocks_of_256_int16:
        for r in tracker.push(block):
            print(r.time_s, r.freq_hz, r.confidence, r.valid)

Defaults are the chosen settings; tools/pitch_experiments.py measures the alternatives.
"""
from dataclasses import dataclass, field, replace
from functools import lru_cache

import numpy as np
from scipy.signal import butter, firwin, kaiserord, sosfilt

FS_IN = 32000            # capture rate, Hz (I2S, PLLI2S exact)
BLOCK = 256              # samples per DMA half at FS_IN
FULL_SCALE = 32768.0     # int16 full scale


# --------------------------------------------------------------------- config

@dataclass(frozen=True)
class Config:
    fs_in: int = FS_IN
    decim: int = 4                 # 32 kHz -> 8 kHz. 1 = no decimation
    window_ms: float = 128.0       # analysis window
    hop_ms: float = 32.0           # time between results (= 4 DMA halves)
    f_min: float = 29.0            # lowest fundamental searched (B0 50 cents flat = 30 Hz)
    f_max: float = 1400.0          # highest fundamental searched
    method: str = "yin"            # "yin" or "mpm"
    yin_threshold: float = 0.15    # YIN absolute threshold on d'(tau)
    mpm_k: float = 0.9             # MPM: first key maximum above k * highest one
    interp: str = "d"              # sub-sample step: "none", "d" (YIN paper), "dprime" / "nsdf"
    refine: bool = True            # refine the period on the raw FS_IN samples
    gate: float = 0.5              # validity: YIN d' <= gate, MPM clarity >= 1 - gate
    hp_hz: float = 0.0             # optional 2nd-order high-pass before the detector (0 = off)
    dtype: type = np.float64       # np.float32 = what the Cortex-M4F computes
    # anti-alias FIR (window method, Kaiser)
    fir_pass_hz: float = 3000.0
    fir_stop_hz: float = 4000.0
    fir_atten_db: float = 60.0

    @property
    def fs(self):
        """Detector sample rate."""
        return self.fs_in // self.decim

    @property
    def window(self):
        return int(round(self.window_ms * self.fs / 1000.0))

    @property
    def hop(self):
        return int(round(self.hop_ms * self.fs / 1000.0))

    @property
    def tau_min(self):
        # one below the shortest period, so a dip at the top of the range still has a left neighbour
        return max(2, int(np.floor(self.fs / self.f_max)))

    @property
    def tau_max(self):
        return int(np.ceil(self.fs / self.f_min))


def config_for_rate(decim, **kw):
    """Same design rules at another decimation factor: FIR passband at 75 % and stopband at
    100 % of the new Nyquist frequency."""
    nyq = FS_IN / decim / 2.0
    return Config(decim=decim, fir_pass_hz=0.75 * nyq, fir_stop_hz=nyq, **kw)


# ------------------------------------------------------------ anti-alias FIR

@lru_cache(maxsize=None)
def design_fir(fs_in, pass_hz, stop_hz, atten_db):
    """Low-pass FIR by the Kaiser window method.

    kaiserord turns (attenuation, transition width) into a length and the Kaiser beta.
    The length is rounded up to odd so the delay (N-1)/2 is a whole number of samples.
    firwin's cutoff is the -6 dB point, so it goes in the middle of the transition band.
    Returns float64 taps (symmetric -> linear phase).
    """
    width = (stop_hz - pass_hz) / (fs_in / 2.0)          # kaiserord wants a fraction of Nyquist
    numtaps, beta = kaiserord(atten_db, width)
    numtaps |= 1                                          # odd length
    cutoff = 0.5 * (pass_hz + stop_hz)
    return firwin(numtaps, cutoff, window=("kaiser", beta), fs=fs_in)


def fir_taps(cfg):
    if cfg.decim == 1:
        return np.array([1.0])
    return design_fir(cfg.fs_in, cfg.fir_pass_hz, cfg.fir_stop_hz, cfg.fir_atten_db)


class Decimator:
    """Streaming FIR decimator: filter, keep every M-th output, compute only the kept ones.

    State between blocks = the last (N - 1) input samples (the delay line). An output is
    produced for every input index n with n % M == M - 1 (the last sample of each group
    of M), so a 256-sample block always yields 256 / M outputs.

    C equivalent: a struct with float delay[N-1 + BLOCK] and a loop of BLOCK/M dot
    products of length N.
    """

    def __init__(self, taps, decim, dtype=np.float64):
        self.h = np.asarray(taps, dtype=dtype)
        self.h_rev = self.h[::-1].copy()                  # dot with oldest..newest samples
        self.m = decim
        self.dtype = dtype
        self.delay = np.zeros(len(self.h) - 1, dtype=dtype)

    def process(self, block):
        block = np.asarray(block, dtype=self.dtype)
        if len(block) % self.m:
            raise ValueError("block length must be a multiple of the decimation factor")
        n_taps = len(self.h)
        buf = np.concatenate((self.delay, block))         # [old N-1 samples | new block]
        out = np.empty(len(block) // self.m, dtype=self.dtype)
        for k in range(len(out)):
            newest = n_taps - 1 + (k + 1) * self.m - 1    # index in buf of the newest input
            out[k] = np.dot(self.h_rev, buf[newest - n_taps + 1:newest + 1])
        self.delay = buf[len(buf) - (n_taps - 1):].copy() if n_taps > 1 else self.delay
        return out


class Highpass:
    """Optional 2nd-order Butterworth high-pass at the detector rate (one biquad, state kept
    between blocks). C equivalent: arm_biquad_cascade_df1_f32 with one stage."""

    def __init__(self, corner_hz, fs, dtype=np.float64):
        self.sos = butter(2, corner_hz, btype="highpass", fs=fs, output="sos").astype(dtype)
        self.zi = np.zeros((1, 2), dtype=dtype)

    def process(self, x):
        y, self.zi = sosfilt(self.sos, x, zi=self.zi)
        return y.astype(x.dtype)


# ---------------------------------------------------------------- detectors

def parabolic_offset(a, b, c):
    """Vertex of the parabola through (-1, a), (0, b), (1, c), relative to the middle point.
    Same formula for a minimum or a maximum. Returns 0 for a flat triple."""
    denom = a - 2.0 * b + c
    if denom == 0:
        return 0.0
    return float(0.5 * (a - c) / denom)


def difference_function(x, tau_max):
    """YIN step 2 (eq. 6), direct form: d[tau] = sum_{j<W} (x[j] - x[j+tau])^2 for
    tau = 0 .. tau_max + 1, with W = len(x) - tau_max - 1 (fixed window, type I).

    Cost: W * (tau_max + 2) multiply-adds. The direct sum is used on purpose: the
    energy-term form (eq. 7) suffers catastrophic cancellation in float32."""
    w = len(x) - tau_max - 1
    d = np.empty(tau_max + 2, dtype=x.dtype)
    head = x[:w]
    for tau in range(tau_max + 2):
        diff = head - x[tau:tau + w]
        d[tau] = np.dot(diff, diff)
    return d


def cmndf(d):
    """YIN step 3 (eq. 8): d'[0] = 1, d'[tau] = d[tau] * tau / sum_{j=1..tau} d[j]."""
    out = np.ones_like(d)
    running = np.cumsum(d[1:])
    tau = np.arange(1, len(d), dtype=d.dtype)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[1:] = np.where(running > 0, d[1:] * tau / running, 1.0)
    return out


@dataclass
class Estimate:
    tau: float            # period in detector samples, sub-sample
    confidence: float     # YIN: d' at the dip (0 = periodic). MPM: clarity (1 = periodic)
    in_range: bool        # False if the best candidate sits on the edge of the lag range
    below_threshold: bool # YIN: a dip under the absolute threshold was found


def yin(x, cfg, return_functions=False):
    """YIN steps 2-5 on one window. Step 6 (best local estimate) is not used."""
    d = difference_function(x, cfg.tau_max)
    dp = cmndf(d)
    lo, hi = cfg.tau_min, cfg.tau_max

    # Step 4: first tau in range with d' under the threshold, then walk down to its minimum
    tau, below = None, False
    for t in range(lo, hi + 1):
        if dp[t] < cfg.yin_threshold:
            while t + 1 <= hi and dp[t + 1] < dp[t]:
                t += 1
            tau, below = t, True
            break
    if tau is None:                                   # no dip under threshold: global minimum
        tau = lo + int(np.argmin(dp[lo:hi + 1]))

    # A "minimum" on the range edge whose outside neighbour is lower is not a real dip
    in_range = not ((tau == lo and dp[lo - 1] < dp[lo]) or (tau == hi and dp[hi + 1] < dp[hi]))

    # Step 5: parabola through the dip. The paper takes the position from d, not d'. d' is d
    # tilted by its running mean, so d's own minimum can sit one lag away: centre on it first.
    src = {"d": d, "dprime": dp}.get(cfg.interp)
    off = 0.0
    if src is not None:
        c = tau
        if cfg.interp == "d":
            c = tau - 1 + int(np.argmin(d[tau - 1:tau + 2]))
            c = min(max(c, 1), len(d) - 2)
        off = (c - tau) + parabolic_offset(src[c - 1], src[c], src[c + 1])
    est = Estimate(tau + off, float(dp[tau]), in_range, below)
    if return_functions:
        return est, {"d": d, "dprime": dp}
    return est


def nsdf(x, tau_max):
    """MPM normalised square difference function (McLeod & Wyvill eq. 9), type II window:
    n[tau] = 2 r'[tau] / m'[tau], r' and m' summed over j = 0 .. N-1-tau.
    m'[tau] is updated incrementally from m'[tau-1] (paper section 6)."""
    n_len = len(x)
    out = np.zeros(tau_max + 2, dtype=x.dtype)
    sq = x * x
    m = 2.0 * np.sum(sq)                                  # m'[0] = sum x_j^2 + sum x_{j+0}^2
    for tau in range(tau_max + 2):
        if tau > 0:
            m -= sq[n_len - tau] + sq[tau - 1]            # drop the samples leaving the overlap
        r = np.dot(x[:n_len - tau], x[tau:])
        out[tau] = 2.0 * r / m if m > 0 else 0.0
    return out


def mpm(x, cfg, return_functions=False):
    """MPM peak picking: key maxima between positive- and negative-going zero crossings."""
    n = nsdf(x, cfg.tau_max)
    lo, hi = cfg.tau_min, cfg.tau_max
    keys = []                                             # (tau, value) of each key maximum
    t = 1
    while t < len(n) and n[t] > 0:                        # skip the lobe around tau = 0
        t += 1
    best_t, best_v, inside = None, -np.inf, False
    for t in range(t, len(n) - 1):
        if not inside and n[t - 1] <= 0 < n[t]:           # positive-going zero crossing
            inside, best_t, best_v = True, None, -np.inf
        elif inside and n[t - 1] > 0 >= n[t]:             # negative-going: close the lobe
            if best_t is not None:
                keys.append((best_t, best_v))
            inside = False
        if inside and n[t] > best_v:
            best_t, best_v = t, n[t]
    if inside and best_t is not None:                     # lobe still open at the end
        keys.append((best_t, best_v))
    keys = [(t, v) for t, v in keys if lo <= t <= hi]
    if not keys:
        est = Estimate(float("nan"), 0.0, False, False)
        return (est, {"nsdf": n}) if return_functions else est

    threshold = cfg.mpm_k * max(v for _, v in keys)
    tau, val = next((t, v) for t, v in keys if v >= threshold)
    in_range = not (tau == hi and n[hi + 1] > n[hi])
    off = parabolic_offset(n[tau - 1], n[tau], n[tau + 1]) if cfg.interp != "none" else 0.0
    est = Estimate(tau + off, float(val), in_range, True)
    return (est, {"nsdf": n}) if return_functions else est


def refine_period(raw, tau_coarse_raw, search, max_lag):
    """Coarse-to-fine: minimum of the difference function on the raw-rate window, searched only
    within +-search lags of the coarse period, then a parabola. Cost: (2*search+3) * W MACs.
    Returns the refined period in raw samples, or the coarse one if no interior minimum."""
    w = len(raw) - max_lag - 2
    centre = int(round(tau_coarse_raw))
    lags = np.arange(max(2, centre - search - 1), min(max_lag + 1, centre + search + 1) + 1)
    head = raw[:w]
    d = np.array([np.dot(head - raw[t:t + w], head - raw[t:t + w]) for t in lags])
    if len(d) < 3:
        return float(tau_coarse_raw)
    k = int(np.argmin(d[1:-1])) + 1                       # interior minimum only
    if not (d[k] <= d[k - 1] and d[k] <= d[k + 1]):
        return float(tau_coarse_raw)
    return float(lags[k] + parabolic_offset(d[k - 1], d[k], d[k + 1]))


def detect(x, cfg, raw=None):
    """One analysis: detector on the decimated window x, optional refinement on raw.
    Returns (freq_hz, Estimate)."""
    est = yin(x, cfg) if cfg.method == "yin" else mpm(x, cfg)
    if not np.isfinite(est.tau) or est.tau <= 0:
        return float("nan"), est
    if cfg.refine and cfg.decim > 1 and raw is not None and est.in_range:
        max_lag = cfg.tau_max * cfg.decim + cfg.decim + 2
        tau_raw = refine_period(raw, est.tau * cfg.decim, search=cfg.decim, max_lag=max_lag)
        return cfg.fs_in / tau_raw, est
    return cfg.fs / est.tau, est


def is_valid(est, cfg):
    """Confidence gate (a first version; a stability stage on top refines it)."""
    if not est.in_range or not np.isfinite(est.tau):
        return False
    if cfg.method == "yin":
        return est.confidence <= cfg.gate
    return est.confidence >= 1.0 - cfg.gate


# ----------------------------------------------------------------- tracker

@dataclass
class Result:
    time_s: float         # time of the newest sample in the window, from the stream start
    freq_hz: float
    confidence: float
    valid: bool
    tau: float = float("nan")


class PitchTracker:
    """The whole chain, fed block by block like the firmware.

    State: decimator delay line, a ring of the latest `window` decimated samples, a ring of
    the latest `window * decim` raw samples (only if refine), and a hop counter."""

    def __init__(self, cfg=Config()):
        self.cfg = cfg
        self.dec = Decimator(fir_taps(cfg), cfg.decim, cfg.dtype)
        self.hp = Highpass(cfg.hp_hz, cfg.fs, cfg.dtype) if cfg.hp_hz else None
        self.win = np.zeros(cfg.window, dtype=cfg.dtype)
        self.raw = np.zeros(cfg.window * cfg.decim, dtype=cfg.dtype)
        self.filled = 0                                   # decimated samples received so far
        self.since_hop = 0
        self.n_in = 0                                     # raw samples received so far

    def push(self, block_int16):
        cfg = self.cfg
        x = np.asarray(block_int16, dtype=cfg.dtype) / cfg.dtype(FULL_SCALE)
        self.n_in += len(x)
        if cfg.refine and cfg.decim > 1:
            self.raw = np.concatenate((self.raw[len(x):], x))
        y = self.dec.process(x)
        if self.hp is not None:
            y = self.hp.process(y)
        self.win = np.concatenate((self.win[len(y):], y))
        self.filled += len(y)
        self.since_hop += len(y)
        results = []
        if self.filled >= cfg.window and self.since_hop >= cfg.hop:
            self.since_hop = 0                            # first result as soon as the window is full
            freq, est = detect(self.win, cfg, self.raw)
            results.append(Result(self.n_in / cfg.fs_in, freq, est.confidence,
                                  is_valid(est, cfg) and np.isfinite(freq), est.tau))
        return results


def track(samples_int16, cfg=Config()):
    """Run a whole recording through the tracker in BLOCK-sized pieces."""
    t = PitchTracker(cfg)
    out = []
    n = len(samples_int16) - len(samples_int16) % BLOCK
    for i in range(0, n, BLOCK):
        out.extend(t.push(samples_int16[i:i + BLOCK]))
    return out


# ------------------------------------------------------------------- cost

def cost_per_hop(cfg):
    """Multiply-adds per hop, split by stage (a starting point for the CPU budget on the target)."""
    taps = len(fir_taps(cfg)) if cfg.decim > 1 else 0
    fir = taps * cfg.hop                                  # one dot product per kept output
    w = cfg.window - cfg.tau_max - 1
    if cfg.method == "yin":
        det = w * (cfg.tau_max + 2)
    else:
        det = sum(cfg.window - t for t in range(cfg.tau_max + 2))
    ref = 0
    if cfg.refine and cfg.decim > 1:
        max_lag = cfg.tau_max * cfg.decim + cfg.decim + 2
        ref = (2 * cfg.decim + 3) * (cfg.window * cfg.decim - max_lag - 2)
    return {"fir": fir, "detector": det, "refine": ref, "total": fir + det + ref}


def ram_bytes(cfg, bytes_per_value=4):
    """Working RAM for the chain in C (float32), excluding the DMA buffer."""
    taps = len(fir_taps(cfg)) if cfg.decim > 1 else 0
    parts = {
        "fir_delay": (taps - 1 + BLOCK) * bytes_per_value if taps else 0,
        "window": cfg.window * bytes_per_value,
        "function": (cfg.tau_max + 2) * bytes_per_value,     # d (YIN keeps d' in place)
        "raw_window": cfg.window * cfg.decim * bytes_per_value if (cfg.refine and cfg.decim > 1) else 0,
    }
    parts["total"] = sum(parts.values())
    return parts


TRACKERS = {
    "yin": Config(method="yin"),
    "mpm": Config(method="mpm", interp="nsdf", hp_hz=20.0),   # MPM needs the rumble removed
}
