"""
Pitch detectors.

Every detector has the same shape:

    def detector(window: np.ndarray, fs: int) -> float | None

    window  one analysis window of samples, float64, int16 scale (+-32768)
    fs      sample rate in Hz
    returns the detected fundamental in Hz, or None for "no answer"
            (e.g. too few crossings, signal too quiet)

Register each detector in DETECTORS at the bottom, then run it with
    python tools/evaluate.py data/captures/*.wav --detector zc
"""
import numpy as np

HYSTERESIS_FRACTION = 0.3   # hysteresis threshold as a fraction of the window's RMS


def _hysteresis_crossings(x, threshold):
    """Times (in samples, sub-sample precision) where x rises through +threshold,
    counting each one only after x has been below -threshold since the previous one."""
    times = []
    armed = False                      # True once the signal has gone below -threshold
    for i in range(len(x) - 1):
        if x[i] < -threshold:
            armed = True
        elif armed and x[i] < threshold <= x[i + 1]:
            # interpolate where the line from x[i] to x[i+1] reaches +threshold
            times.append(i + (threshold - x[i]) / (x[i + 1] - x[i]))
            armed = False              # wait for the next dip below -threshold
    return np.array(times)


def _crossings(window, hysteresis):
    """Upward crossing times: plain zero crossings, or Schmitt-trigger crossings."""
    x = window - np.mean(window)     # remove DC so "zero" is the middle of the waveform
    if hysteresis:
        rms = np.sqrt(np.mean(x ** 2))
        if rms == 0:
            return np.array([])
        return _hysteresis_crossings(x, HYSTERESIS_FRACTION * rms)
    # x[i] < 0 and x[i+1] >= 0, then where the line from x[i] to x[i+1] hits zero
    i = np.nonzero((x[:-1] < 0) & (x[1:] >= 0))[0]
    return i + x[i] / (x[i] - x[i + 1])


def _frequency(t, fs, use_median):
    """Frequency from crossing times: first-to-last average, or median of single periods."""
    if len(t) < 2:
        return None
    if use_median:
        return fs / np.median(np.diff(t))          # np.diff: t[1]-t[0], t[2]-t[1], ...
    return (len(t) - 1) * fs / (t[-1] - t[0])      # N crossings enclose N-1 periods


def zero_crossing(window, fs):
    return _frequency(_crossings(window, hysteresis=False), fs, use_median=False)


def zc_hysteresis(window, fs):
    return _frequency(_crossings(window, hysteresis=True), fs, use_median=False)


def zc_median(window, fs):
    return _frequency(_crossings(window, hysteresis=False), fs, use_median=True)


def zc_hysteresis_median(window, fs):
    return _frequency(_crossings(window, hysteresis=True), fs, use_median=True)


# ------------------------------------------------------------- FFT-based

FFT_MIN_HZ = 30.0       # ignore bins below this (room rumble)
HPS_HARMONICS = 5       # harmonic product spectrum: multiply the spectrum by itself squeezed 2x..5x
HPS_MAX_HZ = 1500.0     # highest fundamental HPS searches for (top of the tuner's range)
HPS_FLOOR_DB = 60.0     # floor magnitudes this far below the strongest bin
HPS_TIE = 0.5           # candidates within this (natural-log sum) of the best count as a tie


def _log_spectrum(window, fs):
    """Natural log of the Hann-windowed magnitude spectrum, and the bin width in Hz."""
    x = window - np.mean(window)
    magnitude = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return np.log(magnitude + 1e-12), fs / len(x)


def _parabolic_offset(log_values, k):
    """Sub-bin position of the peak at k: vertex of the parabola through k-1, k, k+1.

    Works on log magnitudes, where a Hann-windowed peak is close to a parabola.
    Returns an offset in bins, between -0.5 and +0.5.
    """
    if k <= 0 or k >= len(log_values) - 1:
        return 0.0
    a, b, c = log_values[k - 1], log_values[k], log_values[k + 1]
    denom = a - 2.0 * b + c
    return 0.5 * (a - c) / denom if denom != 0 else 0.0


def _strongest_bin(log_mag, bin_hz):
    k_min = int(np.ceil(FFT_MIN_HZ / bin_hz))
    if k_min >= len(log_mag):
        return None
    return k_min + int(np.argmax(log_mag[k_min:]))


def fft_peak(window, fs):
    """Frequency of the strongest bin: resolution is the bin width, fs / N."""
    log_mag, bin_hz = _log_spectrum(window, fs)
    k = _strongest_bin(log_mag, bin_hz)
    return None if k is None else k * bin_hz


def fft_peak_interp(window, fs):
    """Strongest bin refined with parabolic interpolation between its neighbours."""
    log_mag, bin_hz = _log_spectrum(window, fs)
    k = _strongest_bin(log_mag, bin_hz)
    return None if k is None else (k + _parabolic_offset(log_mag, k)) * bin_hz


def fft_hps(window, fs):
    """Harmonic product spectrum.

    For every candidate fundamental bin k, multiply the magnitudes at k, 2k, 3k, ...
    (in log terms: add them). Only the true fundamental has energy at all of
    those places, so its product wins even when the fundamental itself is weak.

    Known weakness: for a nearly pure tone, f0/2, f0/3 ... also collect the one
    real peak, so they tie with f0. Two guards: magnitudes are floored at
    HPS_FLOOR_DB below the strongest bin (noise bins all count the same), and
    among candidates within HPS_TIE of the best, the highest frequency wins.
    """
    log_mag, bin_hz = _log_spectrum(window, fs)
    log_mag = np.maximum(log_mag, log_mag.max() - HPS_FLOOR_DB / 20.0 * np.log(10.0))
    k_min = int(np.ceil(FFT_MIN_HZ / bin_hz))
    k_max = min(len(log_mag) // HPS_HARMONICS, int(HPS_MAX_HZ / bin_hz) + 2)
    if k_max <= k_min + 1:
        return None
    hps = np.zeros(k_max)
    for h in range(1, HPS_HARMONICS + 1):
        hps += log_mag[0:k_max * h:h]          # magnitude at h*k for every k < k_max
    search = hps[k_min:k_max]
    near_best = np.nonzero(search >= search.max() - HPS_TIE)[0]
    c = k_min + int(near_best[-1])             # highest-frequency candidate
    lo, hi = max(k_min, c - 2), min(k_max, c + 3)
    k = lo + int(np.argmax(hps[lo:hi]))        # its exact peak bin
    return (k + _parabolic_offset(hps, k)) * bin_hz


DETECTORS = {
    "zc": zero_crossing,
    "zc_hyst": zc_hysteresis,
    "zc_median": zc_median,
    "zc_hyst_median": zc_hysteresis_median,
    "fft": fft_peak,
    "fft_interp": fft_peak_interp,
    "fft_hps": fft_hps,
}
