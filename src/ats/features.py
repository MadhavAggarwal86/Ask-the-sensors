"""Per-window engineered features.

Features are named and physically interpretable on purpose: the same numbers the classifier
uses are quoted back in the explanations, which is what ties each answer to the signal.
"""
import numpy as np

from . import config as C

AXES = "xyz"


def _dominant(sig: np.ndarray, fmin: float = 0.5, fmax: float = 6.0):
    """Dominant frequency (Hz) and periodicity (share of band power at the peak) per row."""
    sig = sig - sig.mean(1, keepdims=True)
    spec = np.abs(np.fft.rfft(sig, axis=1)) ** 2
    freqs = np.fft.rfftfreq(sig.shape[1], 1 / C.FS)
    band = (freqs >= fmin) & (freqs <= fmax)
    bp = spec[:, band]
    total = bp.sum(1)
    peak = bp.argmax(1)
    # Count the peak bin and its two neighbours, since leakage spreads one tone over ~3 bins.
    padded = np.pad(bp, ((0, 0), (1, 1)))
    rows = np.arange(len(bp))
    peak_pow = padded[rows, peak] + padded[rows, peak + 1] + padded[rows, peak + 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        periodicity = np.where(total > 1e-12, peak_pow / total, 0.0)
    return freqs[band][peak], periodicity, spec, freqs


def _autocorr_peak(sig: np.ndarray, lag_min_s: float = 0.25, lag_max_s: float = 1.5) -> np.ndarray:
    sig = sig - sig.mean(1, keepdims=True)
    n = sig.shape[1]
    f = np.fft.rfft(sig, 2 * n, axis=1)
    ac = np.fft.irfft(f * np.conj(f), axis=1)[:, :n]
    with np.errstate(invalid="ignore", divide="ignore"):
        ac = ac / ac[:, :1]
    lo, hi = int(lag_min_s * C.FS), min(n - 1, int(lag_max_s * C.FS))
    return np.nan_to_num(ac[:, lo:hi].max(1))


def _corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))


def extract(windows: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """windows: (n, win_len, 6) -> (n, n_features) feature matrix and feature names."""
    f: dict[str, np.ndarray] = {}
    acc, gyr = windows[..., :3], windows[..., 3:]
    for i, ch in enumerate(C.CHANNELS):
        f[f"{ch}_mean"] = windows[..., i].mean(1)
        f[f"{ch}_std"] = windows[..., i].std(1)

    # Orientation: angle between gravity (mean acc) and each phone axis. Separates postures.
    g = acc.mean(1)
    gnorm = np.linalg.norm(g, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosang = np.clip(g / gnorm, -1, 1)
    for i, a in enumerate(AXES):
        f[f"tilt_{a}_deg"] = np.degrees(np.arccos(cosang[:, i]))

    amag = np.linalg.norm(acc, axis=2)
    f["acc_mag_mean"] = amag.mean(1)
    f["acc_mag_std"] = amag.std(1)
    p5, p10, p90, p95 = np.percentile(amag, [5, 10, 90, 95], axis=1)
    f["acc_mag_p10"], f["acc_mag_p90"], f["acc_mag_range"] = p10, p90, p95 - p5
    f["acc_mag_max"] = amag.max(1)
    f["acc_jerk"] = np.abs(np.diff(amag, axis=1)).mean(1) * C.FS
    dom, per, spec, freqs = _dominant(amag)
    f["acc_dom_freq_hz"], f["acc_periodicity"] = dom, per
    f["acc_autocorr_peak"] = _autocorr_peak(amag)
    total = spec[:, freqs > 0.3].sum(1) + 1e-12
    f["acc_energy_low"] = spec[:, (freqs > 0.3) & (freqs <= 3)].sum(1) / total     # gait / pedalling
    f["acc_energy_high"] = spec[:, freqs > 3].sum(1) / total                       # impacts
    f["acc_corr_xy"] = _corr(acc[..., 0], acc[..., 1])
    f["acc_corr_xz"] = _corr(acc[..., 0], acc[..., 2])
    f["acc_corr_yz"] = _corr(acc[..., 1], acc[..., 2])

    gmag = np.linalg.norm(gyr, axis=2)  # NaN when there is no gyroscope; the model handles NaN
    f["gyro_mag_mean"] = gmag.mean(1)
    f["gyro_mag_std"] = gmag.std(1)
    f["gyro_mag_max"] = gmag.max(1)
    has_g = np.isfinite(gmag).all(1)
    gdom = np.full(len(gmag), np.nan)
    gper = np.full(len(gmag), np.nan)
    if has_g.any():
        gdom[has_g], gper[has_g], _, _ = _dominant(gmag[has_g])
    f["gyro_dom_freq_hz"], f["gyro_periodicity"] = gdom, gper

    names = list(f)
    return np.column_stack([f[n] for n in names]).astype(np.float32), names
