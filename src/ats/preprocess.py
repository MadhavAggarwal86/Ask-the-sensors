"""Preprocessing layer: unit normalisation, resampling to 25 Hz, gap handling and windowing."""
from dataclasses import dataclass

import numpy as np

from . import config as C

G = 9.80665


@dataclass
class Recording:
    """A 25 Hz recording on a regular grid. Missing samples are NaN.

    t   : (n,) seconds from the start of the recording
    x   : (n, 6) acc_x, acc_y, acc_z [g], gyro_x, gyro_y, gyro_z [rad/s]
    t0  : absolute epoch time of t=0, if known (used to translate clock-time questions)
    """
    t: np.ndarray
    x: np.ndarray
    t0: float | None = None

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if len(self.t) else 0.0

    @property
    def has_gyro(self) -> bool:
        return bool(np.isfinite(self.x[:, 3:]).any())


def _resample_onto(grid: np.ndarray, t: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Linearly interpolate samples (t, v) onto grid, leaving NaN where the source has gaps."""
    out = np.full((len(grid), v.shape[1]), np.nan)
    ok = np.isfinite(t) & np.isfinite(v).all(1)
    t, v = t[ok], v[ok]
    if len(t) < 2:
        return out
    order = np.argsort(t)
    t, v = t[order], v[order]
    # Anti-aliasing: some Android phones sample at 100-200 Hz. Average over one output sample
    # period before interpolating, so high-frequency content does not fold into 0-12.5 Hz.
    k = int(round(1 / (C.FS * np.median(np.diff(t))))) if len(t) > 2 else 1
    if k >= 2:
        kern = np.ones(k) / k
        v = np.column_stack([np.convolve(v[:, c], kern, "same") for c in range(v.shape[1])])
    inside = (grid >= t[0]) & (grid <= t[-1])
    for c in range(v.shape[1]):
        out[inside, c] = np.interp(grid[inside], t, v[:, c])
    # Blank grid points whose nearest real sample is too far away.
    idx = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
    nearest = np.minimum(np.abs(grid - t[idx - 1]), np.abs(t[idx] - grid))
    out[nearest > C.MAX_INTERP_GAP_S] = np.nan
    return out


def build_recording(acc_chunks, gyro_chunks=(), t0: float | None = None) -> Recording:
    """Assemble a recording from lists of (t_abs, values[n,3]) chunks, one list per sensor.

    Chunks can be one continuous stream or many short bursts (ExtraSensory records ~20 s
    every minute). Everything is placed on a single 25 Hz grid starting at t0.
    """
    acc_chunks = [(np.asarray(t, float), np.asarray(v, float)) for t, v in acc_chunks]
    # Decide units once for the whole recording: a phone reports either g or m/s^2.
    to_g = 1 / G if np.median([np.nanmedian(np.linalg.norm(v, axis=1)) for _, v in acc_chunks if len(v)]
                              or [1.0]) > 4.0 else 1.0
    acc_chunks = [(t, v * to_g) for t, v in acc_chunks]
    gyro_chunks = [(np.asarray(t, float), np.asarray(v, float)) for t, v in gyro_chunks]
    starts = [t.min() for t, _ in acc_chunks if len(t)]
    ends = [t.max() for t, _ in acc_chunks if len(t)]
    if not starts:
        raise ValueError("recording has no accelerometer samples")
    if t0 is None:
        t0 = min(starts)
    n = int(np.floor((max(ends) - t0) * C.FS)) + 1
    grid_rel = np.arange(n) / C.FS
    x = np.full((n, 6), np.nan)
    for cols, chunks in ((slice(0, 3), acc_chunks), (slice(3, 6), gyro_chunks)):
        for t, v in chunks:
            if len(t) < 2:
                continue
            lo = max(0, int(np.ceil((t.min() - t0) * C.FS)))
            hi = min(n, int(np.floor((t.max() - t0) * C.FS)) + 1)
            if hi <= lo:
                continue
            x[lo:hi, cols] = _resample_onto(grid_rel[lo:hi], t - t0, v)
    return Recording(t=grid_rel, x=x, t0=t0)


def _fill_small_gaps(w: np.ndarray) -> np.ndarray:
    """Fill NaNs inside an accepted window by linear interpolation (per channel)."""
    w = w.copy()
    idx = np.arange(len(w))
    for c in range(w.shape[1]):
        bad = ~np.isfinite(w[:, c])
        if bad.all() or not bad.any():
            continue
        w[bad, c] = np.interp(idx[bad], idx[~bad], w[~bad, c])
    return w


def make_windows(rec: Recording, win_s: float = C.WIN_S, hop_s: float = C.HOP_S):
    """Slice the recording into overlapping windows.

    Returns (starts, windows) where starts are window start times in seconds and windows is
    (n_windows, win_len, 6). Windows with too little accelerometer data are dropped, so the
    returned starts can have gaps. Gyro channels stay NaN when the gyroscope is absent.
    """
    wl, hl = int(round(win_s * C.FS)), int(round(hop_s * C.FS))
    if len(rec.t) < wl:
        return np.empty(0), np.empty((0, wl, 6))
    valid_acc = np.isfinite(rec.x[:, :3]).all(1).astype(float)
    cov = np.convolve(valid_acc, np.ones(wl), "valid")[::hl] / wl
    first = np.arange(0, len(rec.t) - wl + 1, hl)
    keep = first[cov[: len(first)] >= C.MIN_COVERAGE]
    wins = np.stack([_fill_small_gaps(rec.x[i : i + wl]) for i in keep]) if len(keep) else np.empty((0, wl, 6))
    return rec.t[keep], wins


def degrade(rec: Recording, noise_g: float = 0.0, drop_frac: float = 0.0, fs_low: float | None = None,
            seed: int = 0) -> Recording:
    """Controlled input degradation for the robustness curve (Figure 5)."""
    rng = np.random.default_rng(seed)
    x = rec.x.copy()
    if noise_g:
        x[:, :3] += rng.normal(0, noise_g, x[:, :3].shape)
        x[:, 3:] += rng.normal(0, noise_g * 2, x[:, 3:].shape)
    keep = np.arange(len(x))
    if drop_frac:
        keep = keep[rng.random(len(keep)) >= drop_frac]
    if fs_low and fs_low < C.FS:
        sub = np.round(np.arange(0, len(x), C.FS / fs_low)).astype(int)
        keep = np.intersect1d(keep, sub[sub < len(x)])
    if len(keep) < len(x) and len(keep) > 1:
        # Re-run resampling from the surviving samples, as the loader would on a sparser stream.
        x = np.hstack([_resample_onto(rec.t, rec.t[keep], x[keep, s]) for s in (slice(0, 3), slice(3, 6))])
    return Recording(t=rec.t, x=x, t0=rec.t0)
