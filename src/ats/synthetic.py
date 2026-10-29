"""Synthetic recordings for testing the plumbing ONLY.

These signals are crude physical caricatures of the seven activities. They let us check that
every layer runs and that answers are well formed before the real data is available.
Never report accuracy measured on synthetic data as a result.
"""
import numpy as np

from . import config as C
from .preprocess import Recording, build_recording

# Gravity direction (device frame) per posture family, for a phone carried in a trouser pocket.
UPRIGHT = np.array([0.0, -1.0, 0.0])
SEATED = np.array([0.1, -0.3, -0.95])
LYING = np.array([-0.95, 0.05, -0.3])


def _unit(v):
    return v / np.linalg.norm(v)


def _segment(activity: str, n: int, fs: float, rng) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(n) / fs
    jitter = rng.normal(0, 0.08, 3)
    gyr = rng.normal(0, 0.01, (n, 3))
    if activity == "lying_down":
        g, acc_dyn = _unit(LYING + jitter), rng.normal(0, 0.004, (n, 3))
    elif activity == "sitting":
        g, acc_dyn = _unit(SEATED + jitter), rng.normal(0, 0.006, (n, 3))
    elif activity == "standing_in_place":
        g, acc_dyn = _unit(UPRIGHT + jitter), rng.normal(0, 0.012, (n, 3))
        gyr += rng.normal(0, 0.03, (n, 3))
    elif activity == "standing_moving":
        g = _unit(UPRIGHT + jitter)
        drift = np.cumsum(rng.normal(0, 0.02, (n, 3)), 0)
        acc_dyn = 0.06 * np.tanh(drift) + rng.normal(0, 0.03, (n, 3))
        gyr += 0.25 * np.tanh(np.cumsum(rng.normal(0, 0.05, (n, 3)), 0))
    elif activity in ("walking", "running"):
        g = _unit(UPRIGHT + jitter)
        f = rng.uniform(1.7, 2.1) if activity == "walking" else rng.uniform(2.6, 3.0)
        amp = 0.3 if activity == "walking" else 1.0
        ph = 2 * np.pi * f * t
        vert = amp * (np.maximum(np.sin(ph), 0) ** 3 * 2 - 0.4)        # heel-strike shaped peaks
        acc_dyn = np.column_stack([0.3 * amp * np.sin(ph / 2), vert, 0.2 * amp * np.cos(ph)])
        acc_dyn += rng.normal(0, 0.03 * (1 + amp), (n, 3))
        gyr += np.column_stack([1.2 * amp * np.sin(ph / 2), 0.4 * amp * np.cos(ph), 0.8 * amp * np.sin(ph / 2 + 1)])
    elif activity == "bicycling":
        g = _unit(SEATED + np.array([0, 0.4, 0.3]) + jitter)
        f = rng.uniform(1.0, 1.4)
        ph = 2 * np.pi * f * t
        acc_dyn = np.column_stack([0.12 * np.sin(ph), 0.08 * np.cos(ph), 0.1 * np.sin(ph + 0.5)])
        acc_dyn += rng.normal(0, 0.02, (n, 3)) + 0.04 * rng.standard_normal((n, 1)) * (rng.random((n, 1)) < 0.02)
        gyr += np.column_stack([1.5 * np.sin(ph), 0.3 * np.sin(ph + 1), 0.6 * np.cos(ph)])
    else:
        raise ValueError(activity)
    return np.tile(g, (n, 1)) + acc_dyn, gyr


def make_recording(script: list[tuple[str, float]], seed: int = 0, fs_raw: float = 40.0,
                   bursty: bool = False, drop_gyro: bool = False):
    """Build a recording from a script of (activity, seconds).

    bursty=True mimics ExtraSensory: only the first 20 s of each minute is recorded.
    Returns (Recording, ground-truth intervals [(start, end, activity)]).
    """
    rng = np.random.default_rng(seed)
    t0 = 1_700_000_000.0
    ts, accs, gyrs, gt = [], [], [], []
    cur = 0.0
    for activity, secs in script:
        n = int(secs * fs_raw)
        acc, gyr = _segment(activity, n, fs_raw, rng)
        t = cur + np.arange(n) / fs_raw + rng.normal(0, 0.002, n)   # uneven sampling
        keep = np.ones(n, bool)
        if bursty:
            keep &= (t % 60) < 20
        keep &= rng.random(n) > 0.01                                  # random sample loss
        ts.append(t[keep]); accs.append(acc[keep]); gyrs.append(gyr[keep])
        if bursty:
            m = cur
            while m < cur + secs:
                s, e = max(m, cur), min(m - (m % 60) + 20, cur + secs)
                if e > s:
                    gt.append((s, e, activity))
                m = m - (m % 60) + 60
        else:
            gt.append((cur, cur + secs, activity))
        cur += secs
    t = np.concatenate(ts) + t0
    acc_chunks = [(t, np.concatenate(accs))]
    gyro_chunks = [] if drop_gyro else [(t, np.concatenate(gyrs))]
    if bursty:   # split into per-minute bursts so the gaps are real gaps
        mins = np.floor((t - t0) / 60)
        acc_chunks = [(t[mins == m], acc_chunks[0][1][mins == m]) for m in np.unique(mins)]
        gyro_chunks = [] if drop_gyro else [(t[mins == m], gyro_chunks[0][1][mins == m]) for m in np.unique(mins)]
    return build_recording(acc_chunks, gyro_chunks, t0=t0), gt


def random_script(rng, total_s: float = 3600, min_s: float = 60, max_s: float = 900):
    """A plausible random day fragment, weighted toward sedentary behaviour like the real data."""
    p = np.array([0.22, 0.3, 0.08, 0.12, 0.16, 0.05, 0.07])
    script, cur, prev = [], 0.0, None
    while cur < total_s:
        a = prev
        while a == prev:
            a = C.CLASSES[rng.choice(len(C.CLASSES), p=p)]
        d = float(min(rng.uniform(min_s, max_s), total_s - cur))
        script.append((a, d))
        cur += d
        prev = a
    return script
