"""Loading recordings: ExtraSensory raw files and generic CSV files."""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .preprocess import Recording, build_recording

# ---------------------------------------------------------------------------------------------
# Generic CSV (the format we expect at evaluation time; also what synthetic.py writes)
# ---------------------------------------------------------------------------------------------


def _pick(cols: list[str], *patterns: str) -> str | None:
    for p in patterns:
        for c in cols:
            if re.fullmatch(p, c.lower().replace(" ", "_")):
                return c
    return None


def load_csv(path: Path) -> Recording:
    """CSV with a time column and acc/gyro x,y,z columns. Column names are matched loosely.

    Time may be seconds from start, epoch seconds or epoch milliseconds.
    Gyro columns are optional.
    """
    df = pd.read_csv(path)
    cols = list(df.columns)
    tcol = _pick(cols, r"t", r"time.*", r"timestamp.*", r"sec.*")
    if tcol is None:
        raise ValueError(f"no time column in {path}")
    t = df[tcol].to_numpy(float)
    if np.nanmedian(np.diff(t)) > 1:      # milliseconds
        t = t / 1000.0
    t0 = float(t[0]) if t[0] > 1e8 else None
    acc = [_pick(cols, rf"(raw_)?acc(el(erometer)?)?_?{a}") for a in "xyz"]
    gyr = [_pick(cols, rf"(proc_)?gyr(o(scope)?)?_?{a}") for a in "xyz"]
    if None in acc:
        raise ValueError(f"accelerometer x/y/z columns not found in {path}: {cols}")
    acc_chunks = [(t, df[acc].to_numpy(float))]
    gyro_chunks = [(t, df[gyr].to_numpy(float))] if None not in gyr else []
    return build_recording(acc_chunks, gyro_chunks, t0=t0 if t0 is not None else float(t[0]))


def save_csv(rec: Recording, path: Path) -> None:
    df = pd.DataFrame(rec.x, columns=C.CHANNELS)
    df.insert(0, "t", rec.t + (rec.t0 or 0.0))
    df.to_csv(path, index=False, float_format="%.5f")


# ---------------------------------------------------------------------------------------------
# ExtraSensory
# Layout after scripts/download_data.py:
#   data/raw/labels/<uuid>.original_labels.csv.gz
#   data/raw/raw_acc/<uuid>/<timestamp>.m_raw_acc.dat
#   data/raw/proc_gyro/<uuid>/<timestamp>.m_proc_gyro.dat
# Each .dat holds ~20 s of samples (~40 Hz) recorded at the start of that minute.
# ---------------------------------------------------------------------------------------------


def es_users(raw_dir: Path = C.RAW_DIR) -> list[str]:
    return sorted(p.name.split(".")[0] for p in (raw_dir / "labels").glob("*.original_labels.csv.gz"))


def es_sensor_ok(uuid: str, raw_dir: Path = C.RAW_DIR, n_files: int = 40, max_spread_g: float = 0.5) -> bool:
    """Reject phones whose accelerometer is miscalibrated.

    The median |acc| of a burst is ~1 g whatever the user does, so its 10-90th percentile spread
    across bursts should be small. Healthy phones in ExtraSensory stay under ~0.2 g; one phone
    swings between 0.03 and 1.4 g and would corrupt every orientation and intensity feature.
    """
    files = sorted((raw_dir / "raw_acc" / uuid).glob("*.dat"))
    if not files:
        return False
    meds = []
    for i in np.linspace(0, len(files) - 1, min(n_files, len(files))).astype(int):
        _, v = _read_dat(files[i], 0)
        if len(v):
            meds.append(np.median(np.linalg.norm(v, axis=1)))
    meds = np.array(meds)
    if len(meds) == 0:
        return False
    g = meds / (9.80665 if np.median(meds) > 4 else 1.0)
    return float(np.percentile(g, 90) - np.percentile(g, 10)) <= max_spread_g


def es_labels(uuid: str, raw_dir: Path = C.RAW_DIR) -> pd.DataFrame:
    """Per-minute labels: columns timestamp, label (one of CLASSES, or None if absent/ambiguous)."""
    df = pd.read_csv(raw_dir / "labels" / f"{uuid}.original_labels.csv.gz")
    lab = df[list(C.LABEL_COLUMNS)].fillna(0).to_numpy()
    single = lab.sum(1) == 1
    names = np.array(list(C.LABEL_COLUMNS.values()))
    out = pd.DataFrame({"timestamp": df["timestamp"].astype(int)})
    out["label"] = np.where(single, names[lab.argmax(1)], None)
    return out


def _read_dat(path: Path, minute_ts: int) -> tuple[np.ndarray, np.ndarray]:
    """Read one burst file. Its clock varies by phone, so only relative timing is trusted and the
    burst is anchored at the minute timestamp from the filename."""
    try:
        a = np.loadtxt(path, ndmin=2)
    except (ValueError, OSError):
        return np.empty(0), np.empty((0, 3))
    if a.size == 0 or a.shape[1] < 3:
        return np.empty(0), np.empty((0, 3))
    if a.shape[1] >= 4:
        t = a[:, 0] - a[0, 0]
        d = np.median(np.diff(t)) if len(t) > 1 else 0.025
        for scale in (1.0, 1e-3, 1e-6, 1e-9):    # seconds, ms, us or ns
            if 0.002 <= d * scale <= 0.5:
                t = t * scale
                break
        # Some bursts contain a glitched sample whose clock jumps by days and back. Keep only
        # samples near the burst's median time, then anchor at the first kept sample.
        ok = np.abs(t - np.median(t)) <= 60
        t, a = t[ok] - t[ok].min(), a[ok]
        v = a[:, -3:]
    else:
        v = a
        t = np.arange(len(a)) / 40.0
    return minute_ts + t, v


def es_sessions(labels: pd.DataFrame, max_gap_s: int = 300, min_minutes: int = 10,
                max_minutes: int = 720) -> list[pd.DataFrame]:
    """Split a user's minutes into recordings: runs of minutes with no gap longer than max_gap_s,
    cut into pieces of at most max_minutes (some users have multi-day runs; a day-scale recording
    is the realistic query unit and keeps memory bounded)."""
    ts = labels["timestamp"].to_numpy()
    breaks = np.where(np.diff(ts) > max_gap_s)[0] + 1
    sessions = []
    for idx in np.split(np.arange(len(ts)), breaks):
        sessions += [labels.iloc[idx[i:i + max_minutes]] for i in range(0, len(idx), max_minutes)]
    return [s for s in sessions if len(s) >= min_minutes and s["label"].notna().any()]


def es_recording(uuid: str, minutes: pd.DataFrame, raw_dir: Path = C.RAW_DIR) -> Recording:
    acc, gyr = [], []
    for ts in minutes["timestamp"]:
        a = raw_dir / "raw_acc" / uuid / f"{ts}.m_raw_acc.dat"
        g = raw_dir / "proc_gyro" / uuid / f"{ts}.m_proc_gyro.dat"
        if a.exists():
            acc.append(_read_dat(a, ts))
        if g.exists():
            gyr.append(_read_dat(g, ts))
    return build_recording(acc, gyr, t0=float(minutes["timestamp"].iloc[0]))


def es_ground_truth(minutes: pd.DataFrame, t0: float, burst_s: float = 20.0) -> list[tuple[float, float, str]]:
    """Ground-truth intervals (start, end, label) for a session, in seconds from start."""
    return [(ts - t0, ts - t0 + burst_s, lab) for ts, lab in zip(minutes["timestamp"], minutes["label"]) if isinstance(lab, str)]
