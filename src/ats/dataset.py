"""Iterate labelled recordings from either source, and turn them into window-level training data."""
import json
from pathlib import Path

import numpy as np

from . import config as C
from . import io
from .features import extract
from .preprocess import make_windows

SYNTH_DIR = C.DATA_DIR / "synthetic"
SPLIT_FILE = C.ROOT / "configs" / "split.json"


def users(source: str) -> list[str]:
    if source == "extrasensory":
        return [u for u in io.es_users() if io.es_sensor_ok(u)]
    if source == "synthetic":
        return sorted(p.name for p in SYNTH_DIR.iterdir() if p.is_dir())
    raise ValueError(source)


def recordings(source: str, user: str):
    """Yield (rec_id, Recording, ground-truth intervals) for one user."""
    if source == "extrasensory":
        for i, sess in enumerate(io.es_sessions(io.es_labels(user))):
            try:
                rec = io.es_recording(user, sess)
            except ValueError:      # no raw files for this session
                continue
            yield f"{user}_s{i:03d}", rec, io.es_ground_truth(sess, rec.t0)
    else:
        for csv in sorted((SYNTH_DIR / user).glob("*.csv")):
            gt = [tuple(g) for g in json.loads(csv.with_suffix(".gt.json").read_text())]
            yield csv.stem, io.load_csv(csv), gt


def held_out_recordings(source: str, n: int) -> list:
    """The first n recordings of the test users: the fixed QA evaluation set."""
    items = []
    for u in split_users(source)["test"]:
        items += list(recordings(source, u))
        if len(items) >= n:
            break
    return items[:n]


def window_labels(starts: np.ndarray, gt) -> np.ndarray:
    """Class index for each window whose centre lies inside a ground-truth interval, else -1."""
    y = np.full(len(starts), -1)
    centres = starts + C.WIN_S / 2
    for s, e, a in gt:
        y[(centres >= s) & (centres < e)] = C.CLASSES.index(a)
    return y


def process_user(source: str, user: str, out_dir: Path) -> int:
    Xs, ys, ss, rids = [], [], [], []
    names = None
    for rid, rec, gt in recordings(source, user):
        starts, wins = make_windows(rec)
        if not len(wins):
            continue
        X, names = extract(wins)
        y = window_labels(starts, gt)
        keep = y >= 0
        Xs.append(X[keep]); ys.append(y[keep]); ss.append(starts[keep]); rids += [rid] * int(keep.sum())
    if not Xs:
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{user}.npz", X=np.concatenate(Xs), y=np.concatenate(ys),
                        starts=np.concatenate(ss), rec=np.array(rids), names=np.array(names))
    return int(sum(len(y) for y in ys))


def load_processed(source: str, user_list: list[str]):
    Xs, ys, gs = [], [], []
    names = None
    for u in user_list:
        p = C.PROCESSED_DIR / source / f"{u}.npz"
        if not p.exists():
            continue
        d = np.load(p, allow_pickle=False)
        Xs.append(d["X"]); ys.append(d["y"]); gs += [u] * len(d["y"]); names = list(d["names"])
    return np.concatenate(Xs), np.concatenate(ys), np.array(gs), names


def training_set(X, y, groups, names, rng, cap: int = 3000, motion_min_std: float | None = 0.01,
                 gyro_dropout: float = 0.2):
    """The exact training recipe, shared by train.py and tune_smoothing.py. Returns (X, y)."""
    if motion_min_std is not None:
        # In-the-wild labels cover the whole minute, but the phone is often off the body during
        # the recorded burst. A walking/running/cycling window with no motion cannot show that
        # activity, so it is dropped from TRAINING only; test labels are never touched.
        std = X[:, names.index("acc_mag_std")]
        motion = np.isin(y, [C.CLASSES.index(a) for a in ("walking", "running", "bicycling")])
        drop = motion & (std < motion_min_std)
        print(f"label cleaning: dropped {drop.sum()} motionless motion-class windows "
              f"({drop.sum() / max(motion.sum(), 1):.0%} of motion-class training windows)")
        X, y, groups = X[~drop], y[~drop], groups[~drop]
    # Limit windows per (user, class) so a few heavy sitters do not dominate training.
    keep = []
    for g in np.unique(groups):
        for c in np.unique(y[groups == g]):
            idx = np.where((groups == g) & (y == c))[0]
            keep.append(rng.choice(idx, min(cap, len(idx)), replace=False))
    keep = np.sort(np.concatenate(keep))
    print(f"train windows {len(keep)} (of {len(y)})")
    Xfit = X[keep].copy()
    # Gyro dropout: blank gyro features on a share of windows so the model also learns an
    # accelerometer-only path for phones without a gyroscope.
    gyro_cols = [i for i, n in enumerate(names) if n.startswith("gyro")]
    Xfit[np.ix_(rng.random(len(Xfit)) < gyro_dropout, gyro_cols)] = np.nan
    return Xfit, y[keep]


def split_users(source: str, test_frac: float = 0.2, seed: int = 0) -> dict:
    """Fixed leave-users-out split, saved so every script and every member uses the same one."""
    splits = json.loads(SPLIT_FILE.read_text()) if SPLIT_FILE.exists() else {}
    if source not in splits:
        us = users(source)
        rng = np.random.default_rng(seed)
        test = sorted(rng.choice(us, max(1, int(round(test_frac * len(us)))), replace=False).tolist())
        splits[source] = {"train": [u for u in us if u not in test], "test": test}
        SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
        SPLIT_FILE.write_text(json.dumps(splits, indent=2))
    return splits[source]
