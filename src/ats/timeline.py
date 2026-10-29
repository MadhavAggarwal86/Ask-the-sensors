"""Aggregation layer: turn per-window predictions into activity intervals and query them.

The Timeline is the single source of truth for every temporal answer. The QA layer never
invents a number; it only reads durations, counts and onsets off this structure.
"""
import warnings
from dataclasses import dataclass, field

import numpy as np

from . import config as C


@dataclass
class Interval:
    start: float
    end: float
    activity: str
    confidence: float = 1.0
    idx: np.ndarray = field(default_factory=lambda: np.empty(0, int))  # window indices

    @property
    def duration(self) -> float:
        return self.end - self.start


def _segments(starts: np.ndarray) -> list[np.ndarray]:
    """Split window indices wherever the gap between windows is too long to bridge."""
    if len(starts) == 0:
        return []
    breaks = np.where(np.diff(starts) > C.MERGE_GAP_S + C.HOP_S)[0] + 1
    return np.split(np.arange(len(starts)), breaks)


def smooth_probs(starts: np.ndarray, probs: np.ndarray, k: int = C.SMOOTH_WINDOWS) -> np.ndarray:
    """Centred moving average of class probabilities, never crossing a data gap."""
    out = probs.copy()
    if k <= 1:
        return out
    kern, off = np.ones(k), (k - 1) // 2

    def centred(x):  # "same"-length convolution that also works when len(x) < k
        return np.convolve(x, kern, "full")[off:off + len(x)]

    for seg in _segments(starts):
        norm = centred(np.ones(len(seg)))
        for c in range(probs.shape[1]):
            out[seg, c] = centred(probs[seg, c]) / norm
    return out


def viterbi(starts: np.ndarray, probs: np.ndarray, tau_s: float = C.HMM_TAU_S,
            log_prior: np.ndarray | None = None) -> np.ndarray:
    """Most likely class sequence under a sticky hidden Markov model, decoded per data segment.

    Emissions are the classifier's log-probabilities. The classifier is trained with balanced
    class weights, so its outputs act like scaled likelihoods and a log class prior can be added
    back. Between windows dt seconds apart the activity persists with probability exp(-dt / tau_s),
    otherwise it switches to one of the other classes with equal chance. tau_s is therefore the
    expected bout length, and a longer data gap makes a change more plausible.
    """
    k = probs.shape[1]
    emit = np.log(np.clip(probs, 1e-9, 1.0))
    if log_prior is not None:
        emit = emit + log_prior
    path = np.empty(len(starts), int)
    for seg in _segments(starts):
        p_switch = -np.expm1(-np.diff(starts[seg]) / tau_s)
        stay, switch = np.log1p(-p_switch), np.log(p_switch / (k - 1))
        score = emit[seg[0]].copy()
        back = np.empty((len(seg), k), int)
        for t in range(1, len(seg)):
            # Best way into each class: stay in it, or jump from the overall best class. Staying
            # beats switching from itself, so the best class is always a valid jump origin.
            best = score.argmax()
            kept, moved = score + stay[t - 1], score[best] + switch[t - 1]
            back[t] = np.where(kept >= moved, np.arange(k), best)
            score = np.maximum(kept, moved) + emit[seg[t]]
        state = score.argmax()
        for t in range(len(seg) - 1, -1, -1):
            path[seg[t]] = state
            state = back[t, state]
    return path


def decode(starts: np.ndarray, probs: np.ndarray, method: str = C.SMOOTHING) -> np.ndarray:
    """Class index per window: "hmm" (Viterbi) or "moving_average" (argmax of smoothed probs)."""
    if method == "moving_average":
        return smooth_probs(starts, probs).argmax(1)
    if method == "hmm":
        prior = C.HMM_PRIOR_WEIGHT * np.log([C.CLASS_PRIOR[c] for c in C.CLASSES])
        emit = smooth_probs(starts, probs, C.HMM_PRESMOOTH) if C.HMM_PRESMOOTH > 1 else probs
        return viterbi(starts, emit, C.HMM_TAU_S, prior)
    raise ValueError(method)


def intervals_from_labels(starts: np.ndarray, labels: list[str], conf: np.ndarray | None = None,
                          win_s: float = C.WIN_S) -> list[Interval]:
    """Merge consecutive same-label windows into intervals, bridging short gaps."""
    if conf is None:
        conf = np.ones(len(starts))
    out: list[Interval] = []
    cur: list[int] = []
    for i in range(len(starts)):
        if cur and (labels[i] != labels[cur[-1]] or starts[i] - (starts[cur[-1]] + win_s) > C.MERGE_GAP_S):
            out.append(_close(cur, starts, labels, conf, win_s))
            cur = []
        cur.append(i)
    if cur:
        out.append(_close(cur, starts, labels, conf, win_s))
    return _absorb_flicker(out)


def _close(cur, starts, labels, conf, win_s) -> Interval:
    idx = np.array(cur)
    return Interval(float(starts[idx[0]]), float(starts[idx[-1]] + win_s), labels[idx[0]],
                    float(conf[idx].mean()), idx)


def _absorb_flicker(ivs: list[Interval]) -> list[Interval]:
    """Drop a too-short interval sandwiched between two touching intervals of the same activity."""
    changed = True
    while changed:
        changed = False
        for i in range(1, len(ivs) - 1):
            a, b, c = ivs[i - 1], ivs[i], ivs[i + 1]
            if (b.duration < C.MIN_INTERVAL_S and a.activity == c.activity
                    and c.start - a.end <= C.MERGE_GAP_S + b.duration):
                ivs[i - 1:i + 2] = [Interval(a.start, c.end, a.activity,
                                             (a.confidence + c.confidence) / 2,
                                             np.concatenate([a.idx, b.idx, c.idx]))]
                changed = True
                break
    return ivs


@dataclass
class Timeline:
    intervals: list[Interval]
    starts: np.ndarray = field(default_factory=lambda: np.empty(0))        # window starts
    features: np.ndarray | None = None                                      # (n_windows, n_feat)
    feature_names: list[str] = field(default_factory=list)
    probs: np.ndarray | None = None
    has_gyro: bool = True
    duration: float = 0.0

    @classmethod
    def from_predictions(cls, starts, probs, features, feature_names, has_gyro=True, duration=0.0,
                         method: str = C.SMOOTHING):
        cls_idx = decode(starts, probs, method)
        sp = smooth_probs(starts, probs)
        labels = [C.CLASSES[i] for i in cls_idx]
        ivs = intervals_from_labels(starts, labels, sp[np.arange(len(cls_idx)), cls_idx])
        return cls(ivs, starts, features, feature_names, probs, has_gyro, duration)

    # --- queries -------------------------------------------------------------------------------
    def select(self, activities, t_range=None, min_dur: float = 0.0) -> list[Interval]:
        """Intervals of any of the given activities, clipped to t_range."""
        acts = {activities} if isinstance(activities, str) else set(activities)
        out = []
        for iv in self.intervals:
            if iv.activity not in acts:
                continue
            s, e = iv.start, iv.end
            if t_range is not None:
                s, e = max(s, t_range[0]), min(e, t_range[1])
                if e <= s:
                    continue
            if e - s >= min_dur:
                keep = iv.idx if t_range is None else iv.idx[
                    (self.starts[iv.idx] + C.WIN_S > s) & (self.starts[iv.idx] < e)] if len(iv.idx) else iv.idx
                out.append(Interval(s, e, iv.activity, iv.confidence, keep))
        return out

    def total(self, activities, t_range=None) -> float:
        return sum(iv.duration for iv in self.select(activities, t_range))

    def bouts(self, activities, t_range=None) -> list[Interval]:
        """Occurrences: runs of the activity group, merged across adjacent group members."""
        ivs = sorted(self.select(activities, t_range), key=lambda iv: iv.start)
        merged: list[Interval] = []
        for iv in ivs:
            if merged and iv.start - merged[-1].end <= C.MERGE_GAP_S:
                m = merged[-1]
                merged[-1] = Interval(m.start, max(m.end, iv.end), m.activity if m.duration >= iv.duration
                                      else iv.activity, max(m.confidence, iv.confidence),
                                      np.concatenate([m.idx, iv.idx]))
            else:
                merged.append(iv)
        return [b for b in merged if b.duration >= C.MIN_BOUT_S]

    def breakdown(self, t_range=None) -> dict[str, float]:
        return {a: self.total(a, t_range) for a in C.CLASSES}

    def feature_stats(self, ivs: list[Interval]) -> dict[str, float]:
        """Mean feature values over the windows supporting the given intervals."""
        if self.features is None or not ivs:
            return {}
        idx = np.concatenate([iv.idx for iv in ivs]) if any(len(iv.idx) for iv in ivs) else np.empty(0, int)
        if len(idx) == 0:
            return {}
        with warnings.catch_warnings():   # all-NaN gyro columns when there is no gyroscope
            warnings.simplefilter("ignore", RuntimeWarning)
            vals = np.nanmean(self.features[idx], axis=0)
        return dict(zip(self.feature_names, map(float, vals)))
