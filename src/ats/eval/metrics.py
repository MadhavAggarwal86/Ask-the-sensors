"""Correctness rules per question type (stated in the report and in the Figure 1 caption).

- identification, verification, comparison, prolonged, open_world: exact match of the value
- duration: |error| <= max(DUR_ABS_TOL_S, DUR_REL_TOL * reference)
- count: |error| <= COUNT_TOL
- grounding (onset): yes/no must match, and when yes, IoU(cited, reference) >= IOU_THRESHOLD
- "grounded and correct" (all types with evidence): answer correct AND cited-interval IoU >=
  IOU_THRESHOLD AND modality matches the reference
"""
import numpy as np

DUR_ABS_TOL_S = 30.0
DUR_REL_TOL = 0.10
COUNT_TOL = 1
IOU_THRESHOLD = 0.5
EXACT_TYPES = {"identification", "verification", "comparison", "prolonged", "open_world"}


def _mask(intervals, horizon: float, res: float = 1.0) -> np.ndarray:
    m = np.zeros(int(np.ceil(horizon / res)) + 1, bool)
    for s, e in intervals:
        m[int(max(0, s) // res): int(np.ceil(e / res))] = True
    return m


def interval_iou(pred, ref) -> float:
    """IoU between two sets of intervals (seconds), at 1 s resolution."""
    if not pred and not ref:
        return 1.0
    horizon = max([e for _, e in list(pred) + list(ref)] + [1.0])
    a, b = _mask(pred, horizon), _mask(ref, horizon)
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 0.0


def overlap_fraction(pred, ref) -> float:
    """Share of the cited time that falls inside the reference intervals (grounding precision)."""
    if not pred:
        return np.nan
    horizon = max([e for _, e in list(pred) + list(ref)] + [1.0])
    a, b = _mask(pred, horizon), _mask(ref, horizon)
    return float((a & b).sum() / a.sum()) if a.sum() else np.nan


def _norm(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, str):
        return v.lower().replace("likely ", "")
    return v


def answer_correct(qtype: str, pred, ref, dur_rel_tol=DUR_REL_TOL, iou_thr=IOU_THRESHOLD) -> bool:
    pv, rv = pred.value, ref.value
    if qtype in EXACT_TYPES:
        return _norm(pv) == _norm(rv)
    if qtype == "duration":
        if pv is None or rv is None:
            return pv == rv
        return abs(pv - rv) <= max(DUR_ABS_TOL_S, dur_rel_tol * rv)
    if qtype == "count":
        return pv is not None and rv is not None and abs(pv - rv) <= COUNT_TOL
    if qtype == "grounding":
        if (pv is None) != (rv is None):
            return False
        return rv is None or interval_iou(pred.intervals, ref.intervals) >= iou_thr
    raise ValueError(qtype)


def grounded_correct(qtype: str, pred, ref, iou_thr=IOU_THRESHOLD) -> bool:
    if not answer_correct(qtype, pred, ref):
        return False
    if not ref.intervals:   # nothing to cite (e.g. activity absent): the answer alone decides
        return True
    return interval_iou(pred.intervals, ref.intervals) >= iou_thr and pred.modality == ref.modality


def score(records: list[dict]) -> dict:
    """records: dicts with qtype, correct, grounded. Returns per-type and macro-averaged accuracy."""
    types = sorted({r["qtype"] for r in records})
    per = {t: float(np.mean([r["correct"] for r in records if r["qtype"] == t])) for t in types}
    gper = {t: float(np.mean([r["grounded"] for r in records if r["qtype"] == t])) for t in types}
    return {"per_type": per, "overall_macro": float(np.mean(list(per.values()))) if per else 0.0,
            "grounded_per_type": gper, "grounded_macro": float(np.mean(list(gper.values()))) if gper else 0.0,
            "n": len(records)}
