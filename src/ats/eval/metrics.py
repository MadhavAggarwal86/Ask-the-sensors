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


def _num(v):
    """A numeric answer value, or None (a question routed to the wrong operation returns e.g. a label)."""
    return v if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_)) else None


def answer_correct(qtype: str, pred, ref, dur_rel_tol=DUR_REL_TOL, iou_thr=IOU_THRESHOLD) -> bool:
    pv, rv = pred.value, ref.value
    if qtype in EXACT_TYPES:
        return _norm(pv) == _norm(rv)
    if qtype in ("duration", "count") and pv is not None and _num(pv) is None:
        return False
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


BINARY_TYPES = ("verification", "prolonged")
CATEGORICAL_TYPES = ("identification", "comparison", "open_world")


def _binary(pairs) -> dict:
    """pairs: (predicted, reference) booleans. Precision/recall/F1 on 'yes' plus specificity."""
    tp = sum(p and r for p, r in pairs)
    fp = sum(p and not r for p, r in pairs)
    fn = sum(r and not p for p, r in pairs)
    tn = sum(not p and not r for p, r in pairs)
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = 2 * prec * rec / (prec + rec) if tp else 0.0  # tp > 0 implies prec and rec are defined
    return {"n": len(pairs), "positives": tp + fn, "precision": prec, "recall": rec, "f1": f1,
            "specificity": tn / (tn + fp) if tn + fp else None,
            "accuracy": (tp + tn) / len(pairs) if pairs else None}


def _categorical(pairs) -> dict:
    """pairs: (predicted, reference) labels. Macro-F1 and balanced accuracy over reference classes."""
    from sklearn.metrics import balanced_accuracy_score, f1_score
    p, r = [str(a) for a, _ in pairs], [str(b) for _, b in pairs]
    return {"n": len(pairs), "classes": len(set(r)), "accuracy": float(np.mean([a == b for a, b in zip(p, r)])),
            "macro_f1": float(f1_score(r, p, labels=sorted(set(r)), average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(r, p))}


def answer_metrics(records: list[dict]) -> dict:
    """The per-kind metrics the brief asks for beside plain accuracy: P/R/F1 and specificity for yes/no
    answers, macro-F1 and balanced accuracy for categorical answers, MAE (and MAPE) for numbers."""
    out = {}
    for t in BINARY_TYPES:
        pairs = [(bool(r["pred"].value), bool(r["ref"].value)) for r in records if r["qtype"] == t]
        if pairs:
            out[t] = _binary(pairs)
    # onset questions are yes/no first ("did the user begin X?"), then a time
    pairs = [(r["pred"].value is not None, r["ref"].value is not None) for r in records if r["qtype"] == "grounding"]
    if pairs:
        out["grounding_detection"] = _binary(pairs)
    for t in CATEGORICAL_TYPES:
        pairs = [(_norm(r["pred"].value), _norm(r["ref"].value)) for r in records if r["qtype"] == t]
        if pairs:
            out[t] = _categorical(pairs)
    for t in ("duration", "count"):
        pairs = [(_num(r["pred"].value), r["ref"].value) for r in records
                 if r["qtype"] == t and _num(r["pred"].value) is not None and r["ref"].value is not None]
        if not pairs:
            continue
        err = np.array([abs(p - r) for p, r in pairs], float)
        m = {"n": len(pairs), "mae": float(err.mean()), "median_abs_error": float(np.median(err))}
        if t == "duration":
            nz = [(p, r) for p, r in pairs if r > 0]
            m["mape_pct"] = float(100 * np.mean([abs(p - r) / r for p, r in nz])) if nz else None
            m["mape_n"] = len(nz)
        out[t] = m
    return out
