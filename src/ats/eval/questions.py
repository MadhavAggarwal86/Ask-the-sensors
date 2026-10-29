"""Generate question/reference pairs from ground-truth activity intervals.

The reference answer is computed by running the same operation on a timeline built from the
ground-truth labels, with the Intent constructed directly (not parsed). So the evaluation
tests recognition, aggregation and the parser at once, while the definition of "duration",
"count" etc. stays identical between system and reference.
"""
from dataclasses import dataclass

import numpy as np

from .. import config as C
from ..qa import operations
from ..qa.parser import Intent
from ..timeline import Interval, Timeline

VERB = {"lying_down": "lying down", "sitting": "sitting", "standing_in_place": "standing in place",
        "standing_moving": "standing and moving", "walking": "walking", "running": "running",
        "bicycling": "cycling"}

TEMPLATES = {
    "identification": ["What activity is the user performing between {s} and {e} seconds?",
                       "What was the user doing from {s} to {e} seconds?"],
    "verification": ["Was the user {v} between {s} and {e} seconds?",
                     "Is the user {v} from {s} to {e} seconds?"],
    "duration": ["How long was the user {v}?", "How much time did the user spend {v}?",
                 "What was the total duration of {v}?"],
    "count": ["How many times did the user start {v}?", "How often was the user {v}?"],
    "comparison": ["Did the user spend more time {v} or {v2}?", "Was there more {v} or {v2}?"],
    "grounding": ["Did the user begin {v} at any point, and if so, when?", "When did the user start {v}?"],
    "prolonged": ["Did the user lie down for a prolonged period?", "Was the user sitting for a prolonged period?"],
}
OPEN_INTENTS = [
    "Is the recording consistent with the user exercising?",
    "Does the motion look like a seated desk-type routine?",
    "Is this consistent with travelling or commuting?",
    "Is the activity consistent with sleeping or napping?",
]


@dataclass
class QA:
    qtype: str
    question: str
    reference: operations.Answer


def gt_timeline(gt: list[tuple[float, float, str]], duration: float, has_gyro: bool = True) -> Timeline:
    """Merge ground-truth label intervals exactly as predicted windows are merged."""
    ivs: list[Interval] = []
    for s, e, a in sorted(gt):
        if ivs and ivs[-1].activity == a and s - ivs[-1].end <= C.MERGE_GAP_S:
            ivs[-1].end = max(ivs[-1].end, e)
        else:
            ivs.append(Interval(s, e, a))
    return Timeline(ivs, has_gyro=has_gyro, duration=duration)


def generate(gt, duration: float, has_gyro: bool = True, seed: int = 0, per_type: int = 2) -> list[QA]:
    rng = np.random.default_rng(seed)
    tl = gt_timeline(gt, duration, has_gyro)
    present = sorted({iv.activity for iv in tl.intervals})
    absent = [a for a in C.CLASSES if a not in present]
    out: list[QA] = []

    def add(qtype, text, intent):
        intent.question = text
        out.append(QA(qtype, text, operations.run(intent, tl)))

    long_ivs = [iv for iv in tl.intervals if iv.duration >= 30]
    for iv in rng.permutation(long_ivs)[:per_type] if long_ivs else []:
        s, e = int(np.ceil(iv.start)), int(np.floor(min(iv.end, iv.start + 300)))
        add("identification", rng.choice(TEMPLATES["identification"]).format(s=s, e=e),
            Intent("identify", t_range=(s, e)))
        other = rng.choice([a for a in C.CLASSES if a != iv.activity])
        for act in (iv.activity, other):
            add("verification", rng.choice(TEMPLATES["verification"]).format(v=VERB[act], s=s, e=e),
                Intent("verify", [[act]], t_range=(s, e)))

    for act in list(rng.permutation(present)[:per_type]) + list(rng.permutation(absent)[:1]):
        add("duration", rng.choice(TEMPLATES["duration"]).format(v=VERB[act]), Intent("duration", [[act]]))
        add("count", rng.choice(TEMPLATES["count"]).format(v=VERB[act]), Intent("count", [[act]]))

    for _ in range(per_type):
        if len(present) >= 1:
            a = rng.choice(present)
            b = rng.choice([c for c in C.CLASSES if c != a])
            add("comparison", rng.choice(TEMPLATES["comparison"]).format(v=VERB[a], v2=VERB[b]),
                Intent("compare", [[a], [b]]))

    for act in list(rng.permutation(present)[:per_type]):
        add("grounding", rng.choice(TEMPLATES["grounding"]).format(v=VERB[act]), Intent("onset", [[act]]))

    add("prolonged", TEMPLATES["prolonged"][0], Intent("prolonged", [["lying_down"]]))
    add("prolonged", TEMPLATES["prolonged"][1], Intent("prolonged", [["sitting"]]))

    for q in rng.permutation(OPEN_INTENTS)[:per_type]:
        add("open_world", q, Intent("open"))
    return out
