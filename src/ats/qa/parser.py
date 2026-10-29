"""Question parser: natural language -> structured Intent.

A deterministic rule parser is the default (fast, no model, fully reproducible). An optional
small LLM (see llm.py) can produce the same Intent JSON; its output is validated against
the same schema and the rule parser is the fallback.
"""
import re
from dataclasses import dataclass, field

from .. import config as C

OPS = ["identify", "verify", "duration", "count", "onset", "compare", "longest", "prolonged", "open"]

# Phrase -> set of classes. Longer phrases are matched first so "standing still" wins over "standing".
ACTIVITY_LEXICON: dict[str, tuple[str, ...]] = {
    "standing in place": ("standing_in_place",),
    "standing still": ("standing_in_place",),
    "stand still": ("standing_in_place",),
    "standing and moving": ("standing_moving",),
    "moving around": ("standing_moving",),
    "pottering": ("standing_moving",),
    "lying down": ("lying_down",), "lying": ("lying_down",), "lie down": ("lying_down",),
    "lay down": ("lying_down",), "laying": ("lying_down",), "in bed": ("lying_down",),
    "sitting": ("sitting",), "sit": ("sitting",), "sat": ("sitting",), "seated": ("sitting",),
    "standing": ("standing_in_place", "standing_moving"), "stand": ("standing_in_place", "standing_moving"),
    "walking": ("walking",), "walk": ("walking",), "walked": ("walking",), "stroll": ("walking",),
    "running": ("running",), "run": ("running",), "ran": ("running",), "jog": ("running",),
    "jogging": ("running",), "jogged": ("running",), "sprint": ("running",),
    "bicycling": ("bicycling",), "bicycle": ("bicycling",), "cycling": ("bicycling",),
    "cycle": ("bicycling",), "biking": ("bicycling",), "bike": ("bicycling",),
    "pedal": ("bicycling",), "wheeled": ("bicycling",),
    # Groups
    "resting": ("lying_down", "sitting", "standing_in_place"),
    "rest": ("lying_down", "sitting", "standing_in_place"),
    "sedentary": ("lying_down", "sitting"),
    "inactive": ("lying_down", "sitting", "standing_in_place"),
    "stationary": ("lying_down", "sitting", "standing_in_place"),
    "strenuous": ("running", "bicycling"), "vigorous": ("running", "bicycling"),
    "intense": ("running", "bicycling"),
    "on foot": ("walking", "running"),
    "active": ("standing_moving", "walking", "running", "bicycling"),
    "physically active": ("standing_moving", "walking", "running", "bicycling"),
}
# Words that, alone, describe behaviour beyond the seven labels and route to open-world reasoning.
OPEN_WORLD_CUES = ["consistent with", "exercise", "workout", "commut", "sleep", "nap", "chore", "cooking",
                   "cleaning", "shopping", "unsteady", "fall", "stumble", "why", "explain", "describe",
                   "routine", "behavio", "office", "desk", "outdoor", "travel", "errand", "gym", "hike"]


@dataclass
class Intent:
    op: str
    activities: list[list[str]] = field(default_factory=list)  # one class-group per mentioned activity
    phrases: list[str] = field(default_factory=list)            # the words that matched, for display
    t_range: tuple[float, float] | None = None
    question: str = ""


def _find_activities(q: str) -> tuple[list[list[str]], list[str]]:
    found, phrases, taken = [], [], []
    for phrase in sorted(ACTIVITY_LEXICON, key=len, reverse=True):
        for m in re.finditer(rf"\b{re.escape(phrase)}(?:s|es|ed|d|ing)?\b", q):
            if any(m.start() < e and s < m.end() for s, e in taken):
                continue
            taken.append((m.start(), m.end()))
            found.append((m.start(), list(ACTIVITY_LEXICON[phrase]), phrase))
    found.sort()
    groups, out_phr = [], []
    for _, g, p in found:
        if g not in groups:
            groups.append(g)
            out_phr.append(p)
    return groups, out_phr


def _to_seconds(value: float, unit: str | None) -> float:
    unit = (unit or "s").lower()
    return value * (3600 if unit.startswith("h") else 60 if unit.startswith("m") else 1)


def _clock_to_rel(hh: int, mm: int, ss: int, t0: float | None) -> float | None:
    if t0 is None:
        return None
    import datetime as dt
    start = dt.datetime.fromtimestamp(t0)
    target = start.replace(hour=hh, minute=mm, second=ss)
    rel = (target - start).total_seconds()
    return rel if rel >= 0 else rel + 86400


def _find_range(q: str, duration: float, t0: float | None) -> tuple[float, float] | None:
    unit = r"(s|sec|secs|seconds?|m|min|mins|minutes?|h|hrs?|hours?)?"
    m = re.search(rf"(?:between|from)\s+(\d+(?:\.\d+)?)\s*{unit}\s+(?:and|to|-)\s+(\d+(?:\.\d+)?)\s*{unit}", q)
    if m:
        u2 = m.group(4) or m.group(2)
        return _to_seconds(float(m.group(1)), m.group(2) or u2), _to_seconds(float(m.group(3)), u2)
    m = re.search(rf"first\s+(\d+(?:\.\d+)?)\s*{unit}", q)
    if m:
        return 0.0, _to_seconds(float(m.group(1)), m.group(2) or "m")
    m = re.search(rf"last\s+(\d+(?:\.\d+)?)\s*{unit}", q)
    if m:
        return max(0.0, duration - _to_seconds(float(m.group(1)), m.group(2) or "m")), duration
    m = re.search(rf"(?:at|around|near)\s+(\d+(?:\.\d+)?)\s*{unit}(?!\S*:)", q)
    if m and m.group(2):
        c = _to_seconds(float(m.group(1)), m.group(2))
        return max(0.0, c - 60), c + 60
    # Clock times need the absolute start time of the recording.
    m = re.search(r"(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(am|pm)?", q)
    clock = None
    if m:
        hh = int(m.group(1)) % 12 + (12 if m.group(4) == "pm" else 0) if m.group(4) else int(m.group(1))
        clock = (hh, int(m.group(2)), int(m.group(3) or 0))
    elif "noon" in q or "midday" in q:
        clock = (12, 0, 0)
    if clock:
        c = _clock_to_rel(*clock, t0)
        if c is not None:
            return max(0.0, c - 1800), c + 1800
    for word, (a, b) in {"morning": (6, 12), "afternoon": (12, 18), "evening": (18, 22)}.items():
        if word in q:
            s, e = _clock_to_rel(a, 0, 0, t0), _clock_to_rel(b, 0, 0, t0)
            if s is not None:
                return s, e if e > s else e + 86400
    return None


def parse(question: str, duration: float = float("inf"), t0: float | None = None) -> Intent:
    q = question.lower().strip()
    acts, phrases = _find_activities(q)
    t_range = _find_range(q, duration, t0)
    it = Intent(op="open", activities=acts, phrases=phrases, t_range=t_range, question=question)

    if re.search(r"\b(how long|how much time|how much of|duration|how many (minutes|seconds|hours))\b", q) and acts:
        it.op = "duration"
    elif re.search(r"\b(how many times|how often|number of|how many (bouts|episodes|sessions)|count)\b", q) and acts:
        it.op = "count"
    elif len(acts) >= 2 and re.search(r"\b(more|less|longer|shorter|most|least|compare|than|or)\b", q):
        it.op = "compare"
    elif re.search(r"\blongest\b", q) and acts:
        it.op = "longest"
    elif re.search(r"\b(prolonged|long period|extended|for a long|for long)\b", q) and acts:
        it.op = "prolonged"
    elif re.search(r"\b(when|what time|at what point|begin|began|start|onset|first time)\b", q) and acts:
        it.op = "onset"
    elif re.search(r"\bwhat\b.*\b(doing|activity|performing|posture)\b|\bwhat is (the user|she|he|they) doing", q):
        it.op = "identify"
    elif acts and not any(cue in q for cue in OPEN_WORLD_CUES):
        it.op = "verify"
    return it
