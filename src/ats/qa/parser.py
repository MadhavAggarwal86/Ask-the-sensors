"""Question parser: natural language -> structured Intent.

A deterministic rule parser is the default (fast, no model, fully reproducible). An optional
small LLM (see llm.py) can produce the same Intent JSON; its output is validated against
the same schema and the rule parser is the fallback.
"""
import difflib
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
    "reclining": ("lying_down",), "recline": ("lying_down",), "horizontal": ("lying_down",),
    "sitting": ("sitting",), "sit": ("sitting",), "sat": ("sitting",), "seated": ("sitting",),
    "standing": ("standing_in_place", "standing_moving"), "stand": ("standing_in_place", "standing_moving"),
    "walking": ("walking",), "walk": ("walking",), "walked": ("walking",), "stroll": ("walking",),
    "running": ("running",), "run": ("running",), "ran": ("running",), "jog": ("running",),
    "jogging": ("running",), "jogged": ("running",), "sprint": ("running",),
    "bicycling": ("bicycling",), "bicycle": ("bicycling",), "cycling": ("bicycling",),
    "cycle": ("bicycling",), "biking": ("bicycling",), "bike": ("bicycling",),
    "pedal": ("bicycling",), "pedalling": ("bicycling",), "pedaling": ("bicycling",),
    "wheeled": ("bicycling",), "biked": ("bicycling",), "cycled": ("bicycling",),
    # Groups
    "resting": ("lying_down", "sitting", "standing_in_place"),
    "rest": ("lying_down", "sitting", "standing_in_place"),
    "sedentary": ("lying_down", "sitting"),
    "inactive": ("lying_down", "sitting", "standing_in_place"),
    "idle": ("lying_down", "sitting", "standing_in_place"),
    "motionless": ("lying_down", "sitting", "standing_in_place"),
    "stationary": ("lying_down", "sitting", "standing_in_place"),
    "strenuous": ("running", "bicycling"), "vigorous": ("running", "bicycling"),
    "intense": ("running", "bicycling"),
    "on foot": ("walking", "running"),
    "active": ("standing_moving", "walking", "running", "bicycling"),
    "physically active": ("standing_moving", "walking", "running", "bicycling"),
}
# Words that, alone, describe behaviour beyond the seven labels and route to open-world reasoning.
OPEN_WORLD_CUES = ["consistent with", "exercis", "workout", "commut", "sleep", "asleep", "nap", "chore",
                   "cooking", "cleaning", "housework", "shopping", "unsteady", "fall", "fell", "stumbl", "why",
                   "explain", "routine", "office", "desk", "outdoor", "travel", "errand", "gym", "hike",
                   "suggest", "look like", "looks like", "profile of", "evidence of", "sign of", "heart",
                   "likely", "might", "could", "plausib", "pulse", "temperature", "calorie", "location",
                   "gps", "weather", "mood", "stress", "emotion"]
NUMBER_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
                "thirty": 30, "forty": 40, "forty-five": 45, "fifty": 50, "sixty": 60, "ninety": 90}
UNIT = r"(?:seconds?|secs?|minutes?|mins?|hours?|hrs?)"
# Real words close to an activity word; never "correct" these as typos.
NOT_TYPOS = {"waking", "flying", "ruining", "raining", "setting", "sitter", "walker", "runner", "standard"}


@dataclass
class Intent:
    op: str
    activities: list[list[str]] = field(default_factory=list)  # one class-group per mentioned activity
    phrases: list[str] = field(default_factory=list)            # the words that matched, for display
    t_range: tuple[float, float] | None = None
    question: str = ""


def _normalise(q: str) -> str:
    """Lower-case and rewrite spoken forms into the forms the patterns expect."""
    q = q.lower().strip()
    q = re.sub(r"(?<=[a-z])-(?=[a-z])", " ", q)                       # lying-down -> lying down
    q = re.sub(r"\bhalf(?: an| a)? hour\b", "30 minutes", q)
    q = re.sub(r"\bfinal\b", "last", q)
    words = "|".join(sorted(map(re.escape, NUMBER_WORDS), key=len, reverse=True))
    q = re.sub(rf"\b({words})\s+(?={UNIT}\b)", lambda m: f"{NUMBER_WORDS[m.group(1)]} ", q)
    q = re.sub(rf"\b(first|last)\s+(?={UNIT}\b)", r"\1 1 ", q)          # first hour -> first 1 hour
    q = re.sub(r"\b(second|minute|hour)\s+(\d+(?:\.\d+)?)\b", r"\2 \1s", q)  # minute 3 -> 3 minutes
    return _fix_typos(q)


def _fix_typos(q: str) -> str:
    """Replace a misspelt activity word by its closest lexicon word (same first letter, >= 88% similar)."""
    vocab = [w for w in ACTIVITY_LEXICON if " " not in w and len(w) >= 4]

    def fix(m):
        w = m.group(0)
        if len(w) < 5 or w in NOT_TYPOS or any(re.fullmatch(rf"{re.escape(v)}(?:s|es|ed|d|ing)?", w) for v in vocab):
            return w
        close = [c for c in difflib.get_close_matches(w, vocab, n=3, cutoff=0.88) if c[0] == w[0]]
        return close[0] if close else w

    return re.sub(r"[a-z]+", fix, q)


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
    num = r"(\d+(?:\.\d+)?)"
    m = re.search(rf"(?:between|from)\s+{num}\s*{unit}\s+(?:and|to|-)\s+{num}\s*{unit}", q)
    if m:
        u2 = m.group(4) or m.group(2)
        return _to_seconds(float(m.group(1)), m.group(2) or u2), _to_seconds(float(m.group(3)), u2)
    m = re.search(rf"(?:after|beyond|past)\s+(?:the\s+)?(?:first\s+)?{num}\s*{unit}", q)
    if m:
        return min(duration, _to_seconds(float(m.group(1)), m.group(2) or "m")), duration
    m = re.search(rf"(?:before|within)\s+(?:the\s+)?(?:first\s+)?{num}\s*{unit}", q)
    if m:
        return 0.0, _to_seconds(float(m.group(1)), m.group(2) or "m")
    m = re.search(rf"first\s+{num}\s*{unit}", q)
    if m:
        return 0.0, _to_seconds(float(m.group(1)), m.group(2) or "m")
    m = re.search(rf"last\s+{num}\s*{unit}", q)
    if m:
        return max(0.0, duration - _to_seconds(float(m.group(1)), m.group(2) or "m")), duration
    m = re.search(rf"(?:(?:at|around|near|about)\s+{num}\s*{unit}(?!\S*:)|{num}\s*{unit}\s+in\b)", q)
    if m:
        v, u = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        if u:
            c = _to_seconds(float(v), u)
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
    for word, (a, b) in {"morning": (6, 12), "afternoon": (12, 18), "evening": (18, 22), "night": (22, 30)}.items():
        if word in q and t0 is not None:
            # The period on the recording's first day, or the next day if that one is already over.
            s, e = _period_to_rel(a, b, t0)
            return max(0.0, s), e
    return None


def _period_to_rel(h_start: int, h_end: int, t0: float) -> tuple[float, float]:
    import datetime as dt
    start = dt.datetime.fromtimestamp(t0)
    day = start.replace(hour=0, minute=0, second=0, microsecond=0)
    s = (day + dt.timedelta(hours=h_start) - start).total_seconds()
    e = (day + dt.timedelta(hours=h_end) - start).total_seconds()
    return (s + 86400, e + 86400) if e <= 0 else (s, e)


IDENTIFY = (r"\bwhat\b.*\b(doing|activit(y|ies)|performing|posture|happening|going on|up to|engaged)\b"
            r"|\bwhich activity\b|\b(describe|identify|tell me) what\b|\b(dominant|main|predominant|primary)\b")


def parse(question: str, duration: float = float("inf"), t0: float | None = None) -> Intent:
    q = _normalise(question)
    acts, phrases = _find_activities(q)
    t_range = _find_range(q, duration, t0)
    it = Intent(op="open", activities=acts, phrases=phrases, t_range=t_range, question=question)
    open_cue = any(cue in q for cue in OPEN_WORLD_CUES)

    if open_cue:
        it.op = "open"
    elif re.search(r"\blongest\b", q) and acts:
        it.op = "longest"
    elif len(acts) >= 2 and re.search(r"\b(compare|more|less|longer|shorter|most|least|than|versus|vs)\b", q):
        it.op = "compare"       # before duration: "compare the time spent X with Y"
    elif re.search(r"\b(how long|how much time|how much of|duration|how many (minutes|seconds|hours)|total "
                   r"(time|duration)|time spent|(fraction|share|percentage|proportion|percent) of|"
                   r"what (fraction|share|percentage|proportion))\b|\btotal\b.*\btime\b", q) and acts:
        it.op = "duration"
    elif re.search(r"\b(how many|how often|how frequently|number of|count|times did)\b", q) and acts:
        it.op = "count"
    elif len(acts) >= 2 and re.search(r"\bor\b", q):
        it.op = "compare"
    elif re.search(r"\b(prolonged|long period|extended|for a long|for long|long stretch|at a stretch|without a "
                   r"break|long time|continuously|uninterrupted|more than \d+ (minutes|hours))\b", q) and acts:
        it.op = "prolonged"
    elif re.search(r"\b(when|what time|at what point|beg[ia]n|begins|beginning|start(s|ed|ing)?|onset|"
                   r"first time|moment|commenc\w*)\b", q) and acts:
        it.op = "onset"
    elif re.search(IDENTIFY, q) and not acts:
        it.op = "identify"
    elif acts:
        it.op = "verify"
    if it.op not in ("compare", "open") and len(acts) > 1:
        # "How long was the user walking or running?" asks about the two together.
        it.activities = [list(dict.fromkeys(a for g in acts for a in g))]
    return it
