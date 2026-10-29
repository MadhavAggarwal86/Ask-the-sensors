"""Execute a parsed Intent against the Timeline and build a grounded Answer."""
from dataclasses import dataclass, field

import numpy as np

from .. import config as C
from ..timeline import Interval, Timeline
from .explain import describe_signal, explain_activity, modality
from .parser import Intent

SHORT_CLIP_S = 120.0     # below this, "is the user X?" asks about the dominant activity
LOW_CONFIDENCE = 0.6     # below this mean probability, yes-answers are hedged as "Likely yes"


@dataclass
class Answer:
    answer: str
    activity: str = "N/A"
    timestamps: str = "N/A"
    modality: str = "N/A"
    channels: str = "N/A"
    explanation: str = "N/A"
    # Structured fields used for scoring
    op: str = ""
    value: object = None
    intervals: list[tuple[float, float]] = field(default_factory=list)


def names(group) -> str:
    return " / ".join(C.DISPLAY[a] for a in group)


def fmt_intervals(ivs: list[Interval], max_n: int = 12) -> str:
    if not ivs:
        return "N/A"
    shown = sorted(ivs, key=lambda iv: iv.start)
    extra = ""
    if len(shown) > max_n:
        shown = sorted(sorted(ivs, key=lambda iv: -iv.duration)[:max_n], key=lambda iv: iv.start)
        extra = f" (+{len(ivs) - max_n} shorter intervals)"
    return ", ".join(f"{iv.start:.0f} to {iv.end:.0f}" for iv in shown) + extra + " (seconds from start)"


def _grounded(tl: Timeline, ans: Answer, ivs: list[Interval], activity: str | None, lead: str = "") -> Answer:
    """Fill the evidence fields from the cited intervals."""
    ans.intervals = [(iv.start, iv.end) for iv in ivs]
    if not ivs:
        return ans
    ans.timestamps = fmt_intervals(ivs)
    ans.modality, ans.channels = modality(tl.has_gyro)
    stats = tl.feature_stats(ivs)
    cited = set(np.concatenate([iv.idx for iv in ivs]).tolist()) if ivs else set()
    others = [iv for iv in tl.intervals if not set(iv.idx.tolist()) & cited]
    body = explain_activity(activity, stats, tl.feature_stats(others), tl.has_gyro) if activity \
        else f"The cited windows show {describe_signal(stats, tl.has_gyro)}."
    ans.explanation = (lead + " " + body).strip()
    return ans


def _analysed(tl: Timeline, t_range) -> float:
    return sum(tl.breakdown(t_range).values())


def _hedge(yes: bool, conf: float) -> str:
    if not yes:
        return "No"
    return "Yes" if conf >= LOW_CONFIDENCE else "Likely yes"


def run(intent: Intent, tl: Timeline) -> Answer:
    if not tl.intervals:
        return Answer("N/A", explanation="No usable sensor data: every window had too many missing samples.",
                      op=intent.op)
    fn = {"identify": identify, "verify": verify, "duration": duration, "count": count, "onset": onset,
          "compare": compare, "longest": longest, "prolonged": prolonged}.get(intent.op, open_world)
    ans = fn(intent, tl)
    ans.op = intent.op
    return ans


def _where(intent: Intent) -> str:
    return f" between {intent.t_range[0]:.0f} and {intent.t_range[1]:.0f} s" if intent.t_range else ""


def identify(intent: Intent, tl: Timeline) -> Answer:
    bd = tl.breakdown(intent.t_range)
    total = sum(bd.values())
    if total == 0:
        return Answer("N/A", explanation=f"No sensor data{_where(intent)}.")
    top = max(bd, key=bd.get)
    ivs = sorted(tl.select(top, intent.t_range), key=lambda iv: -iv.duration)[:3]
    ans = Answer(C.DISPLAY[top], C.DISPLAY[top], value=top)
    lead = f"{C.DISPLAY[top]} accounts for {bd[top]:.0f} of {total:.0f} analysed seconds{_where(intent)} ({100 * bd[top] / total:.0f}%)."
    return _grounded(tl, ans, ivs, top, lead)


def verify(intent: Intent, tl: Timeline) -> Answer:
    group = intent.activities[0]
    analysed = _analysed(tl, intent.t_range)
    span = (intent.t_range[1] - intent.t_range[0]) if intent.t_range else tl.duration
    if span < SHORT_CLIP_S or analysed < SHORT_CLIP_S:
        bd = tl.breakdown(intent.t_range)
        yes = bool(analysed) and sum(bd[a] for a in group) >= 0.5 * analysed
        ivs = tl.select(group, intent.t_range)
    else:
        ivs = tl.bouts(group, intent.t_range)
        yes = bool(ivs)
    conf = float(np.mean([iv.confidence for iv in ivs])) if ivs else 0.0
    ans = Answer(_hedge(yes, conf), names(group), value=yes)
    if yes:
        lead = f"{names(group)} detected in {len(ivs)} interval(s){_where(intent)}, {sum(iv.duration for iv in ivs):.0f} s in total."
        return _grounded(tl, ans, ivs, group[0] if len(group) == 1 else None, lead)
    # A "no" is still grounded: cite what the user was doing instead.
    bd = tl.breakdown(intent.t_range)
    top = max(bd, key=bd.get)
    ivs = sorted(tl.select(top, intent.t_range), key=lambda iv: -iv.duration)[:3]
    lead = f"No {names(group).lower()} bout of at least {C.MIN_BOUT_S:.0f} s{_where(intent)}; the dominant activity is {C.DISPLAY[top].lower()}."
    return _grounded(tl, ans, ivs, top, lead)


def duration(intent: Intent, tl: Timeline) -> Answer:
    group = intent.activities[0]
    ivs = tl.select(group, intent.t_range)
    total = sum(iv.duration for iv in ivs)
    ans = Answer(f"{total:.0f} seconds", names(group), value=total)
    if not ivs:
        ans.explanation = f"No {names(group).lower()} was detected{_where(intent)}."
        return ans
    lead = (f"{names(group)} was detected in {len(ivs)} separate interval(s) of "
            f"{', '.join(f'{iv.duration:.0f}' for iv in sorted(ivs, key=lambda iv: iv.start)[:12])} seconds, "
            f"which sum to {total:.0f} seconds.")
    return _grounded(tl, ans, ivs, group[0] if len(group) == 1 else None, lead)


def count(intent: Intent, tl: Timeline) -> Answer:
    group = intent.activities[0]
    bouts = tl.bouts(group, intent.t_range)
    ans = Answer(f"{len(bouts)} time{'s' if len(bouts) != 1 else ''}", names(group), value=len(bouts))
    lead = (f"Counted separate bouts of {names(group).lower()} lasting at least {C.MIN_BOUT_S:.0f} s, "
            f"split wherever another activity or a gap over {C.MERGE_GAP_S:.0f} s intervenes.")
    if not bouts:
        ans.explanation = lead
        return ans
    return _grounded(tl, ans, bouts, group[0] if len(group) == 1 else None, lead)


def onset(intent: Intent, tl: Timeline) -> Answer:
    group = intent.activities[0]
    bouts = tl.bouts(group, intent.t_range)
    if not bouts:
        ans = Answer("No", f"Onset of {names(group).lower()}", value=None)
        ans.explanation = f"No {names(group).lower()} bout of at least {C.MIN_BOUT_S:.0f} s was detected{_where(intent)}."
        return ans
    first = bouts[0]
    ans = Answer(f"Yes, {names(group).lower()} began at {first.start:.0f} seconds",
                 f"Onset of {names(group).lower()}", value=first.start)
    prev = [iv for iv in tl.intervals if iv.end <= first.start + 1e-6]
    lead = ""
    if prev:
        p = prev[-1]
        ps, fs = tl.feature_stats([p]), tl.feature_stats([first])
        if ps and fs:
            lead = (f"The preceding interval ({p.start:.0f} to {p.end:.0f} s) was {C.DISPLAY[p.activity].lower()} with "
                    f"{ps['acc_mag_std']:.3f} g variability; at {first.start:.0f} s this changes to "
                    f"{fs['acc_mag_std']:.3f} g, marking the transition.")
    elif len(tl.starts) and first.start <= tl.starts[0] + 1e-6:
        lead = (f"{names(group)} is already under way in the first analysed window, so it began at or before "
                f"the start of the recording; {first.start:.0f} s is the earliest time the sensors can confirm.")
    return _grounded(tl, ans, [first], first.activity, lead)


def compare(intent: Intent, tl: Timeline) -> Answer:
    groups = intent.activities[:2]
    totals = [tl.total(g, intent.t_range) for g in groups]
    q = intent.question.lower()
    want_less = any(w in q for w in ("less", "least", "shorter", "fewer"))
    win = int(np.argmin(totals) if want_less else np.argmax(totals))
    ans = Answer(names(groups[win]) if totals[0] != totals[1] else "Equal",
                 ", ".join(names(g) for g in groups), value=groups[win][0])
    ivs = [iv for g in groups for iv in tl.select(g, intent.t_range)]
    _grounded(tl, ans, ivs, None, "")
    ans.timestamps = ", ".join(f"{names(g)} = {t:.0f} seconds total" for g, t in zip(groups, totals))
    ans.explanation = (f"Total {names(groups[0]).lower()} time ({totals[0]:.0f} s) "
                       f"{'exceeded' if totals[0] > totals[1] else 'was below' if totals[0] < totals[1] else 'equalled'} "
                       f"total {names(groups[1]).lower()} time ({totals[1]:.0f} s){_where(intent)}.")
    return ans


def longest(intent: Intent, tl: Timeline) -> Answer:
    group = intent.activities[0]
    bouts = tl.bouts(group, intent.t_range)
    if not bouts:
        return Answer("N/A", names(group), explanation=f"No {names(group).lower()} detected{_where(intent)}.")
    b = max(bouts, key=lambda iv: iv.duration)
    ans = Answer(f"{b.duration:.0f} seconds, from {b.start:.0f} to {b.end:.0f}", f"Longest {names(group).lower()} bout",
                 value=b.duration)
    return _grounded(tl, ans, [b], b.activity, f"Longest of {len(bouts)} bout(s).")


def prolonged(intent: Intent, tl: Timeline) -> Answer:
    group = intent.activities[0]
    bouts = tl.bouts(group, intent.t_range)
    b = max(bouts, key=lambda iv: iv.duration) if bouts else None
    yes = b is not None and b.duration >= C.PROLONGED_S
    ans = Answer(_hedge(yes, b.confidence if b else 0), f"Prolonged {names(group).lower()}", value=yes)
    if b is None:
        ans.explanation = f"No {names(group).lower()} detected{_where(intent)}."
        return ans
    lead = (f"The longest continuous {names(group).lower()} bout lasts {b.duration:.0f} s "
            f"({'at least' if yes else 'under'} the {C.PROLONGED_S:.0f} s threshold for 'prolonged').")
    return _grounded(tl, ans, [b], b.activity, lead)


# --- Task 4: open-world reasoning ------------------------------------------------------------
# Each broad behaviour is described by which recognised activities and signal properties would
# support it. The verdict is argued from those measured properties, not from the label alone.
BEHAVIORS = [
    (("exercis", "workout", "gym", "sport", "training", "strenuous"), "Exercise or physical exertion",
     ("running", "bicycling", "walking"), "vigorous"),
    (("commut", "travel", "errand", "trip", "going somewhere"), "Travelling or commuting",
     ("walking", "bicycling", "running"), "sustained"),
    (("sleep", "nap", "bed"), "Sleeping or napping", ("lying_down",), "still"),
    (("desk", "office", "computer", "reading", "tv", "television", "study"), "Seated desk-type activity",
     ("sitting",), "still"),
    (("chore", "cook", "clean", "housework", "dishes", "laundry", "kitchen"), "Household chores",
     ("standing_moving", "walking", "standing_in_place"), "light"),
    (("shopping", "store", "market"), "Shopping", ("walking", "standing_moving", "standing_in_place"), "light"),
    (("hike", "hiking"), "Hiking", ("walking",), "sustained"),
    (("unsteady", "fall", "fell", "stumbl", "balance", "dizzy", "trip over", "tripped", "collaps"),
     "Unsteadiness or possible fall", None, "spikes"),
]


def _intensity(std: float) -> str:
    return "still" if std < 0.02 else "light" if std < 0.1 else "moderate" if std < 0.35 else "vigorous"


# Quantities that motion sensors cannot observe. Answering them would mean inventing evidence.
UNMEASURABLE = {"heart": "heart rate", "pulse": "heart rate", "blood": "blood pressure or chemistry",
                "temperature": "temperature", "calorie": "energy expenditure", "location": "location",
                "gps": "location", "weather": "weather", "mood": "mood", "stress": "stress level",
                "emotion": "emotional state", "speak": "speech", "talk": "speech", "audio": "sound"}


def open_world(intent: Intent, tl: Timeline) -> Answer:
    q = intent.question.lower()
    missing = next((v for k, v in UNMEASURABLE.items() if k in q), None)
    if missing:
        return Answer("Cannot be determined from these sensors", "N/A", value=None,
                      explanation=f"The recording holds only accelerometer and gyroscope signals; {missing} is "
                                  "not measured, so no sensor evidence can support an answer.")
    match = next((b for b in BEHAVIORS if any(k in q for k in b[0])), None)
    analysed = _analysed(tl, intent.t_range)
    if analysed == 0:
        return Answer("N/A", explanation=f"No sensor data{_where(intent)}.")

    if match and match[3] == "spikes":
        return _spikes(intent, tl, match[1])

    if match:
        _, label, acts, mode = match
        ivs = tl.select(acts, intent.t_range)
        frac = sum(iv.duration for iv in ivs) / analysed
        stats = tl.feature_stats(ivs)
        level = _intensity(stats["acc_mag_std"]) if stats else "n/a"
        longest_b = max((iv.duration for iv in tl.bouts(acts, intent.t_range)), default=0.0)
        if not stats:   # no features (e.g. ground-truth timeline): judge on activity share alone
            score = frac
        elif mode == "vigorous":
            score = frac if level in ("moderate", "vigorous") else frac / 2
        elif mode == "sustained":
            score = frac if longest_b >= 300 else frac / 2
        elif mode == "still":
            score = frac if level == "still" else frac / 2
        else:
            score = frac
        verdict = "Consistent" if score >= 0.5 else "Partially consistent" if score >= 0.15 else "Not consistent"
        ans = Answer(verdict, f"{label} (via {', '.join(C.DISPLAY[a].lower() for a in acts)})", value=verdict)
        lead = (f"Activities that would support '{label.lower()}' cover {100 * frac:.0f}% of the analysed time"
                f"{_where(intent)}, with {level} intensity overall and a longest bout of {longest_b:.0f} s.")
        if not ivs:
            ans.explanation = lead
            return ans
        top = sorted(ivs, key=lambda iv: -iv.duration)[:5]
        return _grounded(tl, ans, top, None, lead)

    # Generic profile when the question names no known behaviour.
    bd = tl.breakdown(intent.t_range)
    top = max(bd, key=bd.get)
    ivs = tl.select(list(C.CLASSES), intent.t_range)
    stats = tl.feature_stats(ivs)
    level = _intensity(stats["acc_mag_std"]) if stats else "unknown"
    mix = ", ".join(f"{C.DISPLAY[a].lower()} {100 * v / analysed:.0f}%" for a, v in
                    sorted(bd.items(), key=lambda kv: -kv[1]) if v > 0)
    behaviour = ("Mostly sedentary" if bd["lying_down"] + bd["sitting"] > 0.6 * analysed else
                 "Mostly active" if bd["walking"] + bd["running"] + bd["bicycling"] > 0.4 * analysed else
                 "Mixed light activity")
    ans = Answer(f"{behaviour}; mainly {C.DISPLAY[top].lower()}", behaviour, value=behaviour)
    lead = f"Activity mix{_where(intent)}: {mix}; overall intensity is {level}."
    top_ivs = sorted(tl.select(top, intent.t_range), key=lambda iv: -iv.duration)[:3]
    return _grounded(tl, ans, top_ivs, top, lead)


POST_STILL_G = 0.03      # median variability below this after an impact = the user stopped moving
TILT_CHANGE_DEG = 45.0   # device orientation change across an impact that suggests ending up in a new posture


def _spikes(intent: Intent, tl: Timeline, label: str) -> Answer:
    """Look for falls: an abrupt high-acceleration impact FOLLOWED by stillness or a new orientation.

    Sharp impacts alone also occur in hard footstrikes or bumps; those keep the movement going with
    the same device orientation, so they are reported but not called falls.
    """
    f = dict(zip(tl.feature_names, tl.features.T))
    jerk, peak, std = f["acc_jerk"], f["acc_mag_max"], f["acc_mag_std"]
    grav = np.column_stack([f["acc_x_mean"], f["acc_y_mean"], f["acc_z_mean"]])
    sel = np.ones(len(tl.starts), bool) if intent.t_range is None else \
        (tl.starts + C.WIN_S > intent.t_range[0]) & (tl.starts < intent.t_range[1])
    med = np.nanmedian(jerk[sel]) if sel.any() else 0
    mad = np.nanmedian(np.abs(jerk[sel] - med)) + 1e-6 if sel.any() else 1
    hits = np.where(sel & (jerk > med + 8 * mad) & (peak > 2.0))[0]
    if len(hits) == 0:
        return Answer("No abrupt events detected", label, value=False,
                      explanation=f"No window{_where(intent)} combines a peak above 2 g with a jerk far above the "
                                  f"recording's typical level ({med:.2f} g/s), so there is no signal sign of a "
                                  f"stumble or fall.")
    falls, details = [], []
    for i in hits:
        t = tl.starts[i]
        pre = (tl.starts >= t - 30) & (tl.starts < t)
        post = (tl.starts > t + C.WIN_S / 2) & (tl.starts <= t + 60)
        if not pre.any() or not post.any():
            continue
        g0, g1 = np.nanmean(grav[pre], 0), np.nanmean(grav[post], 0)
        cos = np.dot(g0, g1) / (np.linalg.norm(g0) * np.linalg.norm(g1) + 1e-9)
        tilt = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
        still = float(np.nanmedian(std[post]))
        details.append((i, tilt, still))
        if still < POST_STILL_G or tilt > TILT_CHANGE_DEG:
            falls.append((i, tilt, still))
    if falls:
        ivs = [Interval(float(tl.starts[i]), float(tl.starts[i] + C.WIN_S), "event", 1.0, np.array([i]))
               for i, _, _ in falls]
        ans = Answer(f"Yes, {len(falls)} possible fall(s)", label, value=True)
        i, tilt, still = max(falls, key=lambda x: peak[x[0]])
        lead = (f"{len(falls)} impact(s) with peaks above 2 g are followed by a fall signature; the strongest "
                f"({peak[i]:.2f} g, jerk {jerk[i]:.1f} g/s against a typical {med:.2f} g/s) is followed by "
                f"{still:.3f} g variability and a {tilt:.0f} degree change in device orientation.")
        return _grounded(tl, ans, ivs[:10], None, lead)
    ivs = [Interval(float(tl.starts[i]), float(tl.starts[i] + C.WIN_S), "event", 1.0, np.array([i])) for i in hits]
    ans = Answer("No fall signature", label, value=False)
    after = (f" afterwards movement continued (median {np.median([d[2] for d in details]):.3f} g variability) and "
             f"the device orientation changed by at most {max(d[1] for d in details):.0f} degrees") if details else ""
    lead = (f"{len(hits)} sharp impact(s) reach {peak[hits].max():.2f} g with jerk up to {jerk[hits].max():.1f} g/s "
            f"(typical {med:.2f} g/s), but{after or ' there is too little data around them to judge'}; this points "
            f"to hard footstrikes or bumps rather than a fall.")
    return _grounded(tl, ans, ivs[:10], None, lead)
