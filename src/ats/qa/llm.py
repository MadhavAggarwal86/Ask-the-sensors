"""Small-language-model front end (the only question parser) and explanation writer.

The model reads the question and fills a fixed JSON schema: the operation, the activity groups, the
time expression as written (kind, numbers, unit, clock time or day period) and, for open-world
questions, the behaviour or unmeasurable quantity asked about. llama.cpp turns the schema into a
grammar, so the output is always well-formed and only uses known operations and classes. Python
then does every calculation: it converts the time expression into seconds from the recording start
(the model never does arithmetic), runs the operation on the timeline and fills the answer.

LLMExplainer words the explanation after the operations have computed the answer. It sees only the
computed fields, and its text is kept only if every number in it already appears in those fields;
otherwise the template explanation is used unchanged, so the language layer never produces numbers.
"""
import json
import re
from pathlib import Path

from .. import config as C
from .parser import OPS, Intent, _clock_to_rel, _period_to_rel, _to_seconds

TIME_KINDS = ["none", "between", "after", "before", "first", "last", "at", "clock", "period"]
PERIODS = ["morning", "afternoon", "evening", "night"]
PERIOD_HOURS = {"morning": (6, 12), "afternoon": (12, 18), "evening": (18, 22), "night": (22, 30)}
TOPICS = ["none", "exercise", "commuting", "sleeping", "desk_work", "chores", "shopping", "hiking", "fall",
          "heart_rate", "blood", "temperature", "calories", "location", "weather", "mood", "stress",
          "speech", "other"]

SYSTEM = f"""You convert questions about a person's wearable motion recording into JSON.
Return one JSON object with these keys:
"op": one of
  identify  - which activity happened (no activity named)
  verify    - yes/no: was a named activity happening
  duration  - how long / total time / share of time of an activity
  count     - how many times / bouts / episodes of an activity
  onset     - when an activity started
  compare   - which of two activities took more (or less) time
  longest   - the longest single bout of an activity
  prolonged - whether an activity lasted a long uninterrupted time
  open      - a behaviour (exercise, sleep, commuting, chores, falls ...) or a quantity motion
              sensors cannot measure (heart rate, location, mood ...)
"activities": one group per activity mentioned, in order; [] when none is named. Classes:
  {", ".join(C.CLASSES)}.
  A group with several classes means any of them: "standing" = ["standing_in_place", "standing_moving"];
  "resting" or "idle" = ["lying_down", "sitting", "standing_in_place"]; "sedentary" = ["lying_down", "sitting"];
  "active" or "moving around" = ["standing_moving", "walking", "running", "bicycling"];
  "strenuous" = ["running", "bicycling"]; "on foot" = ["walking", "running"].
Questions about a behaviour or lifestyle (exercise, workouts, gym, sleep, naps, commuting, travel, office or
desk work, chores, shopping, hiking, falls, stumbling, balance) are ALWAYS op "open", even when phrased as
yes/no; then "activities" is [] and "topic" names the behaviour.
Asking whether the recording, motion or activity is "consistent with", "looks like", "fits", "suggests" or
"matches" a behaviour is ALWAYS op "open" (never verify, compare or identify).
"How often", "how many times", "how frequently", "number of" = op "count".
"Was / Is the user <activity> between a and b" or "from a to b" = op "verify" with that activity (not identify).
"Lasting a long time", "for an extended period", "at a stretch", "without a break" = op "prolonged".
"time": null when the question names no time; otherwise the time expression as asked, without doing any arithmetic:
  kind: between (from a to b) | after (after a) | before (before or within a) |
        first (the first a) | last (the last or final a) | at (at or around a, or "a in") |
        clock (a clock time, 24-hour "HH:MM"; noon = "12:00") | period (morning/afternoon/evening/night)
  a, b: the numbers as written (null if absent); "half an hour" = 30 minutes, "an hour" = 1 hour
  unit: seconds | minutes | hours ("10 minutes" -> minutes; null if no unit is written)
  clock: "HH:MM" or null; period: the day period or null
"topic": for op "open", what is asked about: {", ".join(t for t in TOPICS if t != "none")}; otherwise "none".
Output JSON only."""

NO_TIME = {"kind": None, "a": None, "b": None, "unit": None, "clock": None, "period": None}


def _ex(op, acts=(), topic="none", **time):
    return {"op": op, "activities": [list(g) for g in acts], "time": {**NO_TIME, **time} if time else None,
            "topic": topic}


# Few-shot examples, deliberately different from the evaluation question sets.
EXAMPLES = [
    ("What is the wearer doing between 100 and 160 seconds?",
     _ex("identify", kind="between", a=100, b=160, unit="seconds")),
    ("Did the person do any sprinting in the final quarter hour?",
     _ex("verify", [["running"]], kind="last", a=15, unit="minutes")),
    ("How much time went to cycling overall?", _ex("duration", [["bicycling"]])),
    ("How many separate strolls were there after the first 25 minutes?",
     _ex("count", [["walking"]], kind="after", a=25, unit="minutes")),
    ("When did the person first sit?", _ex("onset", [["sitting"]])),
    ("Was there more lying than standing in the afternoon?",
     _ex("compare", [["lying_down"], ["standing_in_place", "standing_moving"]], kind="period", period="afternoon")),
    ("What was the biggest single stretch of running?", _ex("longest", [["running"]])),
    ("Did the person stay seated for ages without getting up?", _ex("prolonged", [["sitting"]])),
    ("What was going on at about 3:40 pm?", _ex("identify", kind="clock", clock="15:40")),
    ("Was the user upright 12 minutes in?",
     _ex("verify", [["standing_in_place", "standing_moving"]], kind="at", a=12, unit="minutes")),
    ("Could this person have been jogging to work?", _ex("open", topic="commuting")),
    ("Did the wearer get any sleep during the night?", _ex("open", topic="sleeping", kind="period", period="night")),
    ("Was the user doing a workout?", _ex("open", topic="exercise")),
    ("Did they lose their balance at any point?", _ex("open", topic="fall")),
    ("Was the user sitting at a computer all morning?", _ex("open", topic="desk_work", kind="period", period="morning")),
    ("Did the person walk continuously for a long stretch?", _ex("prolonged", [["walking"]])),
    ("Was the user bicycling during the first 5 minutes?",
     _ex("verify", [["bicycling"]], kind="first", a=5, unit="minutes")),
    ("How long did the person stand during the first hour?",
     _ex("duration", [["standing_in_place", "standing_moving"]], kind="first", a=1, unit="hours")),
    ("Does the data fit with the user doing housework?", _ex("open", topic="chores")),
    ("How frequently did the person cycle?", _ex("count", [["bicycling"]])),
    ("Is the person lying down from 50 to 80 seconds?",
     _ex("verify", [["lying_down"]], kind="between", a=50, b=80, unit="seconds")),
    ("What was the user's blood sugar?", _ex("open", topic="blood")),
]

NUM_OR_NULL = {"anyOf": [{"type": "number"}, {"type": "null"}]}
# llama.cpp turns this into a grammar, so the model can only emit known operations, classes and fields
INTENT_SCHEMA = {
    "type": "object", "required": ["op", "activities", "time", "topic"], "additionalProperties": False,
    "properties": {
        "op": {"enum": list(OPS)},
        "activities": {"type": "array", "maxItems": 2,
                       "items": {"type": "array", "minItems": 1, "maxItems": 4, "items": {"enum": list(C.CLASSES)}}},
        "time": {"anyOf": [{"type": "null"}, {
            "type": "object", "required": list(NO_TIME), "additionalProperties": False,
            "properties": {"kind": {"enum": TIME_KINDS[1:]}, "a": NUM_OR_NULL, "b": NUM_OR_NULL,
                           "unit": {"enum": ["seconds", "minutes", "hours", None]},
                           "clock": {"anyOf": [{"type": "string", "pattern": "^[0-9]{1,2}:[0-9]{2}$"},
                                               {"type": "null"}]},
                           "period": {"enum": PERIODS + [None]}}}]},
        "topic": {"enum": TOPICS}}}


TIME_WORDS = re.compile(r"\d|\b(seconds?|secs?|minutes?|mins?|hours?|hrs?|noon|midday|midnight|half|quarter)\b")


def grounded_time(t: dict | None, question: str) -> dict | None:
    """Drop a time expression the question does not contain (the model sometimes adds a day period
    or clock time to questions that name none): the model may only copy times, never add them."""
    if not t:
        return None
    q = question.lower()
    if t.get("kind") == "period":
        return t if t.get("period") and t["period"] in q else None
    if t.get("kind") == "clock":
        hh = (t.get("clock") or "").split(":")[0].lstrip("0") or "0"
        return t if re.search(rf"\b{hh}\b|\b{int(hh) % 12 or 12}\b|noon|midday|midnight", q) else None
    return t if TIME_WORDS.search(q) else None


def time_range(t: dict, duration: float, t0: float | None) -> tuple[float, float] | None:
    """Seconds from the recording start for the time expression the model extracted."""
    if not t:
        return None
    kind, a, b = t.get("kind", "none"), t.get("a"), t.get("b")
    unit = t.get("unit") or "minutes"
    sec = (lambda v: _to_seconds(float(v), unit)) if unit else None
    if kind == "between" and a is not None and b is not None:
        return sec(a), sec(b)
    if kind == "after" and a is not None:
        return min(duration, sec(a)), duration
    if kind in ("before", "first") and a is not None:
        return 0.0, sec(a)
    if kind == "last" and a is not None:
        return max(0.0, duration - sec(a)), duration
    if kind == "at" and a is not None:
        c = sec(a)
        return max(0.0, c - 60), c + 60
    if kind == "clock" and t.get("clock"):
        hh, mm = (int(x) for x in t["clock"].split(":"))
        c = _clock_to_rel(hh % 24, mm % 60, 0, t0)
        return None if c is None else (max(0.0, c - 1800), c + 1800)
    if kind == "period" and t.get("period") in PERIOD_HOURS and t0 is not None:
        s, e = _period_to_rel(*PERIOD_HOURS[t["period"]], t0)
        return max(0.0, s), e
    return None


# The two language models the user can pick from (4-bit GGUF files in models/slm/); the first is the default
SLM_MODELS = {"Qwen3-1.7B": C.MODELS_DIR / "slm" / "Qwen3-1.7B-Q4_K_M.gguf",
              "Qwen3-0.6B": C.MODELS_DIR / "slm" / "Qwen3-0.6B-Q4_K_M.gguf"}
DEFAULT_SLM = SLM_MODELS["Qwen3-1.7B"]
THINK = re.compile(r"<think>.*?(</think>|$)", re.S)


class SLM:
    """One loaded small model, shared by the question parser and the explanation writer.

    A .gguf path runs through llama.cpp (llama-cpp-python, Metal on Apple silicon); the default is
    Qwen3-1.7B quantised to 4 bits (Q4_K_M, 1.0 GB). Anything else is taken as a Hugging Face model
    id and run with transformers. Qwen3's thinking mode is switched off with /no_think: the tasks
    are short, and reasoning tokens would multiply the latency.
    """

    def __init__(self, model=DEFAULT_SLM, device: str = "cpu"):
        self.name = Path(model).name
        self.qwen3 = "qwen3" in self.name.lower()
        if str(model).endswith(".gguf"):
            from llama_cpp import Llama
            from llama_cpp import LlamaRAMCache
            self.llama = Llama(model_path=str(model), n_ctx=4096, n_gpu_layers=-1, seed=0, verbose=False)
            self.llama.set_cache(LlamaRAMCache())     # reuse the fixed prompt prefix across questions
        else:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.llama, self.torch, self.device = None, torch, device
            self.tok = AutoTokenizer.from_pretrained(model)
            self.model = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32).to(device).eval()

    def chat(self, msgs: list[dict], max_new_tokens: int, schema: dict | None = None) -> str:
        if self.qwen3:
            msgs = msgs[:-1] + [{**msgs[-1], "content": msgs[-1]["content"] + " /no_think"}]
        if self.llama is not None:
            out = self.llama.create_chat_completion(
                msgs, max_tokens=max_new_tokens, temperature=0.0,
                response_format={"type": "json_object", "schema": schema} if schema else None)
            text = out["choices"][0]["message"]["content"] or ""
        else:
            ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                               return_dict=True).to(self.device)
            with self.torch.no_grad():
                out = self.model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                                          pad_token_id=self.tok.eos_token_id)
            text = self.tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        return THINK.sub("", text).strip()


class LLMParser:
    """Question -> Intent, by the small language model alone (no rule-based parsing)."""

    def __init__(self, model=DEFAULT_SLM, device: str = "cpu"):
        self.slm = model if isinstance(model, SLM) else SLM(model, device)
        self.calls = self.repairs = 0

    def extract(self, question: str) -> dict:
        msgs = [{"role": "system", "content": SYSTEM}]
        for q, a in EXAMPLES:
            msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": json.dumps(a)}]
        msgs.append({"role": "user", "content": question})
        self.calls += 1
        return json.loads(self.slm.chat(msgs, 160, schema=INTENT_SCHEMA))

    def __call__(self, question: str, duration: float = float("inf"), t0: float | None = None) -> Intent:
        obj = self.extract(question)
        groups = []
        for g in obj["activities"]:
            g = list(dict.fromkeys(g))
            if g not in groups:
                groups.append(g)
        op = obj["op"]
        if op not in ("identify", "open") and not groups:
            # an operation that needs an activity but got none: describe the period instead
            op, self.repairs = "identify", self.repairs + 1
        elif op == "compare" and len(groups) < 2:
            # a comparison needs two activities: report the time of the one that was named
            op, self.repairs = "duration", self.repairs + 1
        topic = obj.get("topic", "none")
        return Intent(op=op, activities=groups if op != "identify" else [], phrases=[],
                      t_range=time_range(grounded_time(obj["time"], question), duration, t0), question=question,
                      topic=None if topic == "none" else topic)


EXPLAIN_SYSTEM = """You explain the answer of a wearable motion-sensor system to a non-expert in 2-3 sentences.
Use ONLY the facts given. Keep every number exactly as written; do not add, round or invent numbers,
times or activities. Do not change the answer. Plain text only."""

NUM = re.compile(r"\d+(?:\.\d+)?")


class LLMExplainer:
    def __init__(self, slm: SLM):
        self.slm = slm
        self.calls = self.fallbacks = 0

    def __call__(self, question: str, ans) -> str:
        if ans.explanation in ("", "N/A") or "N/A" in (ans.answer, ans.timestamps):
            return ans.explanation      # no evidence to word: rewording an "N/A" invites a made-up verdict
        facts = (f"Question: {question}\nAnswer: {ans.answer}\nActivity: {ans.activity}\n"
                 f"Time spans (s): {ans.timestamps}\nSensors: {ans.modality}\n"
                 f"Measured evidence: {ans.explanation}")
        self.calls += 1
        text = self.slm.chat([{"role": "system", "content": EXPLAIN_SYSTEM},
                              {"role": "user", "content": facts}], 120)
        allowed = set(NUM.findall(facts))
        if not text or any(n not in allowed for n in NUM.findall(text)):
            self.fallbacks += 1          # a number the computation did not produce: keep the template
            return ans.explanation
        return text
