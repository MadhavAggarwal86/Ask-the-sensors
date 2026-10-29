"""Optional small-language-model front end: question -> the same Intent the rule parser produces.

The model only decides the operation and the activity groups, the part of a question where
phrasing varies freely. Time ranges stay with the deterministic extractor in parser.py: small
models are unreliable at turning "around 10:15" into seconds from the recording start. The model
output is validated against the Intent schema; anything invalid falls back to the rule parser.

mode="llm"     the model parses every question
mode="hybrid"  the rule parser answers first; the model is consulted only when the rule parse
               looks uncertain (no operation matched, or an operation without its activity)
"""
import json
import re

from .. import config as C
from .parser import OPEN_WORLD_CUES, OPS, Intent, _find_range, _normalise, parse

GROUPS = {"standing": ["standing_in_place", "standing_moving"],
          "resting": ["lying_down", "sitting", "standing_in_place"],
          "sedentary": ["lying_down", "sitting"],
          "active": ["standing_moving", "walking", "running", "bicycling"],
          "strenuous": ["running", "bicycling"],
          "on_foot": ["walking", "running"]}

SYSTEM = f"""You convert questions about a person's wearable motion recording into JSON.
Return exactly one JSON object: {{"op": OP, "activities": [[CLASS, ...], ...]}}
OP is one of:
  identify  - which activity happened (no activity named)
  verify    - yes/no: was a named activity happening
  duration  - how long / total time / share of time of an activity
  count     - how many times / bouts / episodes of an activity
  onset     - when an activity started
  compare   - which of two activities took more time
  longest   - the longest single bout of an activity
  prolonged - whether an activity lasted a long uninterrupted time
  open      - anything else: behaviours (exercise, sleep, commuting, chores, falls) or quantities
              motion sensors cannot measure
CLASS is one of: {", ".join(C.CLASSES)}.
"activities" lists one group per activity mentioned, in order. A group with several classes means
any of them: "standing" = ["standing_in_place", "standing_moving"]; "resting" or "idle" =
["lying_down", "sitting", "standing_in_place"]; "sedentary" = ["lying_down", "sitting"];
"active" = ["standing_moving", "walking", "running", "bicycling"]; "strenuous" = ["running", "bicycling"].
Use [] when no activity is named. Output JSON only."""

# Few-shot examples, deliberately different from the evaluation question sets.
EXAMPLES = [
    ("What is the wearer doing between 100 and 160 seconds?", {"op": "identify", "activities": []}),
    ("Did the person do any sprinting?", {"op": "verify", "activities": [["running"]]}),
    ("How much time went to cycling overall?", {"op": "duration", "activities": [["bicycling"]]}),
    ("How many separate strolls were there?", {"op": "count", "activities": [["walking"]]}),
    ("When did the person first sit?", {"op": "onset", "activities": [["sitting"]]}),
    ("Was there more lying than standing?",
     {"op": "compare", "activities": [["lying_down"], ["standing_in_place", "standing_moving"]]}),
    ("What was the biggest single stretch of running?", {"op": "longest", "activities": [["running"]]}),
    ("Did the person stay seated for ages without getting up?", {"op": "prolonged", "activities": [["sitting"]]}),
    ("Could this person have been gardening?", {"op": "open", "activities": []}),
]


def _validate(obj) -> tuple[str, list[list[str]]] | None:
    if not isinstance(obj, dict) or obj.get("op") not in OPS:
        return None
    acts = obj.get("activities", [])
    if not isinstance(acts, list):
        return None
    groups = []
    for g in acts:
        g = [g] if isinstance(g, str) else g
        if not isinstance(g, list):
            return None
        flat = []
        for a in g:
            if not isinstance(a, str):
                return None
            a = a.strip().lower().replace(" ", "_")
            flat += GROUPS.get(a, [a])
        if not flat or any(a not in C.CLASSES for a in flat):
            return None
        flat = list(dict.fromkeys(flat))
        if flat not in groups:
            groups.append(flat)
    return obj["op"], groups


class LLMParser:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-0.5B-Instruct", mode: str = "hybrid", device: str = "cpu"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.mode, self.device = torch, mode, device
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).to(device).eval()
        self.calls = self.fallbacks = 0

    def _generate(self, question: str) -> str:
        msgs = [{"role": "system", "content": SYSTEM}]
        for q, a in EXAMPLES:
            msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": json.dumps(a)}]
        msgs.append({"role": "user", "content": question})
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                           return_dict=True).to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=60, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)

    def _uncertain(self, it: Intent) -> bool:
        q = _normalise(it.question)
        no_op = it.op == "open" and not any(cue in q for cue in OPEN_WORLD_CUES)
        return no_op or (it.op == "identify" and bool(re.search(r"\b(or|than)\b", q)))

    def __call__(self, question: str, duration: float = float("inf"), t0: float | None = None) -> Intent:
        rule = parse(question, duration, t0)
        if self.mode == "hybrid" and not self._uncertain(rule):
            return rule
        self.calls += 1
        m = re.search(r"\{.*\}", self._generate(question), re.S)
        try:
            parsed = _validate(json.loads(m.group(0))) if m else None
        except json.JSONDecodeError:
            parsed = None
        if parsed is None:
            self.fallbacks += 1
            return rule
        op, groups = parsed
        if op not in ("identify", "open") and not groups:     # an operation needs its activity
            self.fallbacks += 1
            return rule
        return Intent(op=op, activities=groups if op != "identify" else [], phrases=[],
                      t_range=_find_range(_normalise(question), duration, t0), question=question)
