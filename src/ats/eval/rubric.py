"""Fixed 1-5 rubric for the explanation field (open-world, prolonged and onset answers).

The brief asks for the faithfulness and plausibility of explanations to be judged on a one-to-five
scale by human graders or by a language model given a fixed rubric. The same rubric text is shown to
both, so their scores can be compared (agreement is computed by scripts/grade_explanations.py).
"""
import json
import re

CRITERIA = ("features", "support", "plausible")

RUBRIC = """Score the explanation of a sensor question-answering system on three criteria, each 1-5.

features: Does the explanation cite real, specific signal evidence (time spans, measured values such as
  variability in g, cadence in Hz, rotation in rad/s, tilt in degrees, named sensors)?
  1 = no signal evidence, generic wording only; 3 = some evidence but vague or partial;
  5 = specific measured features tied to the cited time span.
support: Do the cited features actually support the stated answer (e.g. high variability and a
  1.5-2.5 Hz rhythm support walking; near-zero variability supports rest)?
  1 = features contradict or are unrelated to the answer; 3 = partly supportive or incomplete reasoning;
  5 = the features clearly and directly justify the answer.
plausible: Is the conclusion plausible given the reference (ground-truth) answer shown below?
  1 = contradicts the reference; 3 = partly right (e.g. right activity, wrong time); 5 = agrees with it."""

TEMPLATE = """{rubric}

Question: {question}
System answer: {pred}
Activity/Event: {pred_activity}
Timestamp(s): {pred_timestamps}
Sensor Modality: {pred_modality}; Sensor Channel(s): {pred_channels}
Explanation: {pred_explanation}
Reference answer (ground truth): {ref}

Reply with JSON only, for example {{"features": 4, "support": 3, "plausible": 5}}."""


def prompt(rec: dict) -> str:
    return TEMPLATE.format(rubric=RUBRIC, **rec)


def parse(text: str) -> dict | None:
    """Scores from a judge reply, or None if the reply is not a valid score object."""
    m = re.search(r"\{.*?\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    try:
        scores = {c: int(obj[c]) for c in CRITERIA}
    except (KeyError, TypeError, ValueError):
        return None
    return scores if all(1 <= v <= 5 for v in scores.values()) else None


class LLMJudge:
    """A small instruction-tuned model given the fixed rubric (greedy decoding, so it is repeatable)."""

    def __init__(self, model_id: str = "Qwen/Qwen2.5-1.5B-Instruct", device: str = "cpu"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.device = torch, device
        self.tok = AutoTokenizer.from_pretrained(model_id)
        # bfloat16 halves memory (1.5B -> ~3 GB), which is what lets it run on an 8 GB laptop
        dtype = torch.bfloat16 if device == "cpu" else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()

    def __call__(self, rec: dict) -> tuple[dict | None, str]:
        msgs = [{"role": "system", "content": "You are a strict, careful grader."},
                {"role": "user", "content": prompt(rec)}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                           return_dict=True).to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=40, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        text = self.tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        return parse(text), text
