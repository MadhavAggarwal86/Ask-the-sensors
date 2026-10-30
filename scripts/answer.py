"""Answer questions about a recording in the required output format.

    python scripts/answer.py --recording rec.csv --question "How long was the user walking?"
    python scripts/answer.py --recording rec.csv --questions questions.txt --json out.jsonl
    python scripts/answer.py --recording rec.csv --question "..." --explainer template   # fixed-sentence explanation

The small language model (Qwen3-1.7B, 4-bit, llama.cpp) reads every question (intent extraction),
the Python pipeline computes every number, and the model then words the explanation from those
computed values. A web front end with the same pipeline is app.py (streamlit run app.py).

Timestamps are reported in seconds from the start of the recording.
"""
import argparse
from pathlib import Path

from ats import config as C
from ats.io import load_csv
from ats.pipeline import SensorQA
from ats.qa.format import to_json, to_text
from ats.qa.llm import SLM, SLM_MODELS, LLMExplainer, LLMParser


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", type=Path, required=True, help="CSV: t, acc_x..z, gyro_x..z")
    ap.add_argument("--question", action="append", default=[])
    ap.add_argument("--questions", type=Path, help="text file, one question per line")
    ap.add_argument("--model", type=Path, default=C.MODELS_DIR / "hgb_full.joblib")
    ap.add_argument("--json", type=Path, help="also write JSON lines here")
    ap.add_argument("--explainer", choices=["slm", "template"], default="slm",
                    help="slm: the SLM words the explanation from the computed values; template: fixed sentences")
    ap.add_argument("--slm", choices=list(SLM_MODELS), default="Qwen3-1.7B",
                    help="language model: Qwen3-1.7B (more accurate) or Qwen3-0.6B (faster)")
    ap.add_argument("--llm-model", help="any other .gguf file or Hugging Face model id (overrides --slm)")
    args = ap.parse_args()

    qs = list(args.question)
    if args.questions:
        qs += [l.strip() for l in args.questions.read_text().splitlines() if l.strip()]
    slm = SLM(args.llm_model or SLM_MODELS[args.slm])
    qa = SensorQA.from_path(args.model, parser=LLMParser(slm),
                            explainer=LLMExplainer(slm) if args.explainer == "slm" else None)
    qa.load(load_csv(args.recording))
    print(f"# Time base: {C.TIME_BASE}\n")
    lines = []
    for q in qs:
        a = qa.ask(q)
        print(to_text(q, a), "\n")
        lines.append(to_json(q, a))
    if args.json:
        args.json.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
