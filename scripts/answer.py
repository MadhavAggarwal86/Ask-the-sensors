"""Answer questions about a recording in the required output format.

    python scripts/answer.py --recording rec.csv --question "How long was the user walking?"
    python scripts/answer.py --recording rec.csv --questions questions.txt --json out.jsonl

Timestamps are reported in seconds from the start of the recording.
"""
import argparse
from pathlib import Path

from ats import config as C
from ats.io import load_csv
from ats.pipeline import SensorQA
from ats.qa.format import to_json, to_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", type=Path, required=True, help="CSV: t, acc_x..z, gyro_x..z")
    ap.add_argument("--question", action="append", default=[])
    ap.add_argument("--questions", type=Path, help="text file, one question per line")
    ap.add_argument("--model", type=Path, default=C.MODELS_DIR / "hgb_full.joblib")
    ap.add_argument("--json", type=Path, help="also write JSON lines here")
    args = ap.parse_args()

    qs = list(args.question)
    if args.questions:
        qs += [l.strip() for l in args.questions.read_text().splitlines() if l.strip()]
    qa = SensorQA.from_path(args.model)
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
