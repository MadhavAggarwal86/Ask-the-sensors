"""Grade explanations with the fixed 1-5 rubric (ats.eval.rubric), by a language model and by people.

    # 1. draw a fixed sample and score it with the LLM judge
    python scripts/grade_explanations.py judge
    # 2. write a blank sheet per human grader (same sample, same rubric), fill in the score columns
    python scripts/grade_explanations.py sheet --grader alice
    # 3. agreement between any graders (LLM and/or filled-in sheets)
    python scripts/grade_explanations.py agree results/extrasensory/rubric/*.csv

Inputs come from scripts/evaluate.py (qa_records_<tag>.jsonl); the sample is stratified over the
question types whose explanation is assessed (open-world, prolonged, onset) and seeded, so every
grader sees exactly the same items.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from ats import config as C
from ats.eval.rubric import CRITERIA, RUBRIC, LLMJudge

TYPES = ("open_world", "prolonged", "grounding")
FIELDS = ("question", "pred", "pred_activity", "pred_timestamps", "pred_modality", "pred_channels",
          "pred_explanation", "ref")


def sample(records_path: Path, per_type: int, seed: int) -> list[dict]:
    rs = [json.loads(l) for l in records_path.read_text().splitlines()]
    rng = np.random.default_rng(seed)
    out = []
    for t in TYPES:
        pool = [r for r in rs if r["qtype"] == t]
        for i in sorted(rng.choice(len(pool), min(per_type, len(pool)), replace=False)):
            out.append({"id": f"{t}-{i}", "qtype": t, **{k: pool[i][k] for k in FIELDS},
                        "answer_correct": pool[i]["correct"]})
    return out


def write_sheet(items, path: Path, scores=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, ["id", "qtype", *FIELDS, *CRITERIA])
        w.writeheader()
        for it in items:
            s = (scores or {}).get(it["id"]) or {}
            w.writerow({k: it[k] for k in ("id", "qtype", *FIELDS)} | {c: s.get(c, "") for c in CRITERIA})


def read_sheet(path: Path) -> dict:
    with open(path) as f:
        return {r["id"]: {c: int(r[c]) for c in CRITERIA} for r in csv.DictReader(f)
                if all(r[c].strip() for c in CRITERIA)}


def weighted_kappa(a, b, k=5) -> float:
    """Quadratic-weighted Cohen's kappa for two lists of 1..k scores."""
    a, b = np.asarray(a) - 1, np.asarray(b) - 1
    obs = np.zeros((k, k))
    np.add.at(obs, (a, b), 1)
    exp = np.outer(obs.sum(1), obs.sum(0)) / obs.sum()
    w = np.subtract.outer(np.arange(k), np.arange(k)) ** 2 / (k - 1) ** 2
    den = (w * exp).sum()
    return float(1 - (w * obs).sum() / den) if den else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["judge", "sheet", "agree"])
    ap.add_argument("sheets", nargs="*", type=Path, help="for agree: two or more graded sheets")
    ap.add_argument("--records", type=Path, default=C.RESULTS_DIR / "extrasensory" / "qa_records_hgb_full_hmm.jsonl")
    ap.add_argument("--per-type", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grader", default="human")
    ap.add_argument("--judge-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--device", default="cpu", help="cpu (bfloat16), cuda or mps")
    args = ap.parse_args()
    out = C.RESULTS_DIR / "extrasensory" / "rubric"
    items = sample(args.records, args.per_type, args.seed)

    if args.cmd == "sheet":
        path = out / f"rubric_{args.grader}.csv"
        write_sheet(items, path)
        (out / "RUBRIC.txt").write_text(RUBRIC + "\n")
        print(f"wrote {path} ({len(items)} items) and {out / 'RUBRIC.txt'}: fill in {', '.join(CRITERIA)} (1-5)")

    elif args.cmd == "judge":
        judge = LLMJudge(args.judge_model, args.device)
        scores, raw, t0 = {}, [], time.perf_counter()
        for it in items:
            s, text = judge(it)
            scores[it["id"]] = s
            raw.append({"id": it["id"], "reply": text, "scores": s})
        name = args.judge_model.split("/")[-1]
        write_sheet(items, out / f"rubric_llm_{name}.csv", scores)
        valid = [s for s in scores.values() if s]
        summary = {"judge": args.judge_model, "items": len(items), "valid": len(valid),
                   "seconds": time.perf_counter() - t0,
                   "mean": {c: float(np.mean([s[c] for s in valid])) for c in CRITERIA},
                   "mean_overall": float(np.mean([np.mean(list(s.values())) for s in valid])),
                   "by_type": {t: {c: float(np.mean([scores[i["id"]][c] for i in items
                                                     if i["qtype"] == t and scores[i["id"]]]))
                                   for c in CRITERIA} for t in TYPES},
                   "by_answer_correct": {str(k): float(np.mean([np.mean(list(scores[i["id"]].values()))
                                                                for i in items if i["answer_correct"] == k
                                                                and scores[i["id"]]]))
                                         for k in (True, False)}}
        (out / f"rubric_llm_{name}.json").write_text(json.dumps({"summary": summary, "replies": raw}, indent=2))
        print(json.dumps(summary, indent=2))

    else:
        sheets = {p.stem: read_sheet(p) for p in args.sheets}
        names = list(sheets)
        res = {}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                common = sorted(set(sheets[a]) & set(sheets[b]))
                if not common:
                    continue
                res[f"{a} vs {b}"] = {"items": len(common), **{
                    c: {"weighted_kappa": weighted_kappa([sheets[a][k][c] for k in common],
                                                         [sheets[b][k][c] for k in common]),
                        "within_1": float(np.mean([abs(sheets[a][k][c] - sheets[b][k][c]) <= 1 for k in common]))}
                    for c in CRITERIA}}
        res["mean_score"] = {n: float(np.mean([np.mean(list(s.values())) for s in sheets[n].values()]))
                             for n in names if sheets[n]}
        (out / "rubric_agreement.json").write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
