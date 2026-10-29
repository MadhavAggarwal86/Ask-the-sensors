"""Intent accuracy of a question parser on the hand-written question sets (eval_sets/).

The evaluation questions are hidden until grading, so the parser must cope with phrasings it has
never seen. Two sets were written before any parser hardening: "dev" was then used to improve the
parser, "test" was only scored, never inspected question by question. Each question is annotated
with the intent it should produce; a parse is correct when the operation, the activity groups
(order-insensitive) and the time range (to 1 s) all match. Ranges assume a 2-hour recording that
starts at 08:00 local time.
"""
import argparse
import datetime as dt
import json
import time

from ats import config as C
from ats.qa.parser import parse

DURATION = 7200.0
T0 = dt.datetime(2015, 8, 1, 8, 0).timestamp()
SETS = C.ROOT / "eval_sets"


def match(it, ref) -> dict:
    acts_ok = ref["activities"] is None or \
        sorted(sorted(g) for g in it.activities) == sorted(sorted(g) for g in ref["activities"])
    r, e = it.t_range, ref["t_range"]
    range_ok = (r is None and e is None) or (r is not None and e is not None
                                             and all(abs(a - b) <= 1 for a, b in zip(r, e)))
    ok = {"op": it.op == ref["op"], "activities": acts_ok, "t_range": range_ok}
    ok["intent"] = all(ok.values())
    return ok


def load_parser(name: str, mode: str = "hybrid", model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    if name == "rule":
        return parse
    from ats.qa.llm import LLMParser
    return LLMParser(model_id, mode)


def evaluate(parser, split: str, show_failures: bool = False) -> dict:
    refs = [json.loads(line) for line in (SETS / f"parser_{split}.jsonl").read_text().splitlines()]
    rows, times = [], []
    for ref in refs:
        t = time.perf_counter()
        it = parser(ref["question"], duration=DURATION, t0=T0)
        times.append(time.perf_counter() - t)
        ok = match(it, ref)
        rows.append({**ref, **{f"ok_{k}": v for k, v in ok.items()}})
        if show_failures and not ok["intent"]:
            print(f"  FAIL {ref['question']!r}\n       got  op={it.op} acts={it.activities} range={it.t_range}\n"
                  f"       want op={ref['op']} acts={ref['activities']} range={ref['t_range']}")
    out = {k: sum(r[f"ok_{k}"] for r in rows) / len(rows) for k in ("intent", "op", "activities", "t_range")}
    out["per_tier"] = {t: sum(r["ok_intent"] for r in rows if r["tier"] == t) / sum(r["tier"] == t for r in rows)
                       for t in dict.fromkeys(r["tier"] for r in rows)}
    out["latency_ms_mean"] = 1000 * sum(times) / len(times)
    out["n"] = len(rows)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parser", choices=["rule", "llm"], default="rule")
    ap.add_argument("--split", choices=["dev", "test", "both"], default="dev")
    ap.add_argument("--llm-mode", choices=["llm", "hybrid"], default="hybrid")
    ap.add_argument("--llm-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--show-failures", action="store_true", help="list failed questions (use on dev only)")
    ap.add_argument("--tag", default="", help="suffix for the results file, e.g. 'before'")
    args = ap.parse_args()
    parser = load_parser(args.parser, args.llm_mode, args.llm_model)
    name = args.parser if args.parser == "rule" else f"{args.llm_mode}_{args.llm_model.split('/')[-1]}"
    out = C.RESULTS_DIR / "parser"
    out.mkdir(parents=True, exist_ok=True)
    for split in (["dev", "test"] if args.split == "both" else [args.split]):
        res = evaluate(parser, split, args.show_failures)
        if args.parser == "llm":
            res["llm_calls"], res["llm_fallbacks"] = parser.calls, parser.fallbacks
            parser.calls = parser.fallbacks = 0
        print(split, json.dumps(res, indent=2))
        (out / f"parser_{name}_{split}{'_' + args.tag if args.tag else ''}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
