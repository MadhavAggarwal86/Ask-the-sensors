"""End-to-end QA evaluation on held-out users.

Produces:
  Figure 1  fig1_accuracy_by_type.png   answer accuracy and grounded accuracy per question type
  Figure 3  fig3_strictness.png         acceptance vs duration tolerance and vs IoU threshold
  Figure 5  fig5_robustness.png         (with --robustness) accuracy vs input degradation
  qa_metrics.json, qa_records.jsonl, sample_answers.txt
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ats import config as C
from ats.dataset import held_out_recordings
from ats.eval import metrics as M
from ats.eval.run import ask_generated
from ats.pipeline import SensorQA
from ats.preprocess import degrade
from ats.qa.format import to_text

RULES = {"identification": "exact", "verification": "exact", "comparison": "exact", "prolonged": "exact",
         "open_world": "exact", "duration": f"±max({M.DUR_ABS_TOL_S:.0f}s, {M.DUR_REL_TOL:.0%})",
         "count": f"±{M.COUNT_TOL}", "grounding": f"IoU≥{M.IOU_THRESHOLD}"}


def evaluate(qa: SensorQA, items, transform=None):
    """items: list of (rec_id, rec, gt). Returns per-question records."""
    records = []
    for rid, rec, gt in items:
        if transform:
            rec = transform(rec)
        qa.load(rec)
        records += ask_generated(qa, rid, gt, rec.duration, rec.has_gyro)
    return records


def fig_by_type(sc, path, title):
    types = list(sc["per_type"])
    x = np.arange(len(types) + 1)
    acc = [sc["per_type"][t] for t in types] + [sc["overall_macro"]]
    gnd = [sc["grounded_per_type"][t] for t in types] + [sc["grounded_macro"]]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - 0.2, acc, 0.4, label="Answer correct")
    ax.bar(x + 0.2, gnd, 0.4, label="Answer correct and grounded")
    ax.set_xticks(x, [f"{t}\n({RULES.get(t, '')})" for t in types] + ["overall\n(macro)"], fontsize=8)
    ax.set_ylim(0, 1.2)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend(loc="upper center", ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_strictness(records, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    dur = [r for r in records if r["qtype"] == "duration" and r["ref"].value]
    tols = np.linspace(0, 0.5, 11)
    a1.plot(tols * 100, [np.mean([abs(r["pred"].value - r["ref"].value) <= t * r["ref"].value for r in dur])
                         if dur else np.nan for t in tols], marker="o")
    a1.set_xlabel("Relative error tolerance (%)")
    a1.set_ylabel("Fraction accepted")
    a1.set_title("Duration answers")
    thr = np.linspace(0.1, 0.9, 9)
    cited = [r for r in records if r["ref"].intervals]
    a2.plot(thr, [np.mean([M.interval_iou(r["pred"].intervals, r["ref"].intervals) >= t for r in cited])
                  if cited else np.nan for t in thr], marker="o", label="all cited evidence")
    onset = [r for r in cited if r["qtype"] == "grounding"]
    a2.plot(thr, [np.mean([M.interval_iou(r["pred"].intervals, r["ref"].intervals) >= t for r in onset])
                  if onset else np.nan for t in thr], marker="s", label="onset intervals")
    a2.set_xlabel("IoU threshold")
    a2.set_title("Temporal answers and cited intervals")
    a2.legend()
    for a in (a1, a2):
        a.set_ylim(0, 1.05)
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["extrasensory", "synthetic"], default="extrasensory")
    ap.add_argument("--model", default="hgb_full")
    ap.add_argument("--max-recordings", type=int, default=40)
    ap.add_argument("--smoothing", choices=["hmm", "moving_average"], default=C.SMOOTHING)
    ap.add_argument("--robustness", action="store_true")
    args = ap.parse_args()
    out = C.RESULTS_DIR / args.source
    out.mkdir(parents=True, exist_ok=True)
    qa = SensorQA.from_path(C.MODELS_DIR / f"{args.model}.joblib", smoothing=args.smoothing)
    tag = f"{args.model}_{args.smoothing}"

    items = held_out_recordings(args.source, args.max_recordings)
    records = evaluate(qa, items)
    sc = M.score(records)
    sc["latency_ms_mean"] = float(np.mean([r["latency_ms"] for r in records]))
    dur = [r for r in records if r["qtype"] == "duration"]
    sc["duration_mae_s"] = float(np.mean([abs(r["pred"].value - r["ref"].value) for r in dur])) if dur else None
    gp = [M.overlap_fraction(r["pred"].intervals, r["ref"].intervals) for r in records
          if r["pred"].intervals and r["qtype"] in ("duration", "grounding")]
    sc["grounding_precision"] = float(np.nanmean(gp)) if gp else None
    sc["recordings"] = len(items)
    print(json.dumps(sc, indent=2))
    (out / f"qa_metrics_{tag}.json").write_text(json.dumps(sc, indent=2))
    with open(out / f"qa_records_{tag}.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps({"rec": r["rec"], "qtype": r["qtype"], "question": r["question"],
                                "pred": r["pred"].answer, "ref": r["ref"].answer, "correct": r["correct"],
                                "grounded": r["grounded"]}) + "\n")
    with open(out / f"sample_answers_{tag}.txt", "w") as f:
        seen = set()
        for r in records:
            if r["qtype"] not in seen:
                seen.add(r["qtype"])
                f.write(f"[{r['qtype']}] reference: {r['ref'].answer}\n{to_text(r['question'], r['pred'])}\n\n")
    fig_by_type(sc, out / f"fig1_accuracy_by_type_{tag}.png",
                f"QA accuracy by question type ({args.source}, {len(items)} held-out recordings, {args.smoothing})")
    fig_strictness(records, out / f"fig3_strictness_{tag}.png")

    if args.robustness:
        subset = items[: min(10, len(items))]
        curves = {
            "Gaussian noise (g)": [(v, lambda r, v=v: degrade(r, noise_g=v)) for v in (0, 0.02, 0.05, 0.1, 0.2)],
            "Dropped samples (%)": [(v, lambda r, v=v: degrade(r, drop_frac=v / 100)) for v in (0, 10, 30, 50, 70)],
            "Sampling rate (Hz)": [(v, lambda r, v=v: degrade(r, fs_low=v)) for v in (25, 15, 10, 5)],
        }
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
        rob = {}
        for ax, (name, pts) in zip(axes, curves.items()):
            ys = [M.score(evaluate(qa, subset, fn))["overall_macro"] for _, fn in pts]
            rob[name] = dict(zip([p[0] for p in pts], ys))
            ax.plot([p[0] for p in pts], ys, marker="o")
            ax.set_xlabel(name)
            ax.set_ylim(0, 1.05)
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("Overall QA accuracy (macro)")
        fig.suptitle(f"Robustness on {len(subset)} fixed recordings ({args.smoothing})")
        fig.tight_layout()
        fig.savefig(out / f"fig5_robustness_{tag}.png", dpi=150)
        (out / f"robustness_{tag}.json").write_text(json.dumps(rob, indent=2))
        print(json.dumps(rob, indent=2))


if __name__ == "__main__":
    main()
