"""Tune the temporal decoder (config.HMM_*) on held-out TRAINING users; test users stay unseen.

A validation model is trained on part of the training users, then every decoder setting is scored
on the rest with window macro-F1 and with the QA rules that suffer from a fragmented timeline:
per-activity bout count within ±1 and total duration within ±max(30 s, 10%).
Writes results/<source>/smoothing_tuning.json and fig_smoothing_tuning.png.
"""
import argparse
import itertools
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score

from ats import config as C
from ats.dataset import load_processed, split_users, training_set
from ats.eval import metrics as M
from ats.model import ActivityModel
from ats.timeline import Timeline, intervals_from_labels, smooth_probs, viterbi


def timeline_stats(starts, y):
    tl = Timeline(intervals_from_labels(starts, [C.CLASSES[i] for i in y]))
    return {a: (len(tl.bouts(a)), tl.total(a)) for a in C.CLASSES}


def score(recs, preds):
    """recs: list of (starts, y_true, truth_stats); preds: matching list of predicted class arrays."""
    y_true = np.concatenate([r[1] for r in recs])
    y_pred = np.concatenate(preds)
    count_ok, dur_ok, n_true, n_pred = [], [], 0, 0
    for (starts, _, truth), p in zip(recs, preds):
        pred = timeline_stats(starts, p)
        for a, (n, d) in truth.items():
            if n == 0:
                continue
            count_ok.append(abs(pred[a][0] - n) <= M.COUNT_TOL)
            dur_ok.append(abs(pred[a][1] - d) <= max(M.DUR_ABS_TOL_S, M.DUR_REL_TOL * d))
            n_true += n
            n_pred += pred[a][0]
    out = {"macro_f1": float(f1_score(y_true, y_pred, labels=range(len(C.CLASSES)), average="macro",
                                      zero_division=0)),
           "count_acc": float(np.mean(count_ok)), "duration_acc": float(np.mean(dur_ok)),
           "bouts_pred_per_true": n_pred / max(n_true, 1)}
    out["objective"] = (out["macro_f1"] + out["count_acc"] + out["duration_acc"]) / 3
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["extrasensory", "synthetic"], default="extrasensory")
    ap.add_argument("--val-users", type=int, default=10)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    train_users = split_users(args.source)["train"]
    val = sorted(np.random.default_rng(1).choice(train_users, args.val_users, replace=False).tolist())
    fit_users = [u for u in train_users if u not in val]

    X, y, g, names = load_processed(args.source, fit_users)
    prior = np.bincount(y, minlength=len(C.CLASSES)) / len(y)
    print("class prior (fit users):", {c: round(float(p), 4) for c, p in zip(C.CLASSES, prior)})
    model = ActivityModel.new().fit(*training_set(X, y, g, names, rng), names)
    del X, y, g

    recs, probs = [], []
    for u in val:
        d = np.load(C.PROCESSED_DIR / args.source / f"{u}.npz")
        for r in np.unique(d["rec"]):
            m = d["rec"] == r
            recs.append((d["starts"][m], d["y"][m], timeline_stats(d["starts"][m], d["y"][m])))
            probs.append(model.predict_proba(d["X"][m]))
    print(f"validation: {len(val)} users, {len(recs)} recordings, {sum(len(r[1]) for r in recs)} windows")

    results = [{"method": "moving_average",
                **score(recs, [smooth_probs(r[0], p).argmax(1) for r, p in zip(recs, probs)])}]
    print(results[0])
    taus = [15, 30, 60, 120, 300, 600, 1200, 2400, 4800]
    for tau, pre, w in itertools.product(taus, (1, C.SMOOTH_WINDOWS), (0.0, 0.5, 1.0)):
        preds = [viterbi(r[0], smooth_probs(r[0], p, pre) if pre > 1 else p, tau, w * np.log(prior))
                 for r, p in zip(recs, probs)]
        results.append({"method": "hmm", "tau_s": tau, "presmooth": pre, "prior_weight": w, **score(recs, preds)})
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in results[-1].items()})

    best = max(results, key=lambda r: r["objective"])
    print("best:", best)
    out = C.RESULTS_DIR / args.source
    out.mkdir(parents=True, exist_ok=True)
    (out / "smoothing_tuning.json").write_text(json.dumps(
        {"val_users": val, "class_prior": dict(zip(C.CLASSES, prior.tolist())), "best": best,
         "results": results}, indent=2))

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharex=True)
    keys = ["macro_f1", "count_acc", "duration_acc", "bouts_pred_per_true"]
    for ax, k in zip(axes, keys):
        for pre, w in itertools.product((1, C.SMOOTH_WINDOWS), (0.0, 0.5, 1.0)):
            rs = [r for r in results[1:] if r["presmooth"] == pre and r["prior_weight"] == w]
            ax.plot(taus, [r[k] for r in rs], marker="o", ms=3, label=f"pre-smooth {pre}, prior w {w}")
        ax.axhline(results[0][k], color="k", ls="--", lw=1, label="moving average (baseline)")
        ax.set_xscale("log")
        ax.set_xlabel("HMM expected bout length τ (s)")
        ax.set_title(k)
        ax.grid(alpha=0.3)
    axes[3].set_yscale("log")
    axes[0].legend(fontsize=6)
    fig.suptitle(f"Temporal decoder tuning on {len(val)} held-out training users")
    fig.tight_layout()
    fig.savefig(out / "fig_smoothing_tuning.png", dpi=150)


if __name__ == "__main__":
    main()
