"""Train the recognition backbone on training users; report window-level metrics on test users.

Produces Figure 2 (confusion matrix + per-class precision/recall/F1).
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score

from ats import config as C
from ats.dataset import load_processed, split_users, training_set
from ats.model import ActivityModel
from ats.timeline import decode


def plot_confusion(cm, path, title):
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    labels = [C.DISPLAY[c] for c in C.CLASSES]
    ax.set_xticks(range(7), labels, rotation=40, ha="right")
    ax.set_yticks(range(7), labels)
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted activity")
    ax.set_ylabel("True activity")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["extrasensory", "synthetic"], default="extrasensory")
    ap.add_argument("--cap", type=int, default=3000, help="max windows per user per class")
    ap.add_argument("--gyro-dropout", type=float, default=0.2, help="share of windows trained without gyro")
    ap.add_argument("--no-clean-labels", dest="clean_labels", action="store_false")
    ap.add_argument("--motion-min-std", type=float, default=0.01, help="g; below this a window shows no motion")
    ap.add_argument("--name", default="hgb_full")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    split = split_users(args.source)
    Xtr, ytr, gtr, names = load_processed(args.source, split["train"])
    Xte, yte, _, _ = load_processed(args.source, split["test"])
    Xfit, yfit = training_set(Xtr, ytr, gtr, names, rng, args.cap,
                              args.motion_min_std if args.clean_labels else None, args.gyro_dropout)
    print(f"test windows {len(yte)}")

    model = ActivityModel.new().fit(Xfit, yfit, names)
    model.save(C.MODELS_DIR / f"{args.name}.joblib")

    pred = model.predict_proba(Xte).argmax(1)
    labels = list(range(len(C.CLASSES)))
    report = classification_report(yte, pred, labels=labels, target_names=C.CLASSES, zero_division=0,
                                   output_dict=True)
    summary = {"accuracy": float((pred == yte).mean()),
               "macro_f1": float(f1_score(yte, pred, labels=labels, average="macro", zero_division=0)),
               "balanced_accuracy": float(balanced_accuracy_score(yte, pred)),
               "per_class": {c: report[c] for c in C.CLASSES}}
    print(classification_report(yte, pred, labels=labels, target_names=C.CLASSES, zero_division=0))
    # Temporally decoded predictions (config.SMOOTHING), as the QA timeline actually uses them.
    sm_true, sm_pred = [], []
    for u in split["test"]:
        d = np.load(C.PROCESSED_DIR / args.source / f"{u}.npz")
        for r in np.unique(d["rec"]):
            m = d["rec"] == r
            sm_pred.append(decode(d["starts"][m], model.predict_proba(d["X"][m])))
            sm_true.append(d["y"][m])
    sm_true, sm_pred = np.concatenate(sm_true), np.concatenate(sm_pred)
    summary["smoothed"] = {"macro_f1": float(f1_score(sm_true, sm_pred, labels=labels, average="macro", zero_division=0)),
                           "balanced_accuracy": float(balanced_accuracy_score(sm_true, sm_pred))}
    print("smoothed:", summary["smoothed"])
    out = C.RESULTS_DIR / args.source
    out.mkdir(parents=True, exist_ok=True)
    (out / f"window_metrics_{args.name}.json").write_text(json.dumps(summary, indent=2))
    plot_confusion(confusion_matrix(yte, pred, labels=labels), out / f"fig2_confusion_{args.name}.png",
                   f"Window-level confusion ({args.source}, test users)\n"
                   f"macro-F1 {summary['macro_f1']:.2f}, balanced acc {summary['balanced_accuracy']:.2f}")
    print("saved", C.MODELS_DIR / f"{args.name}.joblib", "and", out)


if __name__ == "__main__":
    main()
