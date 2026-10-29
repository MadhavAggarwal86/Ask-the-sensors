"""Draw the system architecture figure used in the report (report/figures/fig0_architecture.png)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "figures" / "fig0_architecture.png"

BOXES = [  # (x, title, body, colour)
    (0.0, "Preprocess", "units -> g\nanti-alias + 25 Hz grid\ngaps kept as NaN\n6 s windows, 3 s hop", "#dbe9f6"),
    (1.0, "Recognise", "35 named features\n(variability, cadence,\nperiodicity, tilt, rotation)\nHistGradientBoosting", "#dbe9f6"),
    (2.0, "Aggregate", "sticky-HMM Viterbi\nintervals, bouts, onsets\n= single source of truth\nfor every number", "#fde9c9"),
    (3.0, "Answer", "parse -> Intent\n(rules; optional SLM)\noperation on timeline\ntemplate explanation", "#e2f0d9"),
]


def main():
    fig, ax = plt.subplots(figsize=(11, 3.3))
    w, h, gap = 0.8, 0.62, 0.2
    for x, title, body, col in BOXES:
        ax.add_patch(FancyBboxPatch((x, 0.12), w, h, boxstyle="round,pad=0.02", fc=col, ec="0.3"))
        ax.text(x + w / 2, 0.12 + h - 0.08, title, ha="center", va="top", fontsize=11, weight="bold")
        ax.text(x + w / 2, 0.12 + h / 2 - 0.06, body, ha="center", va="center", fontsize=8.5, linespacing=1.4)
    for x, *_ in BOXES[:-1]:
        ax.add_patch(FancyArrowPatch((x + w + 0.01, 0.43), (x + 1 - 0.01, 0.43), arrowstyle="-|>",
                                     mutation_scale=14, color="0.3"))
    ax.text(-0.02, 0.9, "acc + gyro recording", fontsize=9, style="italic")
    ax.text(3.82, 0.9, "6-field grounded answer", fontsize=9, style="italic", ha="right")
    ax.annotate("", xy=(2.4, 0.09), xytext=(3.4, 0.09), arrowprops=dict(arrowstyle="-|>", color="C3", lw=1.2,
                                                                     connectionstyle="arc3,rad=-0.3"))
    ax.text(2.9, -0.3, "durations, counts, timestamps and feature values are read off the timeline;\n"
                        "the language layer never produces a number", ha="center", fontsize=8, color="C3")
    ax.set_xlim(-0.05, 3.85)
    ax.set_ylim(-0.4, 0.98)
    ax.axis("off")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=170)


if __name__ == "__main__":
    main()
