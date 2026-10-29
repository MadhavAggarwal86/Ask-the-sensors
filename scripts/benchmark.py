"""Efficiency study: accuracy versus overhead for full and compressed recognition backbones (Figure 4).

Configurations
  hgb_full            the deployed model (300 boosting rounds x 7 classes)
  hgb_pruned_<n>      first n boosting rounds kept (ensemble pruning)
  hgb_acc_only        retrained without any gyroscope feature, so the gyroscope can stay off
  tree_d<d>           one regression tree distilled from hgb_full's probabilities
  mlp_fp32            small MLP distilled from hgb_full (logit matching)
  mlp_int8            mlp_fp32 after post-training 8-bit quantisation

Every configuration answers the same generated questions on the same held-out recordings through
the full pipeline (HMM decoding, operations, scoring rules); only the classifier changes.

Cost axes, measured on the stated target:
  model size        bytes of the serialised model file (joblib)
  query latency     answering one question about a fresh 1-hour, 25 Hz recording, end to end:
                    windowing, features, classifier, HMM decoding, parsing, answering
  peak memory       peak Python/NumPy allocation during that query (tracemalloc), model excluded;
                    it is dominated by the recording and its windows, so it barely varies
  parameters        learned numbers in the classifier (ats.compress.n_params)
  CPU time / util.  process CPU time of that query, and CPU time / wall time (1.0 = one core busy)
  energy (estimate) CPU time x CORE_POWER_W. Not a measurement: macOS exposes per-process energy only
                    through sudo powermetrics, so the column scales linearly with the assumed power
Target: this laptop's CPU limited to ONE thread (threadpoolctl), to emulate a single-core edge
device; the CPU model is recorded in efficiency.json.
"""
import argparse
import json
import pickle
import platform
import subprocess
import time
import tracemalloc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

from ats import config as C
from ats.compress import Int8MLP, MLPStudent, TreeStudent, n_params, prune_boosting
from ats.dataset import held_out_recordings, load_processed, split_users, training_set
from ats.eval import metrics as M
from ats.eval.run import ask_generated
from ats.features import extract
from ats.model import ActivityModel
from ats.pipeline import SensorQA
from ats.preprocess import make_windows
from ats.synthetic import make_recording

CORE_POWER_W = 4.0   # assumed active power of one busy performance core (energy column only)


def cached_features(source: str, n: int):
    """Window features of the held-out QA recordings, extracted once and reused by every config."""
    path = C.PROCESSED_DIR / f"qa_cache_{source}_{n}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    cache = []
    for rid, rec, gt in held_out_recordings(source, n):
        starts, wins = make_windows(rec)
        X, names = extract(wins) if len(wins) else (None, [])
        cache.append(dict(rid=rid, starts=starts, X=X, names=names, has_gyro=rec.has_gyro,
                          duration=rec.duration, t0=rec.t0, gt=gt))
    path.write_bytes(pickle.dumps(cache))
    return cache


def qa_records(model, cache, acc_only=False):
    """Per-question records. With acc_only the device records no gyroscope at all, so the reference
    evidence is accelerometer-only too."""
    qa = SensorQA(model)
    records = []
    for r in cache:
        has_gyro = r["has_gyro"] and not acc_only
        qa.load_features(r["starts"], r["X"], r["names"], has_gyro, r["duration"], r["t0"])
        records += ask_generated(qa, r["rid"], r["gt"], r["duration"], has_gyro)
    return records


def bootstrap(records_by_name, n_boot=1000, seed=0):
    """95% intervals for overall QA accuracy, resampling recordings (paired across configurations,
    so differences to the full model get their own interval)."""
    rids = sorted({r["rec"] for r in next(iter(records_by_name.values()))})
    rng = np.random.default_rng(seed)
    draws = [rng.choice(len(rids), len(rids)) for _ in range(n_boot)]
    out = {}
    for name, recs in records_by_name.items():
        by_rec = {}
        for r in recs:
            by_rec.setdefault(r["rec"], []).append(r)
        out[name] = np.array([M.score([r for i in d for r in by_rec.get(rids[i], [])])["overall_macro"]
                              for d in draws])
    return out


def query_cost(model, rec, acc_only=False, repeats=7):
    """Single-thread latency (ms), classifier time (ms), peak working memory (MB) and CPU time (ms)
    of one question on a fresh recording."""
    if acc_only:
        rec = type(rec)(rec.t, np.where(np.arange(6) < 3, rec.x, np.nan), rec.t0)
    qa = SensorQA(model)

    def once():
        qa.load(rec)
        qa.ask("How long was the user walking?")

    with threadpool_limits(1):
        once()                                   # warm-up
        times, cpu = [], []
        for _ in range(repeats):
            t, c = time.perf_counter(), time.process_time()
            once()
            times.append(time.perf_counter() - t)
            cpu.append(time.process_time() - c)
        starts, wins = make_windows(rec)
        X, names = extract(wins)
        t = time.perf_counter()
        for _ in range(repeats):
            model.predict_proba(X, names)
        model_ms = 1000 * (time.perf_counter() - t) / repeats
        tracemalloc.start()
        once()
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    return 1000 * float(np.median(times)), model_ms, peak / 2**20, 1000 * float(np.median(cpu))


def build_variants(full, source, split, vdir):
    """Train every compressed configuration and save it to models/variants/."""
    names = full.feature_names
    # Transfer set for distillation: the exact training set of hgb_full (same recipe and seed).
    Xtr, ytr, gtr, _ = load_processed(source, split["train"])
    Xfit, yfit = training_set(Xtr, ytr, gtr, names, np.random.default_rng(0))
    del Xtr, ytr, gtr
    teacher = full.predict_proba(Xfit)

    acc_cols = [i for i, n in enumerate(names) if not n.startswith("gyro")]
    variants = {"hgb_full": (full, False)}
    for n in (100, 30, 10):
        variants[f"hgb_pruned_{n}"] = (ActivityModel(prune_boosting(full.clf, n), names), False)
    print("training hgb_acc_only ...")
    acc = ActivityModel.new().fit(Xfit[:, acc_cols], yfit, [names[i] for i in acc_cols])
    variants["hgb_acc_only"] = (acc, True)
    for d in (10, 6):
        print(f"distilling tree_d{d} ...")
        variants[f"tree_d{d}"] = (ActivityModel(TreeStudent(d).fit(Xfit, teacher), names), False)
    print("distilling mlp_fp32 ...")
    mlp = MLPStudent().fit(Xfit, teacher)
    variants["mlp_fp32"] = (ActivityModel(mlp, names), False)
    calib = Xfit[np.random.default_rng(1).choice(len(Xfit), 20000, replace=False)]
    variants["mlp_int8"] = (ActivityModel(Int8MLP(mlp, calib), names), False)
    for name, (model, _) in variants.items():
        if name != "hgb_full":
            model.save(vdir / f"{name}.joblib")
    return variants



def pareto(points):
    """Indices of non-dominated points (lower cost, higher accuracy), sorted by cost."""
    order = sorted(range(len(points)), key=lambda i: (points[i][0], -points[i][1]))
    front, best = [], -np.inf
    for i in order:
        if points[i][1] > best:
            front.append(i)
            best = points[i][1]
    return front


def plot(rows, path, title):
    axes_spec = [("size_mb", "Model size (MB, log scale)"),
                 ("classifier_ms", "Classifier time for 1 h of data (ms, log scale)"),
                 ("latency_ms", "End-to-end single-query latency, 1 h of data (ms)")]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), sharey=True)
    full_acc = next(r["qa_overall"] for r in rows if r["name"] == "hgb_full")
    for ax, (key, label) in zip(axes, axes_spec):
        pts = [(r[key], r["qa_overall"]) for r in rows]
        front = pareto(pts)
        for i, r in enumerate(rows):
            on = i in front
            # Error bar: paired 95% bootstrap interval of the difference to hgb_full, drawn around the
            # point. Recordings vary a lot, but that variation is shared by every configuration.
            lo, hi = (full_acc + d for d in r["delta_vs_full_ci95"])
            ax.errorbar(*pts[i], yerr=[[max(r["qa_overall"] - lo, 0)], [max(hi - r["qa_overall"], 0)]],
                        fmt="s" if r["acc_only"] else "o", ms=7 if on else 5, color="C3" if on else "C0",
                        ecolor="0.7", elinewidth=1, capsize=2, zorder=3)
            ax.annotate(r["name"], pts[i], textcoords="offset points", fontsize=7.5,
                        xytext=(5, -11) if r["acc_only"] else (5, 4))
        ax.plot([pts[i][0] for i in front], [pts[i][1] for i in front], color="C3", lw=1.5,
                label="Pareto frontier", zorder=2)
        if key != "latency_ms":                 # end-to-end latency has a floor; keep it linear
            ax.set_xscale("log")
        ax.set_xlabel(label)
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("Overall QA accuracy (macro)\nbars: 95% CI of difference to hgb_full")
    axes[0].scatter([], [], marker="s", color="gray", label="accelerometer only (gyroscope off)")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def cost_fields(model, lat_ms, cpu_ms):
    return {"n_params": n_params(model.clf), "cpu_ms": cpu_ms, "cpu_util": cpu_ms / lat_ms,
            "energy_mj_est": cpu_ms * CORE_POWER_W}


def costs_only(out, full, vdir, kinds):
    """Add parameters, CPU time and energy to an existing efficiency.json without re-running the QA
    (the QA columns and the latency already recorded there are kept)."""
    path = out / "efficiency.json"
    eff = json.loads(path.read_text())
    hour, _ = make_recording([("sitting", 1200), ("walking", 1200), ("lying_down", 1200)], seed=0)
    for r in eff["rows"]:
        model = full if r["name"] == "hgb_full" else ActivityModel.load(vdir / f"{r['name']}.joblib")
        lat, _, _, cpu = query_cost(model, hour, kinds[r["name"]])
        r.update(cost_fields(model, lat, cpu))
        print(f"{r['name']:14s} params {r['n_params']:7d}  cpu {cpu:6.1f} ms / wall {lat:6.1f} ms  "
              f"util {r['cpu_util']:.2f}  energy ~{r['energy_mj_est']:.0f} mJ")
    eff["energy_assumption"] = f"CPU time x {CORE_POWER_W} W (estimate, not measured)"
    path.write_text(json.dumps(eff, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["extrasensory", "synthetic"], default="extrasensory")
    ap.add_argument("--max-recordings", type=int, default=40)
    ap.add_argument("--reuse-variants", action="store_true", help="load models/variants/* instead of retraining")
    ap.add_argument("--costs-only", action="store_true",
                    help="only add parameters, CPU time and energy to the existing efficiency.json")
    args = ap.parse_args()
    out = C.RESULTS_DIR / args.source
    vdir = C.MODELS_DIR / "variants"
    vdir.mkdir(parents=True, exist_ok=True)

    full = ActivityModel.load(C.MODELS_DIR / "hgb_full.joblib")
    names = full.feature_names
    split = split_users(args.source)
    kinds = {"hgb_full": False, "hgb_pruned_100": False, "hgb_pruned_30": False, "hgb_pruned_10": False,
             "hgb_acc_only": True, "tree_d10": False, "tree_d6": False, "mlp_fp32": False, "mlp_int8": False}
    if args.costs_only:
        return costs_only(out, full, vdir, kinds)
    if args.reuse_variants:
        variants = {n: (full if n == "hgb_full" else ActivityModel.load(vdir / f"{n}.joblib"), a)
                    for n, a in kinds.items()}
    else:
        variants = build_variants(full, args.source, split, vdir)

    print("caching held-out features ...")
    cache = cached_features(args.source, args.max_recordings)
    Xte, yte, _, _ = load_processed(args.source, split["test"])
    hour, _ = make_recording([("sitting", 1200), ("walking", 1200), ("lying_down", 1200)], seed=0)

    rows, records = [], {}
    for name, (model, acc_only) in variants.items():
        path = C.MODELS_DIR / "hgb_full.joblib" if name == "hgb_full" else vdir / f"{name}.joblib"
        records[name] = qa_records(model, cache, acc_only)
        sc = M.score(records[name])
        pred = model.predict_proba(Xte, names).argmax(1)
        lat, model_ms, peak, cpu_ms = query_cost(model, hour, acc_only)
        rows.append({"name": name, "acc_only": acc_only, "size_mb": path.stat().st_size / 2**20,
                     "latency_ms": lat, "classifier_ms": model_ms, "peak_mem_mb": peak,
                     "qa_overall": sc["overall_macro"], "qa_grounded": sc["grounded_macro"],
                     "window_macro_f1": float(f1_score(yte, pred, labels=range(len(C.CLASSES)),
                                                       average="macro", zero_division=0)),
                     "per_type": sc["per_type"], **cost_fields(model, lat, cpu_ms)})
        r = rows[-1]
        print(f"{name:14s} size {r['size_mb']:8.3f} MB  query {lat:6.1f} ms  (classifier {model_ms:6.2f} ms)  "
              f"peak {peak:5.1f} MB  QA {r['qa_overall']:.3f}  grounded {r['qa_grounded']:.3f}  "
              f"window F1 {r['window_macro_f1']:.3f}")

    boot = bootstrap(records)
    for r in rows:
        b, d = boot[r["name"]], boot[r["name"]] - boot["hgb_full"]
        r["qa_ci95"] = np.percentile(b, [2.5, 97.5]).tolist()
        r["delta_vs_full_ci95"] = np.percentile(d, [2.5, 97.5]).tolist()
        print(f"{r['name']:14s} QA 95% CI [{r['qa_ci95'][0]:.3f}, {r['qa_ci95'][1]:.3f}]  "
              f"minus hgb_full [{r['delta_vs_full_ci95'][0]:+.3f}, {r['delta_vs_full_ci95'][1]:+.3f}]")

    cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    target = f"{cpu or platform.processor()}, 1 thread, Python {platform.python_version()}"
    (out / "efficiency.json").write_text(json.dumps({"target": target, "recordings": len(cache),
                                                     "query": "1 question on a fresh 1-hour 25 Hz recording",
                                                     "energy_assumption": f"CPU time x {CORE_POWER_W} W (estimate, not measured)",
                                                     "rows": rows}, indent=2))
    plot(rows, out / "fig4_accuracy_vs_overhead.png",
         f"Accuracy vs overhead ({len(cache)} held-out recordings; target: {target})")
    print("saved", out / "fig4_accuracy_vs_overhead.png")


if __name__ == "__main__":
    main()
