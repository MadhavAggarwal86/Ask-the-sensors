"""Preprocess every recording into window features + labels (data/processed/<source>/<user>.npz)."""
import argparse

from joblib import Parallel, delayed

from ats import config as C
from ats.dataset import process_user, users


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["extrasensory", "synthetic"], default="extrasensory")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()
    out = C.PROCESSED_DIR / args.source
    us = users(args.source)
    if args.skip_existing:
        us = [u for u in us if not (out / f"{u}.npz").exists()]
    counts = Parallel(n_jobs=args.jobs, verbose=5)(delayed(process_user)(args.source, u, out) for u in us)
    for u, n in zip(us, counts):
        print(f"{u}: {n} labelled windows")
    print(f"total {sum(counts)} windows from {sum(n > 0 for n in counts)} users -> {out}")


if __name__ == "__main__":
    main()
