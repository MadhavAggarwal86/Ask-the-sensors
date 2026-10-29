"""Write synthetic recordings to data/synthetic/ for testing the pipeline before real data exists.

Synthetic results are for debugging only and must never be reported as results.
"""
import argparse
import json

import numpy as np

from ats.dataset import SYNTH_DIR
from ats.io import save_csv
from ats.synthetic import make_recording, random_script


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=10)
    ap.add_argument("--recordings", type=int, default=2)
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--bursty", action="store_true", help="mimic ExtraSensory's 20 s per minute")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    for u in range(args.users):
        d = SYNTH_DIR / f"user{u:02d}"
        d.mkdir(parents=True, exist_ok=True)
        for r in range(args.recordings):
            script = random_script(rng, args.minutes * 60)
            rec, gt = make_recording(script, seed=int(rng.integers(1e9)), bursty=args.bursty,
                                     drop_gyro=(u % 7 == 6))   # some users have no gyroscope
            save_csv(rec, d / f"rec{r}.csv")
            (d / f"rec{r}.gt.json").write_text(json.dumps(gt))
        print("wrote", d)


if __name__ == "__main__":
    main()
