"""Download and unpack the ExtraSensory files this project uses.

    python scripts/download_data.py --labels-only   # ~1 MB, enough to explore the labels
    python scripts/download_data.py                 # + raw_acc (~6.5 GB) and proc_gyro (~9.3 GB)

Dataset: Vaizman, Ellis & Lanckriet, "Recognizing Detailed Human Context In-the-Wild from
Smartphones and Smartwatches", IEEE Pervasive Computing 16(4), 2017. http://extrasensory.ucsd.edu
"""
import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path

from ats import config as C

BASE = "http://extrasensory.ucsd.edu/data"
FILES = {
    "labels": f"{BASE}/additional_data_files/ExtraSensory.per_uuid_original_labels.zip",
    "raw_acc": f"{BASE}/raw_measurements/ExtraSensory.raw_measurements.raw_acc.zip",
    "proc_gyro": f"{BASE}/raw_measurements/ExtraSensory.raw_measurements.proc_gyro.zip",
}


def fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    # curl resumes interrupted downloads (-C -), which matters for the multi-GB files.
    subprocess.run(["curl", "-L", "-C", "-", "--fail", "-o", str(dest), url], check=True)


def unpack(zip_path: Path, out: Path) -> None:
    """Extract, flattening the archive's top folder so files land in out/<uuid>/..."""
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for m in z.infolist():
            if m.is_dir():
                continue
            parts = Path(m.filename).parts
            rel = Path(*parts[-2:]) if len(parts) >= 2 and not parts[-1].endswith(".csv.gz") else Path(parts[-1])
            target = out / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(m) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-only", action="store_true")
    ap.add_argument("--only", choices=list(FILES), help="fetch one archive (run several in parallel)")
    ap.add_argument("--keep-zips", action="store_true")
    args = ap.parse_args()
    names = ["labels"] if args.labels_only else [args.only] if args.only else list(FILES)
    for name in names:
        out = C.RAW_DIR / name
        if out.exists() and any(out.iterdir()):
            print(f"{name}: already present, skipping")
            continue
        zp = C.RAW_DIR / f"{name}.zip"
        print(f"{name}: downloading {FILES[name]}")
        fetch(FILES[name], zp)
        print(f"{name}: extracting")
        unpack(zp, out)
        if not args.keep_zips:
            zp.unlink()
    print("done")


if __name__ == "__main__":
    main()
