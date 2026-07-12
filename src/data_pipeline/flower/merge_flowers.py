"""
merge_flowers.py
----------------
Merge the two versions inside "archive (1).zip" (Flower Classification V1 + V2)
into ONE unified, labeled image dataset.

Output layout (labels = folder names):
    merged_dataset/
        Training Data/<Class>/<files>
        Validation Data/<Class>/<files>
        Testing Data/<Class>/<files>

- Images only (.jpeg/.jpg/.png). TFRecord files are skipped.
- V1 classes (Daisy, Lavender, Lily, Rose, Sunflower) are folded into the
  SAME-named V2 classes, so nothing is duplicated as a separate class.
- Every file is renamed  "<v1|v2>_<origname>"  so V1 and V2 files with the
  same name never overwrite each other.

Usage:
    python merge_flowers.py
    python merge_flowers.py --zip "path/to/archive (1).zip" --out "merged_dataset"
"""

import argparse
import zipfile
from pathlib import Path
from collections import defaultdict

# ---- config -----------------------------------------------------------------
IMAGE_EXTS = {".jpeg", ".jpg", ".png"}

# canonical split folder names in the output
SPLIT_NAMES = {
    "training data": "Training Data",
    "validation data": "Validation Data",
    "testing data": "Testing Data",
}


def detect_source(parts):
    """Return 'v2' or 'v1' depending on which version the path belongs to."""
    top = parts[0].lower()
    return "v2" if "v2" in top else "v1"


def find_split(parts):
    """Return the canonical split name found anywhere in the path, else None."""
    for p in parts:
        key = p.lower()
        if key in SPLIT_NAMES:
            return SPLIT_NAMES[key]
    return None


def merge(zip_path: Path, out_dir: Path):
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip not found: {zip_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # counts[split][class] = number of images written
    counts = defaultdict(lambda: defaultdict(int))
    skipped_non_image = 0

    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        for m in members:
            parts = m.filename.replace("\\", "/").split("/")
            ext = Path(parts[-1]).suffix.lower()

            if ext not in IMAGE_EXTS:
                skipped_non_image += 1
                continue

            split = find_split(parts)
            if split is None:
                # not inside a recognised split folder -> skip
                continue

            # class folder is the directory directly above the file
            class_name = parts[-2]
            source = detect_source(parts)
            orig_name = parts[-1]
            new_name = f"{source}_{orig_name}"

            dest_dir = out_dir / split / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / new_name

            # guard against the (unlikely) exact same name within one source
            if dest_path.exists():
                stem, suf = Path(new_name).stem, Path(new_name).suffix
                i = 1
                while dest_path.exists():
                    dest_path = dest_dir / f"{stem}_{i}{suf}"
                    i += 1

            with zf.open(m) as src, open(dest_path, "wb") as dst:
                dst.write(src.read())

            counts[split][class_name] += 1

    return counts, skipped_non_image


def print_report(counts, skipped_non_image, out_dir):
    print("\n" + "=" * 60)
    print("MERGE COMPLETE ->", out_dir.resolve())
    print("=" * 60)

    grand_total = 0
    for split in ["Training Data", "Validation Data", "Testing Data"]:
        classes = counts.get(split, {})
        split_total = sum(classes.values())
        grand_total += split_total
        print(f"\n[{split}]  total = {split_total}")
        for cls in sorted(classes):
            print(f"    {cls:<12} {classes[cls]}")

    print("\n" + "-" * 60)
    print(f"TOTAL IMAGES MERGED : {grand_total}")
    print(f"Non-image files skipped (e.g. .tfrecord): {skipped_non_image}")
    print(f"Classes: {sorted({c for s in counts.values() for c in s})}")
    print("-" * 60)


# repo root = .../Bee-A-Hero  (this file is at src/data_pipeline/flower/merge_flowers.py)
REPO = Path(__file__).resolve().parents[3]
DEFAULT_ZIP = REPO / "data" / "raw" / "Flower" / "archive (1).zip"
DEFAULT_OUT = REPO / "data" / "processed" / "flower" / "classification"


def main():
    ap = argparse.ArgumentParser(description="Merge Flower Classification V1+V2 into one dataset.")
    ap.add_argument("--zip", default=str(DEFAULT_ZIP),
                    help="path to the source zip (default: data/raw/Flower/archive (1).zip)")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="output folder (default: data/processed/flower/classification)")
    args = ap.parse_args()

    counts, skipped = merge(Path(args.zip), Path(args.out))
    print_report(counts, skipped, Path(args.out))


if __name__ == "__main__":
    main()
