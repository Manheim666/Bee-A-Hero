"""Label regeneration & integrity validation for the cleaned iNaturalist set.

After Phase-4 filtering the original iNaturalist COCO JSONs
(``train_mini.json`` 500k refs, ``val.json`` 100k refs, 10k categories) no
longer match what is on disk (2526 Insecta classes, 151,545 images). This
module rebuilds labels so they reference *only surviving images* and validates
that no orphan/broken reference remains.

It NEVER touches the original raw JSONs (those are the backup in
``data/_backup/original_labels/``). All output goes to
``data/interim/labels/``:

  * ``inat_<split>_filtered.json`` — the original COCO file with every image /
    annotation / category that no longer exists removed.
  * ``{train,val,test}.json``     — clean COCO-classification labels built from
    the Phase-4 manifest (contiguous category ids = training class_id).
  * ``validation_report.json``    — the Phase-5 integrity checklist result.

CLI
    python -m src.data_pipeline.label_tools           # build + validate
    python -m src.data_pipeline.label_tools --validate-only
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from src import config as C

LABELS_DIR: Path = C.INTERIM_DIR / "labels"
_INAT_PREFIX = "data/raw/iNaturist/"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _load_manifest() -> list[dict]:
    with open(C.MANIFEST_DIR / "split_manifest.csv", newline="") as fh:
        return list(csv.DictReader(fh))


def _json_name(manifest_path: str) -> str:
    """Manifest path -> the file_name used inside the original iNat JSONs."""
    return manifest_path[len(_INAT_PREFIX):] if manifest_path.startswith(_INAT_PREFIX) else manifest_path


# --------------------------------------------------------------------------- #
# Filter the original COCO JSONs down to surviving images
# --------------------------------------------------------------------------- #
def filter_original(split_json: str, survivor_names: set[str]) -> tuple[dict, dict]:
    """Drop images/annotations/categories that no longer exist on disk.

    Returns ``(filtered_coco, stats)``.
    """
    src = C.INAT_DIR / split_json
    coco = json.load(open(src))
    imgs = [im for im in coco["images"] if im["file_name"] in survivor_names]
    kept_ids = {im["id"] for im in imgs}
    anns = [a for a in coco["annotations"] if a["image_id"] in kept_ids]
    used_cat = {a["category_id"] for a in anns}
    cats = [c for c in coco["categories"] if c["id"] in used_cat]
    filtered = {k: coco.get(k) for k in ("info", "licenses") if k in coco}
    filtered |= {"images": imgs, "annotations": anns, "categories": cats}
    stats = {
        "source": split_json,
        "images_before": len(coco["images"]), "images_after": len(imgs),
        "annotations_before": len(coco["annotations"]), "annotations_after": len(anns),
        "categories_before": len(coco["categories"]), "categories_after": len(cats),
    }
    return filtered, stats


def _dims_and_origcat(filtered_by_split: dict[str, dict]) -> tuple[dict, dict]:
    """Index file_name -> (width, height) and dir_name -> original category obj."""
    dims: dict[str, tuple[int, int]] = {}
    cat_by_dir: dict[str, dict] = {}
    for coco in filtered_by_split.values():
        for im in coco["images"]:
            dims[im["file_name"]] = (im.get("width"), im.get("height"))
        for c in coco["categories"]:
            cat_by_dir.setdefault(c.get("image_dir_name", c.get("name")), c)
    return dims, cat_by_dir


# --------------------------------------------------------------------------- #
# Build clean per-split labels from the manifest
# --------------------------------------------------------------------------- #
def build_clean_labels(manifest: list[dict], dims: dict, cat_by_dir: dict) -> dict:
    """Write clean COCO-classification JSON for train/val/test. Returns stats."""
    # category catalog: contiguous training id -> taxonomy (+ original iNat id)
    classes: dict[int, dict] = {}
    for r in manifest:
        cid = int(r["class_id"])
        if cid in classes:
            continue
        orig = cat_by_dir.get(r["class_name"], {})
        classes[cid] = {
            "id": cid,
            "name": r["class_name"],
            "order": r["order"], "family": r["family"],
            "genus": r["genus"], "species": r["species"],
            "is_bee": int(r["is_bee"]),
            "inat_category_id": orig.get("id"),
            "common_name": orig.get("common_name"),
        }
    categories = [classes[i] for i in sorted(classes)]

    stats = {}
    for split in ("train", "val", "test"):
        rows = [r for r in manifest if r["split"] == split]
        images, annotations = [], []
        for i, r in enumerate(sorted(rows, key=lambda x: x["path"])):
            jn = _json_name(r["path"])
            w, h = dims.get(jn, (None, None))
            images.append({"id": i, "file_name": r["path"], "width": w, "height": h})
            annotations.append({"id": i, "image_id": i, "category_id": int(r["class_id"])})
        coco = {"info": {"description": f"Bee-A-Hero iNaturalist Insecta — {split}"},
                "images": images, "annotations": annotations, "categories": categories}
        LABELS_DIR.mkdir(parents=True, exist_ok=True)
        json.dump(coco, open(LABELS_DIR / f"{split}.json", "w"))
        stats[split] = {"images": len(images), "annotations": len(annotations)}
    stats["num_categories"] = len(categories)
    return stats


# --------------------------------------------------------------------------- #
# Phase-5 integrity checklist
# --------------------------------------------------------------------------- #
def validate(manifest: list[dict]) -> dict:
    nc = len({int(r["class_id"]) for r in manifest})
    manifest_paths = [r["path"] for r in manifest]
    manifest_set = set(manifest_paths)

    # images on disk (labeled splits) vs manifest
    disk = {str(p.relative_to(C.REPO_ROOT))
            for s in C.INAT_LABELED_SPLITS
            for p in (C.INAT_DIR / s).rglob("*")
            if p.is_file() and p.suffix.lower() in C.IMAGE_EXTS}

    images_without_labels = sorted(disk - manifest_set)          # on disk, no manifest row
    labels_without_images = sorted(manifest_set - disk)          # manifest row, file gone
    dup_counts = Counter(manifest_paths)
    duplicated_labels = sorted(p for p, n in dup_counts.items() if n > 1)
    invalid_category = sorted(r["path"] for r in manifest
                              if not (0 <= int(r["class_id"]) < nc))
    corrupted = sorted(r["path"] for r in manifest
                       if not r["class_name"] or r["split"] not in ("train", "val", "test"))
    # cross-split leakage: a path assigned to >1 split
    split_of = defaultdict(set)
    for r in manifest:
        split_of[r["path"]].add(r["split"])
    leakage = sorted(p for p, s in split_of.items() if len(s) > 1)

    report = {
        "num_classes": nc,
        "manifest_images": len(manifest),
        "disk_images": len(disk),
        "images_without_labels": len(images_without_labels),
        "labels_without_images": len(labels_without_images),
        "duplicated_labels": len(duplicated_labels),
        "invalid_category_ids": len(invalid_category),
        "corrupted_labels": len(corrupted),
        "cross_split_leakage": len(leakage),
        "examples": {
            "images_without_labels": images_without_labels[:5],
            "labels_without_images": labels_without_images[:5],
        },
    }
    report["all_checks_pass"] = all(report[k] == 0 for k in (
        "images_without_labels", "labels_without_images", "duplicated_labels",
        "invalid_category_ids", "corrupted_labels", "cross_split_leakage"))
    return report


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(validate_only: bool = False) -> dict:
    manifest = _load_manifest()
    result: dict = {}

    if not validate_only:
        # survivor file_names per source split, then filter the originals
        survivors_by_split: dict[str, set[str]] = defaultdict(set)
        for r in manifest:
            survivors_by_split[r["source_split"]].add(_json_name(r["path"]))

        LABELS_DIR.mkdir(parents=True, exist_ok=True)
        filtered_by_split: dict[str, dict] = {}
        filter_stats = []
        for split in C.INAT_LABELED_SPLITS:
            filtered, stats = filter_original(f"{split}.json", survivors_by_split[split])
            json.dump(filtered, open(LABELS_DIR / f"inat_{split}_filtered.json", "w"))
            filtered_by_split[split] = filtered
            filter_stats.append(stats)
        dims, cat_by_dir = _dims_and_origcat(filtered_by_split)
        result["filter_original"] = filter_stats
        result["clean_labels"] = build_clean_labels(manifest, dims, cat_by_dir)

    result["validation"] = validate(manifest)
    json.dump(result, open(C.MANIFEST_DIR / "validation_report.json", "w"), indent=2)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()
    print(json.dumps(run(validate_only=args.validate_only), indent=2))


if __name__ == "__main__":
    main()
