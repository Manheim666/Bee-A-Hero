"""iNaturalist dataset preparation: filter → dedup → 70/15/15 split.

Design goals (locked with the team):
  * **Non-destructive.** Nothing is ``rm``-ed. "Unnecessary" images (non-Insecta
    folders, exact duplicates) are *moved* to ``data/_backup/removed/`` mirroring
    their original relative path, and every moved image is logged. Fully
    reversible.
  * **Logical split.** The train/val/test split is a *manifest* (which image
    belongs to which split); images are NOT physically shuffled between split
    folders, so there is no duplication and no cross-split leakage by move.
  * **Reproducible.** Fixed seed, deterministic ordering, hash cache. Running
    twice yields the same manifest and is idempotent on disk.

Pool = Insecta images from ``train_mini`` + ``val`` (public_test is unlabeled →
excluded). The dataset is uniform (60 img/class), so 70/15/15 = 42/9/9 per class
with no truncation; the only removals are non-Insecta and exact duplicates.

CLI
    python -m src.data_pipeline.inaturalist_prep            # dry-run (plan only)
    python -m src.data_pipeline.inaturalist_prep --apply    # execute moves
    python -m src.data_pipeline.inaturalist_prep --apply --skip-dedup
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from src import config as C

# --------------------------------------------------------------------------- #
# Taxonomy parsing
# --------------------------------------------------------------------------- #
# Folder name: ID_Kingdom_Phylum_Class_Order_Family_Genus_specificEpithet
_TAXON_FIELDS = ("tax_id", "kingdom", "phylum", "class", "order",
                 "family", "genus", "species")


def parse_taxon(folder_name: str) -> dict[str, str]:
    """Split an iNaturalist species-folder name into taxonomy fields.

    Names have exactly 8 underscore-separated parts. If a species epithet
    itself contains underscores the extras are re-joined into ``species``.
    """
    parts = folder_name.split("_")
    if len(parts) < len(_TAXON_FIELDS):
        return {f: "" for f in _TAXON_FIELDS} | {"tax_id": folder_name}
    head = parts[: len(_TAXON_FIELDS) - 1]
    species = "_".join(parts[len(_TAXON_FIELDS) - 1:])
    values = head + [species]
    return dict(zip(_TAXON_FIELDS, values))


def is_insecta(taxon: dict[str, str]) -> bool:
    return taxon.get("class", "") == C.TARGET_CLASS


def is_bee(taxon: dict[str, str]) -> bool:
    return taxon.get("order", "") == "Hymenoptera" and taxon.get("family", "") in C.BEE_FAMILIES


# --------------------------------------------------------------------------- #
# Scanning
# --------------------------------------------------------------------------- #
def iter_class_dirs(split_dir: Path):
    """Yield species sub-folders of a split directory, sorted for determinism."""
    if not split_dir.is_dir():
        return
    for d in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        yield d


def _images_in(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in C.IMAGE_EXTS)


def _rel(path: Path) -> str:
    return str(path.relative_to(C.REPO_ROOT))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# Backup move (the only mutating primitive)
# --------------------------------------------------------------------------- #
def move_to_backup(path: Path, reason: str, apply: bool) -> dict:
    """Move ``path`` under REMOVED_DIR mirroring its repo-relative location.

    Returns a log row. In dry-run (``apply=False``) nothing is moved.
    """
    rel = path.relative_to(C.REPO_ROOT)
    dest = C.REMOVED_DIR / rel
    if apply:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(dest))
    return {
        "original_path": str(rel),
        "backup_path": str(dest.relative_to(C.REPO_ROOT)),
        "reason": reason,
        "timestamp": _now(),
    }


# --------------------------------------------------------------------------- #
# Stage 1 — Insecta filter
# --------------------------------------------------------------------------- #
def filter_non_insecta(apply: bool) -> list[dict]:
    """Move every non-Insecta species folder (all its images) to backup.

    Returns per-image removal log rows.
    """
    removed: list[dict] = []
    for split in C.INAT_LABELED_SPLITS:
        for folder in list(iter_class_dirs(C.INAT_DIR / split)):
            taxon = parse_taxon(folder.name)
            if is_insecta(taxon):
                continue
            imgs = _images_in(folder)
            for img in imgs:
                row = move_to_backup(img, reason="non_insecta", apply=apply)
                row["class_name"] = folder.name
                row["split"] = split
                removed.append(row)
            # remove the now-empty (or fully-moved) folder shell on apply
            if apply and folder.exists() and not any(folder.iterdir()):
                folder.rmdir()
    return removed


# --------------------------------------------------------------------------- #
# Stage 2 — build the Insecta image pool
# --------------------------------------------------------------------------- #
def build_pool() -> list[dict]:
    """Return one record per Insecta image across labeled splits.

    Class folders exist identically in both splits, so a class is keyed by its
    folder name; its images are pooled from train_mini + val.
    """
    records: list[dict] = []
    for split in C.INAT_LABELED_SPLITS:
        for folder in iter_class_dirs(C.INAT_DIR / split):
            taxon = parse_taxon(folder.name)
            if not is_insecta(taxon):
                continue
            bee = is_bee(taxon)
            for img in _images_in(folder):
                records.append({
                    "path": _rel(img),
                    "class_name": folder.name,
                    "order": taxon["order"],
                    "family": taxon["family"],
                    "genus": taxon["genus"],
                    "species": taxon["species"],
                    "is_bee": int(bee),
                    "source_split": split,
                })
    return records


# --------------------------------------------------------------------------- #
# Stage 3 — exact duplicate removal (md5, cached)
# --------------------------------------------------------------------------- #
def _md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _load_hash_cache(cache_path: Path) -> dict[str, tuple[int, int, str]]:
    cache: dict[str, tuple[int, int, str]] = {}
    if cache_path.exists():
        with open(cache_path, newline="") as fh:
            for r in csv.DictReader(fh):
                cache[r["path"]] = (int(r["size"]), int(r["mtime"]), r["md5"])
    return cache


def _save_hash_cache(cache_path: Path, cache: dict[str, tuple[int, int, str]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "size", "mtime", "md5"])
        for p, (size, mtime, md5) in sorted(cache.items()):
            w.writerow([p, size, mtime, md5])


def dedup_exact(records: list[dict], apply: bool) -> tuple[list[dict], list[dict]]:
    """Remove exact-duplicate images (identical md5), keeping the first by path.

    Returns ``(kept_records, removed_log_rows)``. Uses a size+mtime-keyed cache
    so re-runs don't re-hash unchanged files.
    """
    cache_path = C.MANIFEST_DIR / "md5_cache.csv"
    cache = _load_hash_cache(cache_path)
    updated = dict(cache)

    for rec in records:
        p = C.REPO_ROOT / rec["path"]
        st = p.stat()
        cached = cache.get(rec["path"])
        if cached and cached[0] == st.st_size and cached[1] == int(st.st_mtime):
            md5 = cached[2]
        else:
            md5 = _md5(p)
            updated[rec["path"]] = (st.st_size, int(st.st_mtime), md5)
        rec["md5"] = md5
    _save_hash_cache(cache_path, updated)

    seen: dict[str, str] = {}          # md5 -> first path kept
    kept: list[dict] = []
    removed: list[dict] = []
    for rec in sorted(records, key=lambda r: r["path"]):
        md5 = rec["md5"]
        if md5 in seen:
            row = move_to_backup(C.REPO_ROOT / rec["path"],
                                 reason="exact_duplicate", apply=apply)
            row["class_name"] = rec["class_name"]
            row["split"] = rec["source_split"]
            row["md5"] = md5
            row["duplicate_of"] = seen[md5]
            removed.append(row)
        else:
            seen[md5] = rec["path"]
            kept.append(rec)
    return kept, removed


# --------------------------------------------------------------------------- #
# Stage 4 — stratified 70/15/15 split (largest remainder, no data loss)
# --------------------------------------------------------------------------- #
def _largest_remainder(n: int, ratios: dict[str, float]) -> dict[str, int]:
    """Split ``n`` items into named buckets summing exactly to ``n``."""
    raw = {k: n * v for k, v in ratios.items()}
    base = {k: int(v) for k, v in raw.items()}
    remaining = n - sum(base.values())
    # distribute leftovers to the largest fractional parts (stable order)
    order = sorted(ratios, key=lambda k: (-(raw[k] - base[k]), k))
    for i in range(remaining):
        base[order[i % len(order)]] += 1
    return base


def assign_splits(kept: list[dict], seed: int) -> list[dict]:
    """Assign each kept image a train/val/test split, stratified per class."""
    by_class: dict[str, list[dict]] = defaultdict(list)
    for rec in kept:
        by_class[rec["class_name"]].append(rec)

    class_names = sorted(by_class)
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    rng = random.Random(seed)

    for name in class_names:
        recs = sorted(by_class[name], key=lambda r: r["path"])
        rng.shuffle(recs)
        alloc = _largest_remainder(len(recs), C.SPLIT_RATIOS)
        i = 0
        for split in ("train", "val", "test"):
            for rec in recs[i:i + alloc[split]]:
                rec["split"] = split
                rec["class_id"] = class_to_idx[name]
                rec["status"] = "kept"
            i += alloc[split]
    return class_names


# --------------------------------------------------------------------------- #
# Stage 5 — downsample the (unlabeled) public_test folder
# --------------------------------------------------------------------------- #
def reduce_public_test(target: int, apply: bool) -> dict:
    """Shrink the huge unlabeled ``public_test/`` folder to ``target`` images.

    Deterministic (seeded) so every teammate keeps the *same* subset: the flat
    image list is sorted, shuffled with ``config.SEED``, and the first
    ``target`` are kept. Surplus images are moved to ``data/_backup/removed/``
    (never deleted here). Idempotent — a no-op once the folder is already at or
    below ``target``. Writes the reproducible kept-list to
    ``manifests/public_test_kept.txt``.
    """
    pt = C.INAT_DIR / C.INAT_UNLABELED_SPLIT
    result = {"target": target, "action": "none"}
    if not pt.is_dir():
        result["action"] = "skip (no public_test dir)"
        return result

    imgs = sorted(p for p in pt.iterdir()
                  if p.is_file() and p.suffix.lower() in C.IMAGE_EXTS)
    total = len(imgs)
    result["public_test_total"] = total

    if total <= target:
        keep = imgs
        surplus: list[Path] = []
        result["action"] = "skip (already <= target)"
    else:
        order = list(imgs)                      # already sorted -> deterministic
        random.Random(C.SEED).shuffle(order)
        keep_set = set(order[:target])
        keep = [p for p in imgs if p in keep_set]
        surplus = [p for p in imgs if p not in keep_set]
        result["action"] = "APPLIED" if apply else "DRY_RUN"

    C.MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    (C.MANIFEST_DIR / "public_test_kept.txt").write_text(
        "\n".join(_rel(p) for p in sorted(keep)) + "\n")

    if apply and surplus:
        dest_dir = C.REMOVED_DIR / pt.relative_to(C.REPO_ROOT)
        dest_dir.mkdir(parents=True, exist_ok=True)
        for p in surplus:
            os.rename(p, dest_dir / p.name)      # same filesystem -> fast rename

    result["kept"] = len(keep)
    result["removed"] = len(surplus)
    return result


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #
_MANIFEST_COLS = ["path", "class_id", "class_name", "order", "family", "genus",
                  "species", "is_bee", "source_split", "split", "status"]
_REMOVED_COLS = ["original_path", "backup_path", "reason", "class_name",
                 "split", "md5", "duplicate_of", "timestamp"]


def _write_csv(path: Path, cols: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_outputs(kept: list[dict], removed: list[dict], class_names: list[str]) -> dict:
    C.MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest = sorted(kept, key=lambda r: (r["class_id"], r["split"], r["path"]))
    _write_csv(C.MANIFEST_DIR / "split_manifest.csv", _MANIFEST_COLS, manifest)
    # Preserve the audit log on idempotent re-runs: only (re)write when this run
    # actually removed something, or when no log exists yet. Otherwise a second
    # --apply (which finds nothing left to remove) would wipe the record.
    removed_log = C.MANIFEST_DIR / "removed_log.csv"
    if removed or not removed_log.exists():
        _write_csv(removed_log, _REMOVED_COLS, removed)

    class_to_idx = {name: i for i, name in enumerate(class_names)}
    with open(C.MANIFEST_DIR / "class_index.json", "w") as fh:
        json.dump({"class_to_idx": class_to_idx,
                   "idx_to_class": {i: n for n, i in class_to_idx.items()},
                   "num_classes": len(class_names)}, fh, indent=2)

    counts = {"train": 0, "val": 0, "test": 0}
    bee = {"train": 0, "val": 0, "test": 0}
    for r in kept:
        counts[r["split"]] += 1
        bee[r["split"]] += r["is_bee"]
    reasons: dict[str, int] = defaultdict(int)
    for r in removed:
        reasons[r["reason"]] += 1

    summary = {
        "generated": _now(),
        "seed": C.SEED,
        "ratios": C.SPLIT_RATIOS,
        "num_classes": len(class_names),
        "kept_images": len(kept),
        "split_counts": counts,
        "split_fractions": {k: round(v / max(len(kept), 1), 4) for k, v in counts.items()},
        "bee_images_per_split": bee,
        "removed_total": len(removed),
        "removed_by_reason": dict(reasons),
    }
    with open(C.MANIFEST_DIR / "split_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(apply: bool = False, skip_dedup: bool = False) -> dict:
    removed: list[dict] = []
    removed += filter_non_insecta(apply=apply)

    records = build_pool()
    if skip_dedup:
        kept = records
        for r in kept:
            r["md5"] = ""
    else:
        kept, dup_removed = dedup_exact(records, apply=apply)
        removed += dup_removed

    class_names = assign_splits(kept, seed=C.SEED)
    summary = write_outputs(kept, removed, class_names)

    # Stage 5: shrink the oversized unlabeled public_test folder to match the
    # val split size (seeded -> reproducible across teammates).
    val_count = summary["split_counts"]["val"]
    summary["public_test_reduction"] = reduce_public_test(val_count, apply=apply)

    summary["mode"] = "APPLIED" if apply else "DRY_RUN"
    with open(C.MANIFEST_DIR / "split_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="execute moves (default: dry-run, manifests only)")
    ap.add_argument("--skip-dedup", action="store_true",
                    help="skip exact-duplicate md5 scan")
    args = ap.parse_args()
    summary = run(apply=args.apply, skip_dedup=args.skip_dedup)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
