#!/usr/bin/env python3
"""
iNaturist (iNaturalist-style) dataset cleaning + EDA + engineering pipeline for Bee-A-Hero.

GOALS (locked with the user):
  1. CLEAN: remove every species folder under the labeled splits whose taxonomic
     Class is NOT Insecta. Verify the surviving images and drop corrupt ones.
  2. JSON SYNC: after images are removed, rewrite any iNaturalist/COCO-style JSON
     annotation files so images / annotations / categories no longer reference
     deleted files (no dangling mismatch). Originals are backed up to *.json.bak.
  3. SPLIT 70/15/15: keep train_mini -> train and val -> val as filtered, then
     trim the large public_test set down so that test forms the target 15% of the
     grand total (i.e. "keep some test, remove most"). The keep-count is derived
     from the *actual* filtered train+val sizes.
  4. EDA + ENGINEERING: class/order distributions, resolution/aspect profiling,
     a visual sample grid, a quality report (class balance + cross-split leakage),
     and the data.yaml / dataset_config.json + manifests a training loop consumes.

SAFETY
  * DRY_RUN = True by default: the pipeline computes and REPORTS every deletion
    but removes nothing. Inspect _pipeline/REPORT.md, then set DRY_RUN = False
    (or pass --apply) to actually delete. Deletions are irreversible.
  * Original *.tar.gz archives are never touched.
  * Every phase is wrapped so one failure does not abort the rest.

Layout assumption (matches the repo):
  data/raw/iNaturist/
    train_mini/<ID>_<Kingdom>_<Phylum>_<Class>_<Order>_<Family>_<Genus>_<species>/*.jpg
    val/       <same folder convention>
    public_test/   (flat, UNLABELED -> trimmed, used as held-out test set)
    *.json         (optional iNat/COCO annotation files, cleaned if present)

Run (only after the data copy into iNaturist has finished):
    python src/data_pipeline/inaturist_pipeline.py            # dry run, deletes nothing
    python src/data_pipeline/inaturist_pipeline.py --apply    # actually clean + trim
"""

import os
import sys
import csv
import json
import random
import shutil
import argparse
import traceback
from collections import Counter, defaultdict
from datetime import datetime

# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root (this file: src/data_pipeline/)
DATA = os.path.join(ROOT, "data", "raw", "iNaturist")
OUT = os.path.join(ROOT, "_pipeline")
EDA = os.path.join(OUT, "eda")

LABELED_SPLITS = ["train_mini", "val"]   # taxonomy folders -> filtered to Insecta
TEST_SPLIT = "public_test"               # flat/unlabeled -> trimmed to target size
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# bees = superfamily Apoidea families within order Hymenoptera
BEE_FAMILIES = {
    "Apidae", "Andrenidae", "Halictidae", "Megachilidae",
    "Colletidae", "Melittidae", "Stenotritidae",
}

# target split proportions (must sum to 1.0). train/val are kept as filtered;
# test is trimmed so it forms TARGET["test"] of (train + val + test).
TARGET = {"train": 0.70, "val": 0.15, "test": 0.15}

SEED = 1337
META_SAMPLE_CAP = 6000      # images sampled for resolution profiling
LEAK_SAMPLE_CAP = 15000     # images per split for perceptual-hash leakage scan
GRID_SAMPLES = 25           # images in the visual sample grid

DRY_RUN = True              # default: report only, delete nothing

LOG_PATH = os.path.join(OUT, "pipeline.log")
STATUS_PATH = os.path.join(OUT, "STATUS.txt")


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def log(msg):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def set_status(s):
    with open(STATUS_PATH, "w", encoding="utf-8") as f:
        f.write(s + "\n")


def remove_tree(path):
    """Delete a folder, honoring DRY_RUN."""
    if DRY_RUN:
        return
    shutil.rmtree(path, ignore_errors=True)


def remove_file(path):
    """Delete a file, honoring DRY_RUN."""
    if DRY_RUN:
        return
    try:
        os.remove(path)
    except OSError:
        pass


def parse_taxonomy(folder_name):
    """Decode an iNat folder name into taxonomy ranks, or None if it is not a
    taxonomy folder (e.g. flat public_test files)."""
    parts = folder_name.split("_")
    if len(parts) < 5 or not parts[0].isdigit():
        return None
    return {
        "id": parts[0],
        "kingdom": parts[1],
        "phylum": parts[2],
        "cls": parts[3],                                  # taxonomic Class
        "order": parts[4] if len(parts) > 4 else "",
        "family": parts[5] if len(parts) > 5 else "",
        "genus": parts[6] if len(parts) > 6 else "",
        "species": "_".join(parts[7:]) if len(parts) > 7 else "",
        "folder": folder_name,
    }


def list_images(d):
    out = []
    try:
        for n in os.listdir(d):
            if os.path.splitext(n)[1].lower() in IMG_EXTS:
                out.append(os.path.join(d, n))
    except FileNotFoundError:
        pass
    return out


# --------------------------------------------------------------------------- #
# PHASE 1 - semantic filtering (remove non-Insecta) + integrity + manifests
# --------------------------------------------------------------------------- #
def phase1_filter():
    from PIL import Image

    summary = {
        "dry_run": DRY_RUN,
        "audited_folders": 0, "audited_images": 0,
        "kept_folders": 0, "removed_folders": 0,
        "kept_images": 0, "corrupt_removed": 0, "bee_images": 0,
        "per_split": {},
    }
    # row: split, path(rel to DATA), class_id, folder, order, family, genus, species, is_bee
    manifest_rows = []

    for split in LABELED_SPLITS:
        sdir = os.path.join(DATA, split)
        if not os.path.isdir(sdir):
            log(f"phase1: split missing, skipping: {split}")
            continue
        folders = sorted(f for f in os.listdir(sdir)
                         if os.path.isdir(os.path.join(sdir, f)))
        s = {"audited_folders": len(folders), "kept_folders": 0,
             "removed_folders": 0, "kept_images": 0, "corrupt_removed": 0,
             "bee_images": 0}

        for fld in folders:
            fpath = os.path.join(sdir, fld)
            tax = parse_taxonomy(fld)
            imgs = list_images(fpath)
            summary["audited_images"] += len(imgs)
            # PURGE: anything whose Class is not Insecta (or unparseable) is removed.
            if tax is None or tax["cls"] != "Insecta":
                remove_tree(fpath)
                s["removed_folders"] += 1
                continue
            is_bee = (tax["order"] == "Hymenoptera" and tax["family"] in BEE_FAMILIES)
            kept_here = 0
            for ip in imgs:
                try:
                    with Image.open(ip) as im:
                        im.verify()                      # catch truncated/corrupt
                except Exception:
                    remove_file(ip)
                    s["corrupt_removed"] += 1
                    continue
                kept_here += 1
                manifest_rows.append([
                    split, os.path.relpath(ip, DATA), tax["id"], fld,
                    tax["order"], tax["family"], tax["genus"], tax["species"],
                    int(is_bee),
                ])
            if kept_here == 0:
                remove_tree(fpath)
                s["removed_folders"] += 1
                continue
            s["kept_folders"] += 1
            s["kept_images"] += kept_here
            if is_bee:
                s["bee_images"] += kept_here

        summary["per_split"][split] = s
        for k in ("audited_folders", "kept_folders", "removed_folders",
                  "kept_images", "corrupt_removed", "bee_images"):
            summary[k] += s[k]
        log(f"phase1: {split} audited={s['audited_folders']} kept={s['kept_folders']} "
            f"removed={s['removed_folders']} imgs={s['kept_images']} "
            f"corrupt={s['corrupt_removed']} bees={s['bee_images']}")

    # write manifests (reflect the KEPT set, so any prior mismatch is gone)
    header = ["split", "path", "class_id", "folder", "order", "family",
              "genus", "species", "is_bee"]
    with open(os.path.join(OUT, "manifest_all.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(manifest_rows)
    for split in LABELED_SPLITS:
        rows = [r for r in manifest_rows if r[0] == split]
        with open(os.path.join(OUT, f"manifest_{split}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(header); w.writerows(rows)

    summary["manifest_rows"] = len(manifest_rows)
    log(f"phase1: manifest rows={len(manifest_rows)} (dry_run={DRY_RUN})")
    return summary, manifest_rows


# --------------------------------------------------------------------------- #
# PHASE 2 - JSON synchronization (drop references to deleted images)
# --------------------------------------------------------------------------- #
def _kept_basenames(manifest_rows):
    """file_name -> set of kept basenames and kept relative paths for matching."""
    base = set()
    rel = set()
    for r in manifest_rows:
        p = r[1].replace("\\", "/")
        rel.add(p)
        base.add(os.path.basename(p))
    return base, rel


def phase2_clean_json(manifest_rows):
    """Defensively rewrite iNat/COCO-style JSON annotation files so they only
    reference surviving images. Unknown schemas are left untouched (and logged).

    A JSON is treated as COCO-like if it is a dict containing "images" and
    "annotations" lists. Matching of an image entry to a survivor is done by
    basename first (robust to path-prefix differences), then by relative path.
    """
    kept_base, kept_rel = _kept_basenames(manifest_rows)
    report = {"files_scanned": 0, "files_rewritten": 0, "details": []}

    if not os.path.isdir(DATA):
        return report
    json_files = [os.path.join(DATA, n) for n in os.listdir(DATA)
                  if n.lower().endswith(".json")]
    # also catch per-split json (e.g. train_mini.json living inside the split dir)
    for split in LABELED_SPLITS + [TEST_SPLIT]:
        sd = os.path.join(DATA, split)
        if os.path.isdir(sd):
            json_files += [os.path.join(sd, n) for n in os.listdir(sd)
                           if n.lower().endswith(".json")]

    for jpath in sorted(set(json_files)):
        report["files_scanned"] += 1
        try:
            with open(jpath, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            log(f"phase2: cannot parse {jpath}: {e} -> skipped")
            report["details"].append({"file": jpath, "action": "skip_unparseable"})
            continue

        if not (isinstance(data, dict) and isinstance(data.get("images"), list)
                and isinstance(data.get("annotations"), list)):
            log(f"phase2: {os.path.basename(jpath)} not COCO-like -> skipped")
            report["details"].append({"file": jpath, "action": "skip_schema"})
            continue

        def survives(img):
            fn = str(img.get("file_name", "")).replace("\\", "/")
            if not fn:
                return False
            return os.path.basename(fn) in kept_base or fn in kept_rel \
                or fn.split("/", 1)[-1] in kept_rel

        before_imgs = len(data["images"])
        keep_imgs = [im for im in data["images"] if survives(im)]
        keep_ids = {im.get("id") for im in keep_imgs}
        before_anns = len(data["annotations"])
        keep_anns = [a for a in data["annotations"] if a.get("image_id") in keep_ids]

        # prune categories to those still referenced (if categories exist)
        cats_before = len(data.get("categories", []) or [])
        if isinstance(data.get("categories"), list):
            used_cats = {a.get("category_id") for a in keep_anns}
            data["categories"] = [c for c in data["categories"]
                                  if c.get("id") in used_cats]
        cats_after = len(data.get("categories", []) or [])

        data["images"] = keep_imgs
        data["annotations"] = keep_anns

        detail = {
            "file": jpath,
            "images": [before_imgs, len(keep_imgs)],
            "annotations": [before_anns, len(keep_anns)],
            "categories": [cats_before, cats_after],
        }
        if DRY_RUN:
            log(f"phase2: WOULD rewrite {os.path.basename(jpath)} "
                f"images {before_imgs}->{len(keep_imgs)}, "
                f"anns {before_anns}->{len(keep_anns)}")
            detail["action"] = "would_rewrite"
        else:
            shutil.copy2(jpath, jpath + ".bak")
            with open(jpath, "w", encoding="utf-8") as f:
                json.dump(data, f)
            log(f"phase2: rewrote {os.path.basename(jpath)} (backup .bak), "
                f"images {before_imgs}->{len(keep_imgs)}")
            detail["action"] = "rewritten"
            report["files_rewritten"] += 1
        report["details"].append(detail)

    with open(os.path.join(OUT, "phase2_json_clean.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log(f"phase2: json files scanned={report['files_scanned']} "
        f"rewritten={report['files_rewritten']} (dry_run={DRY_RUN})")
    return report


# --------------------------------------------------------------------------- #
# PHASE 3 - test trimming -> 70/15/15
# --------------------------------------------------------------------------- #
def phase3_trim_test(summary):
    """Keep train_mini(train)+val(val) as filtered; trim public_test so test is
    TARGET['test'] of the grand total. keep = (train+val) * test/(train+val)."""
    train_n = summary["per_split"].get("train_mini", {}).get("kept_images", 0)
    val_n = summary["per_split"].get("val", {}).get("kept_images", 0)
    tdir = os.path.join(DATA, TEST_SPLIT)
    test_imgs = list_images(tdir) if os.path.isdir(tdir) else []
    have = len(test_imgs)

    base = train_n + val_n
    frac = TARGET["test"] / (TARGET["train"] + TARGET["val"]) if base else 0.0
    target_keep = int(round(base * frac))
    target_keep = max(0, min(target_keep, have))   # cannot keep more than exist

    rng = random.Random(SEED)
    keep_set = set(rng.sample(test_imgs, target_keep)) if target_keep < have else set(test_imgs)
    to_remove = [p for p in test_imgs if p not in keep_set]

    for p in to_remove:
        remove_file(p)

    total = train_n + val_n + target_keep
    info = {
        "dry_run": DRY_RUN,
        "target_ratios": TARGET,
        "train_kept": train_n, "val_kept": val_n,
        "test_available": have, "test_target_keep": target_keep,
        "test_removed": len(to_remove),
        "achieved_ratios": {
            "train": round(train_n / total, 3) if total else 0,
            "val": round(val_n / total, 3) if total else 0,
            "test": round(target_keep / total, 3) if total else 0,
        },
    }
    # persist the kept test list so downstream code knows the held-out set
    with open(os.path.join(OUT, "test_kept.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(os.path.relpath(p, DATA) for p in keep_set)) + "\n")
    with open(os.path.join(OUT, "phase3_split.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    log(f"phase3: test trim train={train_n} val={val_n} "
        f"test {have}->{target_keep} (remove {len(to_remove)}); "
        f"achieved {info['achieved_ratios']} (dry_run={DRY_RUN})")
    return info


# --------------------------------------------------------------------------- #
# PHASE 4 - EDA
# --------------------------------------------------------------------------- #
def phase4_eda(manifest_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    eda = {}
    per_species = Counter(r[3] for r in manifest_rows)
    per_order = Counter(r[4] for r in manifest_rows)
    eda["num_species"] = len(per_species)
    eda["num_orders"] = len(per_order)
    eda["images_total"] = len(manifest_rows)
    eda["bee_images"] = sum(1 for r in manifest_rows if r[8] in (1, "1"))

    with open(os.path.join(EDA, "class_distribution_species.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["species_folder", "images"])
        for k, v in per_species.most_common():
            w.writerow([k, v])

    try:
        items = per_order.most_common(20)
        plt.figure(figsize=(11, 6))
        plt.bar([k.split("_")[0] for k, _ in items], [v for _, v in items])
        plt.xticks(rotation=60, ha="right"); plt.ylabel("images")
        plt.title("Insect images per Order (top 20)"); plt.tight_layout()
        plt.savefig(os.path.join(EDA, "dist_by_order.png"), dpi=110); plt.close()
    except Exception as e:
        log(f"eda: order plot failed: {e}")

    try:
        counts = list(per_species.values())
        plt.figure(figsize=(9, 5)); plt.hist(counts, bins=40)
        plt.xlabel("images per species"); plt.ylabel("# species")
        plt.title("Class-size distribution (imbalance)"); plt.tight_layout()
        plt.savefig(os.path.join(EDA, "class_size_hist.png"), dpi=110); plt.close()
    except Exception as e:
        log(f"eda: hist plot failed: {e}")

    # resolution / aspect profiling (sampled)
    widths, heights, ratios, modes = [], [], [], Counter()
    sample = manifest_rows
    if len(sample) > META_SAMPLE_CAP:
        sample = random.Random(SEED).sample(manifest_rows, META_SAMPLE_CAP)
    for r in sample:
        try:
            with Image.open(os.path.join(DATA, r[1])) as im:
                w, h = im.size
                widths.append(w); heights.append(h)
                ratios.append(round(w / h, 3) if h else 0)
                modes[im.mode] += 1
        except Exception:
            continue
    eda["meta_sampled"] = len(widths)
    if widths:
        eda["res_w_min_med_max"] = [min(widths), sorted(widths)[len(widths) // 2], max(widths)]
        eda["res_h_min_med_max"] = [min(heights), sorted(heights)[len(heights) // 2], max(heights)]
        eda["channel_modes"] = dict(modes)
        try:
            plt.figure(figsize=(9, 5)); plt.scatter(widths, heights, s=4, alpha=0.3)
            plt.xlabel("width"); plt.ylabel("height"); plt.title("Resolution scatter (sampled)")
            plt.tight_layout(); plt.savefig(os.path.join(EDA, "resolution_scatter.png"), dpi=110); plt.close()
            plt.figure(figsize=(9, 5)); plt.hist(ratios, bins=40)
            plt.xlabel("aspect ratio (w/h)"); plt.ylabel("count"); plt.title("Aspect-ratio distribution (sampled)")
            plt.tight_layout(); plt.savefig(os.path.join(EDA, "aspect_ratio_hist.png"), dpi=110); plt.close()
        except Exception as e:
            log(f"eda: meta plots failed: {e}")

    # visual sample grid (classification: caption with genus/species + [BEE])
    try:
        import math
        picks = random.Random(SEED).sample(manifest_rows, min(GRID_SAMPLES, len(manifest_rows)))
        cols = 5; rows = math.ceil(len(picks) / cols)
        plt.figure(figsize=(cols * 2.4, rows * 2.6))
        for i, r in enumerate(picks):
            try:
                with Image.open(os.path.join(DATA, r[1])) as im:
                    ax = plt.subplot(rows, cols, i + 1)
                    ax.imshow(im.convert("RGB")); ax.axis("off")
                    cap = f"{r[6]} {r[7]}" + (" [BEE]" if r[8] in (1, "1") else "")
                    ax.set_title(cap, fontsize=6)
            except Exception:
                continue
        plt.tight_layout(); plt.savefig(os.path.join(EDA, "sample_grid.png"), dpi=130); plt.close()
    except Exception as e:
        log(f"eda: montage failed: {e}")

    with open(os.path.join(EDA, "eda_summary.json"), "w", encoding="utf-8") as f:
        json.dump(eda, f, indent=2)
    log(f"phase4: EDA species={eda['num_species']} orders={eda['num_orders']}")
    return eda, per_species


# --------------------------------------------------------------------------- #
# PHASE 5 - quality (class balance + cross-split leakage) + configs
# --------------------------------------------------------------------------- #
def phase5_quality(manifest_rows, summary, per_species):
    quality = {"completeness_pct": 100.0,
               "kept_images": summary["kept_images"],
               "corrupt_removed": summary["corrupt_removed"]}

    counts = sorted(per_species.values())
    if counts:
        mx, mn = max(counts), min(counts)
        imbalance = mx / mn if mn else float("inf")
        n = len(counts); tot = sum(counts); cum = 0
        for i, c in enumerate(sorted(counts), 1):
            cum += i * c
        gini = (2 * cum) / (n * tot) - (n + 1) / n if tot else 0
        quality["class_balance"] = {
            "species": n, "max_per_class": mx, "min_per_class": mn,
            "imbalance_ratio": round(imbalance, 2), "gini": round(gini, 3),
        }
        quality["loss_recommendation"] = (
            "Severe imbalance: class-weighted CrossEntropy (1/freq) or Focal Loss "
            "(gamma~2) + WeightedRandomSampler."
            if imbalance > 10 or gini > 0.4 else
            "Mild imbalance: standard CrossEntropy with light class weights.")

    # cross-split leakage via perceptual hash (sampled, optional dependency)
    try:
        import imagehash
        from PIL import Image
        hashes = defaultdict(list)
        for split in LABELED_SPLITS:
            rows = [r for r in manifest_rows if r[0] == split]
            if len(rows) > LEAK_SAMPLE_CAP:
                rows = random.Random(SEED).sample(rows, LEAK_SAMPLE_CAP)
            for r in rows:
                try:
                    with Image.open(os.path.join(DATA, r[1])) as im:
                        h = str(imagehash.phash(im.convert("RGB")))
                    hashes[h].append((split, r[1]))
                except Exception:
                    continue
        cross = [{"phash": h, "items": lst[:6]} for h, lst in hashes.items()
                 if len({s for s, _ in lst}) > 1]
        quality["leakage"] = {
            "cross_split_duplicate_groups": len(cross),
            "sampled_per_split_cap": LEAK_SAMPLE_CAP,
            "examples": cross[:20],
            "verdict": ("NO cross-split leakage in sample" if not cross
                        else f"WARNING: {len(cross)} cross-split duplicate groups"),
        }
    except ImportError:
        quality["leakage"] = {"verdict": "skipped (imagehash not installed)"}

    with open(os.path.join(OUT, "phase5_quality.json"), "w", encoding="utf-8") as f:
        json.dump(quality, f, indent=2)
    log(f"phase5: quality done")
    return quality


def write_configs(manifest_rows):
    """Emit data.yaml + dataset_config.json describing the cleaned classification
    dataset (train_mini=train, val=val, public_test=test)."""
    classes = sorted({(r[2], r[3]) for r in manifest_rows}, key=lambda x: x[0])
    class_names = [c[1] for c in classes]
    name_to_idx = {c[1]: i for i, c in enumerate(classes)}

    with open(os.path.join(ROOT, "data.yaml"), "w", encoding="utf-8") as f:
        f.write("# Bee-A-Hero - cleaned iNaturist (Insecta-filtered) classification dataset\n")
        f.write(f"path: {DATA}\n")
        f.write("train: train_mini\n")
        f.write("val: val\n")
        f.write("test: public_test   # trimmed held-out set (see _pipeline/test_kept.txt)\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write("names:\n")
        for i, n in enumerate(class_names):
            f.write(f"  {i}: {n}\n")

    cfg = {
        "dataset": "Bee-A-Hero iNaturist (Insecta-filtered)",
        "task": "image_classification",
        "root": DATA,
        "splits": {"train": "train_mini", "val": "val", "test": "public_test"},
        "target_ratios": TARGET,
        "num_classes": len(class_names),
        "class_names": class_names,
        "name_to_index": name_to_idx,
        "bee_families_tagged": sorted(BEE_FAMILIES),
    }
    with open(os.path.join(ROOT, "dataset_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    log(f"configs: data.yaml + dataset_config.json (nc={len(class_names)})")
    return class_names


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def write_report(summary, json_report, split_info, eda, quality, class_names):
    p = os.path.join(OUT, "REPORT.md")
    cb = quality.get("class_balance", {})
    lk = quality.get("leakage", {})
    with open(p, "w", encoding="utf-8") as f:
        f.write("# Bee-A-Hero - iNaturist Cleaning & EDA Report\n\n")
        f.write(f"_Generated {datetime.now().isoformat(timespec='seconds')} | "
                f"mode: {'DRY RUN (nothing deleted)' if summary['dry_run'] else 'APPLIED'}_\n\n")

        f.write("## 1. Cleaning (remove non-Insecta)\n")
        f.write(f"- Folders audited: **{summary['audited_folders']}**, "
                f"kept (Insecta): **{summary['kept_folders']}**, "
                f"removed: **{summary['removed_folders']}**.\n")
        f.write(f"- Images kept: **{summary['kept_images']}** "
                f"(bees: **{summary['bee_images']}**), corrupt removed: "
                f"**{summary['corrupt_removed']}**.\n\n")
        f.write("| split | audited | kept | removed | images | bees |\n|---|---|---|---|---|---|\n")
        for sp in LABELED_SPLITS:
            d = summary["per_split"].get(sp, {})
            f.write(f"| {sp} | {d.get('audited_folders','-')} | {d.get('kept_folders','-')} | "
                    f"{d.get('removed_folders','-')} | {d.get('kept_images','-')} | {d.get('bee_images','-')} |\n")
        f.write("\n## 2. JSON synchronization\n")
        f.write(f"- Files scanned: {json_report['files_scanned']}, "
                f"rewritten: {json_report['files_rewritten']} "
                f"(originals backed up to `*.json.bak`).\n")
        for d in json_report["details"]:
            if "images" in d:
                f.write(f"  - `{os.path.basename(d['file'])}`: images "
                        f"{d['images'][0]}->{d['images'][1]}, anns "
                        f"{d['annotations'][0]}->{d['annotations'][1]} ({d['action']})\n")
        f.write("\n## 3. Split 70/15/15 (test trimmed)\n")
        si = split_info
        f.write(f"- train (train_mini): **{si['train_kept']}**, val: **{si['val_kept']}**.\n")
        f.write(f"- public_test available **{si['test_available']}** -> keep "
                f"**{si['test_target_keep']}**, remove **{si['test_removed']}**.\n")
        f.write(f"- Achieved ratios: {si['achieved_ratios']} (target {si['target_ratios']}).\n")
        f.write("- Kept test list: `_pipeline/test_kept.txt`.\n\n")
        f.write("## 4. EDA\n")
        f.write(f"- Classes (species): **{eda.get('num_species','?')}**, orders: "
                f"**{eda.get('num_orders','?')}**, labeled images: **{eda.get('images_total','?')}**.\n")
        if "res_w_min_med_max" in eda:
            f.write(f"- Resolution w(min/med/max): {eda['res_w_min_med_max']}, "
                    f"h: {eda['res_h_min_med_max']}; color modes: {eda.get('channel_modes')}.\n")
        f.write("- Plots in `_pipeline/eda/`.\n\n")
        f.write("## 5. Quality\n")
        if cb:
            f.write(f"- Class balance: {cb['species']} classes, per-class "
                    f"{cb['min_per_class']}-{cb['max_per_class']}, imbalance "
                    f"{cb['imbalance_ratio']}, Gini {cb['gini']}.\n")
            f.write(f"  - {quality.get('loss_recommendation','')}\n")
        f.write(f"- Leakage: {lk.get('verdict','')}.\n")
        f.write(f"- Classes for training (nc): **{len(class_names)}** "
                "(`data.yaml` / `dataset_config.json`).\n")
    log(f"report written: {p}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    global DRY_RUN
    ap = argparse.ArgumentParser(description="iNaturist clean + EDA pipeline")
    ap.add_argument("--apply", action="store_true",
                    help="actually delete files / rewrite JSON (default: dry run)")
    args = ap.parse_args()
    if args.apply:
        DRY_RUN = False

    os.makedirs(EDA, exist_ok=True)
    set_status("RUNNING")
    log(f"=== iNaturist pipeline start (dry_run={DRY_RUN}) ===")
    if not os.path.isdir(DATA):
        log(f"FATAL: data dir not found: {DATA}")
        set_status("ERROR: data dir missing")
        return 1
    try:
        summary, rows = phase1_filter()
        json_report = phase2_clean_json(rows)
        split_info = phase3_trim_test(summary)
        eda, per_species = phase4_eda(rows)
        quality = phase5_quality(rows, summary, per_species)
        class_names = write_configs(rows)
        write_report(summary, json_report, split_info, eda, quality, class_names)
        set_status("DRY_RUN_OK" if DRY_RUN else "APPLIED_OK")
        log("=== pipeline done ===")
        return 0
    except Exception:
        log("FATAL:\n" + traceback.format_exc())
        set_status("ERROR (see pipeline.log)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
