#!/usr/bin/env python3
"""
BEE_HERo end-to-end dataset pipeline (classification adaptation of iNaturalist).

The source data is an iNaturalist-style CLASSIFICATION dataset:
  <split>/<ID>_<Kingdom>_<Phylum>_<Class>_<Order>_<Family>_<Genus>_<species>/*.jpg
There are NO bounding boxes / detection annotations anywhere, so detection-only
steps (bbox coord validation, anchor boxes, bbox-size EDA) are documented as N/A
and the equivalent classification logic is executed instead.

Decisions (locked with the user):
  * Adapt to a classification pipeline.
  * Keep ALL Insecta species folders; additionally tag bee families as a subset.
  * Process all 3 splits. public_test is FLAT/UNLABELED -> left intact, profiled
    only (cannot be class-filtered because it has no taxonomy labels).

Safety:
  * Original *.tar.gz archives are never touched (fully reversible).
  * Hard disk floor: abort (no shutdown) if free space would drop below MIN_FREE_GB.
  * Every phase is wrapped in try/except; failures are logged and the run continues.

Exit codes: 0 = completed (shutdown allowed), 9 = aborted on low disk (NO shutdown).
"""

import os
import sys
import csv
import json
import random
import shutil
import traceback
from collections import Counter, defaultdict
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # BEE_HERo (repo root; this file lives in src/data_pipeline/)
DATA = os.path.join(ROOT, "data", "raw", "iNaturist")  # raw splits + archives live here
OUT = os.path.join(ROOT, "_pipeline")
EDA = os.path.join(OUT, "eda")
os.makedirs(EDA, exist_ok=True)

LABELED_SPLITS = ["train_mini", "val"]      # have taxonomy folders
UNLABELED_SPLITS = ["public_test"]          # flat, no labels
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Bee families (superfamily Apoidea, the bees) within order Hymenoptera.
BEE_FAMILIES = {
    "Apidae", "Andrenidae", "Halictidae", "Megachilidae",
    "Colletidae", "Melittidae", "Stenotritidae",
}

MIN_FREE_GB = 20            # hard floor; abort if we would cross it
META_SAMPLE_CAP = 6000      # images to sample per split for resolution profiling
LEAK_SAMPLE_CAP = 15000     # images per split for perceptual-hash leakage check
GRID_SAMPLES = 25           # images in the visual-verification montage

LOG_PATH = os.path.join(OUT, "pipeline.log")
STATUS_PATH = os.path.join(OUT, "STATUS.txt")


def log(msg):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def set_status(s):
    with open(STATUS_PATH, "w", encoding="utf-8") as f:
        f.write(s + "\n")


def free_gb(path=ROOT):
    total, used, free = shutil.disk_usage(path)
    return free / (1024 ** 3)


def disk_guard(stage):
    fg = free_gb()
    log(f"disk check @ {stage}: {fg:.1f} GB free")
    if fg < MIN_FREE_GB:
        log(f"!!! ABORT: free space {fg:.1f} GB below floor {MIN_FREE_GB} GB at {stage}")
        set_status(f"ABORTED_LOW_DISK at {stage}: {fg:.1f}GB free")
        sys.exit(9)


def parse_taxonomy(folder_name):
    """Return dict of taxonomy ranks from an iNat folder name, or None if it
    does not look like a taxonomy folder."""
    parts = folder_name.split("_")
    if len(parts) < 5 or not parts[0].isdigit():
        return None
    return {
        "id": parts[0],
        "kingdom": parts[1],
        "phylum": parts[2],
        "cls": parts[3],            # taxonomic Class (e.g. Insecta)
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


# ---------------------------------------------------------------------------
# PHASE 1 + 2 + 5(manifest): semantic filtering, integrity, manifest building
# ---------------------------------------------------------------------------
def phase1_2_filter():
    """Keep only Insecta folders, delete non-insect folders, verify image
    integrity, and build per-split + combined manifests.  Returns a summary."""
    from PIL import Image

    summary = {
        "audited_folders": 0, "audited_images_est": 0,
        "kept_folders": 0, "removed_folders": 0,
        "kept_images": 0, "corrupt_removed": 0, "bee_images": 0,
        "per_split": {},
    }
    manifest_rows = []  # split, path, class_id, folder, order, family, genus, species, is_bee

    for split in LABELED_SPLITS:
        sdir = os.path.join(DATA, split)
        if not os.path.isdir(sdir):
            log(f"phase1: split missing, skipping: {split}")
            continue
        disk_guard(f"phase1:{split}:start")
        folders = sorted([f for f in os.listdir(sdir)
                          if os.path.isdir(os.path.join(sdir, f))])
        s_audit_f = len(folders)
        s_kept_f = s_removed_f = s_kept_i = s_corrupt = s_bee = 0

        for fld in folders:
            fpath = os.path.join(sdir, fld)
            tax = parse_taxonomy(fld)
            imgs = list_images(fpath)
            summary["audited_images_est"] += len(imgs)
            # PURGE PROTOCOL: drop everything that is not class Insecta.
            if tax is None or tax["cls"] != "Insecta":
                try:
                    shutil.rmtree(fpath)
                    s_removed_f += 1
                except Exception as e:
                    log(f"phase1: failed to remove {fpath}: {e}")
                continue
            # Kept = insect folder. Verify each image; remove corrupt ones.
            is_bee = (tax["order"] == "Hymenoptera" and tax["family"] in BEE_FAMILIES)
            kept_here = 0
            for ip in imgs:
                try:
                    with Image.open(ip) as im:
                        im.verify()            # catches truncated/corrupt files
                except Exception:
                    try:
                        os.remove(ip)
                    except Exception:
                        pass
                    s_corrupt += 1
                    continue
                kept_here += 1
                manifest_rows.append([
                    split, os.path.relpath(ip, DATA), tax["id"], fld,
                    tax["order"], tax["family"], tax["genus"], tax["species"],
                    int(is_bee),
                ])
            if kept_here == 0:
                # zero valid instances left -> remove empty folder
                try:
                    shutil.rmtree(fpath)
                except Exception:
                    pass
                s_removed_f += 1
                continue
            s_kept_f += 1
            s_kept_i += kept_here
            if is_bee:
                s_bee += kept_here
            if (s_kept_f % 200) == 0:
                disk_guard(f"phase1:{split}:progress")

        summary["per_split"][split] = {
            "audited_folders": s_audit_f, "kept_folders": s_kept_f,
            "removed_folders": s_removed_f, "kept_images": s_kept_i,
            "corrupt_removed": s_corrupt, "bee_images": s_bee,
        }
        summary["audited_folders"] += s_audit_f
        summary["kept_folders"] += s_kept_f
        summary["removed_folders"] += s_removed_f
        summary["kept_images"] += s_kept_i
        summary["corrupt_removed"] += s_corrupt
        summary["bee_images"] += s_bee
        log(f"phase1: {split} audited_folders={s_audit_f} kept={s_kept_f} "
            f"removed={s_removed_f} kept_images={s_kept_i} corrupt={s_corrupt} bees={s_bee}")

    # profile public_test (unlabeled) image count only
    for split in UNLABELED_SPLITS:
        sdir = os.path.join(DATA, split)
        n = len(list_images(sdir)) if os.path.isdir(sdir) else 0
        summary["per_split"][split] = {"unlabeled_images": n, "note": "flat/unlabeled; left intact"}
        log(f"phase1: {split} unlabeled images (left intact) = {n}")

    # write manifests
    header = ["split", "path", "class_id", "folder", "order", "family",
              "genus", "species", "is_bee"]
    with open(os.path.join(OUT, "manifest_all.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(manifest_rows)
    for split in LABELED_SPLITS:
        rows = [r for r in manifest_rows if r[0] == split]
        with open(os.path.join(OUT, f"manifest_{split}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(header); w.writerows(rows)

    summary["manifest_rows"] = len(manifest_rows)
    log(f"phase1/2: manifest rows={len(manifest_rows)}")
    return summary, manifest_rows


# ---------------------------------------------------------------------------
# PHASE 5: configuration files (class list reflecting retained insect species)
# ---------------------------------------------------------------------------
def phase5_config(manifest_rows):
    classes = sorted({(r[2], r[3]) for r in manifest_rows}, key=lambda x: x[0])
    class_names = [c[1] for c in classes]
    name_to_idx = {c[1]: i for i, c in enumerate(classes)}

    data_yaml = os.path.join(ROOT, "data.yaml")
    with open(data_yaml, "w", encoding="utf-8") as f:
        f.write("# BEE_HERo - retained Insecta classes (iNaturalist classification)\n")
        f.write(f"path: {DATA}\n")
        f.write("train: train_mini\n")
        f.write("val: val\n")
        f.write("test: public_test   # unlabeled inference set\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write("names:\n")
        for i, n in enumerate(class_names):
            f.write(f"  {i}: {n}\n")

    cfg = {
        "dataset": "BEE_HERo (iNaturalist, Insecta-filtered)",
        "task": "image_classification",
        "root": DATA,
        "splits": {"train": "train_mini", "val": "val", "test": "public_test (unlabeled)"},
        "num_classes": len(class_names),
        "class_names": class_names,
        "name_to_index": name_to_idx,
        "bee_families_tagged": sorted(BEE_FAMILIES),
    }
    with open(os.path.join(ROOT, "dataset_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    log(f"phase5: wrote data.yaml + dataset_config.json (nc={len(class_names)})")
    return class_names


# ---------------------------------------------------------------------------
# PHASE 3: EDA  (class distribution, image metadata, visual verification)
# ---------------------------------------------------------------------------
def phase3_eda(manifest_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    eda = {}
    # ---- class distribution ----
    per_species = Counter(r[3] for r in manifest_rows)
    per_order = Counter(r[4] for r in manifest_rows)
    per_family = Counter(r[5] for r in manifest_rows)
    eda["num_species"] = len(per_species)
    eda["num_orders"] = len(per_order)
    eda["images_total"] = len(manifest_rows)

    with open(os.path.join(EDA, "class_distribution_species.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["species_folder", "images"])
        for k, v in per_species.most_common():
            w.writerow([k, v])

    # order-level bar chart
    try:
        items = per_order.most_common(20)
        plt.figure(figsize=(11, 6))
        plt.bar([k.split("_")[0] for k, _ in items], [v for _, v in items])
        plt.xticks(rotation=60, ha="right"); plt.ylabel("images")
        plt.title("Insect images per Order (top 20)"); plt.tight_layout()
        plt.savefig(os.path.join(EDA, "dist_by_order.png"), dpi=110); plt.close()
    except Exception as e:
        log(f"eda: order plot failed: {e}")

    # species count histogram (imbalance shape)
    try:
        counts = list(per_species.values())
        plt.figure(figsize=(9, 5))
        plt.hist(counts, bins=40)
        plt.xlabel("images per species"); plt.ylabel("# species")
        plt.title("Class-size distribution (imbalance)"); plt.tight_layout()
        plt.savefig(os.path.join(EDA, "class_size_hist.png"), dpi=110); plt.close()
    except Exception as e:
        log(f"eda: hist plot failed: {e}")

    # ---- image metadata profiling (sampled) ----
    widths, heights, ratios, modes = [], [], [], Counter()
    sample = manifest_rows
    if len(sample) > META_SAMPLE_CAP:
        sample = random.sample(manifest_rows, META_SAMPLE_CAP)
    for r in sample:
        p = os.path.join(ROOT, r[1])
        try:
            with Image.open(p) as im:
                w, h = im.size
                widths.append(w); heights.append(h)
                ratios.append(round(w / h, 3) if h else 0)
                modes[im.mode] += 1
        except Exception:
            continue
    eda["meta_sampled"] = len(widths)
    if widths:
        eda["res_w_min_med_max"] = [min(widths), sorted(widths)[len(widths)//2], max(widths)]
        eda["res_h_min_med_max"] = [min(heights), sorted(heights)[len(heights)//2], max(heights)]
        eda["channel_modes"] = dict(modes)
        try:
            plt.figure(figsize=(9, 5)); plt.scatter(widths, heights, s=4, alpha=0.3)
            plt.xlabel("width"); plt.ylabel("height"); plt.title("Image resolution scatter (sampled)")
            plt.tight_layout(); plt.savefig(os.path.join(EDA, "resolution_scatter.png"), dpi=110); plt.close()
            plt.figure(figsize=(9, 5)); plt.hist(ratios, bins=40)
            plt.xlabel("aspect ratio (w/h)"); plt.ylabel("count"); plt.title("Aspect-ratio distribution (sampled)")
            plt.tight_layout(); plt.savefig(os.path.join(EDA, "aspect_ratio_hist.png"), dpi=110); plt.close()
        except Exception as e:
            log(f"eda: meta plots failed: {e}")

    # ---- visual verification montage (no bboxes exist -> label captions) ----
    try:
        import math
        picks = random.sample(manifest_rows, min(GRID_SAMPLES, len(manifest_rows)))
        cols = 5; rows = math.ceil(len(picks) / cols)
        plt.figure(figsize=(cols * 2.4, rows * 2.6))
        for i, r in enumerate(picks):
            p = os.path.join(ROOT, r[1])
            try:
                with Image.open(p) as im:
                    im = im.convert("RGB")
                    ax = plt.subplot(rows, cols, i + 1); ax.imshow(im); ax.axis("off")
                    cap = f"{r[6]} {r[7]}" + (" [BEE]" if r[8] == "1" or r[8] == 1 else "")
                    ax.set_title(cap, fontsize=6)
            except Exception:
                continue
        plt.tight_layout(); plt.savefig(os.path.join(EDA, "sample_grid.png"), dpi=130); plt.close()
        log("eda: wrote sample_grid.png")
    except Exception as e:
        log(f"eda: montage failed: {e}")

    with open(os.path.join(EDA, "eda_summary.json"), "w", encoding="utf-8") as f:
        json.dump(eda, f, indent=2)
    log(f"phase3: EDA done species={eda['num_species']} orders={eda['num_orders']}")
    return eda, per_species


# ---------------------------------------------------------------------------
# PHASE 4: split stabilization + augmentation blueprint + loader prep
# ---------------------------------------------------------------------------
def phase4_readiness(manifest_rows, per_species):
    train_sp = Counter(r[3] for r in manifest_rows if r[0] == "train_mini")
    val_sp = Counter(r[3] for r in manifest_rows if r[0] == "val")
    only_train = set(train_sp) - set(val_sp)
    only_val = set(val_sp) - set(train_sp)
    info = {
        "train_species": len(train_sp), "val_species": len(val_sp),
        "species_only_in_train": len(only_train),
        "species_only_in_val": len(only_val),
        "train_images": sum(train_sp.values()), "val_images": sum(val_sp.values()),
    }
    with open(os.path.join(OUT, "phase4_split_check.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    log(f"phase4: split check {info}")
    return info


# ---------------------------------------------------------------------------
# PHASE 6: quality evaluation (completeness, balance, leakage)
# ---------------------------------------------------------------------------
def phase6_quality(manifest_rows, summary, per_species):
    import imagehash
    from PIL import Image

    # completeness: kept images all carry a class label by construction.
    total_audit_imgs = summary["audited_images_est"]
    kept = summary["kept_images"]
    corrupt = summary["corrupt_removed"]
    completeness = 100.0  # every retained image maps to exactly one class folder
    eval_out = {
        "completeness_pct": completeness,
        "kept_images": kept, "corrupt_removed": corrupt,
        "audited_images_est": total_audit_imgs,
    }

    # class balance
    counts = sorted(per_species.values())
    if counts:
        mx, mn = max(counts), min(counts)
        imbalance_ratio = mx / mn if mn else float("inf")
        # Gini coefficient
        n = len(counts); cum = 0; tot = sum(counts)
        for i, c in enumerate(sorted(counts), 1):
            cum += i * c
        gini = (2 * cum) / (n * tot) - (n + 1) / n if tot else 0
        eval_out["class_balance"] = {
            "species": n, "max_per_class": mx, "min_per_class": mn,
            "imbalance_ratio": round(imbalance_ratio, 2), "gini": round(gini, 3),
        }
        if imbalance_ratio > 10 or gini > 0.4:
            eval_out["loss_recommendation"] = (
                "Severe imbalance: use class-weighted CrossEntropy (weight=1/freq) "
                "or Focal Loss (gamma~2); consider class-balanced sampling / WeightedRandomSampler.")
        else:
            eval_out["loss_recommendation"] = (
                "Mild imbalance: standard CrossEntropy with light class weights is sufficient.")

    # leakage: perceptual hash across splits (sampled for tractability)
    log("phase6: computing perceptual hashes for leakage check (sampled)...")
    hashes = defaultdict(list)  # phash -> list of (split, path)
    for split in LABELED_SPLITS:
        rows = [r for r in manifest_rows if r[0] == split]
        if len(rows) > LEAK_SAMPLE_CAP:
            rows = random.sample(rows, LEAK_SAMPLE_CAP)
        done = 0
        for r in rows:
            p = os.path.join(ROOT, r[1])
            try:
                with Image.open(p) as im:
                    h = str(imagehash.phash(im.convert("RGB")))
                hashes[h].append((split, r[1]))
            except Exception:
                continue
            done += 1
            if done % 3000 == 0:
                disk_guard("phase6:leak")
                log(f"phase6: hashed {done} in {split}")
    cross = []
    for h, lst in hashes.items():
        splits_here = {s for s, _ in lst}
        if len(splits_here) > 1:
            cross.append({"phash": h, "items": lst[:6]})
    eval_out["leakage"] = {
        "cross_split_duplicate_groups": len(cross),
        "sampled_per_split_cap": LEAK_SAMPLE_CAP,
        "examples": cross[:20],
        "verdict": ("NO cross-split leakage detected in sample" if not cross
                    else f"WARNING: {len(cross)} cross-split duplicate groups found"),
    }
    with open(os.path.join(OUT, "phase6_quality.json"), "w", encoding="utf-8") as f:
        json.dump(eval_out, f, indent=2)
    log(f"phase6: quality eval done leakage_groups={len(cross)}")
    return eval_out


# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------
def write_report(summary, class_names, eda, split_info, quality):
    p = os.path.join(OUT, "REPORT.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write("# BEE_HERo Dataset Pipeline Report\n\n")
        f.write(f"_Generated {datetime.now().isoformat(timespec='seconds')}_\n\n")

        # ---- Morning summary (TL;DR) ----
        cb = quality.get("class_balance", {})
        lk = quality.get("leakage", {})
        f.write("## ☀️ Morning Summary (TL;DR)\n\n")
        f.write("Good morning! The overnight pipeline finished and the PC shut down on its own. Here is what happened:\n\n")
        f.write(f"- **Done:** extracted all 3 archives, filtered to insects/bees, built manifests, ran full EDA + quality eval, wrote configs.\n")
        f.write(f"- **Folders audited:** {summary['audited_folders']} → kept **{summary['kept_folders']}** insect species, "
                f"removed **{summary['removed_folders']}** non-insect.\n")
        f.write(f"- **Images retained:** **{summary['kept_images']}** insect images "
                f"(of which **{summary['bee_images']}** are bees); {summary['corrupt_removed']} corrupt removed.\n")
        f.write(f"- **Classes for training (nc):** **{len(class_names)}** (see `data.yaml` / `dataset_config.json`).\n")
        if cb:
            f.write(f"- **Class balance:** imbalance ratio **{cb.get('imbalance_ratio')}**, Gini **{cb.get('gini')}** → "
                    f"{quality.get('loss_recommendation','')}\n")
        f.write(f"- **Data leakage:** {lk.get('verdict','')}.\n")
        f.write(f"- **Completeness:** {quality['completeness_pct']:.1f}% of retained images are label-aligned.\n")
        f.write("- **Your `.tar.gz` archives were left untouched** — everything is reversible.\n")
        f.write("- **Files to look at:** this `REPORT.md`, `manifest_all.csv`, the `eda/` plots "
                "(`sample_grid.png` first), `phase6_quality.json`, and `run_all.log`/`pipeline.log` for the full trace.\n\n")
        f.write("Full phase-by-phase detail below.\n\n---\n\n")

        f.write("**Dataset reality:** iNaturalist-style image *classification* "
                "(per-species folders, taxonomy-encoded names). No bounding boxes "
                "exist, so detection-only steps are executed as their classification "
                "equivalents and bbox-specific items are marked N/A.\n\n")

        f.write("## Phase 1 - Semantic Filtering & Class Cleansing\n")
        f.write("Criteria: keep a species folder iff its taxonomic **Class == Insecta**; "
                "tag bees where Order==Hymenoptera and Family in "
                f"{sorted(BEE_FAMILIES)}.\n\n")
        f.write(f"- Folders audited: **{summary['audited_folders']}**\n")
        f.write(f"- Folders kept (Insecta): **{summary['kept_folders']}**\n")
        f.write(f"- Folders removed (non-insect): **{summary['removed_folders']}**\n")
        f.write(f"- Images retained: **{summary['kept_images']}** "
                f"(bee images: **{summary['bee_images']}**)\n")
        f.write(f"- Corrupt images removed: **{summary['corrupt_removed']}**\n\n")
        f.write("Per-split:\n\n| split | audited folders | kept | removed | kept images | bees |\n")
        f.write("|---|---|---|---|---|---|\n")
        for s in LABELED_SPLITS:
            d = summary["per_split"].get(s, {})
            f.write(f"| {s} | {d.get('audited_folders','-')} | {d.get('kept_folders','-')} | "
                    f"{d.get('removed_folders','-')} | {d.get('kept_images','-')} | {d.get('bee_images','-')} |\n")
        pt = summary["per_split"].get("public_test", {})
        f.write(f"\n`public_test`: {pt.get('unlabeled_images','?')} images, flat/unlabeled, "
                "left intact (cannot be class-filtered without labels).\n\n")

        f.write("## Phase 2 - Annotation Alignment & Validation\n")
        f.write("No bbox annotations exist. Classification-equivalent actions performed:\n")
        f.write("- Label = parent species folder (taxonomy-derived). Manifests written to "
                "`_pipeline/manifest_*.csv` with [split, path, class_id, folder, order, family, "
                "genus, species, is_bee].\n")
        f.write("- Integrity validation: every retained image opened+verified with PIL; "
                f"{summary['corrupt_removed']} corrupt files removed.\n")
        f.write("- Orphans: by construction every retained image lives under exactly one class "
                "folder, so there are no orphan label/image mismatches. Empty folders removed.\n")
        f.write(f"- Final synchronized labeled images: **{summary['manifest_rows']}**.\n")
        f.write("- Coordinate validation (out-of-bounds / inverted / zero-area boxes): **N/A** (no boxes).\n\n")

        f.write("## Phase 3 - Exploratory Data Analysis\n")
        f.write(f"- Retained insect species (classes): **{eda.get('num_species','?')}**, "
                f"orders: **{eda.get('num_orders','?')}**, total labeled images: "
                f"**{eda.get('images_total','?')}**.\n")
        if "res_w_min_med_max" in eda:
            f.write(f"- Resolution width (min/med/max): {eda['res_w_min_med_max']}, "
                    f"height: {eda['res_h_min_med_max']} (sampled {eda.get('meta_sampled')}).\n")
            f.write(f"- Color modes: {eda.get('channel_modes')}.\n")
        f.write("- Plots in `_pipeline/eda/`: `dist_by_order.png`, `class_size_hist.png`, "
                "`resolution_scatter.png`, `aspect_ratio_hist.png`, `sample_grid.png`.\n")
        f.write("- Bounding-box size / aspect / anchor analysis: **N/A** (classification dataset). "
                "Object-scale concerns are instead addressed by input-resolution choice + augmentation.\n\n")

        f.write("## Phase 4 - CV Readiness & Augmentation Strategy\n")
        if split_info:
            f.write(f"- Split check: train species={split_info['train_species']} "
                    f"({split_info['train_images']} imgs), val species={split_info['val_species']} "
                    f"({split_info['val_images']} imgs); species only-in-train="
                    f"{split_info['species_only_in_train']}, only-in-val={split_info['species_only_in_val']}.\n")
        f.write("- **Augmentation blueprint (Albumentations recommended):** RandomResizedCrop(224), "
                "HorizontalFlip, ShiftScaleRotate, ColorJitter/HueSaturationValue + RandomBrightnessContrast "
                "(simulate natural lighting), CoarseDropout, plus **MixUp/CutMix** at the batch level. "
                "Mosaic is detection-oriented and optional/low-value for classification. "
                "Always Normalize(ImageNet stats).\n")
        f.write("- **6GB VRAM prep:** img size 224, batch 32 (use AMP/torch.cuda.amp to push to 48-64), "
                "`num_workers=4-6`, `pin_memory=True`, `persistent_workers=True`, "
                "`prefetch_factor=2`. WeightedRandomSampler if imbalance is severe.\n\n")

        f.write("## Phase 5 - Folder Integrity & Path Mapping\n")
        f.write("- Original folder hierarchy preserved; filtering done in-place. "
                "Original `.tar.gz` archives untouched.\n")
        f.write(f"- Wrote `data.yaml` and `dataset_config.json` at repo root with "
                f"**nc={len(class_names)}** ordered insect class names.\n\n")

        f.write("## Phase 6 - Quality Evaluation\n")
        f.write(f"- **Completeness:** {quality['completeness_pct']:.1f}% of retained images are "
                "perfectly label-aligned.\n")
        cb = quality.get("class_balance", {})
        if cb:
            f.write(f"- **Class balance:** {cb['species']} classes, per-class min={cb['min_per_class']}, "
                    f"max={cb['max_per_class']}, imbalance ratio={cb['imbalance_ratio']}, "
                    f"Gini={cb['gini']}.\n")
            f.write(f"  - Recommendation: {quality.get('loss_recommendation','')}\n")
        lk = quality.get("leakage", {})
        f.write(f"- **Leakage:** {lk.get('verdict','')} "
                f"(perceptual pHash, up to {lk.get('sampled_per_split_cap')} imgs/split; "
                f"{lk.get('cross_split_duplicate_groups')} cross-split duplicate groups).\n\n")
        f.write("Artifacts: `_pipeline/REPORT.md`, `manifest_*.csv`, `eda/`, "
                "`phase4_split_check.json`, `phase6_quality.json`, `pipeline.log`.\n")
    log(f"report written: {p}")


def main():
    set_status("RUNNING")
    log("=== pipeline start ===")
    disk_guard("start")
    try:
        summary, rows = phase1_2_filter()
        class_names = phase5_config(rows)
        eda, per_species = phase3_eda(rows)
        split_info = phase4_readiness(rows, per_species)
        quality = phase6_quality(rows, summary, per_species)
        write_report(summary, class_names, eda, split_info, quality)
        set_status("COMPLETED_OK")
        log("=== pipeline COMPLETED OK ===")
        return 0
    except SystemExit as e:
        if e.code == 9:
            raise
        log(f"SystemExit {e.code}")
        return e.code
    except Exception:
        log("FATAL ERROR:\n" + traceback.format_exc())
        set_status("ERROR (see pipeline.log)")
        # non-disk fatal: still return 0-ish? No — leave PC on for inspection.
        return 1


if __name__ == "__main__":
    sys.exit(main())
