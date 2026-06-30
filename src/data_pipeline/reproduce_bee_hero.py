#!/usr/bin/env python3
"""
BEE_HERo — portable reproduction script.
================================================================================
Run this on ANY PC that has the same three iNaturalist archives sitting next to
it, and it rebuilds the *exact same dataset format* we produced:

    extract (if needed) -> label -> filter to Insecta (+ tag bees) ->
    manifests -> leakage-safe stratified 80/10/10 split -> configs

It is a single-file merge of our two original scripts
(`_pipeline/pipeline.py` + `_pipeline/resplit_option3.py`) with all the
machine-specific bits (worker count, absolute paths, Windows assumptions)
removed so it is cross-platform and turnkey.

--------------------------------------------------------------------------------
WHAT THE SOURCE DATA LOOKS LIKE
--------------------------------------------------------------------------------
iNaturalist-style CLASSIFICATION data (NO bounding boxes):

    <split>/<ID>_<Kingdom>_<Phylum>_<Class>_<Order>_<Family>_<Genus>_<species>/*.jpg

Three archives are expected in the same folder as this script:
    train_mini.tar.gz   (labeled, per-species folders)
    val.tar.gz          (labeled, per-species folders)
    public_test.tar.gz  (FLAT, unlabeled -> inference only, never split)

--------------------------------------------------------------------------------
HOW TO RUN
--------------------------------------------------------------------------------
    pip install pillow imagehash            # required
    pip install matplotlib                  # optional (EDA plots)

    # 1. put this file + the 3 .tar.gz in the same folder, then:
    python reproduce_bee_hero.py

    # options:
    python reproduce_bee_hero.py --root /path/to/data   # data lives elsewhere
    python reproduce_bee_hero.py --no-extract           # folders already extracted
    python reproduce_bee_hero.py --purge                # delete non-insect folders
                                                        #   on disk (matches our
                                                        #   final on-disk state)
    python reproduce_bee_hero.py --no-eda               # skip matplotlib plots

OUTPUTS (identical names/format to the originals)
    _pipeline/manifest_all.csv          [split,path,class_id,folder,order,
    _pipeline/manifest_train_mini.csv    family,genus,species,is_bee]
    _pipeline/manifest_val.csv
    _pipeline/splits/train.txt | val.txt | test.txt
    _pipeline/splits/split_assignments.csv  [path,class_id,species,group_id,
                                             phash,split,status]
    _pipeline/splits/resplit_report.md
    _pipeline/eda/*.png + *.csv + eda_summary.json   (if matplotlib present)
    data.yaml  +  dataset_config.json   (repo root, point at the split lists)

Reproducibility: SEED=1337 + HAMMING_THRESH=5 -> the same machine-independent
split every time. The .tar.gz archives are NEVER modified.
"""

import os
import sys
import csv
import json
import time
import random
import shutil
import tarfile
import argparse
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

# ============================== configuration ===============================
LABELED_SPLITS = ("train_mini", "val")     # have taxonomy folders
UNLABELED_SPLITS = ("public_test",)        # flat, no labels -> inference only
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Bee families (superfamily Apoidea) within order Hymenoptera.
BEE_FAMILIES = {
    "Apidae", "Andrenidae", "Halictidae", "Megachilidae",
    "Colletidae", "Melittidae", "Stenotritidae",
}

SPLIT_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}
HAMMING_THRESH = 5            # <= this distance, same class => "same observation"
SEED = 1337                   # makes the split deterministic across machines
META_SAMPLE_CAP = 6000        # images sampled per split for resolution profiling
GRID_SAMPLES = 25             # images in the visual-verification montage
CACHE_FLUSH = 2000            # rows between phash-cache flushes / progress logs

# These are filled in from CLI args inside main().
ROOT = OUT = EDA = SPLDIR = DATA = None
WORKERS = max(2, (os.cpu_count() or 4) - 2)   # leave a couple of cores free

csv.field_size_limit(10 ** 7)


# ================================ helpers ===================================
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(os.path.join(OUT, "reproduce.log"), "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def set_status(name, value):
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        f.write(value + "\n")


def abspath(rel):
    """Join a manifest-relative path (may use \\ or /) safely onto the data dir."""
    return os.path.join(DATA, *rel.replace("\\", "/").split("/"))


def parse_taxonomy(folder_name):
    """Parse an iNat folder name into taxonomy ranks, or None if it is not a
    taxonomy folder."""
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
        # sorted() makes the result independent of filesystem listing order, so
        # the manifest (and therefore the split) is identical on every machine.
        for n in sorted(os.listdir(d)):
            if os.path.splitext(n)[1].lower() in IMG_EXTS:
                out.append(os.path.join(d, n))
    except FileNotFoundError:
        pass
    return out


def relstore(abs_path):
    """Store a DATA-relative path with '\\' separators so the manifest/split
    CSVs are byte-identical regardless of the OS we run on (the loaders accept
    both separators)."""
    return os.path.relpath(abs_path, DATA).replace(os.sep, "\\").replace("/", "\\")


# ============================ PHASE 0: extract ==============================
def phase0_extract():
    """Extract <split>.tar.gz -> <split>/ when the folder isn't already there.
    Archives themselves are never modified."""
    for split in LABELED_SPLITS + UNLABELED_SPLITS:
        sdir = os.path.join(DATA, split)
        if os.path.isdir(sdir) and os.listdir(sdir):
            log(f"extract: '{split}/' already present -> skip")
            continue
        arc = os.path.join(DATA, f"{split}.tar.gz")
        if not os.path.isfile(arc):
            log(f"extract: WARNING no '{split}/' and no '{split}.tar.gz' -> skipping")
            continue
        log(f"extract: unpacking {split}.tar.gz (this can take a while) ...")
        with tarfile.open(arc, "r:gz") as tf:
            # extract under ROOT; archives are rooted at <split>/...
            tf.extractall(DATA)
        log(f"extract: done -> {split}/")


# ============== PHASE 1+2+5(manifest): label / filter / manifest ============
def phase1_2_label(purge):
    """Keep only Insecta folders, tag bee families, verify image integrity, and
    build per-split + combined manifests (same columns as the original)."""
    from PIL import Image

    summary = {
        "audited_folders": 0, "audited_images_est": 0,
        "kept_folders": 0, "removed_folders": 0,
        "kept_images": 0, "corrupt_removed": 0, "bee_images": 0,
        "per_split": {},
    }
    rows = []  # split, path, class_id, folder, order, family, genus, species, is_bee

    for split in LABELED_SPLITS:
        sdir = os.path.join(DATA, split)
        if not os.path.isdir(sdir):
            log(f"phase1: split missing, skipping: {split}")
            continue
        folders = sorted(f for f in os.listdir(sdir)
                         if os.path.isdir(os.path.join(sdir, f)))
        s_audit_f = len(folders)
        s_kept_f = s_removed_f = s_kept_i = s_corrupt = s_bee = 0

        for fld in folders:
            fpath = os.path.join(sdir, fld)
            tax = parse_taxonomy(fld)
            imgs = list_images(fpath)
            summary["audited_images_est"] += len(imgs)

            # keep iff taxonomic Class == Insecta
            if tax is None or tax["cls"] != "Insecta":
                s_removed_f += 1
                if purge:
                    shutil.rmtree(fpath, ignore_errors=True)
                continue

            is_bee = (tax["order"] == "Hymenoptera"
                      and tax["family"] in BEE_FAMILIES)
            kept_here = 0
            for ip in imgs:
                try:
                    with Image.open(ip) as im:
                        im.verify()                       # catch truncated/corrupt
                except Exception:
                    if purge:
                        try:
                            os.remove(ip)
                        except Exception:
                            pass
                    s_corrupt += 1
                    continue
                kept_here += 1
                rows.append([
                    split, relstore(ip), tax["id"], fld,
                    tax["order"], tax["family"], tax["genus"], tax["species"],
                    int(is_bee),
                ])
            if kept_here == 0:
                s_removed_f += 1
                if purge:
                    shutil.rmtree(fpath, ignore_errors=True)
                continue
            s_kept_f += 1
            s_kept_i += kept_here
            if is_bee:
                s_bee += kept_here

        summary["per_split"][split] = {
            "audited_folders": s_audit_f, "kept_folders": s_kept_f,
            "removed_folders": s_removed_f, "kept_images": s_kept_i,
            "corrupt_removed": s_corrupt, "bee_images": s_bee,
        }
        for k, v in (("audited_folders", s_audit_f), ("kept_folders", s_kept_f),
                     ("removed_folders", s_removed_f), ("kept_images", s_kept_i),
                     ("corrupt_removed", s_corrupt), ("bee_images", s_bee)):
            summary[k] += v
        log(f"phase1: {split} audited={s_audit_f} kept={s_kept_f} "
            f"removed={s_removed_f} images={s_kept_i} corrupt={s_corrupt} bees={s_bee}")

    # profile public_test (unlabeled) — count only
    for split in UNLABELED_SPLITS:
        sdir = os.path.join(DATA, split)
        n = len(list_images(sdir)) if os.path.isdir(sdir) else 0
        summary["per_split"][split] = {"unlabeled_images": n,
                                        "note": "flat/unlabeled; left intact"}
        log(f"phase1: {split} unlabeled images (left intact) = {n}")

    header = ["split", "path", "class_id", "folder", "order", "family",
              "genus", "species", "is_bee"]
    with open(os.path.join(OUT, "manifest_all.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    for split in LABELED_SPLITS:
        srows = [r for r in rows if r[0] == split]
        with open(os.path.join(OUT, f"manifest_{split}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(header); w.writerows(srows)

    summary["manifest_rows"] = len(rows)
    log(f"phase1/2: manifest rows = {len(rows)}")
    return summary, rows


# ============================ PHASE 3: EDA (optional) =======================
def phase3_eda(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
    except Exception as e:
        log(f"eda: matplotlib/PIL unavailable -> skipping plots ({e})")
        return

    per_species = Counter(r[3] for r in rows)
    per_order = Counter(r[4] for r in rows)
    eda = {"num_species": len(per_species), "num_orders": len(per_order),
           "images_total": len(rows)}

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

        counts = list(per_species.values())
        plt.figure(figsize=(9, 5)); plt.hist(counts, bins=40)
        plt.xlabel("images per species"); plt.ylabel("# species")
        plt.title("Class-size distribution (imbalance)"); plt.tight_layout()
        plt.savefig(os.path.join(EDA, "class_size_hist.png"), dpi=110); plt.close()
    except Exception as e:
        log(f"eda: distribution plots failed: {e}")

    # image metadata profiling (sampled)
    widths, heights, ratios, modes = [], [], [], Counter()
    sample = rows if len(rows) <= META_SAMPLE_CAP else random.sample(rows, META_SAMPLE_CAP)
    for r in sample:
        try:
            with Image.open(abspath(r[1])) as im:
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
            plt.xlabel("width"); plt.ylabel("height")
            plt.title("Image resolution scatter (sampled)"); plt.tight_layout()
            plt.savefig(os.path.join(EDA, "resolution_scatter.png"), dpi=110); plt.close()
            plt.figure(figsize=(9, 5)); plt.hist(ratios, bins=40)
            plt.xlabel("aspect ratio (w/h)"); plt.ylabel("count")
            plt.title("Aspect-ratio distribution (sampled)"); plt.tight_layout()
            plt.savefig(os.path.join(EDA, "aspect_ratio_hist.png"), dpi=110); plt.close()
        except Exception as e:
            log(f"eda: metadata plots failed: {e}")

    # visual verification montage (no bboxes -> caption with the label)
    try:
        import math
        picks = random.sample(rows, min(GRID_SAMPLES, len(rows)))
        cols = 5; nrows = math.ceil(len(picks) / cols)
        plt.figure(figsize=(cols * 2.4, nrows * 2.6))
        for i, r in enumerate(picks):
            try:
                with Image.open(abspath(r[1])) as im:
                    ax = plt.subplot(nrows, cols, i + 1)
                    ax.imshow(im.convert("RGB")); ax.axis("off")
                    cap = f"{r[6]} {r[7]}" + (" [BEE]" if str(r[8]) == "1" else "")
                    ax.set_title(cap, fontsize=6)
            except Exception:
                continue
        plt.tight_layout(); plt.savefig(os.path.join(EDA, "sample_grid.png"), dpi=130); plt.close()
    except Exception as e:
        log(f"eda: montage failed: {e}")

    with open(os.path.join(EDA, "eda_summary.json"), "w", encoding="utf-8") as f:
        json.dump(eda, f, indent=2)
    log(f"phase3: EDA done species={eda['num_species']} orders={eda['num_orders']}")


# ===================== PHASE 4: leakage-safe re-split =======================
def _hash_one(rel_root):
    """Top-level worker for ProcessPoolExecutor: (rel, root) -> (rel, hexhash|'')."""
    rel, root = rel_root
    import imagehash
    from PIL import Image
    try:
        p = os.path.join(root, *rel.replace("\\", "/").split("/"))
        with Image.open(p) as im:
            return rel, str(imagehash.phash(im.convert("RGB")))
    except Exception:
        return rel, ""


def _load_cache():
    cache, p = {}, os.path.join(SPLDIR, "phash_cache.csv")
    if os.path.exists(p):
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.reader(f):
                if len(r) == 2:
                    cache[r[0]] = r[1]
    return cache


def _compute_hashes(rels):
    """Resumable parallel phash over all labeled images (cached to disk)."""
    cache = _load_cache()
    todo = [r for r in rels if r not in cache]
    log(f"hashing: {len(rels)} total, {len(cache)} cached, {len(todo)} to do")
    if not todo:
        return cache
    cpath = os.path.join(SPLDIR, "phash_cache.csv")
    done = 0
    with open(cpath, "a", newline="", encoding="utf-8") as cf:
        w = csv.writer(cf)
        buf = []
        with ProcessPoolExecutor(max_workers=WORKERS) as ex:
            for rel, h in ex.map(_hash_one, ((r, ROOT) for r in todo), chunksize=64):
                buf.append((rel, h)); cache[rel] = h; done += 1
                if len(buf) >= CACHE_FLUSH:
                    w.writerows(buf); cf.flush(); buf.clear()
                    log(f"hashing: {done}/{len(todo)}")
        if buf:
            w.writerows(buf); cf.flush()
    log(f"hashing: complete ({done} new)")
    return cache


def _hexham(a, b):
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def _group_within_class(items):
    """Union-find near-duplicates within one class (items: list of (rel,hex)).
    Returns (groups, exact_drop_set)."""
    n = len(items)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    by_hash = defaultdict(list)
    for i, (_, h) in enumerate(items):
        by_hash[h].append(i)
    for idxs in by_hash.values():
        for j in idxs[1:]:
            union(idxs[0], j)

    uniq = list(by_hash.items())
    for a in range(len(uniq)):
        ha, ia = uniq[a][0], uniq[a][1][0]
        for b in range(a + 1, len(uniq)):
            hb, ib = uniq[b][0], uniq[b][1][0]
            if _hexham(ha, hb) <= HAMMING_THRESH:
                union(ia, ib)

    comps = defaultdict(list)
    for i in range(n):
        comps[find(i)].append(i)
    groups = [[items[i][0] for i in members] for members in comps.values()]

    exact_drop = set()
    for idxs in by_hash.values():
        for j in idxs[1:]:
            exact_drop.add(items[j][0])
    return groups, exact_drop


def _assign_splits(class_to_groups):
    """Greedy group-safe stratified allocation per class."""
    rng = random.Random(SEED)
    assignment, order = {}, ["train", "val", "test"]
    for cid in sorted(class_to_groups):
        groups = sorted(class_to_groups[cid], key=len, reverse=True)
        rng.shuffle(groups)
        total = sum(len(g) for g in groups)
        cur = {"train": 0, "val": 0, "test": 0}
        for g in groups:
            deficit = {s: SPLIT_RATIOS[s] * total - cur[s] for s in order}
            pick = max(order, key=lambda s: deficit[s])
            cur[pick] += len(g)
            for rel in g:
                assignment[rel] = pick
    return assignment


def phase4_resplit(rows):
    """Pool train_mini+val, perceptual-dedup, stratified group-safe 80/10/10."""
    labeled = [(r[1], r[2], r[7], r[3]) for r in rows if r[0] in LABELED_SPLITS]
    log(f"resplit: labeled images in manifest = {len(labeled)}")

    cache = _compute_hashes([r[0] for r in labeled])

    class_items, meta, corrupt = defaultdict(list), {}, 0
    for rel, cid, sp, folder in labeled:
        meta[rel] = (cid, sp, folder)
        h = cache.get(rel, "")
        if not h:
            corrupt += 1
            continue
        class_items[cid].append((rel, h))
    log(f"resplit: unreadable images skipped = {corrupt}")

    class_to_groups, total_exact_drop, group_of, gid = {}, 0, {}, 0
    for cid, items in class_items.items():
        groups, exact_drop = _group_within_class(items)
        total_exact_drop += len(exact_drop)
        kept_groups = []
        for g in groups:
            g_kept = [r for r in g if r not in exact_drop]
            if not g_kept:
                continue
            for r in g_kept:
                group_of[r] = gid
            kept_groups.append(g_kept); gid += 1
        class_to_groups[cid] = kept_groups
    kept_total = sum(sum(len(g) for g in gs) for gs in class_to_groups.values())
    log(f"resplit: exact duplicates dropped = {total_exact_drop}")
    log(f"resplit: near-dup groups = {gid} | usable images = {kept_total}")

    assignment = _assign_splits(class_to_groups)
    split_counts = Counter(assignment.values())
    log(f"resplit: split sizes = {dict(split_counts)}")

    arows, lists = [], {"train": [], "val": [], "test": []}
    for cid, groups in class_to_groups.items():
        for g in groups:
            for rel in g:
                s = assignment[rel]
                lists[s].append(rel)
                arows.append((rel, cid, meta[rel][1], group_of[rel],
                              cache.get(rel, ""), s, "kept"))
    kept_set = set(group_of)
    for rel, cid, sp, folder in labeled:
        if rel not in kept_set and cache.get(rel, ""):
            arows.append((rel, cid, meta[rel][1], "", cache.get(rel, ""),
                          "", "exact_dup_dropped"))

    with open(os.path.join(SPLDIR, "split_assignments.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "class_id", "species", "group_id", "phash", "split", "status"])
        w.writerows(arows)
    for s in ("train", "val", "test"):
        with open(os.path.join(SPLDIR, f"{s}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(lists[s])) + "\n")

    # leakage verification: no group_id may span splits
    grp_splits = defaultdict(set)
    for rel, gidx in group_of.items():
        grp_splits[gidx].add(assignment[rel])
    leak_groups = [g for g, ss in grp_splits.items() if len(ss) > 1]
    log(f"resplit: leakage check, groups spanning splits = {len(leak_groups)} (must be 0)")

    per_class_split = defaultdict(lambda: defaultdict(int))
    for rel, s in assignment.items():
        per_class_split[meta[rel][0]][s] += 1
    missing_val = sum(1 for c in per_class_split if per_class_split[c]["val"] == 0)
    missing_test = sum(1 for c in per_class_split if per_class_split[c]["test"] == 0)

    num_classes = len(set(meta[r][0] for r in group_of))
    cfg = {
        "task": "image_classification",
        "split_strategy": "option3_pooled_dedup_stratified_group_safe",
        "ratios": SPLIT_RATIOS,
        "lists": {s: f"_pipeline/splits/{s}.txt" for s in ("train", "val", "test")},
        "counts": {s: len(lists[s]) for s in ("train", "val", "test")},
        "num_classes": num_classes,
        "exact_duplicates_dropped": total_exact_drop,
        "hamming_threshold": HAMMING_THRESH,
        "leakage_groups_spanning_splits": len(leak_groups),
        "note": ("Pooled train_mini+val, perceptual-dedup, stratified group-safe "
                 "80/10/10. public_test/ remains unlabeled inference-only and is "
                 "NOT part of these lists. Original folders untouched (reversible)."),
    }
    with open(os.path.join(ROOT, "dataset_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    with open(os.path.join(ROOT, "data.yaml"), "w", encoding="utf-8") as f:
        f.write("# BEE_HERo - Option-3 re-split (pooled+dedup+stratified, group-safe)\n")
        f.write(f"path: {DATA}\n")
        f.write("train: _pipeline/splits/train.txt\n")
        f.write("val: _pipeline/splits/val.txt\n")
        f.write("test: _pipeline/splits/test.txt\n")
        f.write(f"nc: {num_classes}\n")
        f.write("# class names listed in dataset_config.json / split_assignments.csv\n")

    rep = os.path.join(SPLDIR, "resplit_report.md")
    with open(rep, "w", encoding="utf-8") as f:
        f.write("# BEE_HERo - Option 3 Re-split Report\n\n")
        f.write(f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')}_\n\n")
        f.write("## Inputs\n")
        f.write(f"- Pooled labeled images (train_mini+val): **{len(labeled)}**\n")
        f.write(f"- Unreadable/skipped: {corrupt}\n")
        f.write(f"- Classes: {num_classes}\n\n")
        f.write("## De-duplication\n")
        f.write(f"- Exact duplicates dropped (kept 1 each): **{total_exact_drop}**\n")
        f.write(f"- Near-dup grouping: within-class, hamming <= {HAMMING_THRESH}\n")
        f.write(f"- Near-dup groups (observation units): **{gid}**\n")
        f.write(f"- Usable images after dedup: **{kept_total}**\n\n")
        f.write("## Final split (80/10/10, stratified, group-safe)\n")
        for s in ("train", "val", "test"):
            f.write(f"- {s}: **{len(lists[s])}** "
                    f"({100 * len(lists[s]) / max(1, kept_total):.1f}%)\n")
        f.write("\n## Leakage verification\n")
        f.write(f"- Dup-groups spanning multiple splits: **{len(leak_groups)}** "
                f"(0 = no leakage, guaranteed by construction)\n")
        f.write(f"- Classes with no val sample: {missing_val}\n")
        f.write(f"- Classes with no test sample: {missing_test}\n\n")
        f.write("## Notes\n")
        f.write("- `public_test/` is unlabeled -> inference-only, excluded.\n")
        f.write("- Original `train_mini/` and `val/` folders are untouched; this\n")
        f.write("  re-split is expressed purely as file lists -> fully reversible.\n")
    log(f"resplit: report written -> {rep}")
    set_status(os.path.join("splits", "RESPLIT_STATUS.txt"), "COMPLETED_OK")
    return cfg


# ================================== main ====================================
def main():
    global ROOT, OUT, EDA, SPLDIR, DATA
    ap = argparse.ArgumentParser(description="Reproduce the BEE_HERo dataset format.")
    ap.add_argument("--root",
                    default=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    help="folder holding the archives / extracted splits "
                         "(default: repo root, two levels up from src/data_pipeline/)")
    ap.add_argument("--no-extract", action="store_true",
                    help="assume train_mini/ val/ public_test/ are already extracted")
    ap.add_argument("--no-eda", action="store_true", help="skip matplotlib EDA plots")
    ap.add_argument("--purge", action="store_true",
                    help="DELETE non-insect folders on disk to match our final "
                         "on-disk state (default: keep them, manifest still filters)")
    args = ap.parse_args()

    ROOT = os.path.abspath(args.root)
    DATA = os.path.join(ROOT, "data", "raw", "iNaturist")  # raw splits + archives live here
    OUT = os.path.join(ROOT, "_pipeline")
    EDA = os.path.join(OUT, "eda")
    SPLDIR = os.path.join(OUT, "splits")
    for d in (DATA, OUT, EDA, SPLDIR):
        os.makedirs(d, exist_ok=True)

    set_status("REPRODUCE_STATUS.txt", "RUNNING")
    log(f"=== BEE_HERo reproduce start (root={ROOT}, workers={WORKERS}) ===")
    try:
        if not args.no_extract:
            phase0_extract()
        summary, rows = phase1_2_label(purge=args.purge)
        if not rows:
            log("FATAL: no labeled images found. Are train_mini/ and val/ present?")
            set_status("REPRODUCE_STATUS.txt", "ERROR_NO_DATA")
            return 1
        if not args.no_eda:
            phase3_eda(rows)
        cfg = phase4_resplit(rows)

        log("=== SUMMARY ===")
        log(f"  kept species folders : {summary['kept_folders']}  "
            f"(removed {summary['removed_folders']} non-insect)")
        log(f"  labeled images       : {summary['kept_images']}  "
            f"(bees: {summary['bee_images']}, corrupt: {summary['corrupt_removed']})")
        log(f"  classes (nc)         : {cfg['num_classes']}")
        log(f"  split counts         : {cfg['counts']}")
        log(f"  cross-split leakage  : {cfg['leakage_groups_spanning_splits']} groups")
        set_status("REPRODUCE_STATUS.txt", "COMPLETED_OK")
        log("=== BEE_HERo reproduce COMPLETED_OK ===")
        return 0
    except Exception:
        log("FATAL ERROR:\n" + traceback.format_exc())
        set_status("REPRODUCE_STATUS.txt", "ERROR (see reproduce.log)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
