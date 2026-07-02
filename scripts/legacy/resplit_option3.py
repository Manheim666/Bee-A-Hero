#!/usr/bin/env python3
"""
Option 3 — accurate, leakage-safe re-split of the BEE_HERo labeled data.

Pools the two LABELED splits (train_mini + val), perceptually de-duplicates,
then produces a stratified, GROUP-SAFE 80/10/10 train/val/test split.

Design goals (why each step exists):
  * Leakage is the only "big" risk of re-splitting iNaturalist data, because the
    same observation often has several near-identical photos. We never have the
    observation id, so we approximate "same observation" with perceptual-hash
    (phash) near-duplicate grouping *within a class*. A whole group is always
    assigned to ONE split -> 0 cross-split leakage by construction.
  * Stratified per-class allocation keeps the class distribution stable across
    the three splits (Phase-4 requirement).
  * Fully reversible: we only WRITE file lists + config. No image is moved or
    deleted, and the original train_mini/ and val/ folders are untouched.

Outputs (all under _pipeline/splits/):
  phash_cache.csv        resumable path -> phash cache (so a crash never re-hashes)
  split_assignments.csv  path,class_id,species,group_id,phash,split,status
  train.txt val.txt test.txt   relative image paths, one per line
  resplit_report.md      human summary + leakage verification
Also rewrites repo-root data.yaml / dataset_config.json to point at the lists.
"""
import os, sys, csv, json, random, time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

# ----------------------------- configuration -------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root (this file lives in src/data_pipeline/)
DATA = os.path.join(ROOT, "data", "raw", "iNaturist")  # raw splits + archives live here
OUT  = os.path.join(ROOT, "_pipeline")
SPLDIR = os.path.join(OUT, "splits")
MANIFEST = os.path.join(OUT, "manifest_all.csv")

LABELED_SPLITS = ("train_mini", "val")
SPLIT_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}
HAMMING_THRESH = 5     # <= this distance, same class => "same observation" group
SEED = 1337
WORKERS = 12           # leave headroom on the 20-core box
CACHE_FLUSH = 2000     # rows between cache flushes / progress logs

csv.field_size_limit(10**7)
LOG = os.path.join(SPLDIR, "resplit.log")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(SPLDIR, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def abspath(rel):
    """Manifest stores Windows-style relative paths; join safely to ROOT."""
    return os.path.join(DATA, *rel.replace("\\", "/").split("/"))


# ------------------------- phash worker (top-level) ------------------------
def _hash_one(rel):
    """Return (rel, hexhash|'') -- '' marks an unreadable/corrupt image."""
    import imagehash
    from PIL import Image
    try:
        with Image.open(abspath(rel)) as im:
            return rel, str(imagehash.phash(im.convert("RGB")))  # 64-bit, matches Phase 6
    except Exception:
        return rel, ""


def load_manifest():
    rows = []
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] in LABELED_SPLITS:
                rows.append((r["path"], r["class_id"],
                             r.get("species", ""), r.get("folder", "")))
    return rows


def load_cache():
    cache = {}
    p = os.path.join(SPLDIR, "phash_cache.csv")
    if os.path.exists(p):
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.reader(f):
                if len(r) == 2:
                    cache[r[0]] = r[1]
    return cache


def compute_hashes(rows):
    """Resumable parallel phash over all labeled images."""
    cache = load_cache()
    todo = [r[0] for r in rows if r[0] not in cache]
    log(f"hashing: {len(rows)} total, {len(cache)} cached, {len(todo)} to do")
    if not todo:
        return cache
    cpath = os.path.join(SPLDIR, "phash_cache.csv")
    done = 0
    with open(cpath, "a", newline="", encoding="utf-8") as cf:
        w = csv.writer(cf)
        with ProcessPoolExecutor(max_workers=WORKERS) as ex:
            buf = []
            for rel, h in ex.map(_hash_one, todo, chunksize=64):
                buf.append((rel, h)); cache[rel] = h; done += 1
                if len(buf) >= CACHE_FLUSH:
                    w.writerows(buf); cf.flush(); buf.clear()
                    log(f"hashing: {done}/{len(todo)}")
            if buf:
                w.writerows(buf); cf.flush()
    log(f"hashing: complete ({done} new)")
    return cache


# ----------------------------- dedup grouping ------------------------------
def hexham(a, b):
    """Hamming distance between two 16-char hex phash strings."""
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def group_within_class(items):
    """Union-find near-duplicates within one class (items: list of (rel,hex)).
    Returns groups: list of lists of rel; also a set of exact-duplicate rels to
    drop (keep one representative per identical hash)."""
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

    # bucket by exact hash first (fast path for true duplicates)
    by_hash = defaultdict(list)
    for i, (_, h) in enumerate(items):
        by_hash[h].append(i)
    for idxs in by_hash.values():
        for j in idxs[1:]:
            union(idxs[0], j)

    # near-dup: compare distinct hashes pairwise (class sizes are small ~70)
    uniq = list(by_hash.items())
    for a in range(len(uniq)):
        ha, ia = uniq[a][0], uniq[a][1][0]
        for b in range(a + 1, len(uniq)):
            hb, ib = uniq[b][0], uniq[b][1][0]
            if hexham(ha, hb) <= HAMMING_THRESH:
                union(ia, ib)

    comps = defaultdict(list)
    for i in range(n):
        comps[find(i)].append(i)

    groups, exact_drop = [], set()
    for members in comps.values():
        groups.append([items[i][0] for i in members])
    # drop exact duplicates (identical hash) keeping the first seen
    for idxs in by_hash.values():
        for j in idxs[1:]:
            exact_drop.add(items[j][0])
    return groups, exact_drop


# ----------------------------- stratified split ----------------------------
def assign_splits(class_to_groups):
    """Greedy group-safe stratified allocation per class.
    Each group goes wholly into the split currently most under its target."""
    rng = random.Random(SEED)
    assignment = {}  # rel -> split
    order = ["train", "val", "test"]
    for cid in sorted(class_to_groups):
        groups = class_to_groups[cid]
        # largest groups first so big chunks are placed before fine-tuning balance
        groups = sorted(groups, key=len, reverse=True)
        rng.shuffle(groups)  # deterministic shuffle for ties
        total = sum(len(g) for g in groups)
        cur = {"train": 0, "val": 0, "test": 0}
        for g in groups:
            # choose split with the largest remaining deficit vs its target share
            deficit = {s: SPLIT_RATIOS[s] * total - cur[s] for s in order}
            pick = max(order, key=lambda s: deficit[s])
            cur[pick] += len(g)
            for rel in g:
                assignment[rel] = pick
    return assignment


# ------------------------------- main flow ---------------------------------
def main():
    os.makedirs(SPLDIR, exist_ok=True)
    open(LOG, "w").close()
    log("=== Option-3 re-split start ===")
    rows = load_manifest()
    log(f"labeled images in manifest: {len(rows)}")

    cache = compute_hashes(rows)

    # organize per class, skipping corrupt (empty hash)
    class_items = defaultdict(list)   # cid -> list of (rel, hex)
    meta = {}                         # rel -> (cid, species, folder)
    corrupt = 0
    for rel, cid, sp, folder in rows:
        h = cache.get(rel, "")
        meta[rel] = (cid, sp, folder)
        if not h:
            corrupt += 1
            continue
        class_items[cid].append((rel, h))
    log(f"unreadable images skipped: {corrupt}")

    # dedup + group per class
    class_to_groups = {}
    total_exact_drop = 0
    group_of = {}  # rel -> group_id
    gid_counter = 0
    for cid, items in class_items.items():
        groups, exact_drop = group_within_class(items)
        total_exact_drop += len(exact_drop)
        kept_groups = []
        for g in groups:
            g_kept = [r for r in g if r not in exact_drop]
            if not g_kept:
                continue
            for r in g_kept:
                group_of[r] = gid_counter
            kept_groups.append(g_kept)
            gid_counter += 1
        class_to_groups[cid] = kept_groups
    kept_total = sum(sum(len(g) for g in gs) for gs in class_to_groups.values())
    log(f"exact duplicates dropped: {total_exact_drop}")
    log(f"near-dup groups formed: {gid_counter}  | usable images: {kept_total}")

    # stratified group-safe split
    assignment = assign_splits(class_to_groups)
    split_counts = defaultdict(int)
    for s in assignment.values():
        split_counts[s] += 1
    log(f"split sizes: {dict(split_counts)}")

    # ---- write assignments + file lists ----
    arows = []
    lists = {"train": [], "val": [], "test": []}
    for cid, groups in class_to_groups.items():
        for g in groups:
            for rel in g:
                s = assignment[rel]
                lists[s].append(rel)
                _, sp, _ = meta[rel]
                arows.append((rel, cid, sp, group_of[rel], cache.get(rel, ""), s, "kept"))
    # record dropped exact dups too (status=exact_dup_dropped) for transparency
    kept_set = set(group_of)
    for rel, cid, sp, folder in rows:
        if rel not in kept_set and cache.get(rel, ""):
            arows.append((rel, cid, meta[rel][1], "", cache.get(rel, ""), "", "exact_dup_dropped"))

    with open(os.path.join(SPLDIR, "split_assignments.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "class_id", "species", "group_id", "phash", "split", "status"])
        w.writerows(arows)
    for s in ("train", "val", "test"):
        with open(os.path.join(SPLDIR, f"{s}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(lists[s])) + "\n")

    # ---- leakage verification: no group_id may span splits ----
    grp_splits = defaultdict(set)
    for cid, groups in class_to_groups.items():
        for g in groups:
            for rel in g:
                grp_splits[group_of[rel]].add(assignment[rel])
    leak_groups = [gid for gid, ss in grp_splits.items() if len(ss) > 1]
    log(f"leakage check: groups spanning splits = {len(leak_groups)} (must be 0)")

    # per-class split presence (stratification sanity)
    per_class_split = defaultdict(lambda: defaultdict(int))
    for rel, s in assignment.items():
        per_class_split[meta[rel][0]][s] += 1
    classes_missing_test = sum(1 for c in per_class_split if per_class_split[c]["test"] == 0)
    classes_missing_val = sum(1 for c in per_class_split if per_class_split[c]["val"] == 0)

    # ---- update configs (point at the new lists; keep it reversible) ----
    class_names = sorted({meta[r][2] or meta[r][0] for r in group_of})
    cfg = {
        "task": "image_classification",
        "split_strategy": "option3_pooled_dedup_stratified_group_safe",
        "ratios": SPLIT_RATIOS,
        "lists": {s: f"_pipeline/splits/{s}.txt" for s in ("train", "val", "test")},
        "counts": {s: len(lists[s]) for s in ("train", "val", "test")},
        "num_classes": len(set(meta[r][0] for r in group_of)),
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
        f.write(f"nc: {cfg['num_classes']}\n")
        f.write("# class names listed in dataset_config.json / split_assignments.csv\n")

    # ---- report ----
    rep = os.path.join(SPLDIR, "resplit_report.md")
    with open(rep, "w", encoding="utf-8") as f:
        f.write("# BEE_HERo - Option 3 Re-split Report\n\n")
        f.write(f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')}_\n\n")
        f.write("## Inputs\n")
        f.write(f"- Pooled labeled images (train_mini+val): **{len(rows)}**\n")
        f.write(f"- Unreadable/skipped: {corrupt}\n")
        f.write(f"- Classes: {cfg['num_classes']}\n\n")
        f.write("## De-duplication\n")
        f.write(f"- Exact duplicates dropped (kept 1 each): **{total_exact_drop}**\n")
        f.write(f"- Near-dup grouping: within-class, hamming <= {HAMMING_THRESH}\n")
        f.write(f"- Near-dup groups (observation units): **{gid_counter}**\n")
        f.write(f"- Usable images after dedup: **{kept_total}**\n\n")
        f.write("## Final split (80/10/10, stratified, group-safe)\n")
        for s in ("train", "val", "test"):
            f.write(f"- {s}: **{len(lists[s])}** "
                    f"({100*len(lists[s])/max(1,kept_total):.1f}%)\n")
        f.write("\n## Leakage verification\n")
        f.write(f"- Dup-groups spanning multiple splits: **{len(leak_groups)}** "
                f"(0 = no leakage, guaranteed by construction)\n")
        f.write(f"- Classes with no val sample: {classes_missing_val}\n")
        f.write(f"- Classes with no test sample: {classes_missing_test}\n\n")
        f.write("## Notes\n")
        f.write("- `public_test/` (500k) is unlabeled -> inference-only, excluded.\n")
        f.write("- Original `train_mini/` and `val/` folders are untouched; this\n")
        f.write("  re-split is expressed purely as file lists -> fully reversible.\n")
    log(f"report written: {rep}")
    log("=== Option-3 re-split DONE ===")
    # machine-readable done marker
    with open(os.path.join(SPLDIR, "RESPLIT_STATUS.txt"), "w") as f:
        f.write("COMPLETED_OK\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        log("FATAL: " + repr(e))
        log(traceback.format_exc())
        with open(os.path.join(SPLDIR, "RESPLIT_STATUS.txt"), "w") as f:
            f.write("FAILED\n")
        sys.exit(1)
