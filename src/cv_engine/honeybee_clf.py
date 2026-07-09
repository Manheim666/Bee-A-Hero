"""Honeybee-vs-other-bee subclassifier (binary) for the visit counter.

The insect detector emits a single ``bee`` class. Honeybees (genus *Apis*) are ~10x more
valuable for pollination than other bees, so we split them with a light image classifier
run **only on ``bee`` crops** inside ``video_detect`` (via ``--honeybee-weights``).

Data comes from the iNaturalist manifest (``split_manifest.csv``): every ``is_bee`` row is
labelled ``honeybee`` if ``genus == Apis`` else ``other_bee``. Apis is rare (~240 imgs) so we
class-weight the loss **and** balance the sampler. The checkpoint is saved in the exact shape
``visit_counter.Classifier`` expects: ``{classes, model, state_dict}``.

CLI:
    python -m src.cv_engine.honeybee_clf --epochs 15 --model mobilenetv3_small_100 \
        --out data/interim/cv_runs/honeybee_clf
    python -m src.cv_engine.honeybee_clf --dry-run          # just print class counts
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from src import config as C

MANIFEST = C.INTERIM_DIR / "manifests" / "split_manifest.csv"
CLASSES = ["other_bee", "honeybee"]      # idx 0 = other_bee => empty-crop fallback is conservative


def _is_true(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes")


def _label(row) -> str:
    return "honeybee" if (row.get("genus") or "").strip().lower() == "apis" else "other_bee"


def load_split(split: str) -> list[tuple[str, int]]:
    """Return [(image_path, class_idx)] for every bee row in the given manifest split."""
    items = []
    for r in csv.DictReader(open(MANIFEST)):
        if not _is_true(r.get("is_bee")):
            continue
        if (r.get("split") or "").strip() != split:
            continue
        p = r["path"]
        if not Path(p).exists():
            p2 = C.RAW_DIR / "iNaturist" / p
            p = str(p2) if p2.exists() else ""
        if p:
            items.append((p, CLASSES.index(_label(r))))
    return items


def _counts(items) -> dict:
    c = Counter(lbl for _, lbl in items)
    return {CLASSES[i]: c.get(i, 0) for i in range(len(CLASSES))}


class _BeeSet:
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from PIL import Image
        p, y = self.items[i]
        img = Image.open(p).convert("RGB")
        return self.tf(img), y


def _focal_loss(logits, target, weight, gamma=2.0):
    """Class-weighted focal loss — down-weights easy majority (other_bee), focuses on rare honeybee."""
    import torch
    import torch.nn.functional as F
    p = F.softmax(logits, dim=1)
    pt = p[torch.arange(len(target), device=target.device), target].clamp_min(1e-8)
    w = weight[target]
    return -(w * (1 - pt) ** gamma * torch.log(pt)).mean()


def train(out_dir: Path, model_name="efficientnet_b0", epochs=30, batch=64,
          lr=3e-4, workers=6, loss="focal", gamma=2.0) -> dict:
    import timm
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    tr_items, va_items = load_split("train"), load_split("val")
    if not tr_items:
        raise SystemExit("no bee training rows found in manifest")
    tr_counts = _counts(tr_items)
    print(f"train {len(tr_items)} {tr_counts} | val {len(va_items)} {_counts(va_items)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(model_name, pretrained=True, num_classes=len(CLASSES))
    cfg = timm.data.resolve_data_config({}, model=model)
    tf_tr = timm.data.create_transform(**cfg, is_training=True, auto_augment="rand-m7-n2")
    tf_va = timm.data.create_transform(**cfg, is_training=False)
    model.to(device)

    # balance the rare honeybee class: inverse-freq sample weights + weighted loss
    freq = [max(1, tr_counts[CLASSES[i]]) for i in range(len(CLASSES))]
    sample_w = [1.0 / freq[y] for _, y in tr_items]
    sampler = WeightedRandomSampler(sample_w, num_samples=len(tr_items), replacement=True)
    cls_w = torch.tensor([sum(freq) / f for f in freq], dtype=torch.float32, device=device)

    tr = DataLoader(_BeeSet(tr_items, tf_tr), batch_size=batch, sampler=sampler,
                    num_workers=workers, pin_memory=True)
    va = DataLoader(_BeeSet(va_items, tf_va), batch_size=batch, shuffle=False,
                    num_workers=workers, pin_memory=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce = torch.nn.CrossEntropyLoss(weight=cls_w)
    lossf = ce if loss == "ce" else (lambda o, y: _focal_loss(o, y, cls_w, gamma))

    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"
    hb_idx = CLASSES.index("honeybee")
    best_metric = -1.0
    for ep in range(1, epochs + 1):
        model.train()
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            lossf(model(x), y).backward()
            opt.step()
        sched.step()
        # validate: honeybee precision/recall + balanced acc (rare-class aware)
        model.eval()
        tp = fp = fn = correct = total = 0
        with torch.no_grad():
            for x, y in va:
                x = x.to(device)
                pred = model(x).argmax(1).cpu()
                for p, t in zip(pred.tolist(), y.tolist()):
                    total += 1; correct += int(p == t)
                    if t == hb_idx and p == hb_idx: tp += 1
                    elif p == hb_idx and t != hb_idx: fp += 1
                    elif t == hb_idx and p != hb_idx: fn += 1
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        acc = correct / max(1, total)
        print(f"ep{ep:02d} acc={acc:.3f} honeybee P={prec:.3f} R={rec:.3f} F1={f1:.3f}")
        if f1 > best_metric:                       # pick by honeybee F1 (rare class matters)
            best_metric = f1
            torch.save({"classes": CLASSES, "model": model_name,
                        "state_dict": model.state_dict()}, best_path)
    print(f"best honeybee F1={best_metric:.3f} -> {best_path}")
    return {"weights": str(best_path), "best_f1": round(best_metric, 3),
            "train": _counts(tr_items), "val": _counts(va_items)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(C.INTERIM_DIR / "cv_runs" / "honeybee_clf"))
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--loss", choices=["focal", "ce"], default="focal")
    ap.add_argument("--gamma", type=float, default=2.0, help="focal loss focusing param")
    ap.add_argument("--dry-run", action="store_true", help="print class counts and exit")
    args = ap.parse_args()
    if args.dry_run:
        for sp in ("train", "val", "test"):
            print(sp, _counts(load_split(sp)))
        return
    import json
    print(json.dumps(train(Path(args.out), args.model, args.epochs, args.batch, args.lr,
                           loss=args.loss, gamma=args.gamma), indent=2))


if __name__ == "__main__":
    main()
