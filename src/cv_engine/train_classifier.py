"""Fine-tune an iNaturalist-pretrained classifier for pollinator vs non_pollinator.

Stage 2 of the two-stage insect pipeline. Because the crops are iNaturalist-style
organisms, an iNat21-pretrained backbone (timm) already separates bees from other
insects extremely well — we replace its 10k-class head with a 2-class head and
fine-tune (optionally head-only / linear-probe for big models on a 6GB card).

Reads the ImageFolder built by ``prepare_insect.export_classifier_crops``:
    insect_cls/{train,val,test}/{pollinator,non_pollinator}/*.jpg
Class imbalance (pollinator >> non) is handled with a WeightedRandomSampler.

CLI:
    python -m src.cv_engine.train_classifier --freeze --epochs 12
    # server (full fine-tune, bigger backbone):
    python -m src.cv_engine.train_classifier \
        --model hf-hub:timm/eva02_large_patch14_clip_336.merged2b_ft_inat21 --epochs 15
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
from pathlib import Path

try:
    multiprocessing.set_start_method("fork", force=True)
except RuntimeError:
    pass

import numpy as np
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder

from src import config as C

CLS_DIR = C.INTERIM_DIR / "insect_cls"
RUNS = C.INTERIM_DIR / "cv_runs"
# iNaturalist-2021-pretrained backbone, loaded straight from the HF Hub (the
# ``hf-hub:`` prefix is required — the tag is not in timm's local registry).
DEFAULT_MODEL = "hf-hub:timm/convnext_large_mlp.laion2b_ft_augreg_inat21"


def _loaders(model, batch):
    cfg = timm.data.resolve_data_config({}, model=model)
    train_tf = timm.data.create_transform(**cfg, is_training=True)
    eval_tf = timm.data.create_transform(**cfg, is_training=False)
    ds = {s: ImageFolder(str(CLS_DIR / s), transform=(train_tf if s == "train" else eval_tf))
          for s in ("train", "val", "test")}
    # balance the training classes with a weighted sampler
    targets = np.array([y for _, y in ds["train"].samples])
    cw = 1.0 / np.bincount(targets)
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor([cw[t] for t in targets], dtype=torch.double),
        num_samples=len(targets), replacement=True)
    dl = {
        "train": DataLoader(ds["train"], batch_size=batch, sampler=sampler,
                            num_workers=6, pin_memory=True),
        "val": DataLoader(ds["val"], batch_size=batch, shuffle=False, num_workers=6, pin_memory=True),
        "test": DataLoader(ds["test"], batch_size=batch, shuffle=False, num_workers=6, pin_memory=True),
    }
    return dl, ds["train"].classes


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    correct = total = 0
    per_cls_correct = [0, 0]; per_cls_total = [0, 0]
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item(); total += y.numel()
        for c in (0, 1):
            m = y == c
            per_cls_total[c] += m.sum().item()
            per_cls_correct[c] += (pred[m] == c).sum().item()
    acc = correct / max(total, 1)
    bal = 0.5 * sum(per_cls_correct[c] / max(per_cls_total[c], 1) for c in (0, 1))
    return {"acc": round(acc, 4), "balanced_acc": round(bal, 4),
            "pollinator_recall": round(per_cls_correct[0] / max(per_cls_total[0], 1), 4),
            "non_pollinator_recall": round(per_cls_correct[1] / max(per_cls_total[1], 1), 4)}


def train(model_name: str = DEFAULT_MODEL, name: str = "insect_classifier",
          epochs: int = 12, batch: int = 8, lr: float = 3e-4, freeze: bool = False) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(model_name, pretrained=True, num_classes=2)
    if freeze:                                   # linear-probe: train the head only
        for p in model.parameters():
            p.requires_grad = False
        for p in model.get_classifier().parameters():
            p.requires_grad = True
    model.to(device)

    dl, classes = _loaders(model, batch)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    crit = nn.CrossEntropyLoss()

    out_dir = RUNS / name; out_dir.mkdir(parents=True, exist_ok=True)
    best_bal, best_path = -1.0, out_dir / "best.pt"
    for ep in range(1, epochs + 1):
        model.train()
        for x, y in dl["train"]:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        val = evaluate(model, dl["val"], device)
        print(f"epoch {ep}/{epochs} val={val}", flush=True)
        if val["balanced_acc"] > best_bal:
            best_bal = val["balanced_acc"]
            torch.save({"model": model_name, "classes": classes,
                        "state_dict": model.state_dict()}, best_path)

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    test = evaluate(model, dl["test"], device)
    result = {"model": model_name, "classes": classes, "freeze": freeze,
              "best_val_balanced_acc": best_bal, "test": test, "weights": str(best_path)}
    json.dump(result, open(out_dir / "classifier_result.json", "w"), indent=2)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--name", default="insect_classifier")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--freeze", action="store_true", help="linear-probe (head only)")
    args = ap.parse_args()
    print(json.dumps(train(args.model, args.name, args.epochs, args.batch, args.lr, args.freeze), indent=2))


if __name__ == "__main__":
    main()
