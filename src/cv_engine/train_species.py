"""Fine-tune an iNaturalist-2021-pretrained backbone to our 2526 insect species.

Stage-2 (species) classifier for the two-stage insect pipeline. The backbone is
already pretrained on iNat21, so a linear-probe head over 2526 species converges
fast and scores high. Reports the metrics that matter for a large label space:
top-1, top-5 accuracy and macro-F1.

Reads the ImageFolder at ``data/interim/insect_cls_species/{train,val,test}/<class_id>/``.

CLI:  python -m src.cv_engine.train_species --epochs 5
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
from pathlib import Path

try:
    multiprocessing.set_start_method("fork", force=True)
except (RuntimeError, ValueError):
    pass  # fork unavailable (Windows) -> use the default start method

import numpy as np
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src import config as C

DATA = C.INTERIM_DIR / "insect_cls_species"
RUNS = C.INTERIM_DIR / "cv_runs"
MODEL = "hf-hub:timm/convnext_large_mlp.laion2b_ft_augreg_inat21"


def _loaders(model, batch):
    cfg = timm.data.resolve_data_config({}, model=model)
    tf_tr = timm.data.create_transform(**cfg, is_training=True)
    tf_ev = timm.data.create_transform(**cfg, is_training=False)
    ds = {s: ImageFolder(str(DATA / s), transform=(tf_tr if s == "train" else tf_ev))
          for s in ("train", "val", "test")}
    dl = {s: DataLoader(ds[s], batch_size=batch, shuffle=(s == "train"),
                        num_workers=6, pin_memory=True) for s in ds}
    return dl, ds["train"].classes


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    top1 = top5 = total = 0
    tp = np.zeros(num_classes); fp = np.zeros(num_classes); fn = np.zeros(num_classes)
    for x, y in loader:
        x = x.to(device); logits = model(x).cpu()
        p5 = logits.topk(5, 1).indices
        p1 = p5[:, 0]
        top1 += (p1 == y).sum().item()
        top5 += (p5 == y[:, None]).any(1).sum().item()
        total += y.numel()
        for t, p in zip(y.tolist(), p1.tolist()):
            if t == p: tp[t] += 1
            else: fp[p] += 1; fn[t] += 1
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) > 0)
    return {"top1": round(top1 / total, 4), "top5": round(top5 / total, 4),
            "macro_f1": round(float(f1.mean()), 4)}


def train(epochs=5, batch=8, lr=1e-3, name="species_classifier"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(MODEL, pretrained=True, num_classes=2526)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.get_classifier().parameters():
        p.requires_grad = True
    model.to(device)

    dl, classes = _loaders(model, batch)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    out = RUNS / name; out.mkdir(parents=True, exist_ok=True)
    best, best_path = -1.0, out / "best.pt"
    for ep in range(1, epochs + 1):
        model.train()
        for x, y in dl["train"]:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        val = evaluate(model, dl["val"], device, 2526)
        print(f"epoch {ep}/{epochs} val={val}", flush=True)
        if val["top1"] > best:
            best = val["top1"]
            torch.save({"model": MODEL, "classes": classes, "state_dict": model.state_dict()}, best_path)

    model.load_state_dict(torch.load(best_path, map_location=device)["state_dict"])
    test = evaluate(model, dl["test"], device, 2526)
    res = {"model": MODEL, "num_classes": 2526, "best_val_top1": best, "test": test, "weights": str(best_path)}
    json.dump(res, open(out / "species_result.json", "w"), indent=2)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    print(json.dumps(train(epochs=args.epochs, batch=args.batch), indent=2))


if __name__ == "__main__":
    main()
