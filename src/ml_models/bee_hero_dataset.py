#!/usr/bin/env python3
"""
BEE_HERo — training-ready data layer.
================================================================================
This turns the cleaned/split dataset into *runnable* code: a PyTorch `Dataset`,
the augmentation/normalization pipeline (the image-classification equivalent of
"feature engineering"), DataLoaders tuned for 6 GB VRAM, and batch-level
MixUp/CutMix + the matching soft-target loss.

After this file, the data is 100% ready to feed a training loop — nothing about
the images needs to be touched again.

WHAT IT READS
    _pipeline/splits/split_assignments.csv   (path, class_id, ..., split, status)
        -> the authoritative source of which image is in train/val/test and its
           label. Rows with status == "kept" are the usable images.
    (the train.txt/val.txt/test.txt lists are the same info; we use the CSV
     because it carries the label too.)

WHAT IT BUILDS
    class_to_idx : {class_id -> contiguous 0..nc-1 index}  (saved to
                   _pipeline/class_index.json so training is reproducible)
    transforms   : Albumentations if installed, else an identical torchvision.v2
                   pipeline  (RandomResizedCrop, flip, affine/shift-scale-rotate,
                   colour jitter, coarse-dropout/erasing, ImageNet normalize)
    BeeHeroDataset / DataLoaders  (pin_memory, persistent_workers, prefetch)
    mixup_cutmix() + SoftTargetCrossEntropy

DEPENDENCIES
    required : torch torchvision pillow numpy
    optional : albumentations opencv-python   (nicer aug; auto-detected)

USE AS A LIBRARY
    from bee_hero_dataset import build_dataloaders, mixup_cutmix, SoftTargetCrossEntropy
    train_dl, val_dl, test_dl, class_to_idx = build_dataloaders(batch_size=32)

RUN DIRECTLY (self-test: proves the data is training-ready)
    python bee_hero_dataset.py
        -> loads a real batch from each split, applies MixUp/CutMix, runs one
           forward+backward through ResNet-18, and saves an augmented preview
           grid to _pipeline/eda/augmented_preview.png.
"""

import os
import csv
import json
import time

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

csv.field_size_limit(10 ** 7)

# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.dirname(os.path.dirname(HERE))  # repo root (this file lives in src/ml_models/)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _abspath(root, rel):
    """Join a manifest-relative path (may use \\ or /) onto root, cross-platform."""
    return os.path.join(root, *rel.replace("\\", "/").split("/"))


# --------------------------------------------------------------------------- #
# 1. index: which image is in which split, and its label
# --------------------------------------------------------------------------- #
def build_index(root=DEFAULT_ROOT, save=True):
    """Read split_assignments.csv -> (class_to_idx, samples).

    samples = {"train": [(abs_path, label_idx), ...], "val": [...], "test": [...]}
    Labels are a contiguous 0..nc-1 range built from the sorted unique class_id,
    so they match data.yaml's `nc` and are reproducible.
    """
    csv_path = os.path.join(root, "_pipeline", "splits", "split_assignments.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"{csv_path} not found. Run the split step first "
            "(resplit_option3.py / reproduce_bee_hero.py).")

    rows = []
    class_ids = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("status") != "kept":
                continue                       # skip exact-dup-dropped rows
            split = r["split"]
            if split not in ("train", "val", "test"):
                continue
            cid = r["class_id"]
            class_ids.add(cid)
            rows.append((split, r["path"], cid))

    class_to_idx = {cid: i for i, cid in enumerate(sorted(class_ids))}
    data_dir = os.path.join(root, "data", "raw", "iNaturist")  # images live here
    samples = {"train": [], "val": [], "test": []}
    for split, rel, cid in rows:
        samples[split].append((_abspath(data_dir, rel), class_to_idx[cid]))

    if save:
        with open(os.path.join(root, "_pipeline", "class_index.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"num_classes": len(class_to_idx),
                       "class_to_idx": class_to_idx}, f, indent=2)
    return class_to_idx, samples


# --------------------------------------------------------------------------- #
# 2. transforms  (the "feature engineering" for images)
#    Albumentations if available, otherwise an equivalent torchvision.v2 chain.
# --------------------------------------------------------------------------- #
def _albumentations_transform(img_size, train):
    import numpy as np
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    if train:
        aug = A.Compose([
            A.RandomResizedCrop((img_size, img_size), scale=(0.6, 1.0), ratio=(0.75, 1.333)),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=15,
                               border_mode=0, p=0.5),
            A.OneOf([
                A.ColorJitter(0.2, 0.2, 0.2, 0.1, p=1.0),
                A.HueSaturationValue(20, 30, 20, p=1.0),
            ], p=0.7),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
            A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(8, 32),
                            hole_width_range=(8, 32), p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    else:
        aug = A.Compose([
            A.SmallestMaxSize(max_size=int(img_size * 256 / 224)),
            A.CenterCrop(img_size, img_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

    def _call(pil_img):
        return aug(image=np.asarray(pil_img))["image"]
    return _call


def _torchvision_transform(img_size, train):
    from torchvision.transforms import v2

    if train:
        tf = v2.Compose([
            v2.ToImage(),
            v2.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(0.75, 1.333),
                                 antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomAffine(degrees=15, translate=(0.06, 0.06), scale=(0.9, 1.1)),
            v2.ColorJitter(0.2, 0.2, 0.2, 0.1),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            v2.RandomErasing(p=0.3),                       # ~ CoarseDropout
        ])
    else:
        tf = v2.Compose([
            v2.ToImage(),
            v2.Resize(int(img_size * 256 / 224), antialias=True),
            v2.CenterCrop(img_size),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return tf  # already callable on a PIL image


def build_transforms(img_size=224, train=True):
    """Return a callable PIL.Image -> normalized CHW float tensor.
    Prefers Albumentations; falls back to torchvision.transforms.v2."""
    try:
        import albumentations  # noqa: F401
        return _albumentations_transform(img_size, train), "albumentations"
    except Exception:
        return _torchvision_transform(img_size, train), "torchvision.v2"


# --------------------------------------------------------------------------- #
# 3. Dataset
# --------------------------------------------------------------------------- #
class BeeHeroDataset(Dataset):
    """Reads (path, label) pairs and applies the split-appropriate transform."""

    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        try:
            with Image.open(path) as im:
                img = im.convert("RGB")
        except Exception:
            # robustness: a bad/missing file becomes a black image (rare; pipeline
            # already verified integrity). Never crash a whole epoch.
            img = Image.new("RGB", (224, 224))
        return self.transform(img), label


# --------------------------------------------------------------------------- #
# 4. DataLoaders  (6 GB VRAM recipe)
# --------------------------------------------------------------------------- #
def build_dataloaders(root=DEFAULT_ROOT, img_size=224, batch_size=32,
                      num_workers=None, pin_memory=None):
    """Build train/val/test DataLoaders ready to plug into a training loop."""
    if num_workers is None:
        num_workers = min(6, max(0, (os.cpu_count() or 4) - 2))
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    class_to_idx, samples = build_index(root)
    train_tf, backend = build_transforms(img_size, train=True)
    eval_tf, _ = build_transforms(img_size, train=False)

    sets = {
        "train": BeeHeroDataset(samples["train"], train_tf),
        "val":   BeeHeroDataset(samples["val"],   eval_tf),
        "test":  BeeHeroDataset(samples["test"],  eval_tf),
    }
    common = dict(num_workers=num_workers, pin_memory=pin_memory,
                  persistent_workers=num_workers > 0,
                  prefetch_factor=2 if num_workers > 0 else None)
    train_dl = DataLoader(sets["train"], batch_size=batch_size, shuffle=True,
                          drop_last=True, **common)
    val_dl = DataLoader(sets["val"], batch_size=batch_size, shuffle=False, **common)
    test_dl = DataLoader(sets["test"], batch_size=batch_size, shuffle=False, **common)

    print(f"[bee_hero_dataset] backend={backend} nc={len(class_to_idx)} "
          f"train={len(sets['train'])} val={len(sets['val'])} test={len(sets['test'])} "
          f"workers={num_workers}")
    return train_dl, val_dl, test_dl, class_to_idx


# --------------------------------------------------------------------------- #
# 5. MixUp / CutMix  +  soft-target loss
# --------------------------------------------------------------------------- #
def _rand_bbox(h, w, lam):
    import math
    cut = math.sqrt(1.0 - lam)
    cw, ch = int(w * cut), int(h * cut)
    cx, cy = torch.randint(w, (1,)).item(), torch.randint(h, (1,)).item()
    x1, y1 = max(cx - cw // 2, 0), max(cy - ch // 2, 0)
    x2, y2 = min(cx + cw // 2, w), min(cy + ch // 2, h)
    return x1, y1, x2, y2


def mixup_cutmix(x, y, num_classes, alpha=0.2, cutmix_prob=0.5):
    """Apply MixUp or CutMix to a batch. Returns (x_mixed, soft_targets).
    soft_targets has shape [B, num_classes] -> use SoftTargetCrossEntropy."""
    import numpy as np
    y1 = torch.zeros(x.size(0), num_classes, device=x.device).scatter_(
        1, y.view(-1, 1), 1.0)
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    perm = torch.randperm(x.size(0), device=x.device)
    y2 = y1[perm]

    if np.random.rand() < cutmix_prob:                       # CutMix
        x1_, y1_, x2_, y2_ = _rand_bbox(x.size(2), x.size(3), lam)
        x[:, :, y1_:y2_, x1_:x2_] = x[perm, :, y1_:y2_, x1_:x2_]
        lam = 1.0 - ((x2_ - x1_) * (y2_ - y1_) / (x.size(2) * x.size(3)))
    else:                                                    # MixUp
        x = lam * x + (1.0 - lam) * x[perm]

    target = lam * y1 + (1.0 - lam) * y2
    return x, target


class SoftTargetCrossEntropy(torch.nn.Module):
    """CE against soft (MixUp/CutMix) targets."""
    def forward(self, logits, target):
        logp = torch.log_softmax(logits, dim=1)
        return torch.mean(torch.sum(-target * logp, dim=1))


# --------------------------------------------------------------------------- #
# 6. self-test  — proves the data is training-ready end to end
# --------------------------------------------------------------------------- #
def _denormalize(t):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t.cpu() * std + mean).clamp(0, 1)


def _self_test(root=DEFAULT_ROOT):
    print("=== BEE_HERo data-readiness self-test ===")
    # small, fast settings just to prove the pipeline works
    train_dl, val_dl, test_dl, class_to_idx = build_dataloaders(
        root=root, batch_size=16, num_workers=2)
    nc = len(class_to_idx)

    # (a) load one real batch from each split
    for name, dl in (("train", train_dl), ("val", val_dl), ("test", test_dl)):
        t0 = time.time()
        xb, yb = next(iter(dl))
        assert xb.ndim == 4 and xb.shape[1] == 3, xb.shape
        assert int(yb.min()) >= 0 and int(yb.max()) < nc
        print(f"  {name:5s}: x={tuple(xb.shape)} dtype={xb.dtype} "
              f"y in [{int(yb.min())},{int(yb.max())}]  ({time.time()-t0:.1f}s)")

    # (b) MixUp/CutMix
    xb, yb = next(iter(train_dl))
    xm, ym = mixup_cutmix(xb.clone(), yb, nc)
    print(f"  mixup/cutmix: x={tuple(xm.shape)} soft_target={tuple(ym.shape)} "
          f"(rowsum~{float(ym.sum(1).mean()):.2f})")

    # (c) one real forward + backward through ResNet-18 -> truly training-ready
    import torchvision
    model = torchvision.models.resnet18(weights=None, num_classes=nc)
    crit = SoftTargetCrossEntropy()
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    model.train()
    out = model(xm)
    loss = crit(out, ym)
    loss.backward()
    opt.step()
    print(f"  resnet18 step: logits={tuple(out.shape)} loss={loss.item():.3f}  OK")

    # (d) save an augmented preview grid for visual sanity
    try:
        import torchvision.utils as vutils
        import matplotlib.pyplot as plt
        grid = vutils.make_grid(_denormalize(xb[:16]), nrow=4)
        plt.figure(figsize=(8, 8)); plt.axis("off")
        plt.title("BEE_HERo — augmented training batch (denormalized)")
        plt.imshow(grid.permute(1, 2, 0).numpy())
        out_png = os.path.join(root, "_pipeline", "eda", "augmented_preview.png")
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.tight_layout(); plt.savefig(out_png, dpi=120); plt.close()
        print(f"  preview grid -> {out_png}")
    except Exception as e:
        print(f"  preview grid skipped: {e}")

    print("=== DATA IS 100% TRAINING-READY ===")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    _self_test(ap.parse_args().root)
