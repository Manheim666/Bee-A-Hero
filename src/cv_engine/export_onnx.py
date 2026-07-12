"""Export the trained CV detectors to ONNX for deployment (e.g. the web backend).

ONNX lets the detectors run under `onnxruntime` — no torch/ultralytics, CPU-friendly — which is
what the website's backend uses to score uploaded videos without a GPU. This exports the two
YOLO26 detectors (flower + insect); the honeybee sub-classifier is a small torch classifier and
is exported too when its weights are present.

Outputs land next to each `.pt` (ultralytics writes `<name>.onnx` beside the weights) and are
also copied into `models/onnx/` for a single deployment folder. ONNX files are large and
git-ignored — this script regenerates them from the committed `.pt` weights.

Run:
    pip install onnx onnxruntime onnxscript   # export + verify deps (onnxscript: torch.onnx)
    python -m src.cv_engine.export_onnx        # export all detectors, verify each loads in onnxruntime

Deployment (web backend) only needs ``onnxruntime`` to *run* the exported ``.onnx`` files.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src import config as C

_RUNS = C.INTERIM_DIR / "cv_runs"
ONNX_DIR = C.REPO_ROOT / "models" / "onnx"

# (label, weights path) for the two YOLO detectors shipped with the repo.
DETECTORS = {
    "flower": _RUNS / "flower_det2_v2_yolo26m" / "weights" / "best.pt",
    "insect": _RUNS / "insect_multidet_v2_yolo26m" / "weights" / "best.pt",
}
HONEYBEE = _RUNS / "honeybee_clf" / "best.pt"


def export_yolo(name: str, weights: Path, imgsz: int = 640, opset: int = 12) -> Path | None:
    """Export one YOLO detector to ONNX and copy it into models/onnx/. Returns the ONNX path."""
    if not weights.exists():
        print(f"[skip] {name}: weights not found at {weights}")
        return None
    from ultralytics import YOLO
    print(f"[export] {name}: {weights} -> ONNX (imgsz={imgsz}, opset={opset}) ...")
    onnx_path = Path(YOLO(str(weights)).export(format="onnx", imgsz=imgsz, opset=opset,
                                               dynamic=True, simplify=False))
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    dst = ONNX_DIR / f"{name}.onnx"
    shutil.copy2(onnx_path, dst)
    print(f"          wrote {dst}")
    return dst


def export_honeybee(weights: Path = HONEYBEE, opset: int = 12) -> Path | None:
    """Export the timm honeybee sub-classifier to ONNX via torch.onnx (it is not a YOLO ckpt).

    Rebuilds the model exactly as ``visit_counter.Classifier`` does — ``timm.create_model`` +
    ``state_dict`` from the ``{classes, model, state_dict}`` checkpoint — then traces it to ONNX.
    Never raises: a failure is reported and skipped so the detector exports still succeed.
    """
    if not weights.exists():
        print(f"[skip] honeybee: weights not found at {weights}")
        return None
    try:
        import timm
        import torch
        ckpt = torch.load(str(weights), map_location="cpu", weights_only=True)
        model = timm.create_model(ckpt["model"], pretrained=False,
                                  num_classes=len(ckpt["classes"]))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        sz = timm.data.resolve_data_config({}, model=model).get("input_size", (3, 224, 224))[-1]
        ONNX_DIR.mkdir(parents=True, exist_ok=True)
        dst = ONNX_DIR / "honeybee.onnx"
        print(f"[export] honeybee (timm {ckpt['model']}, imgsz={sz}) -> ONNX ...")
        torch.onnx.export(model, torch.randn(1, 3, sz, sz), str(dst), opset_version=opset,
                          input_names=["input"], output_names=["logits"],
                          dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
        print(f"          wrote {dst}  (classes: {ckpt['classes']})")
        return dst
    except Exception as e:
        print(f"[skip] honeybee export failed ({type(e).__name__}: {e})")
        return None


def verify_onnx(path: Path) -> bool:
    """Load the ONNX model in onnxruntime and print its I/O signature (proves it runs)."""
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        i, o = sess.get_inputs()[0], sess.get_outputs()[0]
        print(f"          verify {path.name}: in {i.name}{i.shape} -> out {o.name}{o.shape} ✓")
        return True
    except Exception as e:
        print(f"          verify {path.name}: FAILED ({type(e).__name__}: {e})")
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=12)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    exported: list[Path] = []
    for name, w in DETECTORS.items():
        p = export_yolo(name, w, imgsz=args.imgsz, opset=args.opset)
        if p:
            exported.append(p)

    # honeybee sub-classifier (timm model, not a YOLO ckpt) — exported via torch.onnx
    p = export_honeybee(HONEYBEE, opset=args.opset)
    if p:
        exported.append(p)

    if not args.no_verify:
        print("\n[verify] loading each ONNX model in onnxruntime ...")
        ok = all(verify_onnx(p) for p in exported)
        print(f"\n{'ALL ONNX MODELS VERIFIED' if ok else 'SOME MODELS FAILED VERIFICATION'} "
              f"— {len(exported)} file(s) in {ONNX_DIR}")


if __name__ == "__main__":
    main()
