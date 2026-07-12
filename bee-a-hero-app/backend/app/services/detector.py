"""Detection service.

Exposes ONE function the rest of the app calls: `run_detection(video_path)`.
It returns a RESULTS_SUMMARY-style dict so callers never change.

Two backends, chosen automatically at import:
  * ONNX (real)  — when `onnxruntime`, `opencv-python`, and the exported detectors in
    `models/onnx/{flower,insect}.onnx` are present. Runs the real Bee-A-Hero detectors on
    sampled frames under onnxruntime (CPU, no torch/ultralytics) and counts a visit whenever an
    insect box sits inside a flower box.
  * MOCK        — otherwise; deterministic-per-file plausible numbers, so the demo runs with no
    models or heavy deps.

Regenerate the ONNX files with `python -m src.cv_engine.export_onnx` (repo root).
"""
from __future__ import annotations

import hashlib
import random
from pathlib import Path

# Headline metrics anchored to the real Bee-A-Hero project (best-checkpoint validation).
FLOWER_MAP = 0.808
INSECT_MAP = 0.669
CLASSIFIER_ACC = 0.978

# insect detector classes counted as pollinators (rest -> non-pollinator)
_POLLINATOR_CLASSES = {"bee", "honeybee", "butterfly", "fly"}

# repo root = .../bee-a-hero-app/backend/app/services/detector.py -> parents[4]
_REPO_ROOT = Path(__file__).resolve().parents[4]
_ONNX_DIR = _REPO_ROOT / "models" / "onnx"


# --------------------------------------------------------------------------- #
# ONNX backend (real) — loaded lazily; falls back to mock if anything is missing
# --------------------------------------------------------------------------- #
class _OnnxDetector:
    """onnxruntime wrapper for one YOLO26 detector (letterbox -> forward -> decode -> NMS)."""

    def __init__(self, onnx_path: Path, imgsz: int = 640):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.imgsz = imgsz
        meta = self.sess.get_modelmeta().custom_metadata_map or {}
        self.names = self._parse_names(meta.get("names", "{}"))  # ultralytics embeds class names

    @staticmethod
    def _parse_names(s: str) -> dict:
        try:
            import ast
            return {int(k): v for k, v in ast.literal_eval(s).items()}
        except Exception:
            return {}

    def __call__(self, frame, conf: float = 0.25):
        """Detect on one BGR frame -> [(xyxy_orig, class_name, conf), ...].

        YOLO26 exports end-to-end (NMS built in): the output is ``[1, 300, 6]`` where each row is
        ``[x1, y1, x2, y2, conf, class_id]`` in the 640-input (letterbox) pixel space. No manual
        decode/NMS needed — just threshold, map the class id, and scale the box back.
        """
        import cv2
        import numpy as np
        h0, w0 = frame.shape[:2]
        r = self.imgsz / max(h0, w0)                              # image placed at top-left, no pad offset
        nw, nh = int(round(w0 * r)), int(round(h0 * r))
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, np.uint8)
        canvas[:nh, :nw] = cv2.resize(frame, (nw, nh))
        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        pred = np.squeeze(self.sess.run(None, {self.inp: blob})[0], 0)   # [N, 6]
        out = []
        for x1, y1, x2, y2, cf, cid in pred:
            if cf <= conf:
                continue
            box = np.array([x1 / r, y1 / r, x2 / r, y2 / r])     # letterbox px -> original px
            out.append((box, self.names.get(int(cid), str(int(cid))), float(cf)))
        return out


_flower_det = None
_insect_det = None


def _load_onnx() -> bool:
    """Build the ONNX detectors once; True if both are ready, else False (-> mock)."""
    global _flower_det, _insect_det
    if _flower_det is not None:
        return True
    fp, ip = _ONNX_DIR / "flower.onnx", _ONNX_DIR / "insect.onnx"
    if not (fp.exists() and ip.exists()):
        return False
    try:
        import onnxruntime  # noqa: F401
        import cv2  # noqa: F401
        _flower_det = _OnnxDetector(fp)
        _insect_det = _OnnxDetector(ip)
        return True
    except Exception:
        return False


def _run_onnx(video_path: str, sample_fps: float = 1.0) -> dict:
    """Run the ONNX detectors on ~`sample_fps` frames/sec and count insect-in-flower visits."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, int(round(fps / sample_fps)))
    flower_ids: dict[tuple, int] = {}    # rough flower identity by a coarse centre grid
    visits: list[dict] = []
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fi % stride == 0:
            t = round(fi / fps, 2)
            flowers = _flower_det(frame)
            insects = _insect_det(frame)
            for fb, _, _ in flowers:
                flower_ids.setdefault((int(fb[0] / 40), int(fb[1] / 40)), len(flower_ids) + 1)
            for ib, iname, _ in insects:
                icx, icy = (ib[0] + ib[2]) / 2, (ib[1] + ib[3]) / 2
                for fb, _, _ in flowers:
                    if fb[0] <= icx <= fb[2] and fb[1] <= icy <= fb[3]:
                        fkey = (int(fb[0] / 40), int(fb[1] / 40))
                        visits.append({
                            "flower_id": flower_ids.setdefault(fkey, len(flower_ids) + 1),
                            "insect_class": iname,
                            "is_pollinator": iname.lower() in _POLLINATOR_CLASSES,
                            "dwell_sec": round(1.0 / sample_fps, 2),
                            "first_seen_sec": t,
                        })
                        break
        fi += 1
    cap.release()
    pol = sum(1 for v in visits if v["is_pollinator"])
    return {
        "flower_mAP": FLOWER_MAP, "insect_mAP": INSECT_MAP, "classifier_acc": CLASSIFIER_ACC,
        "flower_map": len(flower_ids), "insect_tracks": len(visits),
        "pollinator_visits": pol, "non_pollinator_visits": len(visits) - pol,
        "visits": visits, "backend": "onnx",
    }


# --------------------------------------------------------------------------- #
# Mock backend (fallback) — deterministic per file, no models/deps
# --------------------------------------------------------------------------- #
_POLLINATORS = ["honey_bee", "bumblebee", "hoverfly", "solitary_bee"]
_NON_POLLINATORS = ["housefly", "ant", "wasp", "beetle"]


def _seed_from_path(video_path: str) -> int:
    return int(hashlib.sha256(video_path.encode("utf-8")).hexdigest()[:8], 16)


def _run_mock(video_path: str) -> dict:
    rng = random.Random(_seed_from_path(video_path))
    n_flowers = rng.randint(6, 12)
    visits: list[dict] = []
    for flower_id in range(1, n_flowers + 1):
        for _ in range(rng.randint(0, 5)):
            is_pol = rng.random() < 0.6
            visits.append({
                "flower_id": flower_id,
                "insect_class": rng.choice(_POLLINATORS if is_pol else _NON_POLLINATORS),
                "is_pollinator": is_pol,
                "dwell_sec": round(rng.uniform(0.4, 6.0), 2),
                "first_seen_sec": round(rng.uniform(0.0, 60.0), 2),
            })
    pol = sum(1 for v in visits if v["is_pollinator"])
    return {
        "flower_mAP": FLOWER_MAP, "insect_mAP": INSECT_MAP, "classifier_acc": CLASSIFIER_ACC,
        "flower_map": n_flowers, "insect_tracks": len(visits),
        "pollinator_visits": pol, "non_pollinator_visits": len(visits) - pol,
        "visits": visits, "backend": "mock",
    }


def run_detection(video_path: str) -> dict:
    """Real ONNX detection when models + deps are present, else the deterministic mock."""
    if _load_onnx():
        try:
            return _run_onnx(video_path)
        except Exception as e:  # any runtime issue -> never break the app; fall back to mock
            print(f"[detector] ONNX run failed ({type(e).__name__}: {e}); using mock.")
    return _run_mock(video_path)
