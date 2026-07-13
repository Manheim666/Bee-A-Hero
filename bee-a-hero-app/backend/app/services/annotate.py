"""Draw flower + insect YOLO boxes onto a video and save the result.

Uses the repo's trained detectors if their .pt weights are on disk; falls
back to yolov8n so at least *some* boxes appear. Result is written as an
mp4 next to the source. Idempotent — if the target exists it's skipped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2

log = logging.getLogger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
REPO_ROOT = BACKEND_ROOT.parent.parent

DEFAULT_MODEL_CANDIDATES = [
    (
        "flower",
        REPO_ROOT / "data/interim/cv_runs/flower_det2_v2_yolo26m/weights/best.pt",
    ),
    (
        "insect",
        REPO_ROOT / "data/interim/cv_runs/insect_multidet_v2_yolo26m/weights/best.pt",
    ),
]

_PALETTE = [
    (0, 178, 246),
    (89, 191, 107),
    (181, 141, 76),
    (66, 245, 245),
    (255, 128, 128),
    (128, 255, 255),
    (128, 128, 255),
    (255, 255, 128),
]


def _color(idx: int) -> tuple[int, int, int]:
    return _PALETTE[idx % len(_PALETTE)]


@dataclass
class LoadedModel:
    tag: str
    model: object


def _resolve_models() -> list[LoadedModel]:
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover
        log.error("ultralytics not available: %s", exc)
        return []

    loaded: list[LoadedModel] = []
    for tag, path in DEFAULT_MODEL_CANDIDATES:
        if path.exists():
            log.info("Annotator loading %s: %s", tag, path)
            loaded.append(LoadedModel(tag=tag, model=YOLO(str(path))))
    if not loaded:
        log.info("Annotator falling back to yolov8n.pt (generic COCO)")
        loaded.append(LoadedModel(tag="", model=YOLO("yolov8n.pt")))
    return loaded


def _draw_boxes(frame, results, tag: str, palette_offset: int) -> int:
    count = 0
    for r in results:
        names = r.names
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
            color = _color(palette_offset + cls_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            prefix = f"[{tag}] " if tag else ""
            text = f"{prefix}{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y_top = max(0, y1 - th - 6)
            cv2.rectangle(frame, (x1, y_top), (x1 + tw + 6, y_top + th + 6), color, -1)
            cv2.putText(
                frame, text, (x1 + 3, y_top + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA,
            )
            count += 1
    return count


def annotate_video(
    src: str | Path,
    dst: str | Path,
    conf: float = 0.35,
    imgsz: int = 640,
    frame_stride: int = 1,
) -> Path:
    """Read `src`, draw detections, write `dst`. Returns the dst Path.

    `frame_stride` re-runs inference every N frames and reuses the last
    boxes on the intervening frames — a cheap speedup for slow CPUs.
    """

    src = Path(src)
    dst = Path(dst)
    if dst.exists() and dst.stat().st_size > 0:
        log.info("Annotated video already exists: %s", dst)
        return dst

    models = _resolve_models()
    if not models:
        raise RuntimeError("No YOLO models available for annotation")

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source video: {src}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    log.info(
        "Annotating %s -> %s (%.1f fps, %dx%d, %d frames, %d models)",
        src.name, dst.name, fps, width, height, total, len(models),
    )

    # mp4v is universally supported by cv2 wheels; browsers play it fine.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dst), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open writer for {dst}")

    last_results_per_model: list[list] = [[] for _ in models]
    frame_idx = 0
    total_detections = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % max(1, frame_stride) == 0:
                for i, lm in enumerate(models):
                    last_results_per_model[i] = lm.model.predict(
                        frame, conf=conf, imgsz=imgsz, verbose=False,
                    )

            for i, lm in enumerate(models):
                total_detections += _draw_boxes(
                    frame, last_results_per_model[i], lm.tag, palette_offset=i * 3,
                )

            writer.write(frame)
            frame_idx += 1
            if frame_idx % 60 == 0:
                log.info("  … %d / %d frames", frame_idx, total)
    finally:
        cap.release()
        writer.release()

    log.info(
        "Wrote %s (%d frames, %d total detections)",
        dst, frame_idx, total_detections,
    )
    return dst
