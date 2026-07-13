"""Capture + inference threads with latest-frame semantics.

Capture thread pulls frames from the DroidCam MJPEG stream as fast as the
network allows, keeping only the newest raw frame. Inference thread pulls
that newest frame, runs YOLO, draws boxes, encodes JPEG, and stores the
result as the newest annotated frame. The HTTP MJPEG generator reads that
slot at its own pace. Slow inference never blocks capture — old frames
are simply dropped.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .config import settings

log = logging.getLogger(__name__)


# --- palette for boxes (BGR, cycled per class) --------------------------------
_PALETTE = [
    (0, 178, 246),   # honey
    (89, 191, 107),  # leaf
    (181, 141, 76),  # non-pollinator (cool)
    (66, 245, 245),  # amber
    (255, 128, 128), # pink
    (128, 255, 255),
    (128, 128, 255),
    (255, 255, 128),
]


def _color(idx: int) -> tuple[int, int, int]:
    return _PALETTE[idx % len(_PALETTE)]


@dataclass
class Detection:
    label: str
    conf: float
    box: tuple[int, int, int, int]  # x1, y1, x2, y2
    model_label: str = ""


@dataclass
class PipelineState:
    """Shared, thread-safe view of the current stream health."""

    connected: bool = False
    reconnecting: bool = False
    last_error: str = ""
    inference_fps: float = 0.0
    capture_fps: float = 0.0
    detection_count: int = 0
    per_class_counts: dict[str, int] = field(default_factory=dict)
    frame_shape: tuple[int, int] = (0, 0)  # h, w
    started_at: float = field(default_factory=time.time)

    def snapshot(self) -> dict:
        return {
            "connected": self.connected,
            "reconnecting": self.reconnecting,
            "last_error": self.last_error,
            "inference_fps": round(self.inference_fps, 1),
            "capture_fps": round(self.capture_fps, 1),
            "detection_count": self.detection_count,
            "per_class_counts": dict(self.per_class_counts),
            "frame_shape": {"h": self.frame_shape[0], "w": self.frame_shape[1]},
            "uptime_sec": round(time.time() - self.started_at, 1),
        }


class _LatestSlot:
    """Single-slot mailbox with drop-old semantics."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._value = None  # arbitrary payload

    def put(self, value) -> None:
        with self._cv:
            self._value = value
            self._cv.notify_all()

    def get(self, timeout: float | None = None):
        with self._cv:
            if self._value is None:
                self._cv.wait(timeout=timeout)
            value = self._value
            # Do not clear — MJPEG generator wants to keep serving the last
            # good frame if inference stalls briefly.
            return value

    def take_new(self, last_seen_id: int, timeout: float = 1.0):
        """Block until a new value (identified by id()) is available."""
        with self._cv:
            deadline = time.time() + timeout
            while self._value is None or id(self._value) == last_seen_id:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return self._value
                self._cv.wait(timeout=remaining)
            return self._value


class Pipeline:
    def __init__(self) -> None:
        self.state = PipelineState()
        self._raw_slot = _LatestSlot()      # np.ndarray BGR
        self._jpeg_slot = _LatestSlot()     # bytes (encoded JPEG)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._models = []
        self._model_labels: list[str] = []
        self._person_model = None            # COCO detector for the human-veto (lazy)

    # ------------------------------------------------------------------ setup
    def start(self) -> None:
        self._load_models()
        capture_thread = threading.Thread(
            target=self._capture_loop, name="capture", daemon=True
        )
        infer_thread = threading.Thread(
            target=self._infer_loop, name="infer", daemon=True
        )
        capture_thread.start()
        infer_thread.start()
        self._threads = [capture_thread, infer_thread]
        log.info("Pipeline started")

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=2.0)

    def _load_models(self) -> None:
        try:
            from ultralytics import YOLO  # lazy import — heavy
        except Exception as exc:  # pragma: no cover — surfaced at request time
            log.error("ultralytics import failed: %s", exc)
            raise

        paths = settings.model_path_list()
        labels = settings.model_label_list()
        for path, lbl in zip(paths, labels):
            log.info("Loading model %s (label=%s)", path, lbl or "-")
            model = YOLO(path)
            if settings.device and settings.device != "cpu":
                try:
                    model.to(settings.device)
                except Exception as exc:
                    log.warning("Could not move model to %s: %s", settings.device, exc)
            self._models.append(model)
            self._model_labels.append(lbl)

        if settings.person_veto:
            try:
                self._person_model = YOLO(settings.person_model)   # auto-downloads if absent
                log.info("Person-veto enabled (%s)", settings.person_model)
            except Exception as exc:
                log.warning("Person-veto model unavailable (%s); size-gate still active", exc)
                self._person_model = None

    # ------------------------------------------------------------------ capture
    def _capture_loop(self) -> None:
        url = settings.droidcam_url
        cap: Optional[cv2.VideoCapture] = None
        frames = 0
        window_start = time.time()

        while not self._stop.is_set():
            if cap is None or not cap.isOpened():
                self.state.connected = False
                self.state.reconnecting = True
                log.info("Opening DroidCam stream: %s", url)
                cap = cv2.VideoCapture(url)
                # Cheap trick: keep buffer tiny so we always read the latest.
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                if not cap.isOpened():
                    self.state.last_error = f"Could not open {url}"
                    log.warning("Open failed; retrying in %.1fs", settings.reconnect_delay)
                    time.sleep(settings.reconnect_delay)
                    continue
                self.state.connected = True
                self.state.reconnecting = False
                self.state.last_error = ""

            ok, frame = cap.read()
            if not ok or frame is None:
                self.state.last_error = "Stream read failed"
                log.warning("Read failed; reconnecting")
                try:
                    cap.release()
                except Exception:
                    pass
                cap = None
                self.state.connected = False
                self.state.reconnecting = True
                time.sleep(settings.reconnect_delay)
                continue

            self.state.frame_shape = (frame.shape[0], frame.shape[1])
            self._raw_slot.put(frame)
            frames += 1
            now = time.time()
            if now - window_start >= 1.0:
                self.state.capture_fps = frames / (now - window_start)
                frames = 0
                window_start = now

        if cap is not None:
            cap.release()

    # ------------------------------------------------------------------ inference
    def _infer_loop(self) -> None:
        last_id = 0
        infer_count = 0
        window_start = time.time()

        while not self._stop.is_set():
            frame = self._raw_slot.take_new(last_seen_id=last_id, timeout=1.0)
            if frame is None:
                continue
            if id(frame) == last_id:
                continue
            last_id = id(frame)

            try:
                annotated, dets = self._run_inference(frame)
            except Exception as exc:
                log.exception("Inference error: %s", exc)
                annotated = frame
                dets = []

            ok, buf = cv2.imencode(
                ".jpg", annotated,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(settings.jpeg_quality)],
            )
            if not ok:
                continue
            self._jpeg_slot.put(buf.tobytes())

            # Update per-frame stats.
            self.state.detection_count = len(dets)
            counts: dict[str, int] = {}
            for d in dets:
                key = f"{d.model_label}:{d.label}" if d.model_label else d.label
                counts[key] = counts.get(key, 0) + 1
            self.state.per_class_counts = counts

            infer_count += 1
            now = time.time()
            if now - window_start >= 1.0:
                self.state.inference_fps = infer_count / (now - window_start)
                infer_count = 0
                window_start = now

    def _person_boxes(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """COCO person boxes for the veto (empty if the veto model is off/unavailable)."""
        if self._person_model is None:
            return []
        boxes: list[tuple[int, int, int, int]] = []
        for r in self._person_model.predict(
            frame, conf=settings.person_conf, imgsz=settings.img_size,
            classes=[0], verbose=False,        # class 0 = person
        ):
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                boxes.append((x1, y1, x2, y2))
        return boxes

    @staticmethod
    def _vetoed(box, persons, frame_area: float) -> bool:
        """True if `box` is too big to be a flower/insect, or its centre sits in a person."""
        x1, y1, x2, y2 = box
        if (x2 - x1) * (y2 - y1) > settings.max_box_frac * frame_area:
            return True                        # frame-filling blob -> wall/person/FP
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        return any(px1 <= cx <= px2 and py1 <= cy <= py2 for px1, py1, px2, py2 in persons)

    def _run_inference(self, frame: np.ndarray) -> tuple[np.ndarray, list[Detection]]:
        detections: list[Detection] = []
        annotated = frame.copy()
        h, w = frame.shape[:2]
        frame_area = float(h * w)
        persons = self._person_boxes(frame)    # detect humans once, then veto against them

        for model_idx, model in enumerate(self._models):
            results = model.predict(
                frame,
                conf=settings.conf_threshold,
                imgsz=settings.img_size,
                verbose=False,
            )
            model_label = self._model_labels[model_idx]
            for r in results:
                names = r.names
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    cls_id = int(box.cls.item())
                    conf = float(box.conf.item())
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    if self._vetoed((x1, y1, x2, y2), persons, frame_area):
                        continue               # drop human / oversized false positive
                    label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
                    det = Detection(
                        label=label,
                        conf=conf,
                        box=(x1, y1, x2, y2),
                        model_label=model_label,
                    )
                    detections.append(det)
                    self._draw(annotated, det, model_idx * 3 + cls_id)

        self._draw_hud(annotated)
        return annotated, detections

    # ------------------------------------------------------------------ draw
    def _draw(self, img: np.ndarray, det: Detection, palette_idx: int) -> None:
        color = _color(palette_idx)
        x1, y1, x2, y2 = det.box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        prefix = f"[{det.model_label}] " if det.model_label else ""
        text = f"{prefix}{det.label} {det.conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_top = max(0, y1 - th - 6)
        cv2.rectangle(img, (x1, y_top), (x1 + tw + 6, y_top + th + 6), color, -1)
        cv2.putText(
            img, text, (x1 + 3, y_top + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA,
        )

    def _draw_hud(self, img: np.ndarray) -> None:
        text = f"infer {self.state.inference_fps:.1f} fps  |  cap {self.state.capture_fps:.1f} fps  |  {self.state.detection_count} dets"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img, (8, 8), (8 + tw + 12, 8 + th + 12), (24, 30, 23), -1)
        cv2.putText(
            img, text, (14, 8 + th + 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 214, 102), 1, cv2.LINE_AA,
        )

    # ------------------------------------------------------------------ read
    def latest_jpeg(self, last_seen_id: int, timeout: float = 1.0) -> tuple[bytes, int]:
        payload = self._jpeg_slot.take_new(last_seen_id=last_seen_id, timeout=timeout)
        if payload is None:
            return b"", last_seen_id
        return payload, id(payload)


pipeline: Pipeline | None = None


def get_pipeline() -> Pipeline:
    global pipeline
    if pipeline is None:
        pipeline = Pipeline()
        pipeline.start()
    return pipeline
