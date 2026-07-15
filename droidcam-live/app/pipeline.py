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
from pathlib import Path
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
        self._jpeg_slot = _LatestSlot()     # bytes (encoded JPEG) — real-time, newest wins
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._models = []
        self._model_labels: list[str] = []
        self._person_model = None            # COCO detector for the human-veto (lazy)
        # Live camera source, switchable at runtime from the browser (connect a phone via
        # DroidCam without restarting the service). A bare int -> local webcam device.
        self._source: str = settings.droidcam_url
        self._source_lock = threading.Lock()
        # Live landing logging (rolling CSV/JSON) when a flower + insect model are both loaded.
        self._insect_idx: int | None = None
        self._flower_idx: int | None = None
        self._landing_logger = None

    # --------------------------------------------------------------- source switch
    def set_source(self, url: str) -> None:
        """Point the capture thread at a new camera (phone DroidCam URL or webcam index)."""
        with self._source_lock:
            self._source = str(url).strip()
        log.info("Camera source change requested: %s", url)

    def get_source(self) -> str:
        with self._source_lock:
            return self._source

    def _resolved_source(self):
        url = self.get_source()
        return int(url) if url.isdigit() else url

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

        # Enable live landing logging only when a flower model AND an insect model are present,
        # matched by their MODEL_LABELS tags. Without both there is nothing to associate.
        for idx, lbl in enumerate(self._model_labels):
            if lbl == settings.insect_label:
                self._insect_idx = idx
            elif lbl == settings.flower_label:
                self._flower_idx = idx
        if settings.landing_log and self._insect_idx is not None and self._flower_idx is not None:
            from .landings import LandingLogger
            out_dir = Path(__file__).resolve().parent.parent / settings.live_out_dir
            self._landing_logger = LandingLogger(
                out_dir, settings.min_land_s, settings.land_grace_s, settings.stationary_tau)
            log.info("Live landing logging ON -> %s (flower=model#%d, insect=model#%d)",
                     out_dir, self._flower_idx, self._insect_idx)
        else:
            log.info("Live landing logging OFF (need both flower+insect models; "
                     "labels=%s)", self._model_labels)

    # ------------------------------------------------------------------ capture
    def _capture_loop(self) -> None:
        # A bare integer (e.g. "0") means a LOCAL webcam device, not an http stream —
        # so "live camera" works with the machine's own camera when there's no phone.
        # The source is switchable at runtime (set_source) so a phone can be connected
        # via DroidCam from the browser without restarting the service.
        source = self._resolved_source()
        cap: Optional[cv2.VideoCapture] = None
        frames = 0
        window_start = time.time()

        while not self._stop.is_set():
            desired = self._resolved_source()
            if desired != source:                # user switched cameras -> drop and reopen
                log.info("Switching camera source %s -> %s", source, desired)
                source = desired
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
                cap = None
                self.state.connected = False
                self.state.reconnecting = True
            if cap is None or not cap.isOpened():
                self.state.connected = False
                self.state.reconnecting = True
                log.info("Opening camera source: %s", source)
                cap = cv2.VideoCapture(source)
                # Cheap trick: keep buffer tiny so we always read the latest.
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                if not cap.isOpened():
                    self.state.last_error = f"Could not open {source}"
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
    def _dedup_insects(tracks: list) -> list:
        """Drop a box that duplicates/nests inside a higher-confidence insect box.

        The detector can box one bug twice (e.g. a `bee` box and an overlapping `butterfly`
        box) -> "an insect inside an insect". Class-aware NMS keeps both; this class-agnostic
        pass keeps only the highest-confidence box among heavily-overlapping/contained ones.
        ``tracks``: [(tid, box, label, conf)]. Returns the kept subset in original order."""
        order = sorted(range(len(tracks)), key=lambda i: tracks[i][3], reverse=True)
        kept_boxes: list = []
        keep_idx: list = []
        for i in order:
            box = tracks[i][1]
            ia = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
            drop = False
            for kb in kept_boxes:
                ox1, oy1 = max(box[0], kb[0]), max(box[1], kb[1])
                ox2, oy2 = min(box[2], kb[2]), min(box[3], kb[3])
                inter = max(0.0, ox2 - ox1) * max(0.0, oy2 - oy1)
                if inter <= 0:
                    continue
                ka = max(1.0, (kb[2] - kb[0]) * (kb[3] - kb[1]))
                iou = inter / (ia + ka - inter)
                contain = inter / min(ia, ka)      # small box mostly inside the kept one
                if iou >= 0.6 or contain >= 0.75:
                    drop = True
                    break
            if not drop:
                kept_boxes.append(box)
                keep_idx.append(i)
        return [tracks[i] for i in sorted(keep_idx)]

    @staticmethod
    def _is_flower_not_insect(box, flower_boxes) -> bool:
        """True if `box` is really a flower the insect model mislabelled.

        A real insect on a flower is a *small* box inside it: IoU(small, flower) is low and
        its area is a small fraction of the flower's. A mislabelled flower is a box that
        coincides with a flower: high IoU and near-equal area. Veto only the latter, so
        genuine insects sitting on flowers are kept (their landings still count)."""
        ix1, iy1, ix2, iy2 = box
        ia = max(1.0, (ix2 - ix1) * (iy2 - iy1))
        for fx1, fy1, fx2, fy2 in flower_boxes:
            fa = max(1.0, (fx2 - fx1) * (fy2 - fy1))
            ox1, oy1 = max(ix1, fx1), max(iy1, fy1)
            ox2, oy2 = min(ix2, fx2), min(iy2, fy2)
            inter = max(0.0, ox2 - ox1) * max(0.0, oy2 - oy1)
            if inter <= 0:
                continue
            if inter / (ia + fa - inter) >= settings.insect_flower_iou:
                return True                         # box ~ the whole flower, not an insect on it
        return False

    @staticmethod
    def _plausible_flower(box, frame_area: float) -> bool:
        """A real flower is compact and only part of the frame -> reject slivers, noise, and
        whole-scene boxes (the 'the whole screen is a flower' false positive)."""
        x1, y1, x2, y2 = box
        a = max(1.0, (x2 - x1) * (y2 - y1))
        w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
        aspect = max(w / h, h / w)
        return (settings.flower_min_frac * frame_area <= a <= settings.flower_max_frac * frame_area
                and aspect <= settings.flower_max_aspect)

    @staticmethod
    def _nms_boxes(cands: list) -> list:
        """Class-agnostic NMS on [(box, conf)]: keep the highest-confidence box among any
        overlapping/contained set -> no 'flower inside a flower' duplicates."""
        order = sorted(range(len(cands)), key=lambda i: cands[i][1], reverse=True)
        kept_boxes: list = []
        for i in order:
            box = cands[i][0]
            ia = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
            drop = False
            for kb in kept_boxes:
                ox1, oy1 = max(box[0], kb[0]), max(box[1], kb[1])
                ox2, oy2 = min(box[2], kb[2]), min(box[3], kb[3])
                inter = max(0.0, ox2 - ox1) * max(0.0, oy2 - oy1)
                if inter <= 0:
                    continue
                ka = max(1.0, (kb[2] - kb[0]) * (kb[3] - kb[1]))
                if (inter / (ia + ka - inter) >= settings.box_nms_iou
                        or inter / min(ia, ka) >= settings.box_nms_contain):
                    drop = True
                    break
            if not drop:
                kept_boxes.append(box)
        return kept_boxes

    @staticmethod
    def _vetoed(box, persons, frame_area: float) -> bool:
        """True if `box` is too big to be a flower/insect, or its box basically IS a person.

        Person test is IoU-based, not centre-in-person: a detection is vetoed only when it
        overlaps a person box by >= ``person_veto_iou`` (the box ~ the whole person -> a human
        misread as flower/insect). A small object HELD by a person is a small box inside a big
        person box -> low IoU -> kept, so close-up demo subjects (flower/bee in hand) survive."""
        x1, y1, x2, y2 = box
        ba = (x2 - x1) * (y2 - y1)
        if ba > settings.max_box_frac * frame_area:
            return True                        # frame-filling blob -> wall/person/FP
        ba = max(1.0, ba)
        for px1, py1, px2, py2 in persons:
            ox1, oy1 = max(x1, px1), max(y1, py1)
            ox2, oy2 = min(x2, px2), min(y2, py2)
            inter = max(0.0, ox2 - ox1) * max(0.0, oy2 - oy1)
            if inter <= 0:
                continue
            pa = max(1.0, (px2 - px1) * (py2 - py1))
            if inter / (ba + pa - inter) >= settings.person_veto_iou:
                return True                    # box coincides with a person -> human FP
        return False

    def _run_inference(self, frame: np.ndarray) -> tuple[np.ndarray, list[Detection]]:
        if self._landing_logger is not None:
            return self._run_inference_landing(frame)
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

    def _run_inference_landing(self, frame: np.ndarray) -> tuple[np.ndarray, list[Detection]]:
        """Flower + insect models with insect tracking -> feed the landing logger, draw all."""
        annotated = frame.copy()
        h, w = frame.shape[:2]
        frame_area = float(h * w)
        persons = self._person_boxes(frame)
        detections: list[Detection] = []

        # Flowers (per-frame detection; stable ids are assigned by the logger). Gate by shape/size
        # (drop scene-sized false positives) then NMS so one flower is one box, not nested boxes.
        flower_cand: list = []
        for r in self._models[self._flower_idx].predict(
            frame, conf=settings.flower_conf, imgsz=settings.img_size, verbose=False
        ):
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                if self._vetoed((x1, y1, x2, y2), persons, frame_area):
                    continue
                if not self._plausible_flower((x1, y1, x2, y2), frame_area):
                    continue
                flower_cand.append(((x1, y1, x2, y2), float(box.conf.item())))
        flower_boxes = self._nms_boxes(flower_cand)

        # Insects (tracked across frames so landings can span a whole dwell). Held to a higher
        # confidence bar and vetoed when a box IS a flower (high IoU / near flower size) so a
        # colourful flower is never kept as a "butterfly". Small insects sitting ON a flower
        # (low IoU) still pass -> real landings survive.
        insect_tracks: list = []
        for r in self._models[self._insect_idx].track(
            frame, persist=True, conf=settings.insect_conf, imgsz=settings.img_size,
            tracker="botsort.yaml", verbose=False
        ):
            names = r.names
            if r.boxes is None or r.boxes.id is None:
                continue
            ids = r.boxes.id.int().cpu().tolist()
            for box, tid in zip(r.boxes, ids):
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                if self._vetoed((x1, y1, x2, y2), persons, frame_area):
                    continue
                if self._is_flower_not_insect((x1, y1, x2, y2), flower_boxes):
                    continue                       # this box is a flower mislabelled as an insect
                label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
                insect_tracks.append((tid, (x1, y1, x2, y2), label, conf))

        # Drop duplicate/nested boxes on the same bug (an "insect inside an insect").
        insect_tracks = self._dedup_insects(insect_tracks)

        # Update landing episodes; get flowers back with sticky ids for drawing.
        flowers = self._landing_logger.observe(insect_tracks, flower_boxes)

        for fid, fb in flowers:
            # FlowerRegistry keeps EMA (float) boxes; cv2 needs ints. Casting here was the fix
            # for the live feed crashing every frame a flower was present (exception swallowed
            # in _infer_loop -> raw frame, no boxes -> "detects then loses everything").
            x1, y1, x2, y2 = (int(round(v)) for v in fb)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(annotated, str(fid), (x1, max(12, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
            detections.append(Detection(label="flower", conf=1.0, box=(x1, y1, x2, y2),
                                        model_label=settings.flower_label))
        for tid, box, label, conf in insect_tracks:
            det = Detection(label=f"{label} #{tid}", conf=conf, box=box,
                            model_label=settings.insect_label)
            self._draw(annotated, det, tid)    # colour by track id
            detections.append(det)

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

    def landing_snapshot(self) -> dict:
        """Recent live landings (rolling CSV/JSON) for the UI and any consumer."""
        if self._landing_logger is None:
            return {"enabled": False, "total_landings": 0, "real_landings": 0, "recent": []}
        snap = self._landing_logger.snapshot()
        snap["enabled"] = True
        return snap

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
