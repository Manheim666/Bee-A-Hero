from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root (…/Bee-A-Hero) from …/droidcam-live/app/config.py, so the trained detectors are the
# built-in default no matter how the viewer is launched.
_REPO = Path(__file__).resolve().parents[2]
_FLOWER = _REPO / "data/interim/cv_runs/flower_det2_v2_yolo26m/weights/best.pt"
_INSECT = _REPO / "data/interim/cv_runs/insect_multidet_v2_yolo26m/weights/best.pt"
_DEFAULT_MODELS = (
    f"{_FLOWER},{_INSECT}" if _FLOWER.exists() and _INSECT.exists() else "yolov8n.pt"
)


class Settings(BaseSettings):
    # Default to the LOCAL WEBCAM (index 0) so the live viewer works out of the box; switch to a
    # phone's DroidCam URL from the browser. Default to the TRAINED flower+insect detectors, not
    # generic COCO yolov8n, so the camera actually detects flowers and insects.
    droidcam_url: str = "0"
    model_paths: str = _DEFAULT_MODELS
    model_labels: str = "flower,insect" if _FLOWER.exists() else ""
    conf_threshold: float = 0.25       # match the upload pipeline (conf 0.20) so live detects too
    img_size: int = 768                # match the detectors' train size -> better small-bee recall
                                       #   than 640, still real-time on CPU for the live feed
    reconnect_delay: float = 3.0
    jpeg_quality: int = 80
    device: str = "cpu"

    # --- false-positive gating (humans OOD -> misread as flower/insect) ----------
    # YOLO is closed-set: a person has no class, so it snaps onto flower/insect. Veto
    # any detection that overlaps a COCO person box, and drop any box too big to be a
    # real flower/insect (a real one is small; a frame-filling box is a wall/person/FP).
    # OFF by default: the veto killed any detection whose CENTRE fell inside a COCO person box,
    # so a flower/bee HELD IN HAND (hand/arm in frame -> person box over the subject) vanished
    # every frame and never came back. The size-veto (max_box_frac) still cuts wall-sized blobs.
    person_veto: bool = False
    person_model: str = "yolov8n.pt"    # generic COCO detector, auto-downloaded
    person_conf: float = 0.35
    max_box_frac: float = 0.85          # reject only near-frame-filling blobs (wall/person). A
                                        #   flower or bee held to the camera is a BIG box -> must
                                        #   pass, else live detects nothing on close subjects.

    # --- flower <-> insect confusion gating --------------------------------------
    # Two independent detectors run per frame; the insect model can fire (e.g. "butterfly")
    # on a colourful flower. A REAL insect on a flower is a small box INSIDE it (low IoU);
    # a mislabelled flower is a box that basically IS the flower (IoU ~1, similar area). So:
    #  * hold insects to a higher confidence bar than flowers, and
    #  * veto any insect box that overlaps a flower too much / is nearly the flower's size.
    insect_conf: float = 0.20           # match the upload pipeline (low -> detect a settled bee
                                        #   immediately, and survive the OOD gap on close/live subjects)
    flower_conf: float = 0.25           # match the upload pipeline (~0.20) so a live flower fires
    insect_flower_iou: float = 0.80     # insect box matching a flower this closely = the whole
                                        #   flower mislabelled; kept high so bees ON a flower survive
    # flower geometry: a real flower is compact -> reject slivers. Cap raised near full-frame so a
    # flower held close to the camera passes (the live test case), only a scene-sized blob is cut.
    flower_min_frac: float = 0.002      # reject flower boxes smaller than this fraction of the frame
    flower_max_frac: float = 0.92       # reject flower boxes bigger than this (whole-screen = FP)
    flower_max_aspect: float = 3.0      # reject long slivers (aspect above this) -> not a flower
    box_nms_iou: float = 0.55           # merge overlapping same-kind boxes ("flower in a flower")
    box_nms_contain: float = 0.70       # or one box mostly inside another -> keep the stronger

    # --- live landing logging (rolling CSV/JSON for the ML phase) ------------------
    # When a flower model and an insect model are both loaded, track insects, associate
    # them to flowers, and append one row per completed landing (enter/exit/dwell) so the
    # live camera produces the same landing data the offline pipeline does.
    landing_log: bool = True
    insect_label: str = "insect"        # the MODEL_LABELS tag of the insect detector
    flower_label: str = "flower"        # the MODEL_LABELS tag of the flower detector
    min_land_s: float = 2.0             # dwell >= this = a real landing (a counted visit)
    land_grace_s: float = 0.5           # bridge brief flicker inside one landing
    stationary_tau: float = 0.5         # normalised speed below which an insect is "settled"
    live_out_dir: str = "live_out"      # where live_landings.csv / .json are written

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parent.parent / ".env"),
        extra="ignore",
    )

    def model_path_list(self) -> list[str]:
        return [p.strip() for p in self.model_paths.split(",") if p.strip()]

    def model_label_list(self) -> list[str]:
        raw = [p.strip() for p in self.model_labels.split(",") if p.strip()]
        paths = self.model_path_list()
        # Pad/truncate to match paths.
        if len(raw) < len(paths):
            raw += ["" for _ in range(len(paths) - len(raw))]
        return raw[: len(paths)]


settings = Settings()
