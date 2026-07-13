from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    droidcam_url: str = "http://192.168.1.100:4747/video"
    model_paths: str = "yolov8n.pt"
    model_labels: str = ""
    conf_threshold: float = 0.35
    img_size: int = 640
    reconnect_delay: float = 3.0
    jpeg_quality: int = 80
    device: str = "cpu"

    # --- false-positive gating (humans OOD -> misread as flower/insect) ----------
    # YOLO is closed-set: a person has no class, so it snaps onto flower/insect. Veto
    # any detection that overlaps a COCO person box, and drop any box too big to be a
    # real flower/insect (a real one is small; a frame-filling box is a wall/person/FP).
    person_veto: bool = True
    person_model: str = "yolov8n.pt"    # generic COCO detector, auto-downloaded
    person_conf: float = 0.35
    max_box_frac: float = 0.22          # reject boxes bigger than this fraction of the frame

    # --- flower <-> insect confusion gating --------------------------------------
    # Two independent detectors run per frame; the insect model can fire (e.g. "butterfly")
    # on a colourful flower. A REAL insect on a flower is a small box INSIDE it (low IoU);
    # a mislabelled flower is a box that basically IS the flower (IoU ~1, similar area). So:
    #  * hold insects to a higher confidence bar than flowers, and
    #  * veto any insect box that overlaps a flower too much / is nearly the flower's size.
    insect_conf: float = 0.30           # min confidence for an insect box (low -> detect a settled
                                        #   bee immediately instead of a couple seconds late)
    flower_conf: float = 0.45           # min confidence for a flower box (higher -> fewer scene FPs)
    insect_flower_iou: float = 0.80     # insect box matching a flower this closely = the whole
                                        #   flower mislabelled; kept high so bees ON a flower survive
    # flower geometry: a real flower is compact and only part of the frame — not the whole scene.
    flower_min_frac: float = 0.002      # reject flower boxes smaller than this fraction of the frame
    flower_max_frac: float = 0.55       # reject flower boxes bigger than this (whole-screen = FP)
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
