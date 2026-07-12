"""Detection service.

Exposes ONE function the rest of the app calls: `run_detection(video_path)`.
It returns a RESULTS_SUMMARY-style dict so the real Bee-A-Hero pipeline can
drop in later without changing any caller.

The shipped implementation is a MOCK: deterministic-per-file, plausible
numbers, no GPU or model weights required.
"""

import hashlib
import random

# Headline metrics anchored near the real Bee-A-Hero project.
FLOWER_MAP = 0.918
INSECT_MAP = 0.900
CLASSIFIER_ACC = 0.978

_POLLINATORS = ["honey_bee", "bumblebee", "hoverfly", "solitary_bee"]
_NON_POLLINATORS = ["housefly", "ant", "wasp", "beetle"]


def _seed_from_path(video_path: str) -> int:
    """Stable seed derived from the filename, so re-uploading the same file
    yields identical results but different files differ."""
    digest = hashlib.sha256(video_path.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def run_detection(video_path: str) -> dict:
    """Return a RESULTS_SUMMARY-style dict: per-flower visits, pollinator /
    non-pollinator counts, plausible dwell times, and headline metrics."""
    rng = random.Random(_seed_from_path(video_path))

    n_flowers = rng.randint(6, 12)
    visits: list[dict] = []

    for flower_id in range(1, n_flowers + 1):
        n_visits = rng.randint(0, 5)
        for _ in range(n_visits):
            is_pollinator = rng.random() < 0.6
            insect_class = rng.choice(
                _POLLINATORS if is_pollinator else _NON_POLLINATORS
            )
            visits.append(
                {
                    "flower_id": flower_id,
                    "insect_class": insect_class,
                    "is_pollinator": is_pollinator,
                    "dwell_sec": round(rng.uniform(0.4, 6.0), 2),
                    "first_seen_sec": round(rng.uniform(0.0, 60.0), 2),
                }
            )

    pollinator_visits = sum(1 for v in visits if v["is_pollinator"])
    non_pollinator_visits = len(visits) - pollinator_visits

    return {
        "flower_mAP": FLOWER_MAP,
        "insect_mAP": INSECT_MAP,
        "classifier_acc": CLASSIFIER_ACC,
        "flower_map": n_flowers,
        "insect_tracks": len(visits),
        "pollinator_visits": pollinator_visits,
        "non_pollinator_visits": non_pollinator_visits,
        "visits": visits,
    }

    # === REAL MODEL GOES HERE ===
    # Replace the mock above with the real Bee-A-Hero pipeline. Load the three
    # trained checkpoints once (module-level, not per call), then map their
    # output into the exact dict shape returned above:
    #
    #     from ultralytics import YOLO           # pip install ultralytics
    #     import torch, cv2                       # torch + opencv-python
    #     flower = YOLO("data/interim/weights/flower_yolo26.pt")
    #     insect = YOLO("data/interim/weights/insect_yolo26.pt")
    #     classifier = torch.load("data/interim/weights/insect_classifier.pt")
    #
    #     # 1. flower.track(video_path) -> flower ROIs + stable IDs
    #     # 2. insect.track(video_path, tracker="botsort.yaml") -> insect tracks
    #     # 3. classify each track -> pollinator vs non_pollinator
    #     # 4. count a "visit" each time a track enters a flower ROI
    #     # 5. build the same dict: flower_mAP/insect_mAP/classifier_acc,
    #     #    flower_map, insect_tracks, pollinator_visits,
    #     #    non_pollinator_visits, and a `visits` list of per-flower rows.
    #
    # Keep these heavy deps OUT of requirements.txt so the demo stays light.
