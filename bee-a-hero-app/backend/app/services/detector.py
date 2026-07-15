"""Detection service.

Exposes ONE function the rest of the app calls: `run_detection(video_path)`.
It returns a RESULTS_SUMMARY-style dict so callers never change.

Two backends, chosen automatically at import:
  * REAL (`src.cv_engine.video_detect.count_visits_det`) — when `ultralytics`/`torch`
    and the trained `.pt` detectors are present. This is the *same* pipeline that
    produces `test_video_result/` (YOLO + BoT-SORT tracking, confidence-weighted type
    vote, landing episodes with a >=2s dwell gate). Running it here means the website's
    numbers are IDENTICAL to the offline CSVs for the same video + weights — no second,
    divergent counter.
  * MOCK — otherwise; deterministic-per-file plausible numbers, so the demo runs with no
    models or heavy deps.

The old per-frame ONNX counter was removed: it counted one "visit" per sampled frame an
insect box sat on a flower (no tracking / no dwell), which over-counted ~15x and could
never match the offline pipeline. One pipeline, one source of truth.
"""
from __future__ import annotations

import hashlib
import random
import shutil
import sys
from pathlib import Path

# Headline metrics anchored to the real Bee-A-Hero project (best-checkpoint validation).
FLOWER_MAP = 0.808
INSECT_MAP = 0.669
CLASSIFIER_ACC = 0.978

# insect types the tracker emits that count as pollinators (rest -> non-pollinator)
_POLLINATOR_CLASSES = {"bee", "honeybee", "butterfly", "fly"}

# repo root = .../bee-a-hero-app/backend/app/services/detector.py -> parents[4]
_REPO_ROOT = Path(__file__).resolve().parents[4]
_CV_RUNS = _REPO_ROOT / "data" / "interim" / "cv_runs"

# Same trained weights the offline pipeline + annotator use (v2 checkpoints).
_FLOWER_W = _CV_RUNS / "flower_det2_v2_yolo26m" / "weights" / "best.pt"
_INSECT_W = _CV_RUNS / "insect_multidet_v2_yolo26m" / "weights" / "best.pt"
_HONEYBEE_W = _CV_RUNS / "honeybee_clf" / "best.pt"

# per-video CV outputs (CSVs + optional annotated mp4) land here, away from the
# canonical offline test_video_result/ so a web upload never clobbers it.
_CV_OUT = _REPO_ROOT / "bee-a-hero-app" / "backend" / "cv_out"


# --------------------------------------------------------------------------- #
# REAL backend — delegate to the one true pipeline (count_visits_det)
# --------------------------------------------------------------------------- #
def _real_available() -> bool:
    if not (_FLOWER_W.exists() and _INSECT_W.exists()):
        return False
    try:
        import ultralytics  # noqa: F401
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def _run_real(video_path: str) -> dict:
    """Run the offline pipeline on this upload and adapt its landings to RESULTS_SUMMARY.

    Only *real* landings (dwell >= MIN_LAND_S, 2s) become visits — same rule as the CSVs.
    """
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from src.cv_engine.video_detect import count_visits_det

    rows: list[dict] = []
    honeybee = str(_HONEYBEE_W) if _HONEYBEE_W.exists() else ""
    # save_video=True -> the tracked pipeline writes the annotated mp4 itself, so the
    # website preview shows the *exact* boxes/IDs/counts as test_video_result (one source
    # of truth). Move it next to the upload where the player streams it from.
    summary = count_visits_det(
        str(video_path), str(_FLOWER_W), str(_INSECT_W), _CV_OUT,
        honeybee_weights=honeybee, on_landing=rows.append, save_video=True,
        person_veto_iou=0.55,   # drop humans misread as flower/insect (IoU-based; held objects kept)
    )
    try:
        vp = Path(video_path)
        src_annot = _CV_OUT / "videos" / f"{vp.stem}_annotated.mp4"
        if src_annot.exists() and src_annot.stat().st_size > 0:
            dst_annot = vp.with_name(f"annotated_{vp.stem}.mp4")
            shutil.move(str(src_annot), str(dst_annot))
    except Exception as e:  # annotation is a preview nicety -> never fail the run over it
        print(f"[detector] could not place annotated preview: {type(e).__name__}: {e}")

    # keep only real landings, map string flower ids -> stable ints for the DB
    flower_ix: dict[str, int] = {}
    visits: list[dict] = []
    for r in rows:
        if not r.get("is_real_landing"):
            continue
        fid = flower_ix.setdefault(r["flower_id"], len(flower_ix) + 1)
        typ = str(r["insect_type"])
        visits.append({
            "flower_id": fid,
            "insect_class": typ,
            "is_pollinator": typ.lower() in _POLLINATOR_CLASSES,
            "dwell_sec": float(r["landing_s"]),
            "first_seen_sec": float(r["t_enter_s"]),
        })
    pol = sum(1 for v in visits if v["is_pollinator"])
    return {
        "flower_mAP": FLOWER_MAP, "insect_mAP": INSECT_MAP, "classifier_acc": CLASSIFIER_ACC,
        "flower_map": int(summary.get("flowers", len(flower_ix))),
        "insect_tracks": len(visits),
        "pollinator_visits": pol, "non_pollinator_visits": len(visits) - pol,
        "visits": visits, "backend": "cv-real",
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
    """Real tracked pipeline when models + deps are present, else the deterministic mock."""
    if _real_available():
        try:
            return _run_real(video_path)
        except Exception as e:  # any runtime issue -> never break the app; fall back to mock
            print(f"[detector] real pipeline failed ({type(e).__name__}: {e}); using mock.")
    return _run_mock(video_path)
