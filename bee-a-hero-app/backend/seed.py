"""Seed a demo user with one already-processed sample video.

Run once:  python -m seed   (from the backend/ directory)

Idempotent: re-running won't duplicate the demo user or its sample video.
"""

import json
from datetime import datetime, timezone

from app.auth import hash_password
from app.db import SessionLocal, init_db
from app.models import DetectionResult, User, Video, VideoStatus, Visit
from app.services.detector import run_detection

DEMO_EMAIL = "demo@bee.dev"
DEMO_PASSWORD = "beehero123"
DEMO_USERNAME = "Demo Beekeeper"
SAMPLE_NAME = "sample_pomegranate_clip.mp4"


def seed() -> None:
    init_db()
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == DEMO_EMAIL).first()
        if user is None:
            user = User(
                email=DEMO_EMAIL,
                username=DEMO_USERNAME,
                password_hash=hash_password(DEMO_PASSWORD),
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            print(f"Created demo user: {DEMO_EMAIL} / {DEMO_PASSWORD}")
        else:
            print(f"Demo user already exists: {DEMO_EMAIL}")

        existing = (
            db.query(Video)
            .filter(Video.user_id == user.id, Video.original_name == SAMPLE_NAME)
            .first()
        )
        if existing:
            print("Sample video already present — nothing to do.")
            return

        summary = run_detection(SAMPLE_NAME)
        video = Video(
            user_id=user.id,
            original_name=SAMPLE_NAME,
            stored_path=f"(seed)/{SAMPLE_NAME}",
            status=VideoStatus.done,
            duration_sec=62.0,
            processed_at=datetime.now(timezone.utc),
        )
        db.add(video)
        db.commit()
        db.refresh(video)

        db.add(
            DetectionResult(
                video_id=video.id,
                flower_map=summary["flower_map"],
                insect_tracks=summary["insect_tracks"],
                pollinator_visits=summary["pollinator_visits"],
                non_pollinator_visits=summary["non_pollinator_visits"],
                flower_map50=summary["flower_mAP"],
                insect_map50=summary["insect_mAP"],
                classifier_acc=summary["classifier_acc"],
                summary_json=json.dumps(summary),
            )
        )
        for v in summary["visits"]:
            db.add(
                Visit(
                    video_id=video.id,
                    flower_id=v["flower_id"],
                    insect_class=v["insect_class"],
                    is_pollinator=v["is_pollinator"],
                    dwell_sec=v["dwell_sec"],
                    first_seen_sec=v["first_seen_sec"],
                )
            )
        db.commit()
        print(f"Seeded processed sample video with {len(summary['visits'])} visits.")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
