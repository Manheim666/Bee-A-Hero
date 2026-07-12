import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    UploadFile,
    status,
)
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..config import settings
from ..db import SessionLocal, get_db
from ..models import DetectionResult, User, Video, VideoStatus, Visit
from ..schemas import VideoOut, VideoStatusOut
from ..services.detector import run_detection

router = APIRouter(prefix="/api/videos", tags=["videos"])


def _process_video(video_id: int) -> None:
    """Background job: flip to processing, run (mock) detection, persist rows."""
    db: Session = SessionLocal()
    try:
        video = db.get(Video, video_id)
        if video is None:
            return
        video.status = VideoStatus.processing
        db.commit()

        # Visible spinner -> done transition.
        time.sleep(4)

        summary = run_detection(video.stored_path)

        result = DetectionResult(
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
        db.add(result)

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

        video.status = VideoStatus.done
        video.processed_at = datetime.now(timezone.utc)
        db.commit()
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        video = db.get(Video, video_id)
        if video is not None:
            video.status = VideoStatus.failed
            video.error = str(exc)
            db.commit()
    finally:
        db.close()


def _owned_video(db: Session, video_id: int, user: User) -> Video:
    video = db.get(Video, video_id)
    if video is None or video.user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Video not found"
        )
    return video


@router.post("", response_model=VideoOut, status_code=status.HTTP_201_CREATED)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in settings.allowed_video_ext:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported type '{ext}'. Allowed: "
            f"{', '.join(settings.allowed_video_ext)}",
        )

    data = await file.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {settings.max_upload_mb} MB limit",
        )

    stored_name = f"{uuid.uuid4().hex}{ext}"
    stored_path = settings.uploads_dir / stored_name
    stored_path.write_bytes(data)

    video = Video(
        user_id=user.id,
        original_name=file.filename or stored_name,
        stored_path=str(stored_path),
        status=VideoStatus.queued,
    )
    db.add(video)
    db.commit()
    db.refresh(video)

    background_tasks.add_task(_process_video, video.id)
    return video


@router.get("", response_model=list[VideoOut])
def list_videos(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    return db.scalars(
        select(Video)
        .where(Video.user_id == user.id)
        .order_by(Video.uploaded_at.desc())
    ).all()


@router.get("/{video_id}", response_model=VideoOut)
def get_video(
    video_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    return _owned_video(db, video_id, user)


@router.get("/{video_id}/status", response_model=VideoStatusOut)
def get_video_status(
    video_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    return _owned_video(db, video_id, user)


@router.delete("/{video_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_video(
    video_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    video = _owned_video(db, video_id, user)
    stored = Path(video.stored_path)
    if stored.exists():
        stored.unlink()
    db.delete(video)
    db.commit()
