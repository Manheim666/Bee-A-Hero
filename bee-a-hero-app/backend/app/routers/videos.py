import json
import mimetypes
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
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..config import settings
from ..db import SessionLocal, get_db
from ..models import DetectionResult, User, Video, VideoStatus, Visit
from ..schemas import VideoOut, VideoStatusOut
from ..services.annotate import annotate_video
from ..services.detector import run_detection

router = APIRouter(prefix="/api/videos", tags=["videos"])


# In-memory jobs: video_id -> {"status": "running"|"done"|"failed", "error": str}
_annotate_jobs: dict[int, dict] = {}


def _annotated_path(stored_path: str) -> Path:
    p = Path(stored_path)
    return p.with_name(f"annotated_{p.stem}.mp4")


def _run_annotation_job(video_id: int, src: str) -> None:
    dst = _annotated_path(src)
    _annotate_jobs[video_id] = {"status": "running", "error": ""}
    try:
        annotate_video(src, dst)
        _annotate_jobs[video_id] = {"status": "done", "error": ""}
    except Exception as exc:  # noqa: BLE001
        _annotate_jobs[video_id] = {"status": "failed", "error": str(exc)}


def _process_video(video_id: int) -> None:
    """Background job: flip to processing, run (mock) detection, persist rows."""
    db: Session = SessionLocal()
    try:
        video = db.get(Video, video_id)
        if video is None:
            return
        video.status = VideoStatus.processing
        db.commit()

        # run_detection also writes the annotated preview (annotated_<stem>.mp4) so it is ready
        # the moment status flips to done — the player streams it without a second annotation pass.
        summary = run_detection(video.stored_path)

        # Idempotent: clear any prior result/visits for this video so a re-run (e.g. an
        # interrupted job, a reload) replaces them instead of hitting the UNIQUE(video_id).
        db.query(Visit).filter(Visit.video_id == video.id).delete()
        db.query(DetectionResult).filter(DetectionResult.video_id == video.id).delete()
        db.flush()

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


@router.get("/{video_id}/annotated_status")
def annotated_status(
    video_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    video = _owned_video(db, video_id, user)
    dst = _annotated_path(video.stored_path)
    if dst.exists() and dst.stat().st_size > 0:
        return {"status": "done", "error": ""}
    job = _annotate_jobs.get(video_id)
    if job is None:
        return {"status": "idle", "error": ""}
    return job


@router.post("/{video_id}/annotate", status_code=status.HTTP_202_ACCEPTED)
def annotate_video_endpoint(
    video_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    video = _owned_video(db, video_id, user)
    dst = _annotated_path(video.stored_path)
    if dst.exists() and dst.stat().st_size > 0:
        return {"status": "done", "error": ""}
    current = _annotate_jobs.get(video_id)
    if current and current.get("status") == "running":
        return {"status": "running", "error": ""}
    _annotate_jobs[video_id] = {"status": "running", "error": ""}
    background_tasks.add_task(_run_annotation_job, video_id, video.stored_path)
    return {"status": "running", "error": ""}


@router.get("/{video_id}/annotated_stream")
def stream_annotated_video(
    video_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    video = _owned_video(db, video_id, user)
    dst = _annotated_path(video.stored_path)
    if not dst.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Annotated video not ready yet",
        )
    return FileResponse(dst, media_type="video/mp4", filename=f"annotated_{video.original_name}",
                        headers={"Cache-Control": "no-store"})


@router.get("/{video_id}/poster")
def video_poster(
    video_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """A real still frame from the video (the annotated one if ready, else raw) to use as
    the card cover instead of a generic logo. Cached to disk; regenerated once the annotated
    video appears so the cover upgrades from a raw frame to a boxed frame automatically."""
    video = _owned_video(db, video_id, user)
    stored = Path(video.stored_path)
    if not stored.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video file missing")
    poster = stored.with_name(f"poster_{stored.stem}.jpg")
    annotated = _annotated_path(video.stored_path)
    has_annot = annotated.exists() and annotated.stat().st_size > 0
    fresh = (
        poster.exists()
        and (not has_annot or poster.stat().st_mtime >= annotated.stat().st_mtime)
    )
    if not fresh:
        import cv2  # heavy; import lazily so app startup stays light

        src = annotated if has_annot else stored
        cap = cv2.VideoCapture(str(src))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps))  # ~1s in -> skips black lead-in
        ok, frame = cap.read()
        if not ok:  # short clip: fall back to the very first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        cap.release()
        if not ok:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No frame to read")
        cv2.imwrite(str(poster), frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    # never cache: SQLite reuses deleted video ids, so /videos/{id}/poster must not serve a
    # previous video's cover from the browser cache.
    return FileResponse(poster, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.get("/{video_id}/stream")
def stream_video(
    video_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    video = _owned_video(db, video_id, user)
    path = Path(video.stored_path)
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video file missing on disk",
        )
    media_type, _ = mimetypes.guess_type(str(path))
    return FileResponse(
        path,
        media_type=media_type or "video/mp4",
        filename=video.original_name,
        headers={"Cache-Control": "no-store"},
    )


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
