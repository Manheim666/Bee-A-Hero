from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .db import SessionLocal, init_db
from .routers import auth, chat, stats, videos


def _recover_stuck_jobs() -> None:
    """Fail any video left mid-flight by a crash/restart.

    Detection runs as an in-process background task; if the server stops while a job is
    running, that video is frozen in ``queued``/``processing`` and the UI spins forever.
    On startup nothing is resuming it, so mark it ``failed`` with a clear message -> the
    user sees an error and can re-upload instead of an endless spinner."""
    from sqlalchemy import select

    from .models import Video, VideoStatus

    db = SessionLocal()
    try:
        stuck = db.scalars(
            select(Video).where(Video.status.in_([VideoStatus.queued, VideoStatus.processing]))
        ).all()
        for v in stuck:
            v.status = VideoStatus.failed
            v.error = "Processing was interrupted by a server restart. Please re-upload."
        if stuck:
            db.commit()
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    _recover_stuck_jobs()
    yield


app = FastAPI(title="Bee-A-Hero API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin, "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "service": "bee-a-hero"}


app.include_router(auth.router)
app.include_router(videos.router)
app.include_router(stats.router)
app.include_router(chat.router)
