import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from .pipeline import get_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_pipeline()  # spin up capture + inference threads
    yield


app = FastAPI(title="DroidCam Live CV", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "droidcam_url": settings.droidcam_url,
        "models": settings.model_path_list(),
        "labels": settings.model_label_list(),
        "conf": settings.conf_threshold,
        "imgsz": settings.img_size,
    }


@app.get("/api/stats")
def stats():
    return JSONResponse(get_pipeline().state.snapshot())


def _mjpeg_generator():
    pl = get_pipeline()
    last_id = 0
    boundary = b"--frame"
    while True:
        payload, last_id = pl.latest_jpeg(last_seen_id=last_id, timeout=2.0)
        if not payload:
            continue
        yield (
            boundary + b"\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(payload)).encode() + b"\r\n\r\n"
            + payload + b"\r\n"
        )


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
