import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Body, FastAPI
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
        "source": get_pipeline().get_source(),
        "models": settings.model_path_list(),
        "labels": settings.model_label_list(),
        "conf": settings.conf_threshold,
        "imgsz": settings.img_size,
    }


@app.get("/api/source")
def get_source():
    return {"source": get_pipeline().get_source()}


@app.post("/api/source")
def set_source(source: str = Body(..., embed=True)):
    """Switch the live camera at runtime: a phone DroidCam URL
    (e.g. http://192.168.1.5:4747/video) or a bare webcam index ("0")."""
    src = (source or "").strip()
    if not src:
        return JSONResponse({"error": "empty source"}, status_code=400)
    get_pipeline().set_source(src)
    return {"source": src}


@app.get("/api/stats")
def stats():
    return JSONResponse(get_pipeline().state.snapshot())


@app.get("/api/landings")
def landings():
    """Rolling live landings (flower id, enter/exit, dwell) written as insects land + leave."""
    return JSONResponse(get_pipeline().landing_snapshot())


def _mjpeg_generator():
    pl = get_pipeline()
    last_ts = 0.0
    boundary = b"--frame"
    while True:
        # Delayed playback: every annotated frame, in order, released ~display_delay_s late so
        # the feed is smooth at full fps while the pipeline processes the newest frames.
        payload, last_ts = pl.next_delayed_jpeg(last_ts=last_ts, timeout=2.0)
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
