"""Input-source selection for the CV pipeline — **camera first, test videos as fallback**.

One rule, used by both the CLI pipeline (``run_pipeline.py``) and the web viewer
(``src/webapp/app.py``) so they always agree:

  * If ``data/camera/sources.txt`` lists at least one **active** camera source, run live
    on those cameras (each frame's landings stream into the CSV as they happen).
  * Otherwise, fall back to the test-videos folder (``data/raw/Test_Video/``).

A *source* is one line of ``sources.txt`` (blank lines and ``#`` comments ignored):

    0                         # local webcam / capture device index
    rtsp://cam-1.local/stream # IP camera (RTSP)
    http://cam-2/video.mjpg   # MJPEG/HTTP stream
    /abs/path/to/feed.mp4     # a file, handy for testing the live path

"Active" means the source actually opens (``cv2.VideoCapture(src).isOpened()``); a listed
but unreachable camera is skipped, and if none open the pipeline uses the test videos.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src import config as C

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")


@dataclass
class Source:
    """Resolved input for one pipeline run."""
    mode: str                       # "camera" | "video" | "none"
    items: list = field(default_factory=list)   # camera sources (int|str) or video Paths
    reason: str = ""                # human-readable explanation (for logs / the web page)


def _parse_source_line(line: str):
    """Map one ``sources.txt`` line to an OpenCV source: int index or the raw string."""
    line = line.strip()
    return int(line) if line.isdigit() else line


def read_camera_sources(camera_dir: Path = None) -> list:
    """Return the raw source list from ``camera/sources.txt`` (no probing). Empty if absent."""
    camera_dir = Path(camera_dir) if camera_dir is not None else C.CAMERA_DIR
    f = camera_dir / "sources.txt"
    if not f.exists():
        return []
    out = []
    for raw in f.read_text().splitlines():
        s = raw.strip()
        if s and not s.startswith("#"):
            out.append(_parse_source_line(s))
    return out


def probe(src, timeout_ms: int = 3000) -> bool:
    """True if ``src`` opens as a live/readable stream. Best-effort, never raises."""
    try:
        import cv2
        cap = cv2.VideoCapture(src)
        try:
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms)  # ignored by some backends
        except Exception:
            pass
        ok = bool(cap.isOpened())
        cap.release()
        return ok
    except Exception:
        return False


def list_test_videos(video_dir: Path = None) -> list[Path]:
    """Sorted video clips in the fallback test-videos folder."""
    video_dir = Path(video_dir) if video_dir is not None else C.TEST_VIDEO_DIR
    if not video_dir.is_dir():
        return []
    return sorted(p for p in video_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS)


def resolve_source(camera_dir: Path = None, video_dir: Path = None,
                   probe_cameras: bool = True) -> Source:
    """Pick the input: active cameras if any, else the test videos.

    ``probe_cameras`` opens each listed camera to confirm it is reachable; set it False to
    trust the list without touching hardware (faster, e.g. for a status page that only reports
    intent). If cameras are listed but none open, we fall back to video and say so in ``reason``.
    """
    listed = read_camera_sources(camera_dir)
    if listed:
        active = [s for s in listed if probe(s)] if probe_cameras else list(listed)
        if active:
            return Source("camera", active,
                          f"{len(active)}/{len(listed)} camera source(s) active")
        # listed but unreachable -> fall through to video
        reason_cam = f"{len(listed)} camera source(s) listed but none reachable; "
    else:
        reason_cam = "no camera sources in data/camera/sources.txt; "

    vids = list_test_videos(video_dir)
    if vids:
        return Source("video", vids, reason_cam + f"using {len(vids)} test video(s)")
    return Source("none", [], reason_cam + "and no test videos found")
