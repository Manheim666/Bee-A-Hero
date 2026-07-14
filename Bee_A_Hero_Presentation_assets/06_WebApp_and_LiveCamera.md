# 06 · Web App & Live Camera (Full-stack — Speaker 4)

## Stack
- **Backend:** FastAPI (Python 3.11), SQLite via SQLAlchemy, JWT auth (bcrypt), FastAPI `BackgroundTasks` for the detection job (no Celery/Redis). Port **:8000**.
- **Frontend:** React + Vite, port **:5173**.
- **Live camera:** standalone FastAPI MJPEG service (`droidcam-live/`), port **:8001**.

## Upload → results flow (the demo path)
1. User uploads a clip (`POST /api/videos`, ≤ 200 MB, mp4/mov/avi). Stored under a random uuid.
2. A background job flips status → *processing* and calls `run_detection()` → **the real `count_visits_det` pipeline** (same as the offline CSVs).
3. The pipeline writes the **annotated H.264 video** during processing, so it's ready the instant status → *done* (opening plays it immediately — no second annotation pass).
4. Results persisted: a `DetectionResult` (flowers, visits, pollinator/non split, mAPs) + one `Visit` row per real landing. The job is **idempotent** — a re-run clears prior rows first (no `UNIQUE(video_id)` crash).
5. Frontend polls status, then shows the card with a **real still-frame cover** (poster endpoint) and, on open, the annotated video + per-flower stats.

## Key endpoints
`/api/auth/login|register` · `/api/videos` (upload/list) · `/api/videos/{id}/status` · `/api/videos/{id}/annotated_stream` (H.264) · `/api/videos/{id}/poster` (still cover, `no-store`) · `/api/stats/*` (overview, visits, timeseries) · `/api/conversations/*` (assistant + `providers`).

## Robustness details worth mentioning
- **One source of truth:** website numbers == offline CSVs (same pipeline) — no divergent counter.
- **Cache correctness:** SQLite reuses deleted video ids, so poster/stream responses send `Cache-Control: no-store` + a per-video cache-bust → a card never shows another clip's frame.
- **Graceful degrade:** no models/deps → a deterministic mock backend keeps the demo alive.
- **Browser-safe video:** OpenCV here has no H.264 encoder → we render mp4v then transcode via system ffmpeg (even-dimension scaling) so it plays in `<video>`.

## Live camera (DroidCam / webcam)
Standalone MJPEG viewer with a **capture thread** and an **inference thread** joined by a single-slot "latest frame, drop-old" mailbox — **slow inference never blocks capture**, the browser always sees the most recent annotation.
- **Runtime source switch:** paste a phone's DroidCam URL (`http://PHONE_IP:4747/video`) or use the local webcam index — no restart. (Phone & PC must share a Wi-Fi subnet.)
- Runs the **trained flower + insect models** (via env), BoT-SORT tracking, and the **person-veto** (COCO yolov8n) + all the same FP gates as offline.
- **Rolling landing log:** as insects land and leave, one row per landing is appended to `live_out/live_landings.csv` (+ `.json`) — flower id, enter/exit, dwell, type — the *same* landing data the offline pipeline produces. A track occluded behind a petal is re-linked so one bee is counted once.
- Endpoints: `/video_feed` (MJPEG), `/api/stats`, `/api/landings`, `GET/POST /api/source`, `/api/health`.

## One-command launch
`bash ~/Desktop/run-website.sh` starts all three services (no `--reload` — the pipeline writes into the backend tree, which would trigger reloads mid-job) and opens the site. Keys load from the git-ignored `backend/.env` without exposure.

## Anticipated questions
- *"Real-time?"* → live viewer is real-time MJPEG; uploads are batch (CPU). Latest-frame semantics keep the live feed smooth under slow inference.
- *"Website vs CSV mismatch?"* → impossible; identical pipeline.
- *"Why FastAPI BackgroundTasks not Celery?"* → capstone scope; batch jobs, no external broker needed.
- *"Multi-camera?"* → single camera per viewer today; multi-cam is a designed extension (homography handoff for overlapping views — see architecture).
