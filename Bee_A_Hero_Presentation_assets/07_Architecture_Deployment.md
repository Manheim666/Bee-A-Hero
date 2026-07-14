# 07 · Architecture & Deployment (Systems — Speaker 4)

## End-to-end data flow
```
        Camera / uploaded MP4
                 │
        ┌────────▼─────────┐
        │ Stage 1+2: YOLO26 │  flower detector + insect multi-class detector
        └────────┬─────────┘
                 ▼
        ┌──────────────────┐
        │ Stage 3: BoT-SORT │  per-track ids  →  stitch (concurrent + gap re-link)
        └────────┬─────────┘
                 ▼
        ┌──────────────────┐
        │ Landing state m/c │  ROI-in / stationary · 2s dwell gate · fly-off = new visit
        └────────┬─────────┘
                 ▼
        ┌──────────────────┐
        │ CSV / DB (SQLite) │  landings.csv · flower_summary.csv · Visit rows
        └────────┬─────────┘
                 ▼
        ┌──────────────────┐        ┌──────────────────────┐
        │ Stage 6: ML lift  │  ───▶  │ Assistant (Gemini/HF) │ grounded in CV+ML
        └────────┬─────────┘        └──────────────────────┘
                 ▼
        Dashboard · annotated video · pollination-lift report
```

## Services & ports
| Service | Path | Port | Tech |
|---|---|---|---|
| Backend API | `bee-a-hero-app/backend` | **8000** | FastAPI + SQLite + JWT |
| Frontend | `bee-a-hero-app/frontend` | **5173** | React + Vite |
| Live camera | `droidcam-live` | **8001** | FastAPI MJPEG, capture+infer threads |

## Deployment — Docker
Per-service Dockerfiles + one `docker-compose.yml` unifier:
- `docker/backend.Dockerfile`, `frontend.Dockerfile` (nginx SPA fallback), `droidcam.Dockerfile` (bakes the COCO person-veto model).
- **Weights are mounted read-only**, not baked → small images, swappable models.
- CPU torch wheels; consistent ports (droidcam 8001 everywhere).
- YOLO26 exports to ONNX/TensorRT/TFLite for later edge (Jetson) deployment.

## Repo / branch model
`main` ← merges of feature branches: `data`, `cv`, `ml`, `llm`, `web`. Stages 1–4 → cv, Stage 5 → data, Stage 6 → ml, dashboard/report → web. This session's work merged `web` → `main` cleanly (no-ff commits, conflict-free).

## Environment & runtime notes
- One **Python 3.11 venv** runs everything (`~/venv/py311`).
- **Secrets:** `GEMINI_API_KEY`, `HF_API_TOKEN` live only in the git-ignored `backend/.env`, read by absolute path — never printed, never committed, verified absent from git history.
- **CPU-only friendly:** no GPU required; the pipeline resamples to 20 fps and uses a second lightweight render pass. A 66 s clip ≈ ~2 min on CPU.

## Reliability engineering
- Live viewer: capture reopens the stream on any read failure; inference errors stream the raw frame so the viewer never freezes; latest-frame drop-old semantics.
- Backend: idempotent detection job; graceful mock fallback; `no-store` on media responses to survive SQLite id reuse.

## Anticipated questions
- *"How does it scale?"* → batch MP4s now; YOLO26 is edge-optimized for live cameras later; SQLite → Postgres/Timescale if streaming.
- *"Bug-free / production-ready?"* → verified end-to-end (login→upload→process→annotated stream→assistant), idempotent jobs, graceful degradation, secret hygiene audited.
- *"Why SQLite?"* → right-sized for a capstone; schema (visit_log / aggregates) ports directly to a server DB.
