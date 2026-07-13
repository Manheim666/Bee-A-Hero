# Docker — Bee-A-Hero

Three containerised services + one compose unifier.

| Service    | Dockerfile                | Port  | What it is |
|------------|---------------------------|-------|------------|
| `backend`  | `docker/backend.Dockerfile`  | 8000 | FastAPI: upload → **real tracked CV pipeline** (`count_visits_det`) + LLM assistant |
| `frontend` | `docker/frontend.Dockerfile` | 5173 | React/Vite UI, built static, served by nginx (SPA fallback) |
| `droidcam` | `docker/droidcam.Dockerfile` | 8001 | Live DroidCam MJPEG YOLO viewer with **human-veto** gating |

## One command (unifier)

```bash
docker compose up --build
```

- Frontend → http://localhost:5173
- Backend  → http://localhost:8000
- DroidCam → http://localhost:8001  (set `DROIDCAM_URL` to your phone's stream)

## Notes

- **Weights are mounted, not baked.** The gitignored `.pt` detectors under
  `data/interim/cv_runs/` are mounted read-only into `backend` and `droidcam`. Keep them
  on the host; images stay small.
- The backend gives the **same numbers as `test_video_result/`** because both run the one
  `count_visits_det` pipeline (same weights, same ≥2 s landing gate).
- **GPU** is optional — uncomment the `deploy:` block in `docker-compose.yml` and install
  `nvidia-container-toolkit`. Default images use CPU torch wheels.
- `bee-a-hero-app/backend/.env` supplies `GEMINI_API_KEY` (git-ignored — create your own).

## Build one service alone

```bash
docker build -f docker/backend.Dockerfile  -t bee-backend  .
docker build -f docker/frontend.Dockerfile -t bee-frontend --build-arg VITE_API_URL=http://localhost:8000 .
docker build -f docker/droidcam.Dockerfile -t bee-droidcam ./droidcam-live
```
