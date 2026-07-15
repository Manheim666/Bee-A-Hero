# Bee-A-Hero — Demo Web App

A full-stack demo that turns the **Bee-A-Hero** computer-vision project into a
product: a logged-in user uploads a video of pomegranate flowers, the system
"detects" flowers + insects and counts pollination visits per flower, and the
user browses **results, statistics, and an AI assistant**.

The detector is a **mock** behind a clean interface, so the whole app runs
locally with **no GPU and no model weights**. The real YOLO + BoT-SORT pipeline
drops into `backend/app/services/detector.py` later without touching any caller.

## Stack

- **Frontend** — React + Vite, plain CSS (honey/honeycomb theme), `recharts`,
  `react-router-dom`, `axios`.
- **Backend** — FastAPI + Uvicorn, SQLAlchemy, SQLite, Pydantic v2.
- **Auth** — JWT (bcrypt-hashed passwords), token in `localStorage`, sent as
  `Authorization: Bearer`.
- **Assistant** — pick a provider per chat in the Assistant tab: **Gemini**
  (`GEMINI_API_KEY`) or **Hugging Face** (`HF_API_TOKEN`); a mock echo provider
  answers when no key is set, so the chat always works. Answers are grounded in
  your real CV + ML results.
- **Detection job** — FastAPI `BackgroundTasks` (no Celery/Redis).

## Run it

### Easiest — one script, no Docker

Prereqs: Python 3.11+ and Node 20+ installed.

**macOS / Linux**

```bash
cd bee-a-hero-app
./start.sh
```

**Windows**

```bat
cd bee-a-hero-app
start.bat
```

Either script creates the Python venv, installs backend + frontend deps,
seeds the demo user, launches both servers, and opens the browser. Re-run
whenever — it skips any step already done. Press Ctrl+C to stop (macOS/Linux)
or close the two minimized "Bee-A-Hero" windows (Windows).

- App: http://localhost:5173
- API docs: http://localhost:8000/docs
- Login: `demo@bee.dev` / `beehero123`

### Option A — one command (Docker)

```bash
cd bee-a-hero-app
docker compose up --build
```

- API: http://localhost:8000  (docs at `/docs`)
- App: http://localhost:5173

The backend seeds a demo user and one already-processed sample video on start.

### Option B — two terminals

**Backend**

```bash
cd backend
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # optional; add GEMINI_API_KEY / HF_API_TOKEN for real AI
python -m seed                  # demo user + sample video
uvicorn app.main:app --reload
```

**Frontend**

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173.

## Seeded demo login

```
email:    demo@bee.dev
password: beehero123
```

(The login form is pre-filled with these.)

## Enabling the real AI assistant

Add a key to `backend/.env` (git-ignored, read by absolute path, never printed):

```
GEMINI_API_KEY=...      # https://aistudio.google.com/apikey
HF_API_TOKEN=...        # https://huggingface.co/settings/tokens
```

Then choose **Gemini** or **Hugging Face** from the dropdown in the Assistant
tab (unset providers show "(add key)"). Without any key, a built-in mock provider
references your stats so the feature demos with no key. Models are set in one
constant each (`GEMINI_MODEL`, `hf_model`) in `backend/app/services/llm.py` /
`config.py`.

## Demo script

1. **Register** a new account (or use the seeded demo login).
2. Go to **Upload** and drop any `.mp4` — watch it move `queued → processing →
   done` as the poller updates the card.
3. Open **Stats** and change the **video / date / pollinator** filters — the
   stacked bar chart, line chart, and stat tiles update live.
4. Open the **Assistant** and ask *"How many pollinator visits did I get?"*

## Layout

```
bee-a-hero-app/
  docker-compose.yml
  backend/
    app/
      main.py            FastAPI app, CORS, routers
      db.py  models.py  schemas.py  auth.py  config.py
      routers/  auth.py  videos.py  stats.py  chat.py
      services/
        detector.py      real tracked CV pipeline (count_visits_det) + mock fallback
        llm.py           Gemini / Hugging Face / mock chat providers
        stats.py         aggregation / filters
    seed.py              demo user + one processed sample video
    uploads/             saved videos (gitignored)
  frontend/
    src/
      App.jsx  api.js  theme.css
      auth/AuthContext.jsx
      pages/    Login  Register  Dashboard  Upload  Stats  Assistant
      components/  NavBar StatTile VideoCard Hexagon
                   VisitBarChart FilterBar ChatSidebar ChatWindow
```

## Troubleshooting

**"Annotation failed: No YOLO models available for annotation"** (or the live camera stops
detecting). The backend/live process couldn't load the detectors — almost always because the
server process was left running while the GPU was busy (e.g. an overnight training run), which
poisons that process's CUDA/torch state. The weights and code are fine.

Fix: **restart the services** — re-run `run-website.sh` (or restart the backend :8000 and live
:8001 uvicorn processes). A fresh process reloads the weights cleanly. The exact load error is
now logged by the annotator, so check the backend log if it persists. Don't run model training
on the same GPU while the app is serving.
