# Bee-A-Hero backend (FastAPI) — runs the REAL tracked CV pipeline (count_visits_det).
# Build context = repo ROOT (needs both src/ and bee-a-hero-app/backend/).
#   docker build -f docker/backend.Dockerfile -t bee-backend .
# Trained weights + grounding CSVs are mounted at runtime (see docker-compose.yml), not copied.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# cv2/ffmpeg runtime libs for video decode + drawing
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# deps first for layer caching (base app deps + heavy CV deps, CPU torch wheels)
COPY bee-a-hero-app/backend/requirements.txt bee-a-hero-app/backend/requirements-cv.txt /tmp/
RUN pip install -r /tmp/requirements.txt \
    && pip install --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/requirements-cv.txt

# source: the shared pipeline (src/) + the backend app
COPY src/ /app/src/
COPY bee-a-hero-app/backend/ /app/bee-a-hero-app/backend/

WORKDIR /app/bee-a-hero-app/backend
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
