# DroidCam live YOLO viewer — standalone MJPEG service with human-veto gating.
# Build context = ./droidcam-live.
#   docker build -f docker/droidcam.Dockerfile -t bee-droidcam ./droidcam-live
# Real flower/insect weights are mounted at runtime; the COCO person-veto model is baked in.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# pre-download the COCO person-veto model so the container needs no network at runtime
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

COPY . .
EXPOSE 8001
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
