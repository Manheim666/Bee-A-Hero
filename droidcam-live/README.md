# Bee-A-Hero · DroidCam Live

Live phone-camera feed → YOLO flower + insect detection overlay → MJPEG in
your browser. Zero JS build step; one HTML file + one FastAPI service.

```
DroidCam (phone)  ──►  FastAPI capture thread  ──►  latest-frame slot
                                                        │
                                             inference thread (YOLO)
                                                        │
                                          annotated JPEG slot  ──►  /video_feed  (browser <img>)
                                                        │
                                                   /api/stats  (JSON polled 1s)
```

Slow inference **never** blocks capture — old frames are dropped so the browser
always sees the most recent annotation.

## Prereqs

1. Phone running **DroidCam** (Android / iOS) on the same Wi-Fi as your machine.
2. Python 3.11+.
3. Note the MJPEG URL shown in the DroidCam app — typically
   `http://<PHONE_IP>:4747/video`.

## Run it

```bash
cd droidcam-live
./start.sh          # macOS/Linux
# or
start.bat           # Windows
```

First run creates `.env` from the example and exits — edit it to set
`DROIDCAM_URL`, then re-run. Second run opens http://localhost:8001/.

### Using this repo's trained flower + insect detectors

```env
MODEL_PATHS=../data/interim/cv_runs/flower_det2_v2_yolo26m/weights/best.pt,../data/interim/cv_runs/insect_multidet_v2_yolo26m/weights/best.pt
MODEL_LABELS=flower,insect
```

Both models run per frame, boxes drawn in different colors, per-class counts
show in the side panel.

## Config (`.env`)

| Key              | Default                                | Purpose                                                        |
| ---------------- | -------------------------------------- | -------------------------------------------------------------- |
| `DROIDCAM_URL`   | `http://192.168.1.100:4747/video`      | Camera source. Phone MJPEG url, **or a bare index (`0`) for the local webcam** — so the live viewer works with no phone. |
| `MODEL_PATHS`    | `yolov8n.pt`                           | Comma-separated ultralytics weight paths.                      |
| `MODEL_LABELS`   | *(empty)*                              | Optional per-model tag drawn on boxes.                         |
| `CONF_THRESHOLD` | `0.35`                                 | Minimum detection confidence.                                  |
| `IMG_SIZE`       | `640`                                  | Inference image size.                                          |
| `RECONNECT_DELAY`| `3.0`                                  | Seconds between DroidCam reconnect attempts.                   |
| `JPEG_QUALITY`   | `80`                                   | JPEG quality for streamed frames.                              |
| `DEVICE`         | `cpu`                                  | `cpu`, `cuda`, or `mps`.                                       |

## Camera source (switch at runtime)

The **Camera source** panel in the viewer switches the live camera without a
restart: paste your phone's DroidCam URL (`http://PHONE_IP:4747/video`) and
Connect, or click **Use webcam** for the local camera. Phone and PC must be on
the **same Wi‑Fi** (the DroidCam WiFi IP must be on the PC's subnet).

## Live landing log

When a **flower** model and an **insect** model are both loaded (set `MODEL_PATHS`
/ `MODEL_LABELS` to the trained detectors), the viewer tracks insects, associates
them to flowers, and appends one row per landing to `live_out/live_landings.csv`
(+ `.json`) as insects land and leave — flower id, enter/exit, dwell, type — the
same landing data the offline pipeline produces. A track occluded behind a petal
is re‑linked so one bee is counted once.

## Endpoints

- `GET /` — the viewer HTML.
- `GET /video_feed` — `multipart/x-mixed-replace` MJPEG of annotated frames.
- `GET /api/stats` — JSON: `connected`, `reconnecting`, `inference_fps`,
  `capture_fps`, `detection_count`, `per_class_counts`, `frame_shape`,
  `uptime_sec`, `last_error`.
- `GET /api/landings` — recent live landings (`total_landings`, `real_landings`, `recent`).
- `GET`/`POST /api/source` — read / switch the camera source.
- `GET /api/health` — current config snapshot.

## Reliability

- Capture thread reopens the phone stream on any read failure, waiting
  `RECONNECT_DELAY` seconds between attempts. Status shows **Reconnecting**.
- Inference errors are logged and the raw frame is streamed unannotated so
  the viewer never freezes.
- `<img>` reconnects automatically if the browser drops the multipart stream.
