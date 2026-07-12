"""Auto-source pipeline runner: **live cameras if present, else the test videos**.

Ties together the source rule (:mod:`src.cv_engine.source`) and the detector/tracker
(:func:`src.cv_engine.video_detect.count_visits_det`):

  * **Camera mode** — every listed, reachable camera is streamed live; each time an insect
    lands on and leaves a flower, a row is appended to ``test_video_result/csv/live_landings.csv``
    and the per-flower **daily counts** in ``daily_flower_counts.csv`` are updated on the spot.
  * **Video mode** — the test-videos folder is processed in batch and merged into ``ALL_*.csv``
    (the same output the ML/LLM stages already consume).

CLI:
    python -m src.cv_engine.run_pipeline                 # auto: camera if active, else videos
    python -m src.cv_engine.run_pipeline --save-video    # also write annotated videos
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, date
from pathlib import Path

from src import config as C
from src.cv_engine.source import resolve_source
from src.cv_engine import video_detect as vd

# Shipped best-checkpoint weights (overridable on the CLI).
_RUNS = C.INTERIM_DIR / "cv_runs"
DEF_FLOWER = _RUNS / "flower_det2_v2_yolo26m" / "weights" / "best.pt"
DEF_INSECT = _RUNS / "insect_multidet_v2_yolo26m" / "weights" / "best.pt"
DEF_HONEYBEE = _RUNS / "honeybee_clf" / "best.pt"

# Weather covariates travel with each camera landing (real capture wired later; the columns
# exist now so the ML stage can consume them). Populated from data/camera/weather.json if present.
_WEATHER_FIELDS = ["temp_c", "wind_ms", "humidity_pct"]
_DAILY_FIELDS = ["date", "camera_side", "flower_id", "n_landings", "n_real_landings",
                 *[f"n_{t}" for t in vd.INSECT_TYPES], "last_update"]


def read_weather(camera_dir: Path = None) -> dict:
    """Current weather covariates for a camera run.

    Reads ``data/camera/weather.json`` (``{"temp_c":.., "wind_ms":.., "humidity_pct":..}``) when
    present; otherwise returns blanks. Structured scaffolding: the columns exist so ML can gate on
    weather, while real capture (a station / API) is wired in later without a schema change.
    """
    camera_dir = Path(camera_dir) if camera_dir is not None else C.CAMERA_DIR
    f = camera_dir / "weather.json"
    data = {}
    if f.exists():
        try:
            data = json.loads(f.read_text())
        except Exception:
            data = {}
    return {k: data.get(k, "") for k in _WEATHER_FIELDS}


class LiveSink:
    """Incremental CSV writer for live camera runs.

    Two outputs, both under ``<out>/csv/`` (the camera lane is ``data/camera/csv/`` — kept
    separate from the test-video results so live and checkup data never mix):
      * ``live_landings.csv``       — one row per completed landing, enriched with the camera
        ``camera_side`` (east/west/north/south) and the run's weather covariates.
      * ``daily_flower_counts.csv`` — one row per (date, side, flower), counts updated as events land.

    The daily table is keyed by ``(date, camera_side, flower_id)`` and rewritten after each event
    (it is tiny); an existing file is loaded on start so counts accumulate across runs.
    """

    def __init__(self, out_dir: Path, weather: dict | None = None):
        self.csv_dir = Path(out_dir) / "csv"
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        self.live_path = self.csv_dir / "live_landings.csv"
        self.daily_path = self.csv_dir / "daily_flower_counts.csv"
        self.weather = weather or {k: "" for k in _WEATHER_FIELDS}
        self._live_fields: list[str] | None = None
        self._daily: dict[tuple[str, str, str], dict] = {}
        self._load_daily()

    def _load_daily(self) -> None:
        if not self.daily_path.exists():
            return
        for r in csv.DictReader(open(self.daily_path)):
            key = (r["date"], r.get("camera_side", ""), r["flower_id"])
            for k in _DAILY_FIELDS:
                if k.startswith("n_"):
                    r[k] = int(r.get(k) or 0)
            self._daily[key] = r

    def append(self, row: dict, side: str = "cam") -> None:
        """Sink one landing (from camera ``side``): append the enriched row, bump daily counts."""
        # enrich a *copy* with the camera side + weather — never mutate the caller's row
        erow = {**row, "camera_side": side, **self.weather}

        # 1) append the full landing row (header written once)
        if self._live_fields is None:
            self._live_fields = list(erow.keys())
            with open(self.live_path, "w", newline="") as fh:
                csv.DictWriter(fh, fieldnames=self._live_fields).writeheader()
        with open(self.live_path, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=self._live_fields, extrasaction="ignore").writerow(erow)

        # 2) update daily per-(side, flower) counts
        day = (row.get("timestamp") or datetime.now().isoformat())[:10] or date.today().isoformat()
        key = (day, side, str(row["flower_id"]))
        rec = self._daily.get(key)
        if rec is None:
            rec = {"date": day, "camera_side": side, "flower_id": row["flower_id"],
                   "n_landings": 0, "n_real_landings": 0, "last_update": ""}
            for t in vd.INSECT_TYPES:
                rec[f"n_{t}"] = 0
            self._daily[key] = rec
        rec["n_landings"] += 1
        if row.get("is_real_landing"):
            rec["n_real_landings"] += 1
        typ = row.get("insect_type")
        if f"n_{typ}" in rec:
            rec[f"n_{typ}"] += 1
        rec["last_update"] = datetime.now().isoformat(timespec="seconds")
        self._flush_daily()

    def _flush_daily(self) -> None:
        with open(self.daily_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=_DAILY_FIELDS, restval=0)
            w.writeheader()
            for rec in sorted(self._daily.values(),
                              key=lambda r: (r["date"], r["camera_side"], str(r["flower_id"]))):
                w.writerow(rec)


def run_auto(out_dir: Path, flower_w=DEF_FLOWER, insect_w=DEF_INSECT,
             honeybee_w=DEF_HONEYBEE, save_video=False, probe_cameras=True) -> dict:
    """Resolve the input source and run the matching mode."""
    src = resolve_source(probe_cameras=probe_cameras)
    print(f"[source] mode={src.mode} :: {src.reason}", flush=True)

    hb = str(honeybee_w) if Path(honeybee_w).exists() else ""

    if src.mode == "camera":
        # Camera lane is ALWAYS data/camera/csv/ — kept separate from test-video / website-upload
        # results so live counts are never mixed with checkup or uploaded-video CSVs.
        cam_out = C.CAMERA_DIR
        weather = read_weather()
        sink = LiveSink(cam_out, weather)
        runs = []
        for side, cam in src.items:                       # src.items is [(side, src), ...]
            print(f"[camera:{side}] streaming {cam!r} — live landings -> {sink.live_path} "
                  f"(Ctrl-C to stop)", flush=True)
            try:
                r = vd.count_visits_det(cam, str(flower_w), str(insect_w), Path(cam_out),
                                        save_video=save_video, honeybee_weights=hb,
                                        on_landing=lambda row, s=side: sink.append(row, s),
                                        live=True)
                runs.append(r)
            except KeyboardInterrupt:
                print(f"[camera:{side}] stopped", flush=True)
                break
        return {"mode": "camera", "sources": [f"{s}:{c}" for s, c in src.items],
                "csv_dir": str(sink.csv_dir), "live_csv": str(sink.live_path),
                "daily_csv": str(sink.daily_path), "weather": weather, "runs": runs}

    if src.mode == "video":
        runs = []
        for i, v in enumerate(src.items, 1):
            print(f"[video {i}/{len(src.items)}] {v.name}", flush=True)
            runs.append(vd.count_visits_det(str(v), str(flower_w), str(insect_w), Path(out_dir),
                                            save_video=save_video, honeybee_weights=hb))
        agg = vd.aggregate_csvs(Path(out_dir))
        return {"mode": "video", "processed": len(runs), "aggregate": agg, "runs": runs}

    raise SystemExit(f"no input source: {src.reason}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(C.REPO_ROOT / "test_video_result"))
    ap.add_argument("--flower-weights", default=str(DEF_FLOWER))
    ap.add_argument("--insect-weights", default=str(DEF_INSECT))
    ap.add_argument("--honeybee-weights", default=str(DEF_HONEYBEE))
    ap.add_argument("--save-video", action="store_true")
    ap.add_argument("--no-probe", action="store_true",
                    help="trust data/camera/sources.txt without opening the cameras")
    args = ap.parse_args()
    out = run_auto(Path(args.out), args.flower_weights, args.insect_weights,
                   args.honeybee_weights, args.save_video, probe_cameras=not args.no_probe)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
