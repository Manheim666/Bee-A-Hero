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

_DAILY_FIELDS = ["date", "flower_id", "n_landings", "n_real_landings",
                 *[f"n_{t}" for t in vd.INSECT_TYPES], "last_update"]


class LiveSink:
    """Incremental CSV writer for live camera runs.

    Two outputs, both under ``<out>/csv/``:
      * ``live_landings.csv``       — one row appended per completed landing (full schema).
      * ``daily_flower_counts.csv`` — one row per (date, flower), counts updated as events land.

    The daily table is kept in memory keyed by ``(date, flower_id)`` and rewritten after each
    event (it is tiny); an existing file is loaded on start so counts accumulate across runs.
    """

    def __init__(self, out_dir: Path):
        self.csv_dir = Path(out_dir) / "csv"
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        self.live_path = self.csv_dir / "live_landings.csv"
        self.daily_path = self.csv_dir / "daily_flower_counts.csv"
        self._live_fields: list[str] | None = None
        self._daily: dict[tuple[str, str], dict] = {}
        self._load_daily()

    def _load_daily(self) -> None:
        if not self.daily_path.exists():
            return
        for r in csv.DictReader(open(self.daily_path)):
            key = (r["date"], r["flower_id"])
            for k in _DAILY_FIELDS:
                if k.startswith("n_"):
                    r[k] = int(r.get(k) or 0)
            self._daily[key] = r

    def append(self, row: dict) -> None:
        """Sink one landing: append the raw row, then bump the day's per-flower counts."""
        # 1) append the full landing row (header written once)
        if self._live_fields is None:
            self._live_fields = list(row.keys())
            with open(self.live_path, "w", newline="") as fh:
                csv.DictWriter(fh, fieldnames=self._live_fields).writeheader()
        with open(self.live_path, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=self._live_fields).writerow(row)

        # 2) update daily per-flower counts
        day = (row.get("timestamp") or datetime.now().isoformat())[:10] or date.today().isoformat()
        key = (day, str(row["flower_id"]))
        rec = self._daily.get(key)
        if rec is None:
            rec = {"date": day, "flower_id": row["flower_id"],
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
            for rec in sorted(self._daily.values(), key=lambda r: (r["date"], str(r["flower_id"]))):
                w.writerow(rec)


def run_auto(out_dir: Path, flower_w=DEF_FLOWER, insect_w=DEF_INSECT,
             honeybee_w=DEF_HONEYBEE, save_video=False, probe_cameras=True) -> dict:
    """Resolve the input source and run the matching mode."""
    src = resolve_source(probe_cameras=probe_cameras)
    print(f"[source] mode={src.mode} :: {src.reason}", flush=True)

    hb = str(honeybee_w) if Path(honeybee_w).exists() else ""

    if src.mode == "camera":
        sink = LiveSink(out_dir)
        runs = []
        for cam in src.items:
            print(f"[camera] streaming {cam!r} — live landings -> {sink.live_path.name} "
                  f"(Ctrl-C to stop)", flush=True)
            try:
                r = vd.count_visits_det(cam, str(flower_w), str(insect_w), Path(out_dir),
                                        save_video=save_video, honeybee_weights=hb,
                                        on_landing=sink.append, live=True)
                runs.append(r)
            except KeyboardInterrupt:
                print(f"[camera] stopped {cam!r}", flush=True)
                break
        return {"mode": "camera", "sources": [str(c) for c in src.items],
                "live_csv": str(sink.live_path), "daily_csv": str(sink.daily_path),
                "runs": runs}

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
