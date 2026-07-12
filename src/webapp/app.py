"""Minimal Bee-A-Hero web viewer — the **same source rule** as the CLI pipeline.

A single-page Flask app that:
  * shows which input the pipeline would use right now (live cameras if
    ``data/camera/sources.txt`` lists active ones, else the test videos) — via the shared
    :func:`src.cv_engine.source.resolve_source`, so the page and the pipeline never disagree;
  * renders the current per-flower **daily counts** and the most recent **live landings**
    (from ``test_video_result/csv/``), plus the batch ``ALL_flower_summary.csv`` when present;
  * auto-refreshes so a live camera run updates the page as landings stream in.

Setup:  pip install -r src/webapp/requirements-web.txt
Run:    python -m src.webapp.app        # then open http://127.0.0.1:5000
"""
from __future__ import annotations

import csv
from pathlib import Path

from flask import Flask, render_template_string

from src import config as C
from src.cv_engine.source import resolve_source

app = Flask(__name__)
CSV_DIR = C.REPO_ROOT / "test_video_result" / "csv"
REFRESH_S = 5  # page auto-refresh cadence


def _read_csv(path: Path, limit: int | None = None) -> tuple[list[str], list[dict]]:
    if not path.exists():
        return [], []
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    header = rows[0].keys() if rows else []
    if limit is not None:
        rows = rows[-limit:]
    return list(header), rows


_PAGE = """<!doctype html>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="{{ refresh }}">
<title>Bee-A-Hero — live pollination</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: system-ui, sans-serif; margin: 0 auto; max-width: 960px; padding: 1.5rem; }
  h1 { margin: 0 0 .25rem; } .sub { color: #888; margin: 0 0 1rem; }
  .badge { display:inline-block; padding:.2rem .6rem; border-radius:1rem; font-weight:600; }
  .camera { background:#1b7f3b; color:#fff; } .video { background:#2456b8; color:#fff; }
  .none { background:#a33; color:#fff; }
  table { border-collapse: collapse; width: 100%; margin: .5rem 0 1.5rem; font-size: .9rem; }
  th, td { border: 1px solid #8884; padding: .35rem .6rem; text-align: left; }
  th { background: #8882; } caption { text-align:left; font-weight:600; margin-bottom:.3rem; }
  .empty { color:#999; font-style: italic; }
</style>
<h1>🐝 Bee-A-Hero — pollination monitor</h1>
<p class="sub">Input source (same rule as the pipeline) · auto-refreshes every {{ refresh }}s</p>
<p><span class="badge {{ src.mode }}">{{ src.mode|upper }}</span> &nbsp; {{ src.reason }}</p>
{% if src.items %}<p class="sub">sources: {{ src.items|join(', ') }}</p>{% endif %}

{% for title, header, rows in tables %}
<table>
  <caption>{{ title }}</caption>
  {% if rows %}
  <tr>{% for h in header %}<th>{{ h }}</th>{% endfor %}</tr>
  {% for r in rows %}<tr>{% for h in header %}<td>{{ r[h] }}</td>{% endfor %}</tr>{% endfor %}
  {% else %}<tr><td class="empty">no data yet</td></tr>{% endif %}
</table>
{% endfor %}
"""


@app.route("/")
def index():
    # probe_cameras=False: the status page reports intent without blocking on hardware
    src = resolve_source(probe_cameras=False)
    daily_h, daily = _read_csv(CSV_DIR / "daily_flower_counts.csv")
    live_h, live = _read_csv(CSV_DIR / "live_landings.csv", limit=20)
    summ_h, summ = _read_csv(CSV_DIR / "ALL_flower_summary.csv")
    tables = [
        ("Per-flower daily counts (live)", daily_h, daily),
        ("Recent landings (live, last 20)", live_h, list(reversed(live))),
        ("Test-video per-flower summary (batch)", summ_h, summ),
    ]
    return render_template_string(_PAGE, src=src, tables=tables, refresh=REFRESH_S)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
