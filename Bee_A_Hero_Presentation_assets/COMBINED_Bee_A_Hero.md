# Bee-A-Hero — Combined Brief (compact, one file)

*Pollinator tracking → pomegranate yield-lift. Baku, AZ. July 2026. The separate files hold the full detail; this is the whole story in one place.*

## 1. Problem & thesis
Pollinator value is invisible to growers. We measure it from cameras and turn it into an economic signal. **Key thesis — lift, not dependency:** pomegranate is self-fertile (sets ~**45%** fruit without insects); insects add a **bounded lift** toward a **68%** cross-pollination ceiling (+23 pp). We report the *marginal* value of pollinators — the defensible claim, not "more bees = more fruit."

## 2. Pipeline (Detect → Count → Trend → Report)
One source of truth: `count_visits_det()` runs for the website upload, the offline CSVs, and the live camera.

- **Models (fine-tuned YOLO26-m):** flower detector (mAP@0.5 **0.808**), insect multi-class detector `bee/fly/beetle/bug/butterfly` (mAP@0.5 **0.669**), honeybee sub-classifier (acc **0.978**). Tracking: **BoT-SORT** (`persist`).
- **Two passes:** (1) track+collect at 20 fps + confidence-weighted cumulative type vote; **stitch** tracks split by occlusion (concurrent-id merge + gap re-link ≤ 5 s, ~4×√area) and **interpolate** boxes across gaps; derive **landings**; (2) render the annotated H.264 video (mp4v → ffmpeg).
- **Visit state machine:** on-flower = ROI-in *or* near-motionless; grace-bridge 0.5 s; **≥ 2 s = a real landing** (feeding); fly-off + return = new visit, occlusion = same visit.
- **FP gates (clean boxes):** person-veto (COCO), max-size gate (>18% frame), flower-as-insect veto (IoU ≥ 0.80), insect nesting dedup, draw-time NMS, flower plausibility (no whole-screen/sliver), flower canonicalize-by-location + span/cluster count.
- **Pollination score:** Σ species_weight × min(dwell, 30 s); **honeybee 10×**, butterfly 2×, bee 1×, fly/beetle 0.5×, bug 0.2×.
- **Outputs:** `landings.csv` (per episode), `flower_summary.csv` (per flower), `ALL_*.csv`, annotated mp4.

## 3. ML — pollination & yield (4 links)
`V_f = Σ 1[d≥τ]·(1−e^(−d/d_0))` → `P_success = P_self + (P_max−P_self)(1−e^(−k·V_f))` → `Ŷ = Σ P·β_f·m_f` → **lift** `ΔY_bee = Σ(P−P_self)·β_f·m_f`.
Anchored: P_self 0.45, P_max 0.68, m_f 0.22–0.35 kg. Fitted: k (OLS), τ, d_0, β_f. **Report relative lift, not absolute kg.** Worked example: V={1,5,12} → +37% relative lift.

## 4. Assistant (Gemini / Hugging Face)
Per-chat provider pick: **Gemini** (`gemini-2.5-flash`) or **Hugging Face** (`Llama-3.1-8B-Instruct`), plus Auto and a grounded offline mock. Answers are grounded in the **real CV + ML result files** + the user's DB stats. Decoding: **T = 0.3, top-p = 0.9** for factual, repeatable answers. Keys in git-ignored `.env`, never committed; Claude removed.

## 5. App & live camera
FastAPI (:8000, SQLite, JWT, idempotent background detection job) + React/Vite (:5173). Upload → real pipeline → annotated H.264 ready at done-time + real-frame cover + per-flower stats + `no-store` media (SQLite id reuse). Live viewer (:8001): DroidCam/webcam, runtime source switch, capture/inference threads (latest-frame drop-old), trained models + person-veto, **rolling `live_landings.csv/.json`** as insects land/leave.

## 6. Deployment
Per-service Dockerfiles + compose; weights mounted read-only; CPU torch; YOLO26 → ONNX/TensorRT/TFLite for edge. One venv (py3.11). `bash run-website.sh` starts all three.

## 7. Evaluation
Flower mAP 0.808 · insect mAP 0.669 · classifier 0.978. Tracking: HOTA/IDF1/ID-switches (TrackEval). **Core validation = visit precision/recall vs hand-labeled clips.** Yield = relative index, not kg. Demonstrable: occlusion re-link (one visit), fly-through rejection, zero FPs on an empty scene, website == CSV.

## 8. Honest limitations
Insect classifier confuses some honeybees with fly/beetle (mitigated by vote + sub-classifier); flower detector sometimes misses a bloom (falls back to inferred); absolute yield uncalibrated (by design); multi-camera designed not shipped.

## 9. The three winning sentences
1. "We model the **marginal lift** insects add over a 45% self-pollination floor, capped at 68% — not linear dependency."
2. "We count **feeding visits, not frames** — a 2 s dwell gate + occlusion re-linking, so a bee behind a petal is counted once."
3. "**One pipeline, one source of truth** — website, CSVs, and live camera can't disagree."

## 10. Rapid-fire
Visit ≥ 2 s · honeybee 10× · lift +23 pp · YOLO26 + BoT-SORT + honeybee clf · FastAPI + React · Gemini/HF assistant · ports 8000/5173/8001 · Baku, *Punica granatum*.
