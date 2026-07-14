# 09 · Anticipated Questions & Model Answers (the questioning phase)

Grouped by theme. Each answer is short enough to say out loud.

---

## A. Science / framing (the ones that decide the grade)

**Q: More bees = more fruit — isn't that your model?**
No — that would be biologically wrong. Pomegranate is self-fertile (sets ~45% fruit with no insects). We model the **marginal lift** insects add over that floor, saturating toward a 68% cross-pollination ceiling. The insect contribution is a bounded +23 percentage points, not unlimited.

**Q: Why not a bee-*dependent* crop (almond) for a punchier headline?**
It would lose (a) local relevance to Azerbaijan, (b) the citable 45%→68% baseline, and (c) the more rigorous marginal-effect story. We keep pomegranate and lean into the lift framing; the same pipeline generalizes to other crops as a secondary demo.

**Q: Where do 45% and 68% come from?**
Purdue/Morton pomegranate horticulture (bagged vs cross-pollinated fruit set). They're literature anchors, not fitted — that's what makes the model defensible.

**Q: Why saturating (exponential), not linear?**
Pollen deposition saturates — a flower visited 50× isn't 50× better set than one visited 5×. The exponential approaches the biological ceiling; linear would predict impossible set rates.

**Q: What's actually fitted vs assumed?**
Anchored from literature: P_self, P_max, fruit mass m_f. Fitted to the orchard: saturation rate k (OLS), dwell params τ/d_0, viability β_f. We never present a placeholder as a measurement.

**Q: Can you prove the yield number?**
Not in absolute kg in one season — so we don't claim it. We validate **visit counting** against hand-labeled clips and report **relative lift**, which is exactly what the cameras measure.

---

## B. Computer vision / algorithms

**Q: How do you avoid double-counting one bee?**
Three layers: (1) BoT-SORT with `persist` keeps ids across short gaps; (2) offline **track stitching** merges tracks split by occlusion — a bee reappearing within 5 s and near where it vanished is the same track; (3) the **visit state machine** treats a re-entry within the grace window as the same landing. A fly-off that genuinely leaves and returns is counted as a new visit.

**Q: A bee goes behind a petal — counted twice?**
No. The detector drops out, but the track is **parked and re-linked** when the bee re-emerges nearby, and the box is **interpolated** across the gap. One visit.

**Q: Why count "visits" and not detections/frames?**
A detection is one frame; a visit is a behaviour (land, feed, leave). Pollination is about contact events, so we count feeding visits with dwell — the raw pollination signal.

**Q: Why a 2-second dwell threshold?**
Below ~2 s it's a fly-through, not feeding — little pollen transfer. The threshold separates genuine landings from passes. (τ≈0.3 s in the general math; the demo uses a stricter 2 s "real landing" gate.)

**Q: "The whole screen is detected as a flower" — how did you fix it?**
Geometric gating: reject flower boxes bigger than half the frame, slivers (extreme aspect), or noise; raise flower confidence; and NMS overlapping boxes. Pointed at a room, the viewer now shows zero detections.

**Q: "Flower detected as a butterfly" / boxes mixed up?**
A closed-set detector snaps unknowns onto trained classes. We veto an insect box that basically **is** a flower (IoU ≥ 0.80 with a flower box), dedup nested insect boxes (class-agnostic NMS), and NMS at draw time so no two boxes stack on one bug. A small bee **on** a flower (low IoU) still passes.

**Q: Bee vs fly errors?**
The detector sometimes confuses them. We stabilize the label with a **confidence-weighted cumulative vote over the whole track** plus hysteresis (a challenger must out-weigh the leader by 1.5×), and a separate honeybee sub-classifier (0.978 acc) splits honeybees from other bees. It's a known model-accuracy limit, mitigated in software.

**Q: Why two passes over the video?**
To drag the box smoothly from vanish→reappear you must know the reappear point, which a single forward pass doesn't. Pass 1 collects detections; we stitch + interpolate offline; Pass 2 renders. It also guarantees the annotated preview matches the counts exactly.

**Q: Why YOLO26 and BoT-SORT specifically?**
YOLO26: NMS-free, small-object recall (ProgLoss/STAL) — bees are < 20 px. BoT-SORT: Ultralytics default with camera-motion compensation and a refined Kalman filter; a 2025 PLOS One honeybee study validated exactly this stack (98%/36 fps).

**Q: Handling tiny, motion-blurred bees / domain gap?**
P2 small-object head, motion-blur/downscale augmentation, and — highest leverage — a few hundred **in-domain** labeled bees, which dominate a large out-of-domain set.

**Q: Why is the flower count sometimes off?**
When the flower detector fails to fire on a real bloom, landings fall back to inferred synthetic flowers. We canonicalize flowers by location, require a minimum on-screen span, drop empty synthetics, and spatially cluster landing spots (a bee crawling across one bloom = one flower). It's a detector limitation, handled robustly in counting.

---

## C. ML / LLM

**Q: Does the assistant hallucinate the numbers?**
It's given the user's real DB stats and the actual CV+ML result files as context, and instructed to use only measured numbers. Even the offline fallback quotes real stats.

**Q: Why temperature ≤ 0.3 and top-p 0.9?**
Factual, repeatable technical answers. Low temperature sharpens the distribution; top-p nucleus sampling truncates unlikely tokens so they can't derail an answer.

**Q: Why offer both Gemini and Hugging Face?**
No lock-in: an open-source model (Llama-3.1-8B via HF Inference) alongside Gemini, picked per chat. If a key is missing the app falls back gracefully to a grounded mock.

**Q: What if the LLM API is down?**
The provider chain falls back (Gemini → HF → mock). The chat never breaks in a demo.

---

## D. Systems / product

**Q: Do the website numbers match the offline CSVs?**
Yes — by construction. The upload runs the **identical** `count_visits_det` pipeline; there is no second, divergent counter.

**Q: Real-time?**
The live viewer is real-time MJPEG with capture/inference on separate threads and latest-frame drop-old semantics, so slow inference never blocks the feed. Uploads are batch (CPU).

**Q: Multi-camera / don't-double-count across cameras?**
Designed, not shipped: for **overlapping** cameras, homography handoff (no appearance ReID needed); for non-overlapping, a spatio-temporal gate reported as an approximation. Insect ReID by appearance is genuinely hard at camera resolution — we're honest about that.

**Q: GPU required?**
No — runs on CPU (resampled to 20 fps + a light render pass). GPU just makes it faster. YOLO26 exports to TensorRT/ONNX/TFLite for edge later.

**Q: Is it bug-free / production-ready?**
Verified end-to-end (login → upload → process → annotated H.264 stream → assistant). The detection job is idempotent; media responses are `no-store` (SQLite reuses ids); the app degrades gracefully with no models or no API key; secrets are audited out of git.

**Q: Security — are API keys exposed?**
No. Keys live only in the git-ignored `backend/.env`, read by absolute path, never printed or committed; git history was scanned to confirm zero leakage.

---

## E. Non-technical / impact / "why should I care"

**Q: Who uses this and why?**
Growers (value pollinators, time sprays, justify hives), researchers (automated pollination index), conservation/policy (evidence for pollinator value).

**Q: What's novel here vs "detect bees, count them"?**
The **lift-not-dependency** science, one-source-of-truth pipeline, honest visit-counting with occlusion handling, and a grounded assistant — a defensible measurement system, not a bee counter.

**Q: Biggest limitation?**
Absolute yield isn't field-calibrated (we report relative lift), the classifier confuses some insect species, and multi-camera is a future extension. We state these up front.

**Q: What would you do with more time / one bloom season?**
Collect in-domain labeled video before May bloom, fit k on paired visit/fruit-set flowers, add overlapping-camera homography handoff, and validate visit precision/recall on a hand-labeled hold-out.

---

## G. Jury / business / product (the 30% they score — answer with confidence)

**Q: Who is the customer and who pays?**
Pomegranate (and later other-crop) growers and orchard cooperatives; agronomy/ag-tech services; researchers and conservation programs. They pay to *quantify pollinator value* — to decide on hive rental, habitat, and spray timing, which are real line-item costs.

**Q: What's the business value in one sentence?**
We turn a cheap camera into a pollination instrument that tells a grower how much of their fruit set the bees are actually buying them — so they can invest in pollinators with evidence instead of guessing.

**Q: Is this a real problem or a nice-to-have?**
Real and measurable: pollinator decline is global, ~a third of food depends on pollinators, and growers currently have **zero** visibility into on-farm pollinator performance. We give them the missing measurement.

**Q: Why will this actually get adopted?**
Low friction: runs on CPU, uses a phone (DroidCam) or any camera, and outputs a plain-language report a non-technical grower can read. No lab, no GPU, no data-science team required.

**Q: What's the market / scale story?**
Start with pomegranate in the Caspian region (culturally central, clear season), then the *same pipeline* generalizes to any insect-pollinated crop by swapping the flower detector. The pollination-monitoring / precision-ag space is growing precisely because pollinator risk is now a board-level agricultural concern.

**Q: What's the moat / why can't someone copy it in a weekend?**
The defensibility is in the *method*, not a single model: the lift-not-dependency science, the occlusion-aware visit state machine, in-domain data, and one-source-of-truth engineering. A weekend clone counts bee-frames and over-reports 15×.

**Q: Biggest risk to the business and how do you de-risk it?**
Calibration data — you need paired visit/fruit-set labels over a bloom season to fit the yield curve. We de-risk by reporting a **validated relative index** now (defensible without calibration) and lining up ground-truth collection for the season.

**Q: What did you cut, and why (scope/trade-offs)?**
We cut multi-camera ReID (genuinely hard on bees) and absolute-kg calibration (impossible in one season) to ship a working single-camera measurement system with honest relative-lift output. That's a deliberate scope decision, not a gap.

**Q: What would you do next with more time/funding?**
One bloom season of in-domain labeled video, fit k on paired visit/fruit-set flowers, overlapping-camera homography handoff, and a grower-facing weekly report. Then a pilot with a cooperative.

**Q: How is this different from existing insect-camera / smart-trap products?**
Most count or classify insects. We connect visitation to a **crop-yield economic signal** with a defensible saturating-lift model — measurement *plus* meaning.

**Q: Team — how did you split the work?**
Data & training (Raul, Khaver), CV engine + tracking + QA (Asif), ML modeling + LLM reporting (Narmin), with the lead coordinating research direction. Four people, one integrated pipeline that merges cleanly to `main`.

---

## F. Rapid-fire facts (memorize)
- Flower mAP **0.808** · insect mAP **0.669** · classifier **0.978**
- Visit = **≥ 2 s** on a flower · honeybee weight **10×**
- Self **45%** → cross **68%** → lift **+23 pp**
- Stack: **YOLO26 + BoT-SORT + honeybee sub-classifier**; FastAPI + React; Gemini/HF assistant
- Ports: backend **8000**, frontend **5173**, live **8001**
