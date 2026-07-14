# 04 · ML — Pollination & Yield (the science, Speaker 2/ML)

The chain has four links: **visits → effective pollen load → per-flower set probability → orchard yield / lift.** Each link is an equation with named parameters and an empirical anchor.

---

## 4.1 Qualifying visits per flower
Not every track that clips the ROI pollinates. Weight each visit *i* by dwell, keep only visits above threshold τ:

```
V_f = Σ_i  1[d_i ≥ τ] · w_i        w_i = 1 − e^(−d_i/d_0)
```
- `d_i` = dwell (s) inside the dilated ROI · `τ` ≈ 0.3 s (contact threshold) · `d_0` ≈ 1.0 s (pollen saturates with contact time).
- The indicator removes fly-throughs; the exponential weight says a brief touch < a sustained landing, plateauing.
- *In our implementation the "real landing" gate is 2 s and each landing also carries a species weight (see CV engine) — the same idea, tuned for the demo.*

## 4.2 Visits → per-flower success (saturating LIFT)
Pollen deposition **saturates** — a flower visited 50× is not 50× better set than one visited 5×. Model success as an exponential approach from the self floor to the cross ceiling:

```
P_success(V_f) = P_self + (P_max − P_self)·(1 − e^(−k·V_f))
```
| Symbol | Meaning | Value / source |
|---|---|---|
| `P_self` | set with zero insect help | **0.45** (bagged-flower literature) |
| `P_max` | ceiling under full cross-pollination | **0.68** (cross-pollination literature) |
| `k` | saturation rate (visits⁻¹) | **fit from data** (OLS below) |

**Limits behave correctly:** `V_f→0 ⇒ 0.45` (self still works); `V_f→∞ ⇒ 0.68` (insects can't exceed the biological ceiling). Total insect contribution is the **bounded lift = 0.23**.

### Fitting k (OLS through the origin)
Linearize: `y_f = ln((P_max − P_self)/(P_max − P̂_f)) = k·V_f`, then `k̂ = Σ V_f·y_f / Σ V_f²`. Check residuals against the curve — if structured, the exponential form is wrong for this orchard.

## 4.3 Per-flower fruit & orchard yield
```
E[fruits] = Σ_f P_success(V_f)·β_f
Ŷ         = Σ_f P_success(V_f)·β_f·m_f
```
- `β_f` ≈ 0.85–0.95 = set→harvest viability (drop, pests, husk scald).
- `m_f` = mean fruit mass by flower type: ~**0.35 kg** solitary, ~**0.22 kg** lateral (position-dependent). Classifying flower type at the ROI stage lets us weight `m_f`.

## 4.4 The headline: pollinator LIFT (not absolute kg)
Absolute kg can't be calibrated in one season. Report the **lift attributable to insects** — yield minus the self-pollination counterfactual (`V_f = 0`):
```
ΔY_bee = Σ_f (P_success(V_f) − P_self)·β_f·m_f
       = (P_max − P_self)·Σ_f (1 − e^(−k·V_f))·β_f·m_f
```
Report `ΔY_bee` and the **relative lift** `ΔY_bee / Ŷ_self`. This sidesteps absolute-yield calibration while quantifying exactly what the cameras measure: the marginal value of pollinators.

### Worked micro-example (have this ready)
`P_self=0.45, P_max=0.68, k=0.4, β=0.9, m=0.30 kg`, three flowers with `V = {1, 5, 12}`:
- `P(1)=0.526`, `P(5)=0.649`, `P(12)=0.678` (≈ ceiling)
- `Ŷ = 0.9·0.30·(0.526+0.649+0.678) = 0.500 kg`
- Self-only `Ŷ_self = 0.9·0.30·3·0.45 = 0.365 kg`
- **Bee lift ΔY_bee = 0.136 kg → +37% relative lift**, driven by the two well-visited flowers.

## 4.5 What's anchored vs. what's fitted (don't confuse a placeholder for a measurement)
- **Literature-anchored:** `P_self` (0.45), `P_max` (0.68), `m_f` (0.22–0.35 kg).
- **To be fitted to the orchard:** `k`, `τ`, `d_0`, `β_f`.

## 4.6 Reporting artifact
The ML phase (`src/llm_reporting`) produces `models/yield_report.json` with a fruit-set mean and an *illustrative* yield band (synthetic-fit, **not** field-calibrated) — the assistant reads it and always labels it as illustrative.

## Validation strategy (say this before they ask)
You can't ground-truth real kg in one season. So: (a) validate **visit-counting accuracy** against hand-labeled clips; (b) fit `k` on whatever paired visit/fruit-set flowers we can tag; (c) present `Ŷ` and `ΔY_bee` as a **relative index**, not calibrated tonnage. *A validated relative pollination index is defensible; a claimed absolute tonnage is the weakest thing you could show a reviewer.*
