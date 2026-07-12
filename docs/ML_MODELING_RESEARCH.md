# Bee-A-Hero — ML Modeling Research: Visit-to-Fruit-Set Dose–Response

Research for the modeling stage of Bee-A-Hero, in which per-flower **insect-visit counts**
produced by the computer-vision pipeline are related to **fruit set** in **pomegranate**
(*Punica granatum* L.). The document has three parts:

- **Part I — Dose–response modeling research** (Sections 1–8): the statistics and study
  design that must be settled before a curve is fitted.
- **Part II — Pomegranate reproductive biology** (Sections 9–14): the crop-specific
  literature that fixes the model's ceiling, floor, and confounders.
- **Part III — Implementation plan** (Sections 15–20): how to build `src/ml_models/`.

It is self-contained; every formula and figure is traced to a citable source (consolidated
References at the end), and points where no validated pomegranate-specific source exists are
marked **Note:** rather than filled with invented constants.

## Pipeline context

The visit signal comes from `src/cv_engine/video_detect.py`. Per frame, flowers are
detected with a single-class YOLO26 detector and given stable IDs (`FlowerTracker`); a
multi-class YOLO26 insect detector (`bee, fly, beetle, bug, butterfly`) plus BoT-SORT
tracking assigns one track ID and type per insect, the type being majority-voted over the
track's life. A visit is counted when a tracked insect's bounding-box **centre enters a
flower box**, and each `(track_id, flower_id)` pair is counted **once ever** — a fly-off
and return is not a new visit — stamped with an entry time `t_enter_s`. The stream is
resampled to a fixed 24 fps. Outputs are, per video, `<video>_visits.csv`
(`flower_id, total, pollinator, <per-type…>`) and `<video>_timeline.csv`
(`flower_id, track_id, type, t_enter_s`), aggregated across videos into `ALL_visits.csv`
and `ALL_timeline.csv`; `pollinator` rolls up `{bee, butterfly, fly}`. These tables are
the input the modeling stage consumes. A future refinement — qualifying a visit by dwell
time, insect velocity, and spatial overlap (fraction of the insect box on the flower) —
is discussed as it becomes relevant, but is not yet implemented.

---

# Part I — Dose–Response Modeling Research

## 1. Measurement error in the visit (dose) variable

The modeling stage treats the per-flower visit count $x$ as the explanatory variable (the
"dose") and fruit set as the response. That count is the end product of detection,
tracking, and type assignment, each of which errs. Because the error is in the
**predictor**, not the response, it has consequences that error in the response does not.

### 1.1 From a biological visit to a counted visit

The biologically meaningful event is a *foraging visit capable of depositing viable
conspecific pollen on a receptive stigma*. Flower visitation is a known-imperfect proxy
for this: many contacts deposit no pollen (King, Ballantyne & Willmer 2013). The pipeline
uses an **operational** definition — a tracked insect whose box centre enters a flower
box, counted once per insect–flower pair. This is coarser than the biological event in two
specific ways:

- **Contact quality is ignored.** Centre-in-box treats a bee grazing the edge of the box
  the same as one settled on the anthers. A graded overlap measure (the fraction of the
  insect box on the flower) would separate these; it is a natural future gate and feature
  (Section 7).
- **Repeat bouts collapse.** The once-ever rule counts a bee that leaves and returns as a
  single visit, whereas each return can be a separate pollinating event. High-visit
  flowers are therefore undercounted relative to the biological dose.

### 1.2 A taxonomy of dose error

| Error | Mechanism in the pipeline | Effect on the count $x$ |
|---|---|---|
| False positive | Non-insect blob or shadow tracked as an insect | Inflates $x$ |
| False negative | Occlusion by petals, motion blur, small/fast insect missed | Deflates $x$ |
| BoT-SORT ID switch | Two insects swap IDs, or one track fragments | Splits or merges events → miscount |
| Repeat-bout collapse | Leave-and-return counted once (once-ever rule) | Deflates $x$ at the high end |
| Type error | Detector majority-vote flips pollinator ↔ non-pollinator | Moves counts between `pollinator` and per-type columns |

Detection error perturbs `total`; type error perturbs the `pollinator` split and the
per-type breakdown; the once-ever rule, together with a missed first track, also shifts
the per-flower first-visit time derived from `t_enter_s`. The relevant CV quality metrics
here are therefore **tracking** metrics — ID switches, IDF1, MOTA (Bernardin &
Stiefelhagen 2008; Ristani et al. 2016) — not per-frame detection mAP alone, because a
corrupted dose is a *counting* failure, and counting is a property of tracking.

### 1.3 Errors-in-variables and attenuation bias

Write the observed dose as truth plus error, $w_i = x_i + u_i$, with
$\mathbb{E}[u_i \mid x_i] = 0$ and $\operatorname{Var}(u_i) = \sigma_u^2$. Fitting the
response against $w_i$ instead of $x_i$ shrinks the estimated slope toward zero. For a
linear (or logit-linear) dose term the shrinkage is the **reliability ratio** $\lambda$:

$$
\hat{\beta}_1 \;\xrightarrow{\ p\ }\; \lambda\,\beta_1,
\qquad
\lambda = \frac{\sigma_x^2}{\sigma_x^2 + \sigma_u^2} \le 1 .
$$

So random dose error **biases the visit slope toward zero**: the curve looks flatter, the
effect of pollinators looks weaker, and pollination looks less limiting than it is (Fuller
1987; Carroll et al. 2006). This is the single most important statistical fact about the
predictor — imperfect tracking does not merely add noise, it *systematically understates*
the value of pollinators.

Detection **recall $r$** and **precision $p$** map onto this bias:

- **Recall $r < 1$ (misses).** If each true visit is independently detected with
  probability $r$, the observed count is a thinned version with $\mathbb{E}[w \mid x] = rx$
  and $\operatorname{Var}(w \mid x) = x\,r(1-r)$. The deterministic factor $r$ rescales the
  dose axis — it inflates a saturating curve's midpoint $\mathrm{ED}_{50}$ by roughly
  $1/r$ (you need $1/r$ observed visits to reach the same true dose) — while the random
  part $r(1-r)$ supplies the $\sigma_u^2$ that attenuates the slope. Lower recall means
  both more attenuation and a right-shifted, gentler-looking curve.
- **Precision $p < 1$ (false positives).** Spurious visits are not mean-zero given $x$;
  they inflate $w$, biasing the intercept and further diluting the slope. They are most
  corrosive at the low-dose end, where one spurious visit can move a flower out of the
  informative zero-visit stratum.
- **ID switches and repeat-bout collapse** inject positively correlated error (one
  physical event → several counts, or several events → one), behaving like extra
  $\sigma_u^2$ and potentially inducing spurious overdispersion in the fit (Section 3).

**Note:** the clean $\lambda$ formula assumes *classical, non-differential* error —
independent of the response. Ours may be **differential**: a flower that sets fruit may
look different (developing ovary, senescing petals) and be detected differently,
correlating dose error with the outcome. Differential error can bias the slope in either
direction, and no single scalar corrects it.

### 1.4 Mitigation options (roughly by cost)

1. **Measure the operating point.** Report recall and precision at the chosen detection
   and (future) gate settings, from hand-labelled clips, alongside any fitted curve. A
   curve is only interpretable next to the $(r, p)$ that produced its doses.
2. **Validation subsample / regression calibration.** Hand-count visits on a subset to
   estimate $(r, p, \sigma_u^2)$, then replace $w_i$ with $\mathbb{E}[x_i \mid w_i,
   \text{covariates}]$ before fitting (Carroll et al. 2006). Comparatively cheap; turns a
   guess about $\lambda$ into an estimate.
3. **SIMEX (simulation–extrapolation).** Add known extra error to $w$, watch the slope
   attenuate, and extrapolate back to $\sigma_u^2 = 0$ (Cook & Stefanski 1994). Needs only
   an estimate of $\sigma_u^2$, not the true $x$.
4. **Errors-in-variables inside the model.** Put the measurement model $w_i \mid x_i$ and
   a prior on $x_i$ into the fit, propagating dose uncertainty into the parameter estimates
   — naturally Bayesian (Section 4).
5. **Report the direction even when uncorrected.** If none of the above is feasible,
   state that the fitted visit effect is a **lower bound** in magnitude because of
   attenuation.

---

## 2. Observational vs. experimental data, confounders, and non-independence

### 2.1 Two data-generating regimes

- **Controlled-visit (experimental) data.** A flower is bagged to exclude insects, exposed
  to a *known* number of visits by a *known* taxon, then re-bagged; fruit set (or seed /
  pollen-tube count) is scored later. This is the design behind single-visit-deposition and
  pollinator-effectiveness studies (Ne'eman et al. 2010; King, Ballantyne & Willmer 2013).
  Here the visit count is an *assigned treatment*, essentially free of measurement error
  and — crucially — independent of the flower's other properties, because the experimenter
  chose it.
- **Observational (camera) data.** The pipeline produces visit counts at scale, but the
  insect chose which flower to visit. Visit count is now an *observed covariate*,
  correlated with everything that makes a flower attractive. This is the regime the camera
  pipeline operates in.

A curve fitted to controlled-visit data estimates a *causal* dose–response; the same curve
fitted to observational counts estimates a *conditional association*, causal only under
the untestable assumption of no unmeasured confounding.

### 2.2 Confounders to record as covariates

A confounder moves both the visit count and the fruit-set probability. Omitted, it biases
the fitted visit effect in an unknown direction. The observational model should log, per
scored flower:

| Confounder | Why it moves *visits* | Why it moves *fruit set* |
|---|---|---|
| Flower age / stage | Older flowers advertise less | Receptivity and ovule viability decline with age |
| Position in cluster / on plant | Apical flowers are found first | Pomegranate flower type and set vary with position (Section 11) |
| Cultivar | Attractiveness differs | Self-compatibility differs (Sections 4, 10) |
| Local floral density | Dilution vs. facilitation of visits | Pollen limitation vs. resource competition |
| Weather during the visit | Bees forage in a temperature/wind window (Section 5) | Pollen viability and tube growth are temperature-dependent (Section 13) |
| Date / bloom day | Pollinator community turns over | Plant resource state and bloom flush change (Section 13) |

These correspond to the temporal features already produced (visit timings), the spatial
features of Section 7, and the environmental features of Section 5. Recording them does not
*remove* confounding — it lets the model *adjust* for the measured part and be explicit
about the unmeasured part.

### 2.3 Non-independence and the mixed model

Flowers on the same plant share a genotype, water status, and resource pool; flowers in the
same clip share a camera, day, and weather. Their fruit-set outcomes are therefore
correlated, violating the independence assumption of an ordinary binomial GLM. Ignoring
this yields standard errors that are too small and over-optimistic significance
(pseudoreplication; Bolker et al. 2009; Harrison et al. 2018). The fix is a binomial
**generalized linear mixed model (GLMM)** with grouping factors as random intercepts:

$$
\text{fruitset}_i \sim \text{Binomial}(1, p_i),
\qquad
\operatorname{logit}(p_i) = \beta_0 + f(x_i; \boldsymbol{\theta})
+ u_{\text{plant}(i)} + u_{\text{orchard}(i)} + u_{\text{date}(i)},
$$

$$
u_{\text{plant}} \sim \mathcal{N}(0, \sigma^2_{\text{plant}}), \quad
u_{\text{orchard}} \sim \mathcal{N}(0, \sigma^2_{\text{orchard}}), \quad
u_{\text{date}} \sim \mathcal{N}(0, \sigma^2_{\text{date}}),
$$

where $f(x_i; \boldsymbol{\theta})$ is the dose term (linear in $x$, or a nonlinear
saturating / Hill form; Section 6). The variance components $\sigma^2_\bullet$ *are* the
non-independence, estimated rather than assumed away; the intraclass correlation
$\sigma^2_{\text{plant}} / (\sigma^2_{\text{plant}} + \pi^2/3)$ reports how much of the
outcome is "plant, not visits." Nesting is `date/plant`, with orchard crossed or nested as
the layout dictates.

### 2.4 Causal caveat

**Note:** a dose–response curve fitted to camera counts is **correlational**. It answers
"flowers observed receiving more pollinator visits set more fruit, after adjusting for the
covariates we measured" — not "adding a visit causes a fruit." The causal reading is only
licensed when visits are experimentally controlled (Section 2.1). Any automated report
built on the model must inherit this caveat and must not upgrade the association to a causal
claim.

---

## 3. Uncertainty quantification

Point-accuracy metrics (Brier score, log-loss, calibration, AUC) measure how close a single
predicted fruit-set number is to truth. They do not answer a distinct question: **how sure
are we of the numbers themselves** — the fitted parameters and the predictions built on
them. A perfectly calibrated point estimate with an interval spanning $[0.1, 0.9]$ is not
the same result as the same point with an interval of $[0.34, 0.40]$, and only the interval
tells the grower which they have.

### 3.1 Two kinds of uncertainty

- **Parameter uncertainty** — the curve parameters (slope, midpoint, asymptotes) were
  estimated from finite, error-laden data (Section 1).
- **Prediction uncertainty** — even with parameters fixed, a new flower's fruit set is a
  Bernoulli draw, and a new orchard's rate carries the random-effect variance of Section
  2.3.

Point accuracy can be excellent while both are large. They must be reported in addition to,
not instead of, accuracy.

### 3.2 Intervals on the parameters

- **Frequentist.** Intervals on the parameters from the fitted covariance matrix (inverse
  observed Fisher information; Wald intervals) when the sample is comfortable and the
  likelihood is near-quadratic. Wald intervals are poor near a boundary (an asymptote
  pinned at 1, a variance at 0); prefer **profile-likelihood** intervals there, and
  **bootstrap** (bias-corrected accelerated, BCa) intervals for nonlinear curve parameters
  (Efron & Tibshirani 1993; DiCiccio & Efron 1996), resampling at the **cluster** (plant or
  clip) level to respect the non-independence of Section 2.3.
- **Bayesian.** Posterior **credible intervals** directly from the samples (Section 4) —
  the natural choice under small samples, needing no asymptotic-normality assumption and
  honestly widening where data are thin.

### 3.3 Prediction intervals and propagation into yield

A prediction interval must combine both uncertainties of Section 3.1. Draw parameter sets
$\boldsymbol{\theta}^{(b)}$ from the bootstrap or posterior, push each through the curve to
get $\hat{p}^{(b)}$, and add Bernoulli/binomial sampling; the spread is the prediction
interval. Propagating into a simple orchard yield estimate,

$$
\widehat{\text{Yield}} = N_{\text{flowers}} \cdot \hat{p}(\bar{x}) \cdot \bar{m},
\qquad
\text{PI}:\ \big[\,q_{2.5\%},\ q_{97.5\%}\,\big]\ \text{of}\
\big\{ N_{\text{flowers}}\, \hat{p}^{(b)}(\bar{x})\, \bar{m} \big\}_b ,
$$

where $N_{\text{flowers}}$ is the flower count and $\bar{m}$ the mean fruit mass. This is
**Monte-Carlo propagation** — do not plug in a point $\hat{p}$ and call it the yield. A
±5-point uncertainty on fruit-set probability becomes an orchard-scale band on kilograms;
hiding it makes the estimate look far more precise than the science supports.

### 3.4 Small-sample pathologies

- **Separation / perfect prediction.** If every flower above some visit count sets fruit and
  none below, the logistic maximum-likelihood slope diverges to $\pm\infty$ with infinite
  standard error — the fit "succeeds" but is meaningless. Detect it (huge coefficients and
  SEs) and fix it with **penalized likelihood**: Firth's bias reduction (Firth 1993; Heinze
  & Schemper 2002) or a weakly-informative prior (Section 4), both of which keep estimates
  finite. Likely in a small pomegranate sample.
- **Zero-inflation / excess zeros.** Many flowers receive zero visits (and mostly do not
  set fruit), producing more zeros than a plain binomial expects and **overdispersion**.
  Model it explicitly — a hurdle or zero-inflated structure, or a random effect that
  absorbs the excess (Zuur et al. 2009) — rather than letting it deflate standard errors.
  Because ID switches and false positives can *manufacture* overdispersion (Section 1.3),
  check whether excess variance is biology or measurement. In pomegranate a large share of
  the zeros is structural, from functionally male flowers (Section 9).
- **Boundary variance estimates.** A zero plant-level variance ("singular fit") is common
  with few plants; report it honestly rather than dropping the random effect, which would
  reintroduce pseudoreplication.

### 3.5 What the report should carry

**Recommendation:** every model-estimated quantity surfaced to a user — fitted fruit-set
probability, the visit effect, the yield figure — should carry an **interval, not just a
point value** (e.g. "estimated fruit set 32%, 95% interval 24–41%"). A point estimate with
no interval is, for a small pomegranate model, a misleading claim of precision.

---

## 4. Bayesian dose–response fitting with cross-crop priors

Pomegranate presents a two-headed problem: **no controlled-visit anchor** (no experiment
that fixes the dose; Section 2.1) and a **small sample**. A purely data-driven fit that
asks the pomegranate data to identify a full curve on its own will fail or overfit under
those conditions. Bayesian fitting is the principled response, because it lets curves we
*do* trust from other crops inform the pomegranate fit without dictating it.

### 4.1 The idea

A Bayesian fit places a prior over the curve parameters — the asymptotes, the midpoint or
rate, the steepness (Hill coefficient), and the intercept — and updates it with pomegranate
data:

$$
p(\boldsymbol{\theta} \mid \text{data}) \;\propto\;
\underbrace{p(\text{data} \mid \boldsymbol{\theta})}_{\text{binomial GLMM likelihood (Section 2.3)}}
\; \underbrace{p(\boldsymbol{\theta})}_{\text{prior}} .
$$

With a small sample the posterior is pulled toward the prior where the data are silent (for
instance, the upper asymptote, if saturating doses are never observed) and dominated by the
data where they speak (the low-dose slope, where most camera counts live). No arbitrary
"borrow the apple constant" step is needed; the borrowing strength is set by how much the
pomegranate data disagree with the prior.

### 4.2 Where the priors come from

Fitted **apple** and **cranberry** visit-to-fruit-set curves supply *shape* priors, not
point values:

- **Saturation.** Fruit set rises steeply with the first visits and plateaus — encode a
  prior favouring a finite upper asymptote below 1 and a positive steepness.
- **Location of the knee.** The apple/cranberry midpoint (visits to half-maximum) sets a
  weakly informative prior on pomegranate's midpoint — wide enough to be moved, centred
  where cross-crop biology suggests.
- **Self-fertility offset.** Pomegranate is partially self-compatible, so the intercept
  corresponds to fruit set at zero insect visits (Section 10). Its prior should come from
  pomegranate bagging studies, *not* from apple, which is largely self-incompatible — a
  place where a cross-crop prior would be actively wrong.

**Note:** apple and cranberry differ from pomegranate in floral architecture, breeding
system, and pollinator guild. Their curves are legitimate priors on *shape* (does it
saturate? is the knee early or late?) but not on absolute fruit-set level. Priors must be
**weakly informative** — wide enough that a dozen contradicting pomegranate flowers can
overrule them — and a **prior-sensitivity check** (refit under a vague and a tighter prior,
report how far the posterior moves) is mandatory. A conclusion that changes with the prior
is reporting the prior, not pomegranate.

### 4.3 Bayesian vs. frequentist model selection

| Situation | Prefer |
|---|---|
| Large sample, controlled doses spanning the range | Frequentist penalized fit (AIC/BIC); the data identify the curve |
| Small sample, no saturating doses, credible cross-crop shape | Bayesian with cross-crop priors (this section) |
| Honest parameter / prediction intervals under small $n$ | Bayesian posteriors (Section 3); no asymptotic-normality assumption |
| Model comparison among curve families under small $n$ | Bayesian (WAIC / LOO-CV; Vehtari et al. 2017), or penalized frequentist — report both |
| Dose measurement error must enter the fit (Section 1.4) | Bayesian; the measurement model is just another layer |

The two are complementary: fit a frequentist penalized curve as the baseline and a Bayesian
curve as the small-sample, cross-crop-informed version, and report both plus how far apart
they land. Divergence localizes exactly where the pomegranate data are too thin to speak.

---

## 5. Environmental covariates as model features

Temperature, wind, humidity, and solar radiation belong in the feature set for a specific
reason: they drive **both sides** of the dose–response at once, making them the textbook
confounder of Section 2.2. Modelling them is how we avoid attributing to *visits* what
belongs to *weather*.

### 5.1 The dual pathway

| Variable | Effect on bee activity / visits | Effect on pollen viability / fruit set |
|---|---|---|
| Temperature | Foraging has a lower threshold (bees largely inactive below ~12–15 °C) and an upper limit | Pollen germination and tube growth follow a temperature optimum; heat stress kills pollen and aborts ovules |
| Wind speed | High wind suppresses flight and visitation | Desiccates stigmas; some incidental wind pollen transfer |
| Humidity | Modulates activity and nectar concentration | Affects stigma receptivity and pollen hydration; very low humidity shortens the viable window |
| Solar radiation | Warms bees to flight temperature; times foraging peaks | Drives canopy temperature and the daily receptivity rhythm |

Because both arrows exist, weather is a confounder of the visit-to-fruit-set link, not
merely a nuisance. Entered as covariates in the GLMM (Section 2.3), these variables let the
fit separate "more fruit because more visits" from "more fruit because a warm day helped
both visits and pollen."

### 5.2 Feature engineering notes

- Match weather to the **visit window**, not the daily mean: a flower's relevant exposure is
  the temperature and wind while it was being visited and while its pollen was viable
  (align to the visit timestamps).
- Expect **non-linear, often unimodal** responses (an optimum, not a slope) — model with
  splines, quadratic terms, or a biologically-shaped thermal-performance curve.
- Watch **collinearity**: temperature, solar radiation, and humidity co-move; consider a
  reduced set or a derived index rather than all four raw.

### 5.3 Model feature vs. report grounding

**Note:** weather plays two different roles that must not be confused. As a **model
feature**, the fruit-set model may *learn* a relationship between temperature and outcome
from data — a coefficient supported by evidence. That is separate from the **automated
report's grounding rule**, which forbids the report from *inventing* weather-based causal
explanations it was not given. A learned coefficient ("fruit set rose with visit count,
adjusting for temperature") is evidence; a generated sentence ("the cold snap reduced
pollination") with no such input is a hallucination the report layer must block. The model
*uses* weather as data; the report may only *restate* weather that the pipeline actually
measured.

**Data source.** These covariates come from the project's separate weather / time-series
work — the Open-Meteo integration (historical and forecast API keyed to orchard latitude,
longitude, and the clip timestamp), documented there. This section consumes that feed; it
does not define it.

---

## 6. Baseline models and sample size / power

Before comparing candidate curve families, a curve must earn its complexity by beating
trivial baselines, and we must know whether the sample can support it at all.

### 6.1 Baselines any curve must beat

1. **Predict-the-mean (intercept only).** $\hat{p} = \bar{y}$, the grand fruit-set rate,
   ignoring visits. In GLMM terms this is $\operatorname{logit}(p_i) = \beta_0$ plus random
   effects. If the dose term does not beat this out of sample, there is no evidence that
   visits explain fruit set. This is the reference for a McFadden pseudo-$R^2$ and for the
   likelihood-ratio test of "does dose matter."
2. **Simple logistic in raw visit count.** $\operatorname{logit}(p_i) = \beta_0 + \beta_1
   x_i$ — one monotone slope, no saturation. This is the parsimonious rival; a three- or
   four-parameter saturating curve must beat *it* on penalized fit or cross-validated
   log-loss, or it is overfitting the asymptote.
3. **Threshold baseline (optional).** "Any pollinator visit vs. none" ($\mathbb{1}[x_i >
   0]$) — a single step. Useful when counts are sparse and the real question is the
   presence or absence of a pollination event.

Report every candidate curve *relative to* these baselines (Δlog-loss, ΔAIC), never in
isolation. On an imbalanced fruit-set outcome, a high absolute accuracy can be inherited
entirely from baseline 1.

### 6.2 Candidate curve families

The dose term $f(x;\boldsymbol{\theta})$ will be selected from a small set of monotone,
saturating shapes standard in dose–response work (Ritz et al. 2015): the linear-logistic of
baseline 2; a two- or three-parameter log-logistic / Hill curve
$f(x) = P_{\max}\, x^n / (\mathrm{ED}_{50}^{\,n} + x^n)$; a four-parameter version with a
non-zero lower asymptote (the self-fertility offset of Section 4.2); and asymptotic-
exponential ("diminishing returns") forms. Selection is by penalized fit (Section 4.3).

### 6.3 How many flowers are needed?

Sample size for a curve is driven by **events per parameter** on the binary outcome and by
**coverage of the dose axis**, not by raw flower count.

- **Events-per-variable rule of thumb.** Classic guidance for logistic-type models is at
  least 10 events per estimated parameter (Peduzzi et al. 1996), where "events" is the
  *smaller* of {set, no-set}. More recent work shows the real requirement depends on
  prevalence, effect size, and desired precision, and is often lower or higher (van Smeden
  et al. 2016; Riley et al. 2020). Take 10 as a planning anchor, not a guarantee.
- **A two-parameter curve** (slope and intercept, or one fixed asymptote) needs on the
  order of 20 events of the rarer class. At a plausible pomegranate hand-pollination fruit
  set of about 30%, that is roughly **60–100 scored flowers**, *if* the visits span a real
  gradient.
- **A four-parameter curve** (lower and upper asymptote, midpoint, steepness) needs on the
  order of 40 events of the rarer class — roughly **150–250 scored flowers** — and,
  separately, enough flowers near the inflection and both plateaus to identify the
  asymptotes. Flowers bunched at one dose leave asymptotes unidentifiable at any count.
- **Random effects** cost too: estimating the plant- and date-level variances reliably
  wants on the order of 5–10 levels per grouping factor (Harrison et al. 2018) — several
  plants across several dates, not one plant sampled heavily.

**Note:** pomegranate currently has no controlled-visit flowers, so every number above is a
**design target for data yet to be collected**, not a description of data in hand. Until
such flowers exist, the honest deliverable is a curve *shape* borrowed under stated priors
(Section 4), not a pomegranate-specific fit. The arithmetic here is meant to size that
collection effort and to prevent fitting a four-parameter curve to a dozen flowers.

---

## 7. Spatial features

Alongside the temporal features of a visit sequence (timing, first-visit time, inter-visit
gaps), spatial features describe *where* the flower and insect are.

### 7.1 Feature list

| Feature | Definition | Signal it carries |
|---|---|---|
| Position within cluster | Apical / lateral / basal slot in the pomegranate cluster | Confounder for set (Section 2.2); apical flowers found and set first (Section 11) |
| Inter-flower distance | Distance to the nearest flower box (px, or cm if scale known) | Local density → competition / facilitation; drives autocorrelation (Section 7.2) |
| Camera-box geometry | Flower box area, aspect, distance from frame centre | Detection-reliability covariate — edge and small flowers have lower recall (Section 1); box size also proxies flower size (Section 11) |
| Overlap fraction | Fraction of the insect box overlapping the flower box | Graded contact-quality proxy (and a natural future qualifying gate) |
| Entry side / approach | Box edge the track crossed on entry | Weakly separates foraging approaches from fly-throughs |
| Cluster visit density | Visits per flower aggregated to the cluster | Plant-level exposure; pairs with the plant random effect (Section 2.3) |

The **overlap fraction** is worth emphasising: kept continuous it is a graded proxy for how
much of the insect was actually on the flower — and thus for contact with reproductive
parts — recovering some of the biological-versus-operational gap of Section 1.1. Thresholded,
the same quantity becomes a qualifying gate; retaining it as a feature (mean or maximum
overlap over a track's dwell) preserves information a threshold would discard.

### 7.2 Spatial autocorrelation and independence

Neighbouring flowers are not independent draws: they share microclimate, a pollinator's
sequential foraging path (a bee works nearby flowers in a bout), and — on the same plant —
genotype and resources. Both the dose (visits) and the response (fruit set) are therefore
spatially autocorrelated, which is the geometric face of the non-independence handled by the
random effects of Section 2.3. Quantify it with **Moran's I** on the model residuals (Moran
1950; Legendre 1993):

$$
I = \frac{N}{\sum_i \sum_j w_{ij}} \cdot
\frac{\sum_i \sum_j w_{ij}\,(y_i - \bar{y})(y_j - \bar{y})}{\sum_i (y_i - \bar{y})^2},
$$

where $w_{ij}$ is a spatial weight (inverse inter-flower distance, or 1 for neighbours
within radius $d$). $I \approx 0$ means residuals are spatially random (the random effects
and covariates absorbed the structure); $I > 0$ means leftover clustering — the effective
sample is smaller than the flower count, standard errors are still too small, and an
explicit spatial term is warranted (Dormann et al. 2007). The workflow: fit the GLMM of
Section 2.3, compute Moran's I on the residuals, and only if it is non-negligible add
spatial structure (a spatial random field, or a plant/cluster effect fine enough to absorb
the correlation).

**Note:** pixel coordinates from a moving, uncalibrated camera give *relative* geometry
only; inter-flower distance in real units needs a scale reference the pipeline does not
currently record. Until then, spatial features are usable *within* a clip (relative layout)
but not comparable *across* clips, and Moran's I should be computed per-clip, not pooled.

---

## 8. Notation glossary and model card

### 8.1 Notation

| Symbol | Meaning |
|---|---|
| $P_{\text{self}}$ | Fruit set under self-pollination (zero insect visits); the intercept's biological reading |
| $P_{\max}$ | Upper asymptote of the dose–response curve (maximum attainable fruit set) |
| $\beta_0$ | Logit-scale intercept (baseline log-odds of set); maps to $P_{\text{self}}$ |
| $\beta_1$ | Logit-scale slope of the linear visit effect |
| $x$ | True number of visits to a flower (the dose) |
| $w$ | Observed visit count from the pipeline, $w = x + u$ |
| $u$ | Dose measurement error |
| $\lambda$ | Reliability ratio $\sigma_x^2 / (\sigma_x^2 + \sigma_u^2)$; attenuation factor |
| $r,\ p$ | Detection/tracking recall and precision |
| $\mathrm{ED}_{50}$ | Dose (visits) giving half-maximal fruit set |
| $n$ | Hill coefficient (curve steepness) — not sample size |
| $p_i$ | Modelled fruit-set probability for flower $i$ |
| $u_{\text{plant}}, u_{\text{orchard}}, u_{\text{date}}$ | Random intercepts (grouping-level deviations) |
| $\sigma^2_\bullet$ | Variance components of the random effects |
| $I$ | Moran's I spatial autocorrelation statistic |
| $N_{\text{flowers}},\ \bar{m}$ | Flower count and mean fruit mass (yield inputs) |

### 8.2 Fruit-set model card

A specification and guardrail for the model to be built, to be filled with real numbers once
a pomegranate fit exists.

- **Model.** Binomial GLMM (Section 2.3): `fruitset ~ f(visits) + covariates + (1|plant) +
  (1|orchard) + (1|date)`, with $f$ a saturating curve family selected by penalized fit
  and/or Bayesian comparison (Section 4).
- **Response.** Per-flower fruit set (binary: set / no set); optionally seed count as a
  secondary response.
- **Primary predictor.** Per-flower pollinator visit count from `ALL_visits.csv`
  (`pollinator` = bee/butterfly/fly). Today this is a centre-in-box, once-ever-per-track
  count — the observed dose $w$, known to be attenuated and compressed at the high end
  (Section 1). Biology (Section 12) argues for splitting it into type-specific doses.
- **Covariates.** Flower age, position/cluster and flower size (Sections 7, 11), cultivar,
  local density, flower type (Section 9), and visit-window weather (Section 5).
- **Assumptions.** (1) Conditional independence given the random effects (Section 2.3); (2)
  the measured covariates capture the material confounders (Section 2.2) — untestable; (3)
  dose error is approximately classical and non-differential for the $\lambda$ correction to
  apply (Section 1.3, flagged as possibly false); (4) cross-crop priors constrain shape, not
  level (Section 4.2); (5) fruit-capable (bisexual) flowers are separated from functionally
  male ones (Section 9) — otherwise a large structural-zero fraction corrupts the fit.
- **Valid input range.** Interpolation only, within the observed visit range and the
  cultivars and dates sampled. Do not extrapolate past the largest observed dose (the upper
  asymptote is prior-driven, not data-driven, under small $n$) or to other crops.
- **Known failure modes.** Attenuated (understated) visit effect from tracking error
  (Section 1.3); separation / infinite coefficients under small $n$ (Section 3.4);
  zero-inflation and overdispersion from many zero-visit flowers, from ID switches, and from
  functionally male flowers (Sections 3.4, 9); singular random-effect fits with few plants;
  over-precise yield if intervals are dropped (Section 3.3).
- **Uncertainty.** Every estimate ships with an interval (bootstrap or posterior; Section 3).
- **Data provenance.** Visits and timings from `video_detect.py` → `ALL_visits.csv` /
  `ALL_timeline.csv`; detection/tracking performance from held-out CV evaluation; weather
  from the Open-Meteo feed (Section 5.3); pomegranate biology and any self-fertility prior
  from the literature reviewed in Part II.

**Note (status):** there is currently no fitted pomegranate curve. No controlled-visit data
exist, the modeling and reporting modules are still scaffolding, and today's visits come
from the centre-in-box, once-ever counter. This card specifies the model to be built, not
one that exists; its numbers are deliberately blank rather than invented.

---

# Part II — Pomegranate Reproductive Biology

A focused literature review of pomegranate floral and pollination biology, filling the
crop-specific gaps in Part I — the self-fertility offset ($P_{\text{self}}$), the sources of
structural zeros, and which insects actually pollinate. Where a value is cultivar- or
region-specific, or where no source covers this project's cultivar, it is marked **Note:**.
This part informs the dose–response model; it does not fit it.

**Why this matters.** The dose–response model (Section 2.3) relates per-flower visits to
fruit set. Pomegranate biology sets three things that model must respect and that the CV
pipeline does *not* currently observe: (1) a large fraction of flowers **cannot set fruit at
all**, independent of visits (functional andromonoecy — Section 9), a structural source of
the zero-inflation of Section 3.4; (2) fruit set at **zero insect visits is non-zero**
(partial self-fertility — Section 10), which fixes the intercept / $P_{\text{self}}$ prior
of Section 4.2; (3) **not every visitor is a pollinator** (Section 12), sharpening the
biological-vs-operational visit distinction of Section 1.1.

## 9. Functional andromonoecy: most flowers are structurally sterile

Pomegranate is **functionally andromonoecious**: a single plant bears two flower types —
**hermaphroditic (bisexual)** flowers that can set fruit, and **functionally male
(staminate)** flowers whose ovary and ovules are degenerate and which **cannot set fruit**
regardless of pollination (Wetzstein et al. 2011). The types are morphologically distinct:
bisexual flowers are urceolate ("vase-shaped") with a well-developed ovary; male flowers
are campanulate ("bell-shaped") and smaller. Critically, the **functionally male fraction
is large — commonly 60–70% of flowers, varying with cultivar and season** (Wetzstein et al.
2011).

**Modeling implication.** The flower detector in `video_detect.py` detects *flowers*, not
*flower type*. If 60–70% of detected flowers are functionally male, the majority of the
modeling rows have a fruit-set probability that is **structurally zero for reasons unrelated
to visit count**. Feeding these into a visit→fruit-set curve without a type label will
inflate the apparent zero-inflation (Section 3.4), attenuate the visit slope (a biological
analogue of Section 1.3), and bias $P_{\text{self}}$ downward.

> **Note (recommendation).** The model needs a **flower-type covariate** (bisexual vs.
> functionally male), or it must treat type as a latent zero-inflation component. The
> cleanest fix is upstream: add a bisexual-vs-male flower classifier (the two types are
> visually separable — vase vs. bell shape, size), so only fruit-capable flowers enter the
> dose–response. This is a concrete CV task not yet in the pipeline.

## 10. Breeding system: self-fertile, but insect pollination raises fruit set

Pomegranate is **partially self-compatible** — bisexual flowers set some fruit without
insect visits — but **cross- and insect-pollination consistently increase fruit set, seed
number, and fruit size** relative to selfing or pollinator exclusion (Derin & Eti 2001;
Chater et al. 2015; Holland et al. 2009). Reported figures (all cultivar- and
region-specific):

- **Cross-pollination** raised fruit set to roughly **68%** in 'Hicaz', well above
  self-pollination, with more seeds and larger fruit; pollen germination was ~61.5% (Derin
  & Eti 2001).
- **Self-pollination fruit set was consistently lower than open pollination** across
  cultivars, and supplementary/open pollination improved fruit growth and characteristics
  (Chater et al. 2015).
- Open/insect and supplementary pollination improved qualitative and physicochemical fruit
  attributes over strict self-pollination (recent multi-cultivar work).

**Modeling implication.** This is exactly the shape assumed in Section 4.2: a **non-zero
intercept** ($P_{\text{self}} > 0$) rising to a **higher plateau** ($P_{\max}$) with visits.
Pomegranate's partial self-fertility is what makes the intercept prior *pomegranate-specific*
and not borrowable from apple (largely self-incompatible).

> **Note (no canonical $P_{\text{self}}$).** The self-fertility level varies strongly by
> cultivar and environment, and the numbers above come from 'Hicaz' (Turkey), Californian,
> and Indian cultivars — **not** this project's cultivar, which is not yet specified. Use
> these to set a **wide, weakly-informative prior on the intercept** (e.g. $P_{\text{self}}$
> somewhere in a broad 20–50% band, cultivar-dependent), not a fixed constant. A
> pomegranate-specific $P_{\text{self}}$ needs bagging/self-pollination data from *our*
> cultivar (Section 6.3).

## 11. Flower position and size: a strong, observable fruit-set driver

Within a cluster, **single and terminal flowers are larger, carry more ovules, and set far
more fruit than lateral flowers** (Wetzstein et al. 2013). Specifically: **49% of lateral
flowers had abortive internal ovary tissue** (vs. 10% of single and 7% of terminal
flowers), and under hand pollination the **largest flowers exceeded 90% fruit set while the
smallest set only 12–20%** (Wetzstein et al. 2013).

**Modeling implication.** This validates the **position-within-cluster** and **flower-size**
covariates of Sections 2.2 and 7. Two of these are *already measurable from the video*:
flower **box size** is a direct pipeline output, and **position within cluster** is
recoverable from the spatial features of Section 7.1. Flower size is therefore a rare case
of a confounder the camera can observe directly — it should be a first-class covariate, and
it partially proxies the unobserved bisexual-vs-male distinction of Section 9 (male flowers
are smaller).

## 12. Effective pollinators vs. mere visitors

The insects that *visit* pomegranate are not equally effective at *pollinating* it. Across
regional pollinator surveys, the effective pollinators are **bees** — honey bees (*Apis
dorsata*, *A. mellifera*, *A. florea*, *A. cerana*) and stingless bees (*Tetragonula*) —
which carry and deposit pomegranate pollen. **Lepidopterans (butterflies) are frequent
flower visitors but poor pollinators** of pomegranate, and pollinator exclusion (bagging)
lowers fruit set relative to open flowers, confirming insect dependence for maximal set.

**Modeling implication — the `pollinator` rollup is too generous.** The pipeline defines
`pollinator = {bee, butterfly, fly}` and rolls these together. The biology says **butterfly
visits should not carry the same pollination weight as bee visits**, and flies are
intermediate. Counting a butterfly visit as equivalent to a bee visit adds low-information
events to the dose — a systematic, *biological* measurement error on top of the tracking
errors of Section 1.

> **Note (recommendation).** Two options, in order of rigor: (a) **model type-specific
> doses** — separate bee, fly, and butterfly visit counts as distinct predictors, and let
> the fit estimate their relative effectiveness (a data-driven single-visit-deposition
> weight); or (b) **re-weight the rollup** so `pollinator` reflects known effectiveness
> (bees ≫ flies > butterflies) rather than a flat sum. Option (a) is preferred because it
> *measures* the weights instead of assuming them, and the `by_type` columns already in
> `ALL_visits.csv` make it free to try.

## 13. Pollen viability and weather

Pomegranate pollen germination and tube growth are temperature-sensitive, and pollen
viability varies with cultivar and flowering flush (Derin & Eti 2001 report ~61.5%
germination in 'Hicaz'; viability differs between the multiple bloom flushes pomegranate
produces per season). This is the biological basis for the temperature/humidity covariates
of Section 5: weather affects fruit set partly *through pollen viability*, independently of
how many bees visited.

> **Note.** Pomegranate flowers in **multiple flushes** across a season, and early vs. late
> flushes differ in fruit-set potential and fruit quality. "Date / bloom day" (Section 2.2)
> is therefore not just a random-effect nuisance but a biologically real covariate; if
> flush can be identified, it is worth recording explicitly.

## 14. Summary: what the biology fixes in the model

| Biological fact | Source | Model consequence |
|---|---|---|
| 60–70% of flowers functionally male, cannot set fruit | Wetzstein et al. 2011 | Structural zeros → flower-type covariate/classifier; zero-inflation (Section 3.4), slope attenuation (Section 1.3) |
| Partial self-fertility; insect pollination raises set | Derin & Eti 2001; Chater et al. 2015 | Non-zero intercept $P_{\text{self}}$; cross-crop-inadmissible prior (Section 4.2) |
| Lateral flowers abort (49%); size drives set 12–20% → >90% | Wetzstein et al. 2013 | Flower size + cluster position covariates, camera-observable (Sections 2.2, 7) |
| Bees pollinate; butterflies visit but pollinate poorly | Regional entomophily surveys | `pollinator` rollup too flat → type-specific doses (Section 1.1) |
| Pollen viability temperature-dependent; multiple flushes | Derin & Eti 2001; Holland et al. 2009 | Weather-through-viability pathway (Section 5); bloom flush as real covariate (Section 2.2) |

> **Note (overall honesty).** These findings come from Turkish, Californian, Indian, and
> Mediterranean cultivars and growing conditions. None is from this project's cultivar or
> region, and none provides a validated visit→fruit-set *curve* for pomegranate (that curve
> does not exist in the literature — the two-headed problem of Section 4). This review
> constrains **priors and covariate structure**; it does not substitute for pomegranate
> fruit-set data collected from the orchards this project actually films.

---

# Part III — Implementation Plan (`src/ml_models/`)

How to turn the research above (Sections 1–14) into code under `src/ml_models/`: the
prerequisite data, the tooling choice and its trade-offs, the module layout, and a build
order. This is an engineering plan, not a fit; the honest gating condition — that no
fruit-set labels exist yet — is stated up front.

## 15. Current state

- `src/ml_models/` contains only empty scaffolding (`train.py`, `visit_dataset.py`,
  `__init__.py`) on every branch.
- The project `.venv` has **only `scikit-learn` and `scipy`** for modeling — no
  `statsmodels`, no Bayesian stack. Adding dependencies is therefore step one of any real
  work.
- The visit signal (`ALL_visits.csv`, `ALL_timeline.csv` from `video_detect.py`) exists;
  the **response variable (fruit set) does not**, and neither does the flower-type label
  that Section 9 shows is essential.

## 16. Prerequisite data (the real blocker)

No modeling can produce a pomegranate curve until these exist. This is not a tooling
problem; it is a data-collection problem.

| Needed | Why | Status |
|---|---|---|
| **Per-flower fruit set** (set / no-set, scored weeks after bloom) | The response variable $y$ | Not collected |
| **Flower type** (bisexual vs. functionally male) | 60–70% of flowers can't set fruit (Section 9); without it the curve is corrupted | Not collected / not detected |
| **Flower identity linkage** | To join a visit count to the *same* flower's later fruit set | Flower IDs are per-clip only; no cross-time identity |
| **Cultivar, plant, date, orchard keys** | Random effects + confounders (Section 2) | Partially available (per video) |
| **Controlled-visit flowers** (bagging + known visits) | The only route to a *causal* curve (Section 2.1) and an unattenuated dose | Not collected |
| **Weather at clip time** | Environmental covariates (Section 5) | Available via Open-Meteo (separate work) |

> **Note.** Until fruit-set labels exist, the modeling code can still be **written and
> validated on simulated data** (generate flowers with a known curve, known self-fertility,
> known male fraction, and known detection error, then check the pipeline recovers the
> parameters). This is the right interim deliverable and doubles as the measurement-error
> study of Section 1.4.

## 17. Tooling choice

Python's support for binomial **GLMMs with crossed random effects** and for **nonlinear
dose–response with informative priors** is thinner than R's. The choice below reflects that
Section 4 recommends a **Bayesian** path for pomegranate's small-sample, no-anchor regime —
which is also where Python is strongest.

| Task | Recommended tool | Notes |
|---|---|---|
| Baselines: predict-the-mean, simple logistic GLM (Section 6.1) | **statsmodels** (`GLM`, `Logit`) | Add to deps; also gives GEE and a variational binomial mixed GLM |
| Frequentist nonlinear curves: Hill / log-logistic + AIC/BIC (Section 6.2) | **scipy** (`optimize.curve_fit`) | Already installed; wrap for AIC/BIC and profile CIs |
| Separation / small-n logistic (Section 3.4) | **`firthlogist`** (Firth penalized logistic) | Small dependency; prevents infinite coefficients |
| **Primary model**: binomial GLMM + nonlinear dose + cross-crop priors + measurement-error layer (Sections 2.3, 4, 1.4) | **PyMC + ArviZ**, optionally via **Bambi** | Handles crossed random effects, informative priors, zero-inflation, and errors-in-variables as layers; posteriors give credible intervals for free (Section 3.2) |
| Bootstrap CIs, cluster resampling, yield propagation (Sections 3.2–3.3) | **scipy** + numpy | `scipy.stats.bootstrap`; cluster (plant/clip) resampling by hand |
| Spatial autocorrelation: Moran's I (Section 7.2) | **`esda`/`libpysal`** or a short numpy implementation | Optional; Moran's I is ~10 lines if avoiding the dependency |
| Metrics: Brier, log-loss, calibration, AUC (Section 3) | **scikit-learn** | Already installed |

**Why not R via `rpy2` (`lme4`/`glmmTMB`/`brms`/`drc`)?** R is the gold standard for GLMM
and dose–response and would be the choice for a frequentist GLMM-heavy plan. It is
*deliberately avoided here* to keep the project single-language and reproducible in the
existing `.venv`; PyMC/Bambi covers the recommended Bayesian path natively. Revisit only if
a frequentist crossed-random-effects GLMM becomes the primary deliverable.

**Dependencies to add** (`requirements.txt` or a new `requirements-ml.txt`):
`statsmodels`, `pymc`, `arviz`, `bambi`, `firthlogist` (and optionally `libpysal`/`esda`).
Keep them out of `requirements-cv.txt` so the CV and modeling environments stay separable.

## 18. Module layout for `src/ml_models/`

```
src/ml_models/
  visit_dataset.py      # load ALL_visits.csv + ALL_timeline.csv, join fruit-set labels,
                        #   flower-type, cultivar/plant/date keys, weather -> modeling frame
  features.py           # dose (type-specific visit counts), temporal, spatial (Section 7),
                        #   environmental (Section 5) feature construction
  baselines.py          # predict-the-mean, simple logistic, threshold model (Section 6.1)
  dose_response.py      # frequentist Hill/log-logistic via curve_fit + AIC/BIC (Section 6.2)
  glmm.py               # binomial GLMM (statsmodels / Bambi): random intercepts (Section 2.3)
  bayesian.py           # PyMC/Bambi model: nonlinear dose + cross-crop priors (Section 4) +
                        #   optional measurement-error layer (Section 1.4); prior-sensitivity
  uncertainty.py        # bootstrap (cluster-level) + prediction intervals + yield
                        #   propagation (Sections 3.2-3.3)
  diagnostics.py        # separation check, overdispersion, Moran's I on residuals (Section
                        #   7.2), calibration (Section 3.4)
  evaluate.py           # grouped / leave-one-plant-out CV; metrics vs. baselines
  simulate.py           # synthetic flowers with known curve + male fraction + detection
                        #   error, to validate the code before real labels exist (Section 1)
  train.py              # CLI orchestrator: load -> baselines -> curves -> GLMM/Bayesian ->
                        #   uncertainty -> diagnostics -> report inputs
```

## 19. Build order

Each step produces something runnable and is validated against `simulate.py` output before
touching (eventual) real data.

1. **`simulate.py` + `visit_dataset.py`** — a synthetic generator and the real-data
   loader sharing one schema. This unblocks everything else without waiting for labels.
2. **`baselines.py` + `evaluate.py`** — predict-the-mean and simple logistic under grouped
   CV, reporting Δlog-loss. Nothing more complex ships until it beats these (Section 6.1).
3. **`dose_response.py`** — frequentist Hill/log-logistic with AIC/BIC; confirm on
   simulated data that it recovers the known curve, and that attenuation appears when
   detection error is injected (Section 1.3).
4. **`glmm.py`** — add plant/date random intercepts; check variance components recover the
   simulated non-independence (Section 2.3).
5. **`bayesian.py`** — the primary model: nonlinear dose + weakly-informative cross-crop
   priors (Section 4.2) + measurement-error layer (Section 1.4); run the mandatory
   prior-sensitivity check.
6. **`uncertainty.py`** — bootstrap/posterior intervals and Monte-Carlo yield propagation;
   verify coverage on simulated data (do 95% intervals contain truth ~95% of the time?).
7. **`diagnostics.py`** — separation, overdispersion, Moran's I, calibration as gating
   checks before any fit is reported.
8. **`train.py`** — wire it together and emit the interval-carrying fields the report layer
   consumes (Section 3.5).

## 20. Honesty notes

> **Modeling cannot start on real data yet.** The blocker is Section 16 (no fruit-set
> labels, no flower-type labels, no cross-time flower identity), not tooling. The
> highest-value work available *now* is `simulate.py` + the baseline/curve/uncertainty code
> validated against it — which also quantifies how much the measurement error of Section 1
> will bias a future real fit.

> **Two upstream CV tasks block the clean model**, both flagged in Part II: a
> **bisexual-vs-male flower classifier** (Section 9) and **cross-time flower identity** so a
> visit count joins to the same flower's later fruit set. Without them, even perfect
> fruit-set labels cannot be modeled correctly.

> **The `pollinator` rollup should become type-specific** (Section 12): build `features.py`
> to expose separate bee/fly/butterfly visit counts from the existing `by_type` columns, so
> the fit can estimate their relative pollination effectiveness rather than assuming a flat
> sum.

---

## References

- Bernardin, K., & Stiefelhagen, R. (2008). Evaluating multiple object tracking performance:
  the CLEAR MOT metrics. *EURASIP Journal on Image and Video Processing*, 2008, 246309.
  https://doi.org/10.1155/2008/246309
- Bolker, B. M., Brooks, M. E., Clark, C. J., Geange, S. W., Poulsen, J. R., Stevens,
  M. H. H., & White, J.-S. S. (2009). Generalized linear mixed models: a practical guide for
  ecology and evolution. *Trends in Ecology & Evolution*, 24(3), 127–135.
  https://doi.org/10.1016/j.tree.2008.10.008
- Carroll, R. J., Ruppert, D., Stefanski, L. A., & Crainiceanu, C. M. (2006). *Measurement
  Error in Nonlinear Models: A Modern Perspective* (2nd ed.). Chapman & Hall/CRC.
  https://doi.org/10.1201/9781420010138
- Chater, J. M., Merhaut, D. J., Jia, Z., Mauk, P. A., & Preece, J. E. (2015). Effects of
  self, open, and supplementary pollination on growth pattern and characteristics of
  pomegranate fruit. *International Journal of Fruit Science*, 15(4), 435–447.
  https://doi.org/10.1080/15538362.2015.1009974
- Cook, J. R., & Stefanski, L. A. (1994). Simulation–extrapolation estimation in parametric
  measurement error models. *Journal of the American Statistical Association*, 89(428),
  1314–1328. https://doi.org/10.1080/01621459.1994.10476871
- Derin, K., & Eti, S. (2001). Determination of pollen quality, quantity and effect of
  cross pollination on the fruit set and quality in the pomegranate. *Turkish Journal of
  Agriculture and Forestry*, 25(3), 169–173.
  https://journals.tubitak.gov.tr/agriculture/vol25/iss3/
- DiCiccio, T. J., & Efron, B. (1996). Bootstrap confidence intervals. *Statistical Science*,
  11(3), 189–228. https://doi.org/10.1214/ss/1032280214
- Dormann, C. F., McPherson, J. M., Araújo, M. B., Bivand, R., Bolliger, J., Carl, G., et al.
  (2007). Methods to account for spatial autocorrelation in the analysis of species
  distributional data: a review. *Ecography*, 30(5), 609–628.
  https://doi.org/10.1111/j.2007.0906-7590.05171.x
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman &
  Hall/CRC. https://doi.org/10.1201/9780429246593
- Firth, D. (1993). Bias reduction of maximum likelihood estimates. *Biometrika*, 80(1),
  27–38. https://doi.org/10.1093/biomet/80.1.27
- Fuller, W. A. (1987). *Measurement Error Models*. Wiley.
  https://doi.org/10.1002/9780470316665
- Harrison, X. A., Donaldson, L., Correa-Cano, M. E., Evans, J., Fisher, D. N., Goodwin,
  C. E. D., Robinson, B. S., Hodgson, D. J., & Inger, R. (2018). A brief introduction to
  mixed effects modelling and multi-model inference in ecology. *PeerJ*, 6, e4794.
  https://doi.org/10.7717/peerj.4794
- Heinze, G., & Schemper, M. (2002). A solution to the problem of separation in logistic
  regression. *Statistics in Medicine*, 21(16), 2409–2419. https://doi.org/10.1002/sim.1047
- Holland, D., Hatib, K., & Bar-Ya'akov, I. (2009). Pomegranate: botany, horticulture,
  breeding. *Horticultural Reviews*, 35, 127–191. https://doi.org/10.1002/9780470593776.ch2
- King, C., Ballantyne, G., & Willmer, P. G. (2013). Why flower visitation is a poor proxy
  for pollination: measuring single-visit pollen deposition, with implications for
  pollination networks and conservation. *Methods in Ecology and Evolution*, 4(9), 811–818.
  https://doi.org/10.1111/2041-210X.12074
- Legendre, P. (1993). Spatial autocorrelation: trouble or new paradigm? *Ecology*, 74(6),
  1659–1673. https://doi.org/10.2307/1939924
- Moran, P. A. P. (1950). Notes on continuous stochastic phenomena. *Biometrika*, 37(1–2),
  17–23. https://doi.org/10.1093/biomet/37.1-2.17
- Ne'eman, G., Jürgens, A., Newstrom-Lloyd, L., Potts, S. G., & Dafni, A. (2010). A framework
  for comparing pollinator performance: effectiveness and efficiency. *Biological Reviews*,
  85(3), 435–451. https://doi.org/10.1111/j.1469-185X.2009.00108.x
- Peduzzi, P., Concato, J., Kemper, E., Holford, T. R., & Feinstein, A. R. (1996). A
  simulation study of the number of events per variable in logistic regression analysis.
  *Journal of Clinical Epidemiology*, 49(12), 1373–1379.
  https://doi.org/10.1016/S0895-4356(96)00236-3
- Riley, R. D., Snell, K. I. E., Ensor, J., Burke, D. L., Harrell, F. E., Moons, K. G. M., &
  Collins, G. S. (2020). Calculating the sample size required for developing a clinical
  prediction model. *BMJ*, 368, m441. https://doi.org/10.1136/bmj.m441
- Ristani, E., Solera, F., Zou, R., Cucchiara, R., & Tomasi, C. (2016). Performance measures
  and a data set for multi-target, multi-camera tracking. In *ECCV 2016 Workshops*, 17–35.
  https://doi.org/10.1007/978-3-319-48881-3_2
- Ritz, C., Baty, F., Streibig, J. C., & Gerhard, D. (2015). Dose–response analysis using R.
  *PLoS ONE*, 10(12), e0146021. https://doi.org/10.1371/journal.pone.0146021
- van Smeden, M., de Groot, J. A. H., Moons, K. G. M., Collins, G. S., Altman, D. G.,
  Eijkemans, M. J. C., & Reitsma, J. B. (2016). No rationale for 1 variable per 10 events
  criterion for binary logistic regression analysis. *BMC Medical Research Methodology*, 16,
  163. https://doi.org/10.1186/s12874-016-0267-3
- Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian model evaluation using
  leave-one-out cross-validation and WAIC. *Statistics and Computing*, 27(5), 1413–1432.
  https://doi.org/10.1007/s11222-016-9696-4
- Wetzstein, H. Y., Ravid, N., Wilkins, E., & Martinelli, A. P. (2011). A morphological and
  histological characterization of bisexual and male flower types in pomegranate. *Journal
  of the American Society for Horticultural Science*, 136(2), 83–92.
  https://doi.org/10.21273/JASHS.136.2.83
- Wetzstein, H. Y., Yi, W., Porter, J. A., & Ravid, N. (2013). Flower position and size
  impact ovule number per flower, fruitset, and fruit size in pomegranate. *Journal of the
  American Society for Horticultural Science*, 138(3), 159–166.
  https://doi.org/10.21273/JASHS.138.3.159
- Zuur, A. F., Ieno, E. N., Walker, N. J., Saveliev, A. A., & Smith, G. M. (2009). *Mixed
  Effects Models and Extensions in Ecology with R*. Springer.
  https://doi.org/10.1007/978-0-387-87458-6
- Aharon, N., Orfaig, R., & Bobrovsky, B.-Z. (2022). BoT-SORT: robust associations
  multi-pedestrian tracking. *arXiv:2206.14651*. https://arxiv.org/abs/2206.14651

> **Note (sources).** Regional pollinator-effectiveness figures (e.g. *Apis dorsata* as the
> primary visitor; butterflies as frequent-but-poor pollinators; open/bee vs. bagged fruit
> set) are drawn from South-Asian entomophily surveys of pomegranate available largely as
> grey literature and regional-journal articles; they are cited qualitatively in Section 12
> and should be re-verified against a primary source before any figure is quoted in a
> farmer-facing report.
