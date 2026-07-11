"""
generate_bee_data.py
=====================
Multi-crop synthetic bee-visit dataset generator for the ECLIPSE / "Bee a Hero"
project. Produces a *noisy, real-life-shaped* synthetic dataset of insect visits
to flowers of multiple crops across 20 years, mirroring the labels our computer-
vision pipeline (YOLO detection + BoT-SORT tracking) emits on test-video clips,
so the pollination / fruit-set models can be built and validated before the full
real dataset exists.

The noise is deliberate and comes from many independent sources, exactly as in
the field:
  * lognormal dwell times (per species),
  * fly-bys / mis-tracks that fail the velocity gate,
  * partial-overlap tracks that fail the fraction-on gate,
  * short contacts that fail the dwell gate,
  * qualifying visits that still transfer no pollen (no reproductive contact),
  * visits outside the stigma-receptive window (logged, transfer nothing),
  * and a *Bernoulli* fruit-set draw, so identical dose gives different outcomes.

Outputs (written to OUT_DIR):
  visits.csv            one row per visit  (main CV-style label file)
  flowers.csv           one row per flower per year (dose V, FruitSet(V), 0/1)
  clips.csv             one row per synthetic video clip (test-video metadata)
  raw_labels_sample.csv MOT-style per-frame rows for a small sample of clips
  daily_summary.csv     per (crop, year, day): volumes, active bees, pollination

Run:  python generate_bee_data.py
Deps: standard library + numpy + pandas only.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

# ===========================================================================
# CONFIG BLOCK  (edit these)
# ===========================================================================
SEED = 42
N_YEARS = 20
START_YEAR = 2005
N_FLOWERS_PER_YEAR_PER_CROP = 2000
MEAN_VISITS_PER_FLOWER_SEASON = 15

# --- CV gate thresholds (match your CV export's units!) --------------------
DWELL_MIN = 2.0        # s   ; a real feeding visit dwells >= this
VEL_MAX = 0.05         # frac-of-frame / s ; slow = real, fast = fly-by/mis-track
FRAC_MIN = 0.60        # fraction of tracked frames overlapping the flower bbox
IOU_MIN = 0.50         # IoU threshold used to decide "overlapping" in raw labels
TAU = 5.0              # s   ; dwell-saturation constant for per-visit pollen
FPS = 30
N_RAW_SAMPLE_CLIPS = 20

# --- population structure --------------------------------------------------
N_BEES_PER_YEAR_PER_CROP = 150
N_COLONIES_PER_YEAR_PER_CROP = 10
N_ORCHARDS_PER_CROP = 3
N_BLOCKS_PER_ORCHARD = 4
N_CAMERAS_PER_BLOCK = 2

# --- noise knobs -----------------------------------------------------------
P_FLYBY = 0.15                 # fraction of tracks that are fast fly-bys/mis-tracks
DWELL_SIGMA_LOG = 0.5          # spread of the lognormal dwell distribution
VEL_SLOW_MEAN = 0.020          # mean velocity of a true (slow) visit
VEL_FAST_LO, VEL_FAST_HI = 0.08, 0.30   # fly-by velocity range (fails vel gate)
FRAC_BETA_A, FRAC_BETA_B = 6.0, 1.5     # Beta(a,b) for fraction_on (skewed high)

FRAME_W, FRAME_H = 1920, 1080

# --- output location -------------------------------------------------------
# repo_root / data / synthetic   (script lives in repo_root/src/ml_models/)
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "synthetic"


# ===========================================================================
# CROP TABLE  (edit / extend — melon, pumpkin, watermelon can be added)
# Numbers: pomegranate = ECLIPSE project; cucumber = Bomfim et al. 2016 chapter.
# k = 3 / V*  so a flower getting ~V* effective visits reaches ~95% of Fmax.
# ===========================================================================
CROPS = [
    dict(
        crop="pomegranate", season="spring/early-summer", region="Goychay",
        bloom_doy=(105, 181),                 # mid-Apr .. Jun
        anthesis=(6.0, 19.0),                 # daily open window (hours)
        day_peak=(10.0, 14.0),                # within-day activity peak
        receptive=(7.0, 16.0),                # stigma-receptive window
        V_star=8.0, F0=0.45, Fmax=0.95,
    ),
    dict(
        crop="cucumber", season="summer", region="Lankaran",
        bloom_doy=(152, 243),                 # Jun .. Aug
        anthesis=(6.0, 14.0),                 # ~7 h (Table 11.1)
        day_peak=(7.0, 10.0),                 # cucurbit pollen scarce after midday
        receptive=(6.0, 14.0),                # receptive ~06:00-14:00
        V_star=18.0, F0=0.05, Fmax=0.95,      # V* from Table 11.2
    ),
]
for _c in CROPS:
    _c["k"] = 3.0 / _c["V_star"]              # k = 3 / V*

# ===========================================================================
# SPECIES TABLE  (global mix; per-visit effectiveness)
# ===========================================================================
SPECIES = [
    # name,                     weight, dwell_mean_s, effectiveness, p_no_contact, is_bee
    ("honeybee",                0.55,   5.0,          0.80,          0.08,         True),
    ("bumblebee",               0.20,   8.0,          0.95,          0.05,         True),
    ("squash_solitary_bee",     0.10,   4.0,          0.75,          0.10,         True),
    ("stingless_bee",           0.05,   4.0,          0.70,          0.10,         True),
    ("non_bee_insect",          0.10,   3.0,          0.20,          0.50,         False),
]
SP_NAME = np.array([s[0] for s in SPECIES])
SP_WEIGHT = np.array([s[1] for s in SPECIES], dtype=float)
SP_WEIGHT = SP_WEIGHT / SP_WEIGHT.sum()
SP_DWELL = np.array([s[2] for s in SPECIES], dtype=float)
SP_EFF = np.array([s[3] for s in SPECIES], dtype=float)
SP_NOCONTACT = np.array([s[4] for s in SPECIES], dtype=float)
SP_ISBEE = np.array([s[5] for s in SPECIES], dtype=bool)


# ===========================================================================
# HELPERS
# ===========================================================================
def daylight_hours(doy: np.ndarray) -> np.ndarray:
    """Photoperiod at ~40 deg N, solar noon 12:00."""
    return 12.0 + 4.5 * np.sin(2.0 * np.pi * (doy - 81) / 365.0)


def sunrise_sunset(doy: np.ndarray):
    dl = daylight_hours(doy)
    return 12.0 - dl / 2.0, 12.0 + dl / 2.0


def trunc_normal(rng, mean, sd, lo, hi, size):
    """Vectorized truncated normal via rejection, then a final clip.
    mean/sd/lo/hi may be scalars or arrays broadcastable to `size`."""
    mean = np.broadcast_to(np.asarray(mean, float), size).astype(float).copy()
    sd = np.broadcast_to(np.asarray(sd, float), size).astype(float).copy()
    lo = np.broadcast_to(np.asarray(lo, float), size).astype(float).copy()
    hi = np.broadcast_to(np.asarray(hi, float), size).astype(float).copy()
    out = rng.normal(mean, sd)
    for _ in range(12):
        bad = (out < lo) | (out > hi)
        if not bad.any():
            break
        out[bad] = rng.normal(mean[bad], sd[bad])
    return np.clip(out, lo, hi)


def hours_to_hms(hf: np.ndarray):
    """Fractional hours -> integer (H, M, S)."""
    hf = np.clip(hf, 0.0, 23.999)
    H = np.floor(hf).astype(int)
    m = (hf - H) * 60.0
    M = np.floor(m).astype(int)
    S = np.floor((m - M) * 60.0).astype(int)
    return H, M, S


def iou(b1, b2):
    """IoU of two [x, y, w, h] boxes (top-left origin)."""
    ax1, ay1, aw, ah = b1
    bx1, by1, bw, bh = b2
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


# ===========================================================================
# MAIN GENERATION
# ===========================================================================
def generate():
    rng = np.random.default_rng(SEED)
    years = list(range(START_YEAR, START_YEAR + N_YEARS))

    vcols = {c: [] for c in [
        "video_id", "crop", "season", "year", "day_of_year", "hour",
        "H", "M", "S", "bee_id", "species", "flower_id",
        "dwell_seconds", "velocity", "fraction_on",
        "gate_dwell", "gate_velocity", "gate_fraction", "qualifying_visit",
        "stigma_receptive", "pollinator_contact", "pollination_prob",
        "colony_id", "orchard_id", "block_id", "camera_id", "region",
    ]}
    flower_records = []
    global_flower_idx = 0

    for crop in CROPS:
        cname = crop["crop"]
        d0, d1 = crop["bloom_doy"]
        a0, a1 = crop["anthesis"]
        pk = 0.5 * (crop["day_peak"][0] + crop["day_peak"][1])
        r0, r1 = crop["receptive"]

        for yi, year in enumerate(years):
            # ---- bee population for this crop-year (species + colony fixed) ----
            bee_sp_idx = rng.choice(len(SPECIES), size=N_BEES_PER_YEAR_PER_CROP, p=SP_WEIGHT)
            bee_colony = rng.integers(0, N_COLONIES_PER_YEAR_PER_CROP, N_BEES_PER_YEAR_PER_CROP)
            bee_id_arr = np.array(
                [f"BEE_{cname[:3]}_{yi:02d}_{b:05d}" for b in range(N_BEES_PER_YEAR_PER_CROP)]
            )
            colony_id_arr = np.array([f"COL_{cname[:3]}_{yi:02d}_{c:02d}" for c in bee_colony])

            # ---- flowers for this crop-year, with grouping keys ----
            n_f = N_FLOWERS_PER_YEAR_PER_CROP
            f_gids = np.arange(global_flower_idx, global_flower_idx + n_f)
            global_flower_idx += n_f
            flower_id_arr = np.array([f"FLW_{g:07d}" for g in f_gids])
            f_orch = rng.integers(0, N_ORCHARDS_PER_CROP, n_f)
            f_block = rng.integers(0, N_BLOCKS_PER_ORCHARD, n_f)
            f_cam = rng.integers(0, N_CAMERAS_PER_BLOCK, n_f)
            orch_id = np.array([f"ORCH_{cname[:3]}_{o:02d}" for o in f_orch])
            block_id = np.array([f"BLK_{cname[:3]}_{o:02d}_{b:02d}" for o, b in zip(f_orch, f_block)])
            cam_id = np.array([f"CAM_{cname[:3]}_{o:02d}_{b:02d}_{c:02d}"
                               for o, b, c in zip(f_orch, f_block, f_cam)])

            # ---- number of visits per flower, then expand to per-visit index ----
            n_visits_f = rng.poisson(MEAN_VISITS_PER_FLOWER_SEASON, n_f)
            total = int(n_visits_f.sum())
            if total == 0:
                continue
            vf = np.repeat(np.arange(n_f), n_visits_f)   # flower index per visit

            # ---- day-of-year: seasonal bell inside the bloom window ----
            center = 0.5 * (d0 + d1)
            sd_day = (d1 - d0) / 4.0
            doy = np.round(trunc_normal(rng, center, sd_day, d0, d1, total)).astype(int)

            # ---- within-day time: daylight  AND  anthesis, peaked at crop peak ----
            sr, ss = sunrise_sunset(doy.astype(float))
            eff_start = np.maximum(sr, a0)
            eff_end = np.minimum(ss, a1)
            eff_end = np.maximum(eff_end, eff_start + 0.5)   # guard degenerate window
            sd_time = (eff_end - eff_start) / 4.0
            peak = np.clip(pk, eff_start, eff_end)
            hf = trunc_normal(rng, peak, sd_time, eff_start, eff_end, total)
            H, M, S = hours_to_hms(hf)

            # ---- which bee (=> species, colony) ----
            bidx = rng.integers(0, N_BEES_PER_YEAR_PER_CROP, total)
            sp_idx = bee_sp_idx[bidx]

            # ---- dwell: lognormal per species (noise source #1) ----
            mu = np.log(SP_DWELL[sp_idx]) - 0.5 * DWELL_SIGMA_LOG ** 2
            dwell = np.exp(rng.normal(mu, DWELL_SIGMA_LOG))
            dwell = np.clip(dwell, 0.3, 60.0)

            # ---- velocity: true=slow, fly-by=fast (noise source #2) ----
            flyby = rng.random(total) < P_FLYBY
            vel_slow = np.exp(rng.normal(np.log(VEL_SLOW_MEAN) - 0.5 * 0.4 ** 2, 0.4, total))
            vel_fast = rng.uniform(VEL_FAST_LO, VEL_FAST_HI, total)
            velocity = np.where(flyby, vel_fast, vel_slow)

            # ---- fraction_on: skewed high, some partial tracks fail (noise #3) ----
            fraction_on = rng.beta(FRAC_BETA_A, FRAC_BETA_B, total)

            # ---- three gates ----
            g_dwell = dwell >= DWELL_MIN
            g_vel = velocity <= VEL_MAX
            g_frac = fraction_on >= FRAC_MIN
            qualifying = g_dwell & g_vel & g_frac

            # ---- stigma receptive? ----
            receptive = (hf >= r0) & (hf < r1)

            # ---- pollinator contact: some qualifying visits still transfer none ----
            contact = rng.random(total) >= SP_NOCONTACT[sp_idx]

            # ---- per-visit pollination probability (saturating in dwell) ----
            valid = qualifying & receptive & contact
            poll = SP_EFF[sp_idx] * (1.0 - np.exp(-dwell / TAU))
            pollination_prob = np.where(valid, poll, 0.0)

            # ---- clip id: one clip per (flower, date) that has >=1 visit ----
            video_id = np.array([f"VID_{cname[:3]}_{year}_{doy[i]:03d}_{flower_id_arr[vf[i]]}"
                                 for i in range(total)])

            # ---- accumulate visit columns ----
            vcols["video_id"].append(video_id)
            vcols["crop"].append(np.full(total, cname))
            vcols["season"].append(np.full(total, crop["season"]))
            vcols["year"].append(np.full(total, year))
            vcols["day_of_year"].append(doy)
            vcols["hour"].append(H)
            vcols["H"].append(H); vcols["M"].append(M); vcols["S"].append(S)
            vcols["bee_id"].append(bee_id_arr[bidx])
            vcols["species"].append(SP_NAME[sp_idx])
            vcols["flower_id"].append(flower_id_arr[vf])
            vcols["dwell_seconds"].append(dwell)
            vcols["velocity"].append(velocity)
            vcols["fraction_on"].append(fraction_on)
            vcols["gate_dwell"].append(g_dwell)
            vcols["gate_velocity"].append(g_vel)
            vcols["gate_fraction"].append(g_frac)
            vcols["qualifying_visit"].append(qualifying)
            vcols["stigma_receptive"].append(receptive)
            vcols["pollinator_contact"].append(contact)
            vcols["pollination_prob"].append(pollination_prob)
            vcols["colony_id"].append(colony_id_arr[bidx])
            vcols["orchard_id"].append(orch_id[vf])
            vcols["block_id"].append(block_id[vf])
            vcols["camera_id"].append(cam_id[vf])
            vcols["region"].append(np.full(total, crop["region"]))

            # ---- per-flower dose V and fruit set (noise source #4: Bernoulli) ----
            V = np.zeros(n_f)
            np.add.at(V, vf, pollination_prob)
            nq = np.zeros(n_f)
            np.add.at(nq, vf, qualifying.astype(float))
            fs_prob = crop["F0"] + (crop["Fmax"] - crop["F0"]) * (1.0 - np.exp(-crop["k"] * V))
            fruit_set = (rng.random(n_f) < fs_prob).astype(int)
            for j in range(n_f):
                flower_records.append((
                    flower_id_arr[j], cname, year, crop["region"], orch_id[j],
                    block_id[j], cam_id[j], int(n_visits_f[j]), int(nq[j]),
                    float(V[j]), float(fs_prob[j]), int(fruit_set[j]),
                ))

    # ---- assemble visits dataframe ----
    v = {c: np.concatenate(vcols[c]) for c in vcols}
    n_total = len(v["crop"])
    visits = pd.DataFrame({
        "visit_id": [f"V_{i:09d}" for i in range(n_total)],
        "video_id": v["video_id"],
        "crop": v["crop"], "season": v["season"], "year": v["year"],
        "day_of_year": v["day_of_year"],
        "date": [dt.date(int(y), 1, 1) + dt.timedelta(int(d) - 1)
                 for y, d in zip(v["year"], v["day_of_year"])],
        "timestamp": [dt.datetime(int(y), 1, 1, int(h), int(mi), int(se)) +
                      dt.timedelta(int(d) - 1)
                      for y, d, h, mi, se in zip(v["year"], v["day_of_year"], v["H"], v["M"], v["S"])],
        "hour": v["hour"],
        "bee_id": v["bee_id"], "species": v["species"], "flower_id": v["flower_id"],
        "dwell_seconds": np.round(v["dwell_seconds"], 3),
        "velocity": np.round(v["velocity"], 5),
        "fraction_on": np.round(v["fraction_on"], 4),
        "gate_dwell": v["gate_dwell"], "gate_velocity": v["gate_velocity"],
        "gate_fraction": v["gate_fraction"], "qualifying_visit": v["qualifying_visit"],
        "stigma_receptive": v["stigma_receptive"], "pollinator_contact": v["pollinator_contact"],
        "pollination_prob": np.round(v["pollination_prob"], 5),
        "colony_id": v["colony_id"], "orchard_id": v["orchard_id"],
        "block_id": v["block_id"], "camera_id": v["camera_id"], "region": v["region"],
    })
    visits["_hf"] = v["H"] + v["M"] / 60.0 + v["S"] / 3600.0   # kept for validation

    flowers = pd.DataFrame(flower_records, columns=[
        "flower_id", "crop", "year", "region", "orchard_id", "block_id", "camera_id",
        "n_visits", "n_qualifying_visits", "V", "fruit_set_prob", "fruit_set",
    ])

    # ---- clips.csv : one row per (flower, date) that produced visits ----
    grp = visits.groupby("video_id")
    clip_rows = []
    for vid, g in grp:
        r0 = g.iloc[0]
        start_h = g["_hf"].min()
        span_s = max(1.0, (g["_hf"].max() - g["_hf"].min()) * 3600.0 + g["dwell_seconds"].max() + 5.0)
        H = int(start_h); Mi = int((start_h - H) * 60); Se = int((((start_h - H) * 60) - Mi) * 60)
        clip_rows.append((
            vid, r0["crop"], r0["orchard_id"], r0["block_id"], r0["camera_id"],
            r0["date"], int(r0["day_of_year"]), f"{H:02d}:{Mi:02d}:{Se:02d}",
            FPS, round(span_s, 2), FRAME_W, FRAME_H, int(round(span_s * FPS)),
        ))
    clips = pd.DataFrame(clip_rows, columns=[
        "video_id", "crop", "orchard_id", "block_id", "camera_id", "date",
        "day_of_year", "clip_start_time", "fps", "duration_s", "frame_w", "frame_h", "n_frames",
    ])

    # ---- daily_summary.csv ----
    def _agg(g):
        tot = len(g)
        q = g["qualifying_visit"].sum()
        mp = g.loc[g["qualifying_visit"], "pollination_prob"].mean() if q > 0 else 0.0
        return pd.Series({
            "total_visits": tot,
            "qualifying_visits": int(q),
            "active_unique_bees": g["bee_id"].nunique(),
            "qualifying_rate": q / tot if tot else 0.0,
            "mean_pollination_prob": mp,
        })
    daily = (visits.groupby(["crop", "year", "day_of_year"], as_index=False)
             .apply(_agg, include_groups=False).reset_index(drop=True))

    return visits, flowers, clips, daily


# ===========================================================================
# MERGED MODELING FRAME  (the analysis-ready table: one row per flower)
# Joins the fruit-set label to per-flower aggregated visit features, incl.
# type-specific qualifying-visit doses (research doc Section 12). This is the
# table baselines.py / dose_response.py actually consume.
# ===========================================================================
def build_modeling_frame(visits: pd.DataFrame, flowers: pd.DataFrame) -> pd.DataFrame:
    q = visits[visits["qualifying_visit"]]

    # features over ALL visits to the flower
    feat = visits.groupby("flower_id").agg(
        mean_dwell_s=("dwell_seconds", "mean"),
        total_dwell_s=("dwell_seconds", "sum"),
        mean_fraction_on=("fraction_on", "mean"),
        mean_velocity=("velocity", "mean"),
    )
    # features over QUALIFYING visits only
    featq = q.groupby("flower_id").agg(
        n_receptive_qual=("stigma_receptive", "sum"),
        n_contact_qual=("pollinator_contact", "sum"),
    )
    # type-specific qualifying-visit counts (bee vs bumblebee vs non-bee ...)
    sp = (q.groupby(["flower_id", "species"]).size().unstack(fill_value=0)
          .add_prefix("nq_"))

    season_of = {c["crop"]: c["season"] for c in CROPS}
    df = (flowers
          .merge(feat, on="flower_id", how="left")
          .merge(featq, on="flower_id", how="left")
          .merge(sp, on="flower_id", how="left"))
    df.insert(2, "season", df["crop"].map(season_of))

    # flowers with zero visits -> fill the aggregates with 0
    count_like = [c for c in df.columns if c.startswith("nq_") or c in
                  ("n_receptive_qual", "n_contact_qual", "total_dwell_s")]
    df[count_like] = df[count_like].fillna(0)
    df[["mean_dwell_s", "mean_fraction_on", "mean_velocity"]] = \
        df[["mean_dwell_s", "mean_fraction_on", "mean_velocity"]].fillna(0.0)

    # tidy column order: keys -> label/truth -> features
    front = ["flower_id", "crop", "season", "year", "region", "orchard_id",
             "block_id", "camera_id", "n_visits", "n_qualifying_visits",
             "V", "fruit_set_prob", "fruit_set"]
    rest = [c for c in df.columns if c not in front]
    return df[front + rest]


# ===========================================================================
# RAW MOT-STYLE LABEL SAMPLE  (a few clips only)
# ===========================================================================
def generate_raw_sample(visits: pd.DataFrame, clips: pd.DataFrame, rng):
    """Emit per-frame MOT rows for a sample of clips whose visits are tightly
    spaced (small n_frames), so the aggregation logic can be re-checked."""
    small = clips[clips["n_frames"] <= 60 * FPS].copy()          # <= 60 s clips
    counts = visits.groupby("video_id").size().rename("nv")
    small = small.join(counts, on="video_id").sort_values("nv", ascending=False)
    sample_ids = small["video_id"].head(N_RAW_SAMPLE_CLIPS).tolist()

    # a fixed flower bbox (normalized, centred); bees overlap it during dwell
    FB = (0.42, 0.42, 0.16, 0.16)
    rows = []
    expected = []   # (video_id, track_id, dwell_seconds, fraction_on) to re-check
    for vid in sample_ids:
        g = visits[visits["video_id"] == vid].sort_values("_hf").reset_index(drop=True)
        clip = clips[clips["video_id"] == vid].iloc[0]
        n_frames = int(clip["n_frames"])
        t0 = g["_hf"].min() * 3600.0
        # flower track (id=1) present every frame
        for fr in range(n_frames):
            rows.append((vid, fr, 1, "flower", FB[0], FB[1], FB[2], FB[3],
                         round(rng.uniform(0.80, 0.98), 3)))
        # one insect track per visit
        for k, (_, vr) in enumerate(g.iterrows(), start=2):
            n_overlap = max(1, int(round(vr["dwell_seconds"] * FPS)))
            frac = float(vr["fraction_on"])
            n_track = max(n_overlap, int(round(n_overlap / max(frac, 1e-3))))
            pad = n_track - n_overlap
            pad_b = pad // 2
            start = int(round((vr["_hf"] * 3600.0 - t0) * FPS))
            start = min(max(0, start), max(0, n_frames - n_track))
            for j in range(n_track):
                fr = start + j
                overlapping = pad_b <= j < pad_b + n_overlap
                if overlapping:
                    # sit on the flower with tiny jitter -> IoU >= IOU_MIN
                    bx = FB[0] + rng.normal(0, 0.005)
                    by = FB[1] + rng.normal(0, 0.005)
                    bw, bh = 0.15, 0.15
                else:
                    # tracked but off the flower -> IoU below threshold
                    bx = FB[0] + 0.30
                    by = FB[1] + 0.30
                    bw, bh = 0.10, 0.10
                rows.append((vid, fr, k, vr["species"],
                             round(float(bx), 4), round(float(by), 4), bw, bh,
                             round(rng.uniform(0.55, 0.95), 3)))
            expected.append((vid, k, vr["dwell_seconds"], vr["fraction_on"]))

    raw = pd.DataFrame(rows, columns=[
        "video_id", "frame_id", "track_id", "class_name", "x", "y", "w", "h", "conf"])
    return raw, expected


def recheck_raw(raw: pd.DataFrame, expected):
    """Re-derive dwell_seconds & fraction_on from the raw frames and compare."""
    flower_boxes = {}
    for vid, g in raw[raw["class_name"] == "flower"].groupby("video_id"):
        r = g.iloc[0]
        flower_boxes[vid] = (r["x"], r["y"], r["w"], r["h"])
    ok = 0
    for vid, tid, dwell_exp, frac_exp in expected:
        fb = flower_boxes[vid]
        g = raw[(raw["video_id"] == vid) & (raw["track_id"] == tid)].sort_values("frame_id")
        overl = np.array([iou((r.x, r.y, r.w, r.h), fb) >= IOU_MIN for r in g.itertuples()])
        n_track = len(g)
        if overl.any():
            frames = g["frame_id"].to_numpy()
            of = frames[overl]
            dwell_re = (of.max() - of.min() + 1) / FPS
            frac_re = overl.sum() / n_track
            if abs(dwell_re - dwell_exp) <= 1.5 / FPS and abs(frac_re - frac_exp) <= 0.06:
                ok += 1
    return ok, len(expected)


# ===========================================================================
# VALIDATION REPORT
# ===========================================================================
def validate_and_report(visits, flowers, clips, daily, raw, expected):
    print("=" * 74)
    print("VALIDATION REPORT")
    print("=" * 74)

    # 1. No night visits: inside sunrise-sunset AND anthesis window
    bad_night = 0
    for crop in CROPS:
        m = visits["crop"] == crop["crop"]
        hf = visits.loc[m, "_hf"].to_numpy()
        doy = visits.loc[m, "day_of_year"].to_numpy()
        sr, ss = sunrise_sunset(doy.astype(float))
        a0, a1 = crop["anthesis"]
        bad = (hf < np.maximum(sr, a0) - 1e-6) | (hf > np.minimum(ss, a1) + 1e-6)
        bad_night += int(bad.sum())
    print(f"[1] Night / out-of-anthesis visits : {bad_night}  (must be 0)")
    assert bad_night == 0, "night-visit assertion failed"

    # 2. Season gating: no visits outside a crop's bloom window
    leak = 0
    for crop in CROPS:
        m = visits["crop"] == crop["crop"]
        d = visits.loc[m, "day_of_year"].to_numpy()
        d0, d1 = crop["bloom_doy"]
        leak += int(((d < d0) | (d > d1)).sum())
    leak_rate = leak / len(visits)
    print(f"[2] Cross-season leakage           : {leak_rate:.4%}  (must be < 1%)")
    assert leak_rate < 0.01

    # 3. Gate + receptive pass-rates
    print("[3] Pass rates:")
    print(f"      gate_dwell        : {visits['gate_dwell'].mean():.3f}")
    print(f"      gate_velocity     : {visits['gate_velocity'].mean():.3f}")
    print(f"      gate_fraction     : {visits['gate_fraction'].mean():.3f}")
    print(f"      qualifying_visit  : {visits['qualifying_visit'].mean():.3f}")
    print(f"      stigma_receptive  : {visits['stigma_receptive'].mean():.3f}")
    print(f"      pollinator_contact: {visits['pollinator_contact'].mean():.3f}")

    # 4. Curve sanity per crop
    print("[4] Dose-response curve sanity (FruitSet(0) ~ F0 ; FruitSet(V*) ~ 0.95 band):")
    for crop in CROPS:
        f0 = crop["F0"]
        fvs = crop["F0"] + (crop["Fmax"] - crop["F0"]) * (1 - np.exp(-crop["k"] * crop["V_star"]))
        print(f"      {crop['crop']:<12} k={crop['k']:.3f}  FruitSet(0)={f0:.3f}  "
              f"FruitSet(V*={crop['V_star']:.0f})={fvs:.3f}")

    # 5. Contrast pomegranate vs cucumber
    print("[5] Realised fruit-set rate (Bernoulli outcomes):")
    for crop in CROPS:
        fm = flowers[flowers["crop"] == crop["crop"]]
        print(f"      {crop['crop']:<12} fruit_set rate = {fm['fruit_set'].mean():.3f}  "
              f"mean dose V = {fm['V'].mean():.2f}  (n={len(fm)})")
    pom = flowers.loc[flowers["crop"] == "pomegranate", "fruit_set"].mean()
    cuc = flowers.loc[flowers["crop"] == "cucumber", "fruit_set"].mean()
    print(f"      => pomegranate {'>' if pom > cuc else '<='} cucumber "
          f"(expected higher: F0=0.45 vs 0.05)")

    # 6. Raw-vs-visit consistency
    ok, tot = recheck_raw(raw, expected)
    print(f"[6] Raw->visit re-derivation match : {ok}/{tot} sampled visits")

    # 7. Volume report
    print("[7] Volumes:")
    print(f"      visits rows     : {len(visits):,}")
    print(f"      flowers rows    : {len(flowers):,}")
    print(f"      clips rows      : {len(clips):,}")
    print(f"      daily rows      : {len(daily):,}")
    print(f"      raw sample rows : {len(raw):,}")
    print(f"      unique bees     : {visits['bee_id'].nunique():,}")
    print(f"      unique flowers  : {visits['flower_id'].nunique():,}")
    for crop in CROPS:
        m = visits["crop"] == crop["crop"]
        print(f"      {crop['crop']:<12} visits={int(m.sum()):,}")
    print("=" * 74)


# ===========================================================================
# ENTRY POINT
# ===========================================================================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating synthetic dataset (SEED={SEED}) -> {OUT_DIR}")
    visits, flowers, clips, daily = generate()

    dataset = build_modeling_frame(visits, flowers)

    raw_rng = np.random.default_rng(SEED + 1)
    raw, expected = generate_raw_sample(visits, clips, raw_rng)

    validate_and_report(visits, flowers, clips, daily, raw, expected)
    print(f"[8] Merged modeling frame          : {len(dataset):,} rows x "
          f"{dataset.shape[1]} cols -> dataset.csv")

    visits.drop(columns=["_hf"]).to_csv(OUT_DIR / "visits.csv", index=False)
    flowers.to_csv(OUT_DIR / "flowers.csv", index=False)
    clips.to_csv(OUT_DIR / "clips.csv", index=False)
    raw.to_csv(OUT_DIR / "raw_labels_sample.csv", index=False)
    daily.to_csv(OUT_DIR / "daily_summary.csv", index=False)
    dataset.to_csv(OUT_DIR / "dataset.csv", index=False)
    print("Wrote: visits.csv, flowers.csv, clips.csv, raw_labels_sample.csv, "
          "daily_summary.csv, dataset.csv")


if __name__ == "__main__":
    main()
