import { useEffect, useState } from "react";
import api from "../api";
import StatTile from "./StatTile.jsx";

// Per-crop pollination -> fruit-set -> yield. Choosing a crop is optional (defaults to the
// backend default); the flower count is estimated automatically, not entered by hand.
export default function CropYield() {
  const [crops, setCrops] = useState([]);
  const [crop, setCrop] = useState("");
  const [est, setEst] = useState(null);

  useEffect(() => {
    api.get("/api/stats/crops").then((res) => {
      setCrops(res.data.crops);
      setCrop(res.data.default);
    });
  }, []);

  useEffect(() => {
    if (!crop) return;
    api
      .get("/api/stats/yield", { params: { crop } })
      .then((res) => setEst(res.data))
      .catch(() => setEst(null));
  }, [crop]);

  return (
    <section className="card" style={{ marginBottom: "1.5rem" }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          flexWrap: "wrap",
          gap: "0.75rem",
        }}
      >
        <h2 style={{ margin: 0 }}>Fruit-set &amp; yield</h2>
        <label className="muted" style={{ fontSize: "0.85rem" }}>
          Crop (optional)&nbsp;
          <select
            value={crop}
            onChange={(e) => setCrop(e.target.value)}
            style={{
              fontFamily: "var(--font)",
              padding: "0.4rem 0.7rem",
              borderRadius: "10px",
              border: "1px solid var(--border)",
              background: "var(--card)",
              color: "var(--bee-black)",
            }}
          >
            {crops.map((c) => (
              <option key={c.value} value={c.value}>
                {c.label}
              </option>
            ))}
          </select>
        </label>
      </div>

      {est && (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))",
            gap: "0.9rem",
            marginTop: "1rem",
          }}
        >
          <StatTile value={`${est.fruit_set_pct}%`} label="Fruit set" background="var(--leaf)" />
          <StatTile
            value={`${est.yield_kg} kg`}
            label="Estimated yield / tree"
            sub={`${est.n_flowers} flowers (auto)`}
            background="var(--honey)"
          />
          <StatTile
            value={`${est.pollination_deficit_pct}%`}
            label="Pollination deficit"
            background="var(--non)"
          />
          <StatTile
            value={`${Math.round(est.pollinator_dependence * 100)}%`}
            label="Pollinator dependence"
            sub={est.crop_label}
            background="var(--pollinator)"
          />
          <StatTile
            value={`+${(est.marginal_fruitset_per_visit * 100).toFixed(2)}%`}
            label="Gain per extra visit"
            background="var(--amber-glow)"
          />
        </div>
      )}
      <p className="muted" style={{ fontSize: "0.8rem", marginTop: "0.9rem" }}>
        FruitSet(V) = F0 + (Fmax−F0)(1−e^(−kV)); parameters differ per crop (self-fertility,
        pollinator dependence, fruit size). Illustrative until field-calibrated.
      </p>
    </section>
  );
}
