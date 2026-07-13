import { useEffect, useState } from "react";
import api from "../api";

// Per-crop pollination -> fruit-set -> yield. Each crop has its own formula (backend yield_model),
// so switching the crop re-runs the estimate on the user's detected visits.
export default function CropYield() {
  const [crops, setCrops] = useState([]);
  const [crop, setCrop] = useState("pomegranate");
  const [nFlowers, setNFlowers] = useState(1200);
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
      .get("/api/stats/yield", { params: { crop, n_flowers: nFlowers } })
      .then((res) => setEst(res.data))
      .catch(() => setEst(null));
  }, [crop, nFlowers]);

  return (
    <section className="card" style={{ marginBottom: "1.5rem" }}>
      <h2>Fruit-set &amp; yield by crop</h2>
      <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap", alignItems: "flex-end" }}>
        <label>
          Crop
          <br />
          <select value={crop} onChange={(e) => setCrop(e.target.value)}>
            {crops.map((c) => (
              <option key={c.value} value={c.value}>
                {c.label}
              </option>
            ))}
          </select>
        </label>
        <label>
          Flowers / tree
          <br />
          <input
            type="number"
            min="1"
            value={nFlowers}
            onChange={(e) => setNFlowers(Number(e.target.value) || 1)}
            style={{ width: "8rem" }}
          />
        </label>
      </div>

      {est && (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
            gap: "0.75rem",
            marginTop: "1rem",
          }}
        >
          <Tile label="Fruit set" value={`${est.fruit_set_pct}%`} />
          <Tile label="Estimated yield" value={`${est.yield_kg} kg/tree`} />
          <Tile label="Pollination deficit" value={`${est.pollination_deficit_pct}%`} />
          <Tile label="Pollinator dependence" value={`${Math.round(est.pollinator_dependence * 100)}%`} />
          <Tile label="Effective dose (V)" value={est.effective_dose} />
          <Tile label="+1 visit ⇒ fruit set" value={`+${(est.marginal_fruitset_per_visit * 100).toFixed(2)}%`} />
        </div>
      )}
      <p style={{ color: "#888", fontSize: ".8rem", marginTop: ".75rem" }}>
        FruitSet(V) = F0 + (Fmax−F0)(1−e^(−kV)); parameters differ per crop (self-fertility,
        pollinator dependence, fruit size). Illustrative until field-calibrated.
      </p>
    </section>
  );
}

function Tile({ label, value }) {
  return (
    <div className="stat-tile" style={{ padding: ".75rem", border: "1px solid #8883", borderRadius: ".5rem" }}>
      <div style={{ fontSize: "1.3rem", fontWeight: 700 }}>{value}</div>
      <div style={{ color: "#888", fontSize: ".8rem" }}>{label}</div>
    </div>
  );
}
