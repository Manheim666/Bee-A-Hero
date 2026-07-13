import Hexagon from "./Hexagon.jsx";

/** Big number + label, framed by a honeycomb hexagon. */
export default function StatTile({ value, label, sub, background }) {
  return (
    <div
      className="card"
      style={{ display: "flex", gap: 16, alignItems: "center" }}
    >
      <Hexagon size={64} background={background}>
        <span style={{ fontSize: "1.05rem" }}>{value}</span>
      </Hexagon>
      <div>
        <div style={{ fontWeight: 600 }}>{label}</div>
        {sub && (
          <div className="muted" style={{ fontSize: "0.85rem" }}>
            {sub}
          </div>
        )}
      </div>
    </div>
  );
}
