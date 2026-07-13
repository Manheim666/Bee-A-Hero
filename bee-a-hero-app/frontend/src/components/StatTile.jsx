import Hexagon from "./Hexagon.jsx";

/** Big number + label, framed by a honeycomb hexagon. */
export default function StatTile({ value, label, sub, background }) {
  return (
    <div
      className="card card-hover"
      style={{ display: "flex", gap: 16, alignItems: "center" }}
    >
      <Hexagon size={64} background={background}>
        <span className="mono" style={{ fontSize: "1.05rem", fontWeight: 700 }}>
          {value}
        </span>
      </Hexagon>
      <div>
        <div
          style={{
            fontWeight: 600,
            fontSize: "0.72rem",
            letterSpacing: "0.12em",
            textTransform: "uppercase",
            color: "var(--charcoal)",
          }}
        >
          {label}
        </div>
        {sub && (
          <div style={{ fontSize: "0.95rem", fontWeight: 600, marginTop: 2 }}>
            {sub}
          </div>
        )}
      </div>
    </div>
  );
}
