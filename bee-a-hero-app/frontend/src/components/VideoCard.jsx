import Hexagon from "./Hexagon.jsx";

export default function VideoCard({ video, onDelete }) {
  const r = video.result;
  const processing =
    video.status === "queued" || video.status === "processing";

  return (
    <div className="card" style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <div
        style={{
          height: 120,
          borderRadius: 10,
          background: "linear-gradient(135deg, var(--queued-bg), var(--amber-glow))",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <Hexagon size={54} background="rgba(255,255,255,0.7)" color="var(--honey-deep)">
          🎞️
        </Hexagon>
      </div>

      <div style={{ display: "flex", justifyContent: "space-between", gap: 8 }}>
        <strong
          style={{
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
          title={video.original_name}
        >
          {video.original_name}
        </strong>
        <span className={`pill pill-${video.status}`}>
          {processing && <span className="spinner" style={{ marginRight: 6 }} />}
          {video.status}
        </span>
      </div>

      {video.status === "done" && r && (
        <div className="muted" style={{ fontSize: "0.85rem" }}>
          {r.flower_map} flowers · {r.insect_tracks} visits ·{" "}
          <span style={{ color: "var(--pollinator)", fontWeight: 600 }}>
            {r.pollinator_visits} pollinator
          </span>{" "}
          /{" "}
          <span style={{ color: "var(--non)", fontWeight: 600 }}>
            {r.non_pollinator_visits} non
          </span>
        </div>
      )}
      {video.status === "failed" && (
        <div className="muted" style={{ fontSize: "0.85rem", color: "var(--danger)" }}>
          {video.error || "Detection failed"}
        </div>
      )}
      {processing && (
        <div className="muted" style={{ fontSize: "0.85rem" }}>
          Detecting flowers and insects…
        </div>
      )}

      <button
        className="btn btn-ghost"
        style={{ alignSelf: "flex-start" }}
        onClick={() => onDelete(video.id)}
      >
        Delete
      </button>
    </div>
  );
}
