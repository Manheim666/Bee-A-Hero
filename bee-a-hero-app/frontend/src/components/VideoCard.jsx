import { useEffect, useRef, useState } from "react";
import Hexagon from "./Hexagon.jsx";
import api from "../api";

export default function VideoCard({ video, onDelete, onOpen }) {
  const r = video.result;
  const processing =
    video.status === "queued" || video.status === "processing";
  const canOpen = video.status === "done" && !!onOpen;

  // Real still frame from the clip as the cover (upgrades to the boxed frame once the
  // annotated video is ready), instead of a generic logo. Falls back to the emoji.
  const [poster, setPoster] = useState(null);
  const posterRef = useRef(null);
  useEffect(() => {
    let cancelled = false;
    api
      .get(`/api/videos/${video.id}/poster`, { responseType: "blob" })
      .then((res) => {
        if (cancelled) return;
        const url = URL.createObjectURL(res.data);
        posterRef.current = url;
        setPoster(url);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
      if (posterRef.current) {
        URL.revokeObjectURL(posterRef.current);
        posterRef.current = null;
      }
    };
  }, [video.id, video.status]);

  return (
    <div
      className={`card ${canOpen ? "card-hover" : ""}`}
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 12,
        cursor: canOpen ? "pointer" : "default",
      }}
      onClick={canOpen ? () => onOpen(video) : undefined}
      role={canOpen ? "button" : undefined}
      tabIndex={canOpen ? 0 : undefined}
      onKeyDown={
        canOpen
          ? (e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                onOpen(video);
              }
            }
          : undefined
      }
    >
      <div
        style={{
          aspectRatio: "1 / 1",
          width: "100%",
          borderRadius: 10,
          background: poster
            ? "#000"
            : "linear-gradient(135deg, var(--queued-bg), var(--amber-glow))",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          position: "relative",
          overflow: "hidden",
        }}
      >
        {poster ? (
          <img
            src={poster}
            alt={video.original_name}
            style={{ width: "100%", height: "100%", objectFit: "cover" }}
          />
        ) : (
          <Hexagon size={54} background="rgba(255,255,255,0.7)" color="var(--honey-deep)">
            🎞️
          </Hexagon>
        )}
        {canOpen && (
          <div
            style={{
              position: "absolute",
              inset: 0,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: "#fff",
              fontSize: "2.4rem",
              textShadow: "0 2px 8px rgba(0,0,0,0.5)",
              opacity: 0.85,
              pointerEvents: "none",
            }}
          >
            ▶
          </div>
        )}
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

      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        {canOpen && (
          <button
            className="btn"
            style={{ padding: "6px 14px", fontSize: "0.85rem" }}
            onClick={(e) => {
              e.stopPropagation();
              onOpen(video);
            }}
          >
            ▶ Play
          </button>
        )}
        <button
          className="btn btn-ghost"
          onClick={(e) => {
            e.stopPropagation();
            onDelete(video.id);
          }}
        >
          Delete
        </button>
      </div>
    </div>
  );
}
