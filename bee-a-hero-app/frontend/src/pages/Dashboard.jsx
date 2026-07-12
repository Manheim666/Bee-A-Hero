import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import api from "../api";
import { useAuth } from "../auth/AuthContext.jsx";
import StatTile from "../components/StatTile.jsx";

export default function Dashboard() {
  const { user } = useAuth();
  const [overview, setOverview] = useState(null);
  const [videos, setVideos] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([api.get("/api/stats/overview"), api.get("/api/videos")])
      .then(([ov, vids]) => {
        setOverview(ov.data);
        setVideos(vids.data.slice(0, 4));
      })
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="container">
      <div className="honeycomb-header">
        <h1>Welcome back, {user?.username} 🐝</h1>
        <p className="muted">
          Your pollination dashboard — detections, visit stats, and the
          assistant, all in one place.
        </p>
      </div>

      {loading ? (
        <p>
          <span className="spinner" /> Loading your stats…
        </p>
      ) : (
        <>
          <div
            className="grid"
            style={{ gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))" }}
          >
            <StatTile
              value={overview.videos_processed}
              label="Videos processed"
              sub="detections completed"
            />
            <StatTile
              value={overview.total_visits}
              label="Total insect visits"
              sub="across all videos"
            />
            <StatTile
              value={`${overview.pollinator_pct}%`}
              label="Pollinator share"
              sub="of all visits"
              background="linear-gradient(135deg, var(--pollinator), var(--honey-deep))"
            />
            <StatTile
              value={overview.avg_visits_per_flower}
              label="Avg visits / flower"
              sub={
                overview.top_flower
                  ? `top: flower_${overview.top_flower}`
                  : "no data yet"
              }
              background="linear-gradient(135deg, var(--leaf), #3f8f30)"
            />
          </div>

          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 30 }}>
            <h2 style={{ margin: 0 }}>Recent videos</h2>
            <Link to="/upload" className="btn">
              Upload a video
            </Link>
          </div>

          {videos.length === 0 ? (
            <div className="card" style={{ marginTop: 14 }}>
              <p className="muted" style={{ margin: 0 }}>
                No videos yet. Head to{" "}
                <Link to="/upload">Upload</Link> to run your first detection.
              </p>
            </div>
          ) : (
            <div
              className="grid"
              style={{
                marginTop: 14,
                gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))",
              }}
            >
              {videos.map((v) => (
                <div className="card" key={v.id}>
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      gap: 8,
                    }}
                  >
                    <strong
                      style={{
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        whiteSpace: "nowrap",
                      }}
                      title={v.original_name}
                    >
                      {v.original_name}
                    </strong>
                    <span className={`pill pill-${v.status}`}>{v.status}</span>
                  </div>
                  {v.result && (
                    <p className="muted" style={{ fontSize: "0.85rem", marginBottom: 0 }}>
                      {v.result.insect_tracks} visits ·{" "}
                      {v.result.pollinator_visits} pollinator
                    </p>
                  )}
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
