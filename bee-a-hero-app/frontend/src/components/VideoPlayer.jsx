import { useEffect, useRef, useState } from "react";
import api from "../api";

/**
 * Modal player. Requests server-side annotation (flower + insect YOLO
 * boxes drawn on every frame), polls until ready, then plays the
 * annotated mp4. Falls back to the raw video if annotation errors out.
 */
export default function VideoPlayer({ video, onClose }) {
  const [blobUrl, setBlobUrl] = useState(null);
  const [visits, setVisits] = useState([]);
  const [error, setError] = useState("");
  const [annotStatus, setAnnotStatus] = useState("idle"); // idle|running|done|failed
  const [showRaw, setShowRaw] = useState(false);
  const pollRef = useRef(null);
  const blobRef = useRef(null);

  function releaseBlob() {
    if (blobRef.current) {
      URL.revokeObjectURL(blobRef.current);
      blobRef.current = null;
    }
  }

  async function loadAnnotated() {
    releaseBlob();
    setError("");
    try {
      const res = await api.get(`/api/videos/${video.id}/annotated_stream`, {
        responseType: "blob",
      });
      const url = URL.createObjectURL(res.data);
      blobRef.current = url;
      setBlobUrl(url);
    } catch (err) {
      setError(err.response?.data?.detail || "Could not load annotated video");
    }
  }

  async function loadRaw() {
    releaseBlob();
    setError("");
    try {
      const res = await api.get(`/api/videos/${video.id}/stream`, {
        responseType: "blob",
      });
      const url = URL.createObjectURL(res.data);
      blobRef.current = url;
      setBlobUrl(url);
    } catch (err) {
      setError(err.response?.data?.detail || "Could not load video");
    }
  }

  async function kickAnnotation() {
    try {
      const res = await api.post(`/api/videos/${video.id}/annotate`);
      setAnnotStatus(res.data.status);
      return res.data.status;
    } catch (err) {
      setAnnotStatus("failed");
      setError(err.response?.data?.detail || "Annotation request failed");
      return "failed";
    }
  }

  useEffect(() => {
    if (!video) return;
    setShowRaw(false);
    setBlobUrl(null);
    (async () => {
      const st = await api
        .get(`/api/videos/${video.id}/annotated_status`)
        .then((r) => r.data)
        .catch(() => ({ status: "idle" }));
      if (st.status === "done") {
        setAnnotStatus("done");
        loadAnnotated();
        return;
      }
      const started = st.status === "running" ? "running" : await kickAnnotation();
      if (started === "failed") return;
      pollRef.current = setInterval(async () => {
        try {
          const res = await api.get(`/api/videos/${video.id}/annotated_status`);
          setAnnotStatus(res.data.status);
          if (res.data.status === "done") {
            clearInterval(pollRef.current);
            pollRef.current = null;
            loadAnnotated();
          } else if (res.data.status === "failed") {
            clearInterval(pollRef.current);
            pollRef.current = null;
            setError(res.data.error || "Annotation failed");
          }
        } catch (err) {
          /* ignore transient */
        }
      }, 2000);
    })();

    api
      .get("/api/stats/visits", { params: { video_id: video.id } })
      .then((res) => setVisits(res.data.bars || []))
      .catch(() => {});

    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
      releaseBlob();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [video]);

  useEffect(() => {
    function onKey(e) {
      if (e.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  if (!video) return null;
  const r = video.result;

  return (
    <div
      onClick={onClose}
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(0,0,0,0.68)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: 20,
        zIndex: 1000,
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          background: "var(--card)",
          border: "1px solid var(--border)",
          borderRadius: "var(--radius)",
          boxShadow: "0 12px 32px rgba(0,0,0,0.4)",
          width: "100%",
          maxWidth: 900,
          maxHeight: "92vh",
          overflow: "auto",
          padding: 20,
        }}
      >
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            gap: 10,
            marginBottom: 12,
            flexWrap: "wrap",
          }}
        >
          <h2
            style={{
              margin: 0,
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
              flex: 1,
            }}
            title={video.original_name}
          >
            {video.original_name}
          </h2>
          <span className={`pill pill-${video.status}`}>{video.status}</span>
          <button className="btn btn-ghost" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </div>

        <div
          style={{
            background: "#000",
            borderRadius: 12,
            overflow: "hidden",
            aspectRatio: "16 / 9",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            flexDirection: "column",
            color: "#fff",
            gap: 10,
            padding: 12,
          }}
        >
          {!blobUrl && annotStatus === "running" && (
            <>
              <p style={{ margin: 0, textAlign: "center" }}>
                <span className="spinner" /> Running YOLO on every frame…
              </p>
              <p className="muted" style={{ margin: 0, fontSize: "0.8rem", color: "#ccc", textAlign: "center" }}>
                First-time annotation of this clip. ~10–90 seconds on CPU.
                <br />Result is cached — future plays are instant.
              </p>
              <button
                className="btn btn-ghost"
                style={{ background: "rgba(255,255,255,0.08)", color: "#fff", border: "1px solid rgba(255,255,255,0.2)" }}
                onClick={() => {
                  setShowRaw(true);
                  loadRaw();
                }}
              >
                Skip and play raw video
              </button>
            </>
          )}
          {!blobUrl && annotStatus === "failed" && (
            <>
              <p style={{ color: "var(--danger)" }}>Annotation failed: {error}</p>
              <button
                className="btn"
                onClick={() => {
                  setShowRaw(true);
                  loadRaw();
                }}
              >
                Play raw video instead
              </button>
            </>
          )}
          {blobUrl && (
            <video
              src={blobUrl}
              controls
              autoPlay
              style={{ width: "100%", height: "100%", objectFit: "contain" }}
            />
          )}
        </div>

        <div
          style={{
            marginTop: 8,
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            gap: 10,
            flexWrap: "wrap",
          }}
        >
          <span className="muted" style={{ fontSize: "0.82rem" }}>
            {annotStatus === "done" && !showRaw && "🟢 Showing annotated stream (flower + insect boxes)"}
            {annotStatus === "running" && "🟡 Annotating in the background…"}
            {showRaw && "⚪ Showing raw video (no boxes)"}
          </span>
          {annotStatus === "done" && (
            <button
              className="btn btn-ghost"
              style={{ padding: "4px 12px", fontSize: "0.8rem" }}
              onClick={() => {
                if (showRaw) {
                  setShowRaw(false);
                  loadAnnotated();
                } else {
                  setShowRaw(true);
                  loadRaw();
                }
              }}
            >
              {showRaw ? "Show annotations" : "Show raw"}
            </button>
          )}
        </div>

        {r && (
          <div
            className="grid"
            style={{
              marginTop: 16,
              gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))",
              gap: 10,
            }}
          >
            <MiniStat value={r.flower_map} label="Flowers" />
            <MiniStat value={r.insect_tracks} label="Visits" />
            <MiniStat
              value={r.pollinator_visits}
              label="Pollinator"
              color="var(--pollinator)"
            />
            <MiniStat
              value={r.non_pollinator_visits}
              label="Non-pollinator"
              color="var(--non)"
            />
          </div>
        )}

        {visits.length > 0 && (
          <div style={{ marginTop: 18 }}>
            <h3 style={{ marginBottom: 8, fontSize: "1rem" }}>Visits by flower</h3>
            <div
              style={{
                display: "grid",
                gap: 6,
                gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))",
              }}
            >
              {visits.map((b) => (
                <div
                  key={b.flower_id}
                  className="card"
                  style={{ padding: 10, textAlign: "center" }}
                >
                  <div style={{ fontSize: "0.75rem" }} className="muted">
                    flower_{b.flower_id}
                  </div>
                  <div style={{ fontWeight: 700 }}>
                    <span style={{ color: "var(--pollinator)" }}>{b.pollinator}</span>
                    {" / "}
                    <span style={{ color: "var(--non)" }}>{b.non_pollinator}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {video.status === "failed" && (
          <p style={{ color: "var(--danger)", marginTop: 12 }}>
            {video.error || "Detection failed."}
          </p>
        )}
      </div>
    </div>
  );
}

function MiniStat({ value, label, color }) {
  return (
    <div
      className="card"
      style={{ padding: 10, textAlign: "center" }}
    >
      <div style={{ fontSize: "1.4rem", fontWeight: 700, color: color || "var(--honey-deep)" }}>
        {value}
      </div>
      <div className="muted" style={{ fontSize: "0.78rem" }}>
        {label}
      </div>
    </div>
  );
}
