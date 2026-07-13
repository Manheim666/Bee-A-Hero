import { useCallback, useEffect, useRef, useState } from "react";
import api from "../api";
import VideoCard from "../components/VideoCard.jsx";
import VideoPlayer from "../components/VideoPlayer.jsx";

export default function Upload() {
  const [videos, setVideos] = useState([]);
  const [progress, setProgress] = useState(null);
  const [error, setError] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const [openVideo, setOpenVideo] = useState(null);
  const fileRef = useRef(null);
  const pollRef = useRef(null);

  const loadVideos = useCallback(async () => {
    const res = await api.get("/api/videos");
    setVideos(res.data);
    return res.data;
  }, []);

  useEffect(() => {
    loadVideos();
  }, [loadVideos]);

  // Poll while anything is queued/processing, then stop.
  useEffect(() => {
    const anyPending = videos.some(
      (v) => v.status === "queued" || v.status === "processing"
    );
    if (anyPending && !pollRef.current) {
      pollRef.current = setInterval(loadVideos, 1500);
    } else if (!anyPending && pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [videos, loadVideos]);

  async function uploadFile(file) {
    if (!file) return;
    setError("");
    const form = new FormData();
    form.append("file", file);
    try {
      await api.post("/api/videos", form, {
        onUploadProgress: (e) => {
          if (e.total) setProgress(Math.round((100 * e.loaded) / e.total));
        },
      });
      setProgress(null);
      loadVideos();
    } catch (err) {
      setProgress(null);
      setError(err.response?.data?.detail || "Upload failed");
    }
  }

  function onDrop(e) {
    e.preventDefault();
    setDragOver(false);
    uploadFile(e.dataTransfer.files?.[0]);
  }

  async function handleDelete(id) {
    await api.delete(`/api/videos/${id}`);
    loadVideos();
  }

  return (
    <div className="container">
      <h1>Upload &amp; detect</h1>
      <p className="muted">
        Drop a video of pomegranate flowers. The system detects flowers, tracks
        insects, and counts pollination visits.
      </p>

      <div
        className="card"
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        onClick={() => fileRef.current?.click()}
        style={{
          textAlign: "center",
          padding: 40,
          cursor: "pointer",
          borderStyle: "dashed",
          borderWidth: 2,
          borderColor: dragOver ? "var(--honey-deep)" : "var(--border)",
          background: dragOver ? "var(--amber-glow)" : "var(--card)",
        }}
      >
        <p style={{ fontSize: "2rem", margin: 0 }}>🎞️</p>
        <p style={{ fontWeight: 600, margin: "8px 0 4px" }}>
          Drag &amp; drop a video here, or click to choose
        </p>
        <p className="muted" style={{ margin: 0, fontSize: "0.85rem" }}>
          mp4 / mov / avi · up to 200 MB
        </p>
        <input
          ref={fileRef}
          type="file"
          accept=".mp4,.mov,.avi,video/*"
          style={{ display: "none" }}
          onChange={(e) => uploadFile(e.target.files?.[0])}
        />
      </div>

      {progress !== null && (
        <div className="card" style={{ marginTop: 12 }}>
          <div className="muted" style={{ fontSize: "0.85rem", marginBottom: 6 }}>
            Uploading… {progress}%
          </div>
          <div
            style={{
              height: 10,
              borderRadius: 999,
              background: "var(--queued-bg)",
              overflow: "hidden",
            }}
          >
            <div
              style={{
                width: `${progress}%`,
                height: "100%",
                background:
                  "linear-gradient(90deg, var(--honey), var(--honey-deep))",
                transition: "width 0.2s",
              }}
            />
          </div>
        </div>
      )}
      {error && (
        <p style={{ color: "var(--danger)", marginTop: 10 }}>{error}</p>
      )}

      <h2 style={{ marginTop: 30 }}>Your library</h2>
      {videos.length === 0 ? (
        <div className="card">
          <p className="muted" style={{ margin: 0 }}>
            No videos yet — upload one above to get started.
          </p>
        </div>
      ) : (
        <div
          className="grid"
          style={{ gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))" }}
        >
          {videos.map((v) => (
            <VideoCard
              key={v.id}
              video={v}
              onDelete={handleDelete}
              onOpen={setOpenVideo}
            />
          ))}
        </div>
      )}
      {openVideo && (
        <VideoPlayer video={openVideo} onClose={() => setOpenVideo(null)} />
      )}
    </div>
  );
}
