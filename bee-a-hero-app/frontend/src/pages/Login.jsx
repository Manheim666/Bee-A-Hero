import { useState } from "react";
import { Link, Navigate, useNavigate } from "react-router-dom";
import { useAuth } from "../auth/AuthContext.jsx";
import Hexagon from "../components/Hexagon.jsx";
import ThemeToggle from "../components/ThemeToggle.jsx";

export default function Login() {
  const { user, login } = useAuth();
  const navigate = useNavigate();
  const [email, setEmail] = useState("demo@bee.dev");
  const [password, setPassword] = useState("beehero123");
  const [showPw, setShowPw] = useState(false);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  if (user) return <Navigate to="/dashboard" replace />;

  async function submit(e) {
    e.preventDefault();
    setError("");
    setBusy(true);
    try {
      await login(email, password);
      navigate("/dashboard");
    } catch (err) {
      setError(err.response?.data?.detail || "Login failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div
      className="container"
      style={{ maxWidth: 420, marginTop: 60, textAlign: "center", position: "relative" }}
    >
      <div style={{ position: "absolute", top: 0, right: 24 }}>
        <ThemeToggle />
      </div>
      <div style={{ display: "flex", justifyContent: "center", marginBottom: 14 }}>
        <Hexagon size={72} style={{ fontSize: "1.8rem" }}>
          🐝
        </Hexagon>
      </div>
      <h1>Bee-A-Hero</h1>
      <p className="muted">Turning flower videos into a pollination signal.</p>

      <form className="card" onSubmit={submit} style={{ marginTop: 18, textAlign: "left" }}>
        <label className="muted" style={{ fontSize: "0.85rem" }}>
          Email
        </label>
        <input
          className="input"
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          style={{ marginBottom: 12 }}
          required
        />
        <label className="muted" style={{ fontSize: "0.85rem" }}>
          Password
        </label>
        <div className="password-wrap" style={{ marginBottom: 16 }}>
          <input
            className="input"
            type={showPw ? "text" : "password"}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
          <button
            type="button"
            className="password-toggle"
            aria-label={showPw ? "Hide password" : "Show password"}
            onClick={() => setShowPw((s) => !s)}
          >
            {showPw ? "🙈" : "👁️"}
          </button>
        </div>
        {error && (
          <p style={{ color: "var(--danger)", fontSize: "0.9rem" }}>{error}</p>
        )}
        <button className="btn" style={{ width: "100%" }} disabled={busy}>
          {busy ? "Signing in…" : "Sign in"}
        </button>
      </form>

      <p className="muted" style={{ marginTop: 14 }}>
        No account? <Link to="/register">Register</Link>
      </p>
      <p className="muted" style={{ fontSize: "0.8rem" }}>
        Demo login is pre-filled: demo@bee.dev / beehero123
      </p>
    </div>
  );
}
