import { useState } from "react";
import { Link, Navigate, useNavigate } from "react-router-dom";
import { useAuth } from "../auth/AuthContext.jsx";
import Hexagon from "../components/Hexagon.jsx";

export default function Register() {
  const { user, register } = useAuth();
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  if (user) return <Navigate to="/dashboard" replace />;

  async function submit(e) {
    e.preventDefault();
    setError("");
    setBusy(true);
    try {
      await register(email, username, password);
      navigate("/dashboard");
    } catch (err) {
      setError(err.response?.data?.detail || "Registration failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div
      className="container"
      style={{ maxWidth: 420, marginTop: 60, textAlign: "center" }}
    >
      <div style={{ display: "flex", justifyContent: "center", marginBottom: 14 }}>
        <Hexagon size={72} style={{ fontSize: "1.8rem" }}>
          🐝
        </Hexagon>
      </div>
      <h1>Create account</h1>

      <form className="card" onSubmit={submit} style={{ marginTop: 18, textAlign: "left" }}>
        <label className="muted" style={{ fontSize: "0.85rem" }}>
          Username
        </label>
        <input
          className="input"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          style={{ marginBottom: 12 }}
          required
        />
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
        <input
          className="input"
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          style={{ marginBottom: 16 }}
          required
        />
        {error && (
          <p style={{ color: "#a13020", fontSize: "0.9rem" }}>{error}</p>
        )}
        <button className="btn" style={{ width: "100%" }} disabled={busy}>
          {busy ? "Creating…" : "Register"}
        </button>
      </form>

      <p className="muted" style={{ marginTop: 14 }}>
        Already have an account? <Link to="/login">Sign in</Link>
      </p>
    </div>
  );
}
