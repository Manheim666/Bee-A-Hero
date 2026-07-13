import { NavLink, useNavigate } from "react-router-dom";
import { useAuth } from "../auth/AuthContext.jsx";
import Hexagon from "./Hexagon.jsx";
import ThemeToggle from "./ThemeToggle.jsx";

const PILL = {
  padding: "8px 14px",
  borderRadius: 999,
  fontWeight: 600,
  whiteSpace: "nowrap",
  display: "inline-flex",
  alignItems: "center",
  gap: 6,
  transition: "background 0.15s ease, color 0.15s ease",
};

const linkStyle = ({ isActive }) => ({
  ...PILL,
  color: isActive ? "#fff" : "var(--charcoal)",
  background: isActive
    ? "linear-gradient(135deg, var(--honey), var(--honey-deep))"
    : "transparent",
});

export default function NavBar() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  function handleLogout() {
    logout();
    navigate("/login");
  }

  return (
    <nav
      style={{
        background: "var(--card)",
        borderBottom: "1px solid var(--border)",
        boxShadow: "var(--shadow)",
      }}
    >
      <div
        className="container"
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 16,
          paddingTop: 12,
          paddingBottom: 12,
          flexWrap: "wrap",
        }}
      >
        {/* Left: brand */}
        <div style={{ display: "flex", alignItems: "center", gap: 10, flexShrink: 0 }}>
          <Hexagon size={38}>🐝</Hexagon>
          <strong style={{ fontSize: "1.15rem" }}>Bee-A-Hero</strong>
        </div>

        {/* Center: primary tabs */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 6,
            flexWrap: "wrap",
            justifyContent: "center",
          }}
        >
          <NavLink to="/dashboard" style={linkStyle}>
            Dashboard
          </NavLink>
          <NavLink to="/upload" style={linkStyle}>
            Upload
          </NavLink>
          <NavLink to="/stats" style={linkStyle}>
            Stats
          </NavLink>
          <NavLink to="/assistant" style={linkStyle}>
            Assistant
          </NavLink>
          <a
            href="http://localhost:8001/"
            target="_blank"
            rel="noopener noreferrer"
            style={{ ...PILL, color: "var(--charcoal)", border: "1px solid var(--border)" }}
            title="Open the DroidCam live detection viewer"
          >
            📷 Live Camera
          </a>
        </div>

        {/* Right: session actions */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 12,
            flexShrink: 0,
          }}
        >
          <ThemeToggle />
          <span className="pill pill-done" style={{ whiteSpace: "nowrap" }}>
            {user?.username}
          </span>
          <button
            className="btn btn-ghost"
            style={{ whiteSpace: "nowrap" }}
            onClick={handleLogout}
          >
            Logout
          </button>
        </div>
      </div>
    </nav>
  );
}
