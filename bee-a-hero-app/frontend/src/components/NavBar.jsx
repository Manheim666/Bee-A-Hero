import { NavLink, useNavigate } from "react-router-dom";
import { useAuth } from "../auth/AuthContext.jsx";
import Hexagon from "./Hexagon.jsx";
import ThemeToggle from "./ThemeToggle.jsx";

const linkStyle = ({ isActive }) => ({
  padding: "8px 14px",
  borderRadius: 999,
  fontWeight: 600,
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
          gap: 14,
          paddingTop: 12,
          paddingBottom: 12,
          flexWrap: "wrap",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <Hexagon size={38}>🐝</Hexagon>
          <strong style={{ fontSize: "1.15rem" }}>Bee-A-Hero</strong>
        </div>

        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
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
        </div>

        <div
          style={{
            marginLeft: "auto",
            display: "flex",
            alignItems: "center",
            gap: 12,
          }}
        >
          <ThemeToggle />
          <span className="pill pill-done">{user?.username}</span>
          <button className="btn btn-ghost" onClick={handleLogout}>
            Logout
          </button>
        </div>
      </div>
    </nav>
  );
}
