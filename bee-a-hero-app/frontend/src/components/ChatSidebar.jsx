export default function ChatSidebar({
  conversations,
  activeId,
  onSelect,
  onNew,
  onDelete,
}) {
  return (
    <div
      className="card"
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 10,
        height: "100%",
      }}
    >
      <button className="btn" onClick={onNew}>
        + New chat
      </button>
      <div style={{ display: "flex", flexDirection: "column", gap: 6, overflowY: "auto" }}>
        {conversations.length === 0 && (
          <p className="muted" style={{ fontSize: "0.85rem" }}>
            No conversations yet.
          </p>
        )}
        {conversations.map((c) => (
          <div
            key={c.id}
            onClick={() => onSelect(c.id)}
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              gap: 6,
              padding: "9px 12px",
              borderRadius: 10,
              cursor: "pointer",
              background:
                c.id === activeId ? "var(--amber-glow)" : "transparent",
            }}
          >
            <span
              style={{
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
                fontSize: "0.9rem",
                fontWeight: 500,
              }}
              title={c.title}
            >
              {c.title}
            </span>
            <button
              className="btn btn-ghost"
              style={{ padding: "2px 9px", fontSize: "0.75rem" }}
              onClick={(e) => {
                e.stopPropagation();
                onDelete(c.id);
              }}
            >
              ✕
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
