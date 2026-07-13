import { useEffect, useRef, useState } from "react";
import api from "../api";

function Bubble({ role, children }) {
  const isUser = role === "user";
  return (
    <div
      style={{
        display: "flex",
        justifyContent: isUser ? "flex-end" : "flex-start",
      }}
    >
      <div
        style={{
          maxWidth: "78%",
          padding: "10px 14px",
          borderRadius: 14,
          background: isUser
            ? "linear-gradient(135deg, var(--honey), var(--honey-deep))"
            : "var(--queued-bg)",
          color: isUser ? "#1a1208" : "var(--bee-black)",
          border: isUser ? "none" : "1px solid var(--border)",
          whiteSpace: "pre-wrap",
          lineHeight: 1.45,
        }}
      >
        {children}
      </div>
    </div>
  );
}

export default function ChatWindow({ conversation, sending, onSend }) {
  const [text, setText] = useState("");
  const [provider, setProvider] = useState("auto");
  const [providers, setProviders] = useState([{ id: "auto", label: "Auto", available: true }]);
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [conversation?.messages?.length, sending]);

  useEffect(() => {
    api
      .get("/api/conversations/providers")
      .then((res) => setProviders(res.data))
      .catch(() => {});
  }, []);

  if (!conversation) {
    return (
      <div
        className="card"
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          height: "100%",
        }}
      >
        <p className="muted">Start a new chat or pick one on the left.</p>
      </div>
    );
  }

  function submit(e) {
    e.preventDefault();
    const value = text.trim();
    if (!value || sending) return;
    onSend(value, provider);
    setText("");
  }

  return (
    <div
      className="card"
      style={{ display: "flex", flexDirection: "column", height: "100%" }}
    >
      <div
        style={{
          flex: 1,
          overflowY: "auto",
          display: "flex",
          flexDirection: "column",
          gap: 12,
          paddingRight: 6,
          minHeight: 300,
        }}
      >
        {conversation.messages.length === 0 && (
          <p className="muted">
            Ask about your pollination stats — e.g. “How many pollinator visits
            did I get?”
          </p>
        )}
        {conversation.messages.map((m) => (
          <Bubble key={m.id} role={m.role}>
            {m.content}
          </Bubble>
        ))}
        {sending && (
          <Bubble role="assistant">
            <span className="spinner" /> thinking…
          </Bubble>
        )}
        <div ref={endRef} />
      </div>

      <form onSubmit={submit} style={{ display: "flex", gap: 8, marginTop: 12, flexWrap: "wrap" }}>
        <select
          className="input"
          value={provider}
          onChange={(e) => setProvider(e.target.value)}
          title="Choose which model answers"
          style={{ width: "auto", flex: "0 0 auto", cursor: "pointer" }}
        >
          {providers.map((p) => (
            <option key={p.id} value={p.id} disabled={!p.available}>
              {p.label}
              {p.available ? "" : " (add key)"}
            </option>
          ))}
        </select>
        <input
          className="input"
          style={{ flex: 1, minWidth: 180 }}
          placeholder="Ask the Bee-A-Hero assistant…"
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
        <button className="btn" type="submit" disabled={sending || !text.trim()}>
          Send
        </button>
      </form>
    </div>
  );
}
