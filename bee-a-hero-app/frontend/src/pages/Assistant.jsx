import { useCallback, useEffect, useState } from "react";
import api from "../api";
import ChatSidebar from "../components/ChatSidebar.jsx";
import ChatWindow from "../components/ChatWindow.jsx";

export default function Assistant() {
  const [conversations, setConversations] = useState([]);
  const [active, setActive] = useState(null); // full conversation detail
  const [sending, setSending] = useState(false);

  const loadConversations = useCallback(async () => {
    const res = await api.get("/api/conversations");
    setConversations(res.data);
    return res.data;
  }, []);

  useEffect(() => {
    loadConversations();
  }, [loadConversations]);

  async function selectConversation(id) {
    const res = await api.get(`/api/conversations/${id}`);
    setActive(res.data);
  }

  async function newConversation() {
    const res = await api.post("/api/conversations");
    await loadConversations();
    setActive(res.data);
  }

  async function deleteConversation(id) {
    await api.delete(`/api/conversations/${id}`);
    if (active?.id === id) setActive(null);
    loadConversations();
  }

  async function sendMessage(content, provider) {
    let conv = active;
    if (!conv) {
      const res = await api.post("/api/conversations");
      conv = res.data;
      setActive(conv);
    }
    // Optimistically show the user's message.
    setActive((prev) => ({
      ...conv,
      messages: [
        ...(prev?.id === conv.id ? prev.messages : conv.messages),
        { id: `tmp-${Date.now()}`, role: "user", content },
      ],
    }));
    setSending(true);
    try {
      await api.post(`/api/conversations/${conv.id}/messages`, { content, provider });
      const res = await api.get(`/api/conversations/${conv.id}`);
      setActive(res.data);
      loadConversations();
    } finally {
      setSending(false);
    }
  }

  return (
    <div className="container">
      <h1>Assistant</h1>
      <p className="muted">
        Ask about your pollination results and how the detection works. Your
        stats are shared with the assistant for grounded answers.
      </p>

      <div
        className="grid"
        style={{
          gridTemplateColumns: "minmax(200px, 260px) 1fr",
          alignItems: "stretch",
          minHeight: 480,
        }}
      >
        <ChatSidebar
          conversations={conversations}
          activeId={active?.id}
          onSelect={selectConversation}
          onNew={newConversation}
          onDelete={deleteConversation}
        />
        <ChatWindow
          conversation={active}
          sending={sending}
          onSend={sendMessage}
        />
      </div>
    </div>
  );
}
