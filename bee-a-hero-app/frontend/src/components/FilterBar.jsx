/** Controls that drive the Stats charts: video, date range, pollinator toggle. */
export default function FilterBar({ videos, filters, onChange }) {
  function set(key, value) {
    onChange({ ...filters, [key]: value });
  }

  return (
    <div
      className="card"
      style={{
        display: "flex",
        gap: 16,
        flexWrap: "wrap",
        alignItems: "flex-end",
      }}
    >
      <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        <span className="muted" style={{ fontSize: "0.8rem" }}>
          Video
        </span>
        <select
          className="input"
          value={filters.video_id}
          onChange={(e) => set("video_id", e.target.value)}
        >
          <option value="">All videos</option>
          {videos.map((v) => (
            <option key={v.id} value={v.id}>
              {v.original_name}
            </option>
          ))}
        </select>
      </label>

      <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        <span className="muted" style={{ fontSize: "0.8rem" }}>
          From
        </span>
        <input
          type="date"
          className="input"
          value={filters.from}
          onChange={(e) => set("from", e.target.value)}
        />
      </label>

      <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        <span className="muted" style={{ fontSize: "0.8rem" }}>
          To
        </span>
        <input
          type="date"
          className="input"
          value={filters.to}
          onChange={(e) => set("to", e.target.value)}
        />
      </label>

      <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        <span className="muted" style={{ fontSize: "0.8rem" }}>
          Type
        </span>
        <select
          className="input"
          value={filters.pollinator}
          onChange={(e) => set("pollinator", e.target.value)}
        >
          <option value="">Both</option>
          <option value="true">Pollinator</option>
          <option value="false">Non-pollinator</option>
        </select>
      </label>
    </div>
  );
}
