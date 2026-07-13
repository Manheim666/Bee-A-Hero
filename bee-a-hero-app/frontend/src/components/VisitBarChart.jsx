import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

/**
 * Stacked bar chart of per-flower visits.
 * Warm = pollinator, cool = non-pollinator — the same two colors everywhere.
 */
export default function VisitBarChart({ bars }) {
  if (!bars || bars.length === 0) {
    return (
      <p className="muted">No visits match the current filters.</p>
    );
  }

  const data = bars.map((b) => ({
    flower: `flower_${b.flower_id}`,
    pollinator: b.pollinator,
    non_pollinator: b.non_pollinator,
  }));

  return (
    <div style={{ width: "100%", height: 340 }}>
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(128,128,128,0.25)" />
          <XAxis dataKey="flower" tick={{ fontSize: 12 }} />
          <YAxis allowDecimals={false} tick={{ fontSize: 12 }} />
          <Tooltip
            contentStyle={{
              borderRadius: 10,
              border: "1px solid var(--border)",
              background: "var(--card)",
              color: "var(--bee-black)",
            }}
          />
          <Legend />
          <Bar
            dataKey="pollinator"
            stackId="v"
            fill="var(--pollinator)"
            name="Pollinator"
            radius={[0, 0, 0, 0]}
          />
          <Bar
            dataKey="non_pollinator"
            stackId="v"
            fill="var(--non)"
            name="Non-pollinator"
            radius={[6, 6, 0, 0]}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
