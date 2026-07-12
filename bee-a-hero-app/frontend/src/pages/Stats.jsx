import { useEffect, useMemo, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import api from "../api";
import FilterBar from "../components/FilterBar.jsx";
import StatTile from "../components/StatTile.jsx";
import VisitBarChart from "../components/VisitBarChart.jsx";

const EMPTY_FILTERS = { video_id: "", from: "", to: "", pollinator: "" };

export default function Stats() {
  const [videos, setVideos] = useState([]);
  const [filters, setFilters] = useState(EMPTY_FILTERS);
  const [visits, setVisits] = useState(null);
  const [series, setSeries] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.get("/api/videos").then((res) => setVideos(res.data));
    api.get("/api/stats/timeseries").then((res) => setSeries(res.data.points));
  }, []);

  useEffect(() => {
    const params = {};
    if (filters.video_id) params.video_id = filters.video_id;
    if (filters.from) params.from = filters.from;
    if (filters.to) params.to = filters.to;
    if (filters.pollinator) params.pollinator = filters.pollinator;

    setLoading(true);
    api
      .get("/api/stats/visits", { params })
      .then((res) => setVisits(res.data))
      .finally(() => setLoading(false));
  }, [filters]);

  const pollinatorPct = useMemo(() => {
    if (!visits || visits.total === 0) return 0;
    return Math.round((100 * visits.pollinator) / visits.total);
  }, [visits]);

  return (
    <div className="container">
      <h1>Pollination analytics</h1>
      <p className="muted">
        Filter by video, date range, and insect type. Warm = pollinator, cool =
        non-pollinator, in every chart.
      </p>

      <FilterBar videos={videos} filters={filters} onChange={setFilters} />

      <div
        className="grid"
        style={{
          marginTop: 20,
          gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
        }}
      >
        <StatTile value={visits?.total ?? 0} label="Visits (filtered)" />
        <StatTile
          value={visits?.pollinator ?? 0}
          label="Pollinator visits"
          background="linear-gradient(135deg, var(--pollinator), var(--honey-deep))"
        />
        <StatTile
          value={visits?.non_pollinator ?? 0}
          label="Non-pollinator visits"
          background="linear-gradient(135deg, var(--non), #33627f)"
        />
        <StatTile value={`${pollinatorPct}%`} label="Pollinator share" />
      </div>

      <div className="card" style={{ marginTop: 20 }}>
        <h2>Visits per flower</h2>
        {loading ? (
          <p>
            <span className="spinner" /> Loading…
          </p>
        ) : (
          <VisitBarChart bars={visits?.bars} />
        )}
      </div>

      <div className="card" style={{ marginTop: 20 }}>
        <h2>Visits over time</h2>
        {series.length === 0 ? (
          <p className="muted">No time-series data yet.</p>
        ) : (
          <div style={{ width: "100%", height: 260 }}>
            <ResponsiveContainer>
              <LineChart data={series} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#efe3cc" />
                <XAxis dataKey="bucket" tick={{ fontSize: 12 }} />
                <YAxis allowDecimals={false} tick={{ fontSize: 12 }} />
                <Tooltip
                  contentStyle={{ borderRadius: 10, border: "1px solid var(--border)" }}
                />
                <Line
                  type="monotone"
                  dataKey="visits"
                  stroke="var(--honey-deep)"
                  strokeWidth={3}
                  dot={{ fill: "var(--honey)" }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </div>
  );
}
