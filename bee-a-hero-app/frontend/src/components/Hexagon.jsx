/**
 * Reusable honeycomb hexagon. Wraps children in a hexagonal clip-path so the
 * bee motif stays consistent across stat tiles, avatars, and the logo mark.
 */
export default function Hexagon({
  size = 72,
  background = "linear-gradient(135deg, var(--honey), var(--honey-deep))",
  color = "#fff",
  children,
  style = {},
}) {
  return (
    <div
      style={{
        width: size,
        height: size,
        clipPath:
          "polygon(50% 0%, 100% 25%, 100% 75%, 50% 100%, 0% 75%, 0% 25%)",
        background,
        color,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        textAlign: "center",
        fontWeight: 700,
        flexShrink: 0,
        ...style,
      }}
    >
      {children}
    </div>
  );
}
