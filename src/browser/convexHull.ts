import type { Point } from "./types.js";

function cross(o: Point, a: Point, b: Point): number {
  return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

/**
 * Andrew's monotone chain. Returns hull points as a simple (non-self-
 * intersecting) cyclic polygon — the exact winding direction doesn't matter
 * to callers (minAreaRect/minAreaQuad work on either), since final corner
 * ordering is fixed downstream by canonicalizeQuad (plan §2).
 */
export function convexHull(points: Point[]): Point[] {
  const pts = Array.from(
    new Map(points.map((p) => [`${p.x},${p.y}`, p])).values(),
  ).sort((a, b) => a.x - b.x || a.y - b.y);

  if (pts.length <= 2) return pts;

  const lower: Point[] = [];
  for (const p of pts) {
    while (
      lower.length >= 2 &&
      cross(lower[lower.length - 2]!, lower[lower.length - 1]!, p) <= 0
    ) {
      lower.pop();
    }
    lower.push(p);
  }

  const upper: Point[] = [];
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i]!;
    while (
      upper.length >= 2 &&
      cross(upper[upper.length - 2]!, upper[upper.length - 1]!, p) <= 0
    ) {
      upper.pop();
    }
    upper.push(p);
  }

  upper.pop();
  lower.pop();
  return lower.concat(upper);
}
