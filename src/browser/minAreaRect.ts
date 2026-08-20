import type { Point } from "./types.js";

export interface RotatedRect {
  corners: [Point, Point, Point, Point];
  area: number;
}

/**
 * Minimum-area bounding rectangle of a convex hull, brute-force over every
 * hull edge direction (O(n²): n edges × n points). The textbook rotating-
 * calipers algorithm advances two pointers for O(n), but hulls here have at
 * most a few dozen vertices (pixel-mask contours of a handful of book
 * spines) — the optimization isn't worth the extra bug surface.
 *
 * Always returns 4 corners, unlike minAreaQuad — this is the fallback for
 * degenerate hulls (plan §12), and always produces a ROTATED RECTANGLE,
 * not a general quad. It loses perspective on a tilted spine (§1) — kept
 * only as the fallback and as the comparison baseline in tests.
 */
export function minAreaRect(hull: Point[]): RotatedRect {
  if (hull.length < 3) {
    throw new Error("minAreaRect requires at least 3 points");
  }

  let best: RotatedRect | null = null;

  for (let i = 0; i < hull.length; i++) {
    const p1 = hull[i]!;
    const p2 = hull[(i + 1) % hull.length]!;
    const dx = p2.x - p1.x;
    const dy = p2.y - p1.y;
    const len = Math.hypot(dx, dy);
    if (len === 0) continue;

    const ux = dx / len;
    const uy = dy / len;
    const vx = -uy;
    const vy = ux;

    let minU = Infinity;
    let maxU = -Infinity;
    let minV = Infinity;
    let maxV = -Infinity;
    for (const p of hull) {
      const u = p.x * ux + p.y * uy;
      const v = p.x * vx + p.y * vy;
      if (u < minU) minU = u;
      if (u > maxU) maxU = u;
      if (v < minV) minV = v;
      if (v > maxV) maxV = v;
    }

    const area = (maxU - minU) * (maxV - minV);
    if (!best || area < best.area) {
      const corners: [Point, Point, Point, Point] = [
        { x: minU * ux + minV * vx, y: minU * uy + minV * vy },
        { x: maxU * ux + minV * vx, y: maxU * uy + minV * vy },
        { x: maxU * ux + maxV * vx, y: maxU * uy + maxV * vy },
        { x: minU * ux + maxV * vx, y: minU * uy + maxV * vy },
      ];
      best = { corners, area };
    }
  }

  if (!best) {
    throw new Error("minAreaRect: degenerate hull (all points coincide)");
  }
  return best;
}
