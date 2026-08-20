import { lineIntersection, polygonArea, polygonPerimeter } from "./polygonMath.js";
import type { Point } from "./types.js";

/**
 * Expands a convex quad outward by the DBNet unclip formula:
 * distance = area * unclipRatio / perimeter. Each edge is translated
 * outward along its normal by `distance`, then consecutive translated
 * edges are re-intersected to find the new corners (plan §6 step 4).
 *
 * This is the fix for "a spine cropped exactly at its polygon edge loses a
 * letter at the edge of the title" (plan §7) — shrink-mask training
 * deliberately under-predicts the true spine boundary, and unclip un-does
 * that shrink at inference time.
 */
export function unclipQuad(quad: Point[], unclipRatio: number): Point[] {
  const area = polygonArea(quad);
  const perimeter = polygonPerimeter(quad);
  if (perimeter === 0) return quad;
  const distance = (area * unclipRatio) / perimeter;

  const n = quad.length;
  const cx = quad.reduce((s, p) => s + p.x, 0) / n;
  const cy = quad.reduce((s, p) => s + p.y, 0) / n;

  const offsetEdges: Array<{ p1: Point; p2: Point }> = [];
  for (let i = 0; i < n; i++) {
    const a = quad[i]!;
    const b = quad[(i + 1) % n]!;
    const ex = b.x - a.x;
    const ey = b.y - a.y;
    const len = Math.hypot(ex, ey);
    if (len === 0) {
      offsetEdges.push({ p1: a, p2: b });
      continue;
    }
    // Perpendicular to the edge; sign chosen to point away from centroid.
    let nx = -ey / len;
    let ny = ex / len;
    const midx = (a.x + b.x) / 2;
    const midy = (a.y + b.y) / 2;
    if ((midx - cx) * nx + (midy - cy) * ny < 0) {
      nx = -nx;
      ny = -ny;
    }
    offsetEdges.push({
      p1: { x: a.x + nx * distance, y: a.y + ny * distance },
      p2: { x: b.x + nx * distance, y: b.y + ny * distance },
    });
  }

  const result: Point[] = [];
  for (let i = 0; i < n; i++) {
    const prevEdge = offsetEdges[(i - 1 + n) % n]!;
    const currEdge = offsetEdges[i]!;
    const corner = lineIntersection(prevEdge.p1, prevEdge.p2, currEdge.p1, currEdge.p2);
    // Parallel neighboring edges (shouldn't happen for a real quad, but
    // degenerate input is possible): fall back to the un-intersected point
    // rather than throwing.
    result.push(corner ?? currEdge.p1);
  }
  return result;
}
