import type { Point } from "./types.js";

/** Shoelace formula. Works regardless of winding direction (abs()). */
export function polygonArea(poly: Point[]): number {
  let sum = 0;
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i]!;
    const b = poly[(i + 1) % poly.length]!;
    sum += a.x * b.y - b.x * a.y;
  }
  return Math.abs(sum) / 2;
}

export function polygonPerimeter(poly: Point[]): number {
  let sum = 0;
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i]!;
    const b = poly[(i + 1) % poly.length]!;
    sum += Math.hypot(b.x - a.x, b.y - a.y);
  }
  return sum;
}

/**
 * Intersection of infinite lines (p1,p2) and (p3,p4). Returns null when the
 * lines are near-parallel — checked via the sine of the angle between the
 * two directions, which is scale-independent (unlike a raw cross-product
 * epsilon, which breaks down at pixel-coordinate magnitudes). Pixel-grid
 * hull edges are frequently exactly or near parallel, so this can't be
 * skipped.
 */
export function lineIntersection(
  p1: Point,
  p2: Point,
  p3: Point,
  p4: Point,
): Point | null {
  const d1x = p2.x - p1.x;
  const d1y = p2.y - p1.y;
  const d2x = p4.x - p3.x;
  const d2y = p4.y - p3.y;
  const len1 = Math.hypot(d1x, d1y);
  const len2 = Math.hypot(d2x, d2y);
  if (len1 < 1e-9 || len2 < 1e-9) return null;

  const sinAngle = (d1x * d2y - d1y * d2x) / (len1 * len2);
  if (Math.abs(sinAngle) < 1e-6) return null;

  const denom = d1x * d2y - d1y * d2x;
  const t = ((p3.x - p1.x) * d2y - (p3.y - p1.y) * d2x) / denom;
  return { x: p1.x + t * d1x, y: p1.y + t * d1y };
}
