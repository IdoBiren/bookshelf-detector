import { lineIntersection, polygonArea } from "./polygonMath.js";
import type { Point } from "./types.js";

/**
 * Removes hull edge `i` by extending its two NEIGHBORING edges (the one
 * before it and the one after it) until they meet. This merges the two
 * endpoints of the removed edge into a single intersection point, reducing
 * the vertex count by exactly one while keeping the polygon convex and
 * circumscribing the original hull (we extend outward, never cut inward —
 * unlike corner-clipping simplification, this never loses spine pixels).
 *
 * Returns null when the two neighboring edges are parallel (no
 * intersection) — the caller skips this edge as a removal candidate rather
 * than treating it as an error.
 */
function removeEdge(poly: Point[], i: number): Point[] | null {
  const n = poly.length;
  const prev = poly[(i - 1 + n) % n]!;
  const curr = poly[i]!;
  const next = poly[(i + 1) % n]!;
  const afterNext = poly[(i + 2) % n]!;

  const intersection = lineIntersection(prev, curr, next, afterNext);
  if (!intersection) return null;

  const newPoly: Point[] = [];
  for (let k = 0; k < n - 2; k++) {
    newPoly.push(poly[(i + 2 + k) % n]!);
  }
  newPoly.push(intersection);
  return newPoly;
}

/**
 * Reduces a convex hull to a 4-vertex bounding quadrilateral by greedily
 * removing, one at a time, whichever edge adds the least area when its
 * neighbors are extended to replace it (plan §1/§12). Unlike minAreaRect,
 * this preserves a genuine trapezoid — the shape a tilted, perspective-
 * skewed book spine actually has — instead of forcing a rotated rectangle.
 *
 * Returns null (caller falls back to minAreaRect) when:
 * - the hull already has fewer than 4 vertices (a blob too small/degenerate
 *   to have a meaningful quad), or
 * - every remaining edge has parallel neighbors, so no further reduction
 *   is possible via this method.
 */
export function minAreaQuad(hull: Point[]): Point[] | null {
  if (hull.length < 4) return null;

  let poly = [...hull];

  while (poly.length > 4) {
    const n = poly.length;
    const currentArea = polygonArea(poly);
    let bestCandidate: Point[] | null = null;
    let bestAreaIncrease = Infinity;

    for (let i = 0; i < n; i++) {
      const candidate = removeEdge(poly, i);
      if (!candidate) continue;
      const increase = polygonArea(candidate) - currentArea;
      // Extending two convex-hull edges outward to remove a vertex must
      // never shrink the enclosed area. A negative "increase" means the
      // two lines crossed on the wrong side, producing a self-intersecting
      // (invalid) polygon rather than a valid circumscribing one — reject
      // it exactly like the parallel-edges case, don't let its spuriously
      // small shoelace area win the "minimum increase" search. (Found by
      // a failing test: a bottom edge's neighbors extended "backward" and
      // produced a bowtie polygon whose area came out at 720 against a
      // hull area of 3280.)
      if (increase < -1e-6) continue;
      if (increase < bestAreaIncrease) {
        bestAreaIncrease = increase;
        bestCandidate = candidate;
      }
    }

    if (!bestCandidate) return null;
    poly = bestCandidate;
  }

  return poly;
}
