import type { QuadCoords, RawQuad } from "./types.js";

/**
 * Bounding rect of a quad. Works in whatever coordinate space the quad is
 * already in (model-input px, original-image px, or normalized 0..1000) —
 * callers normalize first, then derive the AABB, so the two never disagree.
 */
export function quadToAABB(quad: QuadCoords): {
  ymin: number;
  xmin: number;
  ymax: number;
  xmax: number;
} {
  const xs = [quad[0], quad[2], quad[4], quad[6]];
  const ys = [quad[1], quad[3], quad[5], quad[7]];
  return {
    xmin: Math.min(...xs),
    xmax: Math.max(...xs),
    ymin: Math.min(...ys),
    ymax: Math.max(...ys),
  };
}

function dist(a: [number, number], b: [number, number]): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

/**
 * Reorders 4 arbitrary (unordered) quad corners into the canonical order
 * from plan §2: corner 0 = top-left once the spine is rotated to
 * horizontal, then clockwise (TL, TR, BR, BL).
 *
 * A book spine quad is elongated (tall/narrow or wide/short), so the two
 * short edges are the spine's physical top and bottom; the two long edges
 * are its left/right sides. We find those by edge length, pick the short
 * edge closest to the image top as "top", then walk the polygon from
 * there — no assumption is made about the input's original point order.
 */
export function canonicalizeQuad(points: RawQuad): QuadCoords {
  const cx = points.reduce((s, p) => s + p[0], 0) / 4;
  const cy = points.reduce((s, p) => s + p[1], 0) / 4;

  // Sort by angle around centroid -> clockwise cyclic order in image
  // coordinates (y-down), since increasing atan2 sweeps right->down->left->up.
  const ordered = [...points].sort(
    (a, b) => Math.atan2(a[1] - cy, a[0] - cx) - Math.atan2(b[1] - cy, b[0] - cx),
  );

  const edgeLen = [0, 1, 2, 3].map((i) => dist(ordered[i]!, ordered[(i + 1) % 4]!));
  const pairSumA = edgeLen[0]! + edgeLen[2]!; // edges 0&2 are opposite
  const pairSumB = edgeLen[1]! + edgeLen[3]!; // edges 1&3 are opposite
  const shortPair: [number, number] = pairSumA < pairSumB ? [0, 2] : [1, 3];

  const midY = (i: number) => (ordered[i]![1] + ordered[(i + 1) % 4]![1]) / 2;
  const topEdge = midY(shortPair[0]) <= midY(shortPair[1]) ? shortPair[0] : shortPair[1];

  const a = ordered[topEdge]!;
  const b = ordered[(topEdge + 1) % 4]!;

  let tl: [number, number];
  let tr: [number, number];
  let br: [number, number];
  let bl: [number, number];

  if (a[0] <= b[0]) {
    tl = a;
    tr = b;
    br = ordered[(topEdge + 2) % 4]!;
    bl = ordered[(topEdge + 3) % 4]!;
  } else {
    // 'a' is actually the top-right corner: walk the OTHER neighbor of 'a'
    // (not b) to reach BR, keeping the result a simple clockwise polygon.
    tl = b;
    tr = a;
    br = ordered[(topEdge + 3) % 4]!;
    bl = ordered[(topEdge + 2) % 4]!;
  }

  return [tl[0], tl[1], tr[0], tr[1], br[0], br[1], bl[0], bl[1]];
}
