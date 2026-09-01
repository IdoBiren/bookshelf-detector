"""
Python mirror of src/browser/unclip.ts + polygonMath.ts (plan §13 phase B).

`pyclipper` is deliberately NOT used here, even though it is what every
public DBNet implementation reaches for. Two reasons, not one:

1. `postprocess.ts:64` fits a `minAreaQuad` BEFORE calling `unclipQuad` — at
   inference the model only ever expands a 4-vertex QUAD, never a raw
   many-vertex contour. Training targets must therefore be built from the
   same quad, or the model learns to predict a shape the browser's
   post-processing can never produce. Once that quad-reduction happens,
   `pyclipper`'s `JT_MITER` join and this module's line-intersection offset
   are the SAME operation (miter = extend edges to their intersection,
   which is exactly what `offset_polygon` below does) — so the choice
   between them is a dependency-footprint question, not a correctness one,
   and this module exists mainly to guarantee parity with the ALREADY
   SHIPPING `unclip.ts` (see the golden fixture in
   src/browser/__tests__/fixtures/unclipParity.json, generated from
   unclipQuad's own output and shared with unclip.test.ts).
2. Reflex (non-convex) vertices break `unclip.ts`'s own centroid-based
   normal-sign test — 22.6% of pretrain polygons are non-convex — so `raw`
   mode is unsafe with EITHER implementation. The fix is quad-reduction
   first, not swapping libraries.

Every function takes plain `(float, float)` tuples and works for any vertex
count `>= 3`, exactly like the TS source (`unclipQuad`'s own `n` is not
fixed at 4 despite the misleading name).
"""

from __future__ import annotations

import math

Point = tuple[float, float]


def polygon_area(poly: list[Point]) -> float:
    """Shoelace formula — mirrors polygonMath.ts:4."""
    total = 0.0
    n = len(poly)
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        total += ax * by - bx * ay
    return abs(total) / 2.0


def polygon_perimeter(poly: list[Point]) -> float:
    """Mirrors polygonMath.ts:14."""
    total = 0.0
    n = len(poly)
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        total += math.hypot(bx - ax, by - ay)
    return total


def line_intersection(p1: Point, p2: Point, p3: Point, p4: Point) -> Point | None:
    """Intersection of infinite lines (p1,p2) and (p3,p4). Mirrors
    polygonMath.ts:32 exactly, including the scale-independent sinAngle
    near-parallel check (pixel-grid hull edges are frequently exactly or
    near parallel, so a raw cross-product epsilon isn't enough)."""
    d1x, d1y = p2[0] - p1[0], p2[1] - p1[1]
    d2x, d2y = p4[0] - p3[0], p4[1] - p3[1]
    len1 = math.hypot(d1x, d1y)
    len2 = math.hypot(d2x, d2y)
    if len1 < 1e-9 or len2 < 1e-9:
        return None

    sin_angle = (d1x * d2y - d1y * d2x) / (len1 * len2)
    if abs(sin_angle) < 1e-6:
        return None

    denom = d1x * d2y - d1y * d2x
    t = ((p3[0] - p1[0]) * d2y - (p3[1] - p1[1]) * d2x) / denom
    return (p1[0] + t * d1x, p1[1] + t * d1y)


def offset_polygon(poly: list[Point], distance: float) -> list[Point]:
    """Mirrors unclipQuad's body exactly, but takes a signed DISTANCE
    instead of fusing "compute distance" and "offset by it" into one
    function. Positive = outward (identical to unclipQuad); negative =
    inward (the shrink direction DBNet targets need). Splitting these apart
    is what makes shrink and unclip provably the same operation with
    opposite sign, checkable by reading two one-line callers instead of two
    separately-written algorithms.
    """
    n = len(poly)
    if n == 0:
        return poly

    cx = sum(p[0] for p in poly) / n
    cy = sum(p[1] for p in poly) / n

    offset_edges: list[tuple[Point, Point]] = []
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        ex, ey = bx - ax, by - ay
        length = math.hypot(ex, ey)
        if length == 0:
            offset_edges.append(((ax, ay), (bx, by)))
            continue

        # Perpendicular to the edge; sign chosen to point away from centroid
        # (for a NEGATIVE distance this points the offset inward instead —
        # the sign of `distance` composes with this normal automatically).
        nx = -ey / length
        ny = ex / length
        midx, midy = (ax + bx) / 2, (ay + by) / 2
        if (midx - cx) * nx + (midy - cy) * ny < 0:
            nx, ny = -nx, -ny

        offset_edges.append(
            ((ax + nx * distance, ay + ny * distance), (bx + nx * distance, by + ny * distance))
        )

    result: list[Point] = []
    for i in range(n):
        prev_edge = offset_edges[(i - 1) % n]
        curr_edge = offset_edges[i]
        corner = line_intersection(prev_edge[0], prev_edge[1], curr_edge[0], curr_edge[1])
        # Parallel neighboring edges: fall back to the un-intersected point
        # rather than raising, exactly like unclip.ts's `corner ?? currEdge.p1`.
        result.append(corner if corner is not None else curr_edge[0])
    return result


def unclip_distance(poly: list[Point], unclip_ratio: float) -> float:
    """Mirrors unclip.ts:19 — area*ratio/perimeter, bare. Returns 0 for a
    degenerate (zero-perimeter) polygon rather than raising, matching
    unclipQuad's own `if (perimeter === 0) return quad` short-circuit."""
    perimeter = polygon_perimeter(poly)
    if perimeter == 0:
        return 0.0
    return polygon_area(poly) * unclip_ratio / perimeter


def shrink_distance(poly: list[Point], shrink_ratio: float) -> float:
    """DBNet's shrink formula: area*(1-r^2)/perimeter. NOT the same
    arithmetic as unclip_distance despite both taking a "ratio" — shrink
    folds in (1-r^2) before dividing; reading shrink_ratio=0.4 as a bare
    offset factor (skipping the square) is a 2.1x error. See
    TestInverseProperty.test_shrink_ratio_point_four_means_offset_factor...
    for the pinned identity: shrink_distance(P, r) == unclip_distance(P, 1-r**2).
    """
    perimeter = polygon_perimeter(poly)
    if perimeter == 0:
        return 0.0
    return polygon_area(poly) * (1 - shrink_ratio**2) / perimeter


def unclip_polygon(poly: list[Point], unclip_ratio: float) -> list[Point]:
    """The direct mirror of unclipQuad — offset_polygon recomposed with
    unclip_distance, so this is checkable against the TS golden fixture as
    a single call, the same way a caller would use unclipQuad."""
    return offset_polygon(poly, unclip_distance(poly, unclip_ratio))


def shrink_polygon(poly: list[Point], shrink_ratio: float) -> list[Point]:
    """Inward offset by DBNet's shrink formula — offset_polygon with a
    NEGATIVE distance, the sign that makes this the geometric inverse
    direction of unclip_polygon."""
    return offset_polygon(poly, -shrink_distance(poly, shrink_ratio))


def exterior_angle_sum(poly: list[Point]) -> float:
    """Q = sum(tan(exterior_angle_i / 2)) over every vertex. For any
    rectangle every turn is 90 degrees and tan(45 deg)=1, so Q=4 regardless
    of the rectangle's aspect ratio — this is the invariant
    exact_unclip_distance's closed form depends on."""
    n = len(poly)
    total = 0.0
    for i in range(n):
        px, py = poly[(i - 1) % n]
        cx, cy = poly[i]
        nx, ny = poly[(i + 1) % n]
        in_x, in_y = cx - px, cy - py
        out_x, out_y = nx - cx, ny - cy
        cross = in_x * out_y - in_y * out_x
        dot = in_x * out_x + in_y * out_y
        turning_angle = math.atan2(cross, dot)
        total += math.tan(turning_angle / 2)
    return total


def exact_unclip_distance(shrunk: list[Point], shrink_ratio: float) -> float:
    """The exact inverse of shrink_polygon(P, shrink_ratio) == shrunk,
    computed from the SHRUNK polygon alone (the original is not needed).

    Derivation (plan §13): with inward offset D, a convex polygon's
    perimeter and area after offsetting relate to Q = exterior_angle_sum by
        L_S = L_P - 2*D*Q
        A_S = A_P - L_P*D + D^2*Q
    and the shrink itself is defined by D = k*A_P/L_P where k = 1-shrink_ratio^2.
    Eliminating A_P and L_P between these three equations gives a quadratic
    in D solvable from (L_S, A_S, Q) alone:
        Q*(2-k)*D^2 + L_S*(1-k)*D - k*A_S = 0
    Verified by test: max corner error is exactly 0.0 (float64) on real
    quads; degrades gracefully (not exactly) on many-vertex hulls because
    inward offsetting collapses short edges, changing Q.
    """
    k = 1 - shrink_ratio**2
    q = exterior_angle_sum(shrunk)
    area_s = polygon_area(shrunk)
    perimeter_s = polygon_perimeter(shrunk)

    a = q * (2 - k)
    b = perimeter_s * (1 - k)
    c = -k * area_s

    if abs(a) < 1e-12:
        # Q ~ 0 (degenerate/near-straight polygon) -- falls back to the
        # linear solve of b*D + c = 0.
        return -c / b if b != 0 else 0.0

    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        discriminant = 0.0
    return (-b + math.sqrt(discriminant)) / (2 * a)


def _cross(o: Point, a: Point, b: Point) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def convex_hull(points: list[Point]) -> list[Point]:
    """Andrew's monotone chain. Mirrors src/browser/convexHull.ts exactly,
    including its tie-breaking (sort by x then y, `<= 0` pop condition), so
    a hull computed here and one computed in the browser agree on
    degenerate collinear runs — required for quad-mode targets to match
    what inference actually does with the same raw points."""
    unique = sorted(set(points))
    if len(unique) <= 2:
        return unique

    lower: list[Point] = []
    for p in unique:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list[Point] = []
    for p in reversed(unique):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    upper.pop()
    lower.pop()
    return lower + upper


def _remove_edge(poly: list[Point], i: int) -> list[Point] | None:
    """Mirrors minAreaQuad.ts's removeEdge: deletes vertex i by extending
    its two NEIGHBORING edges until they meet, reducing vertex count by one
    while staying convex and circumscribing (extends outward, never cuts
    inward)."""
    n = len(poly)
    prev_p, curr_p, next_p, after_next_p = (
        poly[(i - 1) % n],
        poly[i],
        poly[(i + 1) % n],
        poly[(i + 2) % n],
    )
    intersection = line_intersection(prev_p, curr_p, next_p, after_next_p)
    if intersection is None:
        return None
    new_poly = [poly[(i + 2 + k) % n] for k in range(n - 2)]
    new_poly.append(intersection)
    return new_poly


def min_area_quad(hull: list[Point]) -> list[Point] | None:
    """Mirrors src/browser/minAreaQuad.ts exactly, including the
    `increase < -1e-6` rejection rule documented in HANDOFF.md: without it
    the greedy reduction can prefer a self-intersecting bowtie whose
    shoelace area is spuriously SMALLER than the hull's, because extending
    two edges the wrong way flips the polygon inside-out rather than
    growing it. Returns None (caller falls back to min_area_rect) below 4
    hull vertices or when every remaining edge has parallel neighbors."""
    if len(hull) < 4:
        return None

    poly = list(hull)
    while len(poly) > 4:
        n = len(poly)
        current_area = polygon_area(poly)
        best_candidate: list[Point] | None = None
        best_increase = math.inf

        for i in range(n):
            candidate = _remove_edge(poly, i)
            if candidate is None:
                continue
            increase = polygon_area(candidate) - current_area
            if increase < -1e-6:
                continue
            if increase < best_increase:
                best_increase = increase
                best_candidate = candidate

        if best_candidate is None:
            return None
        poly = best_candidate

    return poly


def min_area_rect(hull: list[Point]) -> list[Point]:
    """Mirrors src/browser/minAreaRect.ts — brute-force over every hull edge
    direction, O(n^2). Always a rotated RECTANGLE (loses perspective on a
    tilted spine); kept only as min_area_quad's fallback."""
    if len(hull) < 3:
        raise ValueError("min_area_rect requires at least 3 points")

    best_corners: list[Point] | None = None
    best_area = math.inf

    for i in range(len(hull)):
        p1, p2 = hull[i], hull[(i + 1) % len(hull)]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        length = math.hypot(dx, dy)
        if length == 0:
            continue
        ux, uy = dx / length, dy / length
        vx, vy = -uy, ux

        min_u = min_v = math.inf
        max_u = max_v = -math.inf
        for px, py in hull:
            u = px * ux + py * uy
            v = px * vx + py * vy
            min_u, max_u = min(min_u, u), max(max_u, u)
            min_v, max_v = min(min_v, v), max(max_v, v)

        area = (max_u - min_u) * (max_v - min_v)
        if area < best_area:
            best_area = area
            best_corners = [
                (min_u * ux + min_v * vx, min_u * uy + min_v * vy),
                (max_u * ux + min_v * vx, max_u * uy + min_v * vy),
                (max_u * ux + max_v * vx, max_u * uy + max_v * vy),
                (min_u * ux + max_v * vx, min_u * uy + max_v * vy),
            ]

    if best_corners is None:
        raise ValueError("min_area_rect: degenerate hull (all points coincide)")
    return best_corners


def is_simple_polygon(poly: list[Point]) -> bool:
    """O(n^2) segment-pair intersection test — n is at most a few dozen for
    a hull/quad here, so the naive approach is fine. A shrunk polygon that
    self-intersects (possible on non-convex input, or a many-vertex hull
    whose short edges collapsed) must never be rasterized as a positive
    region; this is the backstop dbnet_targets.py checks regardless of
    which polygon-preparation mode produced the input.
    """
    n = len(poly)
    if n < 3:
        return False
    for i in range(n):
        a1, a2 = poly[i], poly[(i + 1) % n]
        for j in range(i + 1, n):
            # Adjacent edges legitimately share an endpoint -- not a
            # self-intersection.
            if j == i or (j + 1) % n == i:
                continue
            b1, b2 = poly[j], poly[(j + 1) % n]
            if _segments_properly_intersect(a1, a2, b1, b2):
                return False
    return True


def _segments_properly_intersect(a1: Point, a2: Point, b1: Point, b2: Point) -> bool:
    def orientation(p: Point, q: Point, r: Point) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    o1 = orientation(a1, a2, b1)
    o2 = orientation(a1, a2, b2)
    o3 = orientation(b1, b2, a1)
    o4 = orientation(b1, b2, a2)
    return (o1 * o2 < 0) and (o3 * o4 < 0)
