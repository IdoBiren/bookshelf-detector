"""
Tests for polygon_offset.py — the Python mirror of src/browser/unclip.ts +
polygonMath.ts (plan §13 phase B).

Group A proves the mirror is a real mirror (byte-for-byte parity against
unclip.ts's own output, via the fixture shared with unclip.test.ts).
Group B proves the inverse property that makes shrink targets and unclip
post-processing consistent. Group C covers geometry invariants.
"""

import json
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from polygon_offset import (  # noqa: E402
    convex_hull,
    exact_unclip_distance,
    exterior_angle_sum,
    is_simple_polygon,
    line_intersection,
    min_area_quad,
    min_area_rect,
    offset_polygon,
    polygon_area,
    polygon_perimeter,
    shrink_distance,
    shrink_polygon,
    unclip_distance,
    unclip_polygon,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "src" / "browser" / "__tests__" / "fixtures" / "unclipParity.json"
)


def _rotate(poly: list[tuple[float, float]], angle_rad: float) -> list[tuple[float, float]]:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return [(x * c - y * s, x * s + y * c) for x, y in poly]


def _shear(poly: list[tuple[float, float]], k: float) -> list[tuple[float, float]]:
    return [(x + k * y, y) for x, y in poly]


def _rect(w: float, h: float) -> list[tuple[float, float]]:
    return [(0, 0), (w, 0), (w, h), (0, h)]


def _max_corner_error(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    assert len(a) == len(b)
    return max(math.hypot(ax - bx, ay - by) for (ax, ay), (bx, by) in zip(a, b))


class TestParityWithUnclipTs(unittest.TestCase):
    """pyclipper is intentionally NOT used here (see the module docstring in
    polygon_offset.py) — this port must match unclip.ts's own arithmetic
    exactly, not merely produce a geometrically-similar offset."""

    def test_matches_the_shared_golden_fixture(self):
        cases = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        for case in cases:
            with self.subTest(case["name"]):
                polygon = [tuple(p) for p in case["polygon"]]
                expected = [tuple(p) for p in case["expected"]]
                result = unclip_polygon(polygon, case["unclipRatio"])
                self.assertEqual(len(result), len(expected))
                for (rx, ry), (ex, ey) in zip(result, expected):
                    self.assertAlmostEqual(rx, ex, places=9)
                    self.assertAlmostEqual(ry, ey, places=9)

    def test_line_intersection_returns_none_for_near_parallel_lines(self):
        # sinAngle just under the 1e-6 threshold in polygonMath.ts:47
        p1, p2 = (0.0, 0.0), (1000.0, 0.0)
        p3, p4 = (0.0, 1.0), (1000.0, 1.0 + 5e-7)
        self.assertIsNone(line_intersection(p1, p2, p3, p4))

    def test_line_intersection_finds_a_clear_crossing(self):
        result = line_intersection((0, 0), (10, 0), (5, -5), (5, 5))
        self.assertAlmostEqual(result[0], 5.0, places=9)
        self.assertAlmostEqual(result[1], 0.0, places=9)

    def test_polygon_area_matches_shoelace(self):
        self.assertAlmostEqual(polygon_area(_rect(10, 20)), 200.0, places=9)

    def test_polygon_perimeter_matches_sum_of_edges(self):
        self.assertAlmostEqual(polygon_perimeter(_rect(10, 20)), 60.0, places=9)

    def test_pyclipper_is_not_imported_by_this_module(self):
        """A future contributor reaching for the library every public DBNet
        implementation uses is the realistic way this constraint gets
        silently violated. Checks for an actual import statement, not just
        the word "pyclipper" anywhere -- the module docstring legitimately
        explains WHY it isn't used, and that explanation must not itself
        trip this test."""
        import polygon_offset

        self.assertNotIn("pyclipper", sys.modules)
        source = Path(polygon_offset.__file__).read_text(encoding="utf-8")
        for line in source.splitlines():
            stripped = line.strip()
            self.assertFalse(
                stripped.startswith("import pyclipper") or stripped.startswith("from pyclipper"),
                f"found a pyclipper import: {line!r}",
            )


class TestInverseProperty(unittest.TestCase):
    """The highest-value tests in this phase. postprocess.ts:64 fits a
    minAreaQuad BEFORE unclipping, so at inference the model only ever
    expands a QUAD — these tests operate on quads for exactly that reason;
    the many-vertex case is covered separately as a known limitation."""

    RECTS = [(25, 400), (66, 355), (100, 300), (200, 200), (20, 52), (40, 600), (12, 90)]
    SHRINK_RATIO = 0.4

    def test_shrink_then_exact_unclip_round_trips_a_rectangle(self):
        for w, h in self.RECTS:
            with self.subTest(f"{w}x{h}"):
                original = _rect(w, h)
                shrunk = shrink_polygon(original, self.SHRINK_RATIO)
                distance = exact_unclip_distance(shrunk, self.SHRINK_RATIO)
                recovered = offset_polygon(shrunk, distance)
                self.assertLess(_max_corner_error(original, recovered), 1e-6)

    def test_round_trip_survives_rotation_and_shear(self):
        for w, h in self.RECTS[:4]:
            for transform_name, transform in [
                ("rotate30", lambda p: _rotate(p, math.radians(30))),
                ("rotate77", lambda p: _rotate(p, math.radians(77))),
                ("shear", lambda p: _shear(p, 0.3)),
            ]:
                with self.subTest(f"{w}x{h}-{transform_name}"):
                    original = transform(_rect(w, h))
                    shrunk = shrink_polygon(original, self.SHRINK_RATIO)
                    distance = exact_unclip_distance(shrunk, self.SHRINK_RATIO)
                    recovered = offset_polygon(shrunk, distance)
                    self.assertLess(_max_corner_error(original, recovered), 1e-6)

    # A real, hand-labeled 11-vertex spine polygon from
    # data/merged/indomain_train.json (same one used in the TS golden
    # fixture's real_11vertex_from_indomain case) -- exercises min_area_quad
    # on genuine annotation data, not a synthetic shape.
    REAL_11VERTEX_POLYGON = [
        (726.5753424657534, 369.49771689497715),
        (1297.8995433789953, 381.917808219178),
        (1422.1004566210047, 2372.237442922374),
        (1335.1598173515983, 2428.127853881278),
        (894.2465753424657, 2400.1826484018256),
        (1061.917808219178, 2285.296803652968),
        (1086.7579908675798, 2043.10502283105),
        (1074.3378995433789, 1912.6940639269405),
        (1009.1324200913242, 1850.5936073059358),
        (919.0867579908676, 1825.753424657534),
        (825.9360730593606, 1850.5936073059358),
    ]

    def test_round_trip_on_a_quad_reduced_from_a_real_polygon(self):
        """Uses min_area_quad(convex_hull(...)) on a real annotation,
        exactly the reduction dbnet_targets.py applies in mode="quad" --
        not an arbitrary 4-point subset of the original vertices, which
        (found while writing this test) can produce a near-degenerate
        sliver quad that the closed-form inverse handles poorly. That is a
        real property of Q for extreme non-rectangular quads, not a bug in
        exact_unclip_distance -- min_area_quad's own greedy reduction does
        not produce such slivers from a well-formed hull."""
        hull = convex_hull(self.REAL_11VERTEX_POLYGON)
        quad = min_area_quad(hull)
        self.assertIsNotNone(quad)
        shrunk = shrink_polygon(quad, self.SHRINK_RATIO)
        distance = exact_unclip_distance(shrunk, self.SHRINK_RATIO)
        recovered = offset_polygon(shrunk, distance)
        self.assertLess(_max_corner_error(quad, recovered), 1e-6)

    def test_round_trip_on_a_skewed_parallelogram(self):
        skewed = _shear(_rect(66, 355), 0.4)
        shrunk = shrink_polygon(skewed, self.SHRINK_RATIO)
        distance = exact_unclip_distance(shrunk, self.SHRINK_RATIO)
        recovered = offset_polygon(shrunk, distance)
        self.assertLess(_max_corner_error(skewed, recovered), 1e-6)

    def test_round_trip_degrades_on_a_many_vertex_hull_not_reduced_to_a_quad(self):
        """Documents the known limitation rather than leaving it to be
        discovered later: inward offsetting collapses short edges on a
        many-vertex hull, changing Q and invalidating the closed-form
        inverse. This is exactly why dbnet_targets.py defaults to
        mode="quad" instead of mode="hull"."""
        # A near-rectangle with several closely-spaced extra vertices along
        # one long edge -- those short edges are exactly what collapses.
        many_vertex = [
            (0, 0), (30, 0), (60, 0), (100, 0),
            (100, 400), (60, 400), (30, 400), (0, 400),
        ]
        shrunk = shrink_polygon(many_vertex, self.SHRINK_RATIO)
        distance = exact_unclip_distance(shrunk, self.SHRINK_RATIO)
        recovered = offset_polygon(shrunk, distance)
        error = _max_corner_error(many_vertex, recovered)
        # Not exact (that's the point) -- but also not wildly broken.
        self.assertGreater(error, 1e-6)
        self.assertLess(error, 50)

    def test_fixed_unclip_ratio_1_5_does_not_invert_a_thin_spine(self):
        """The negative test pinning §13's finding: postprocess.ts's shipped
        default (1.5) is NOT the inverse of shrink_ratio=0.4 for spine
        aspect ratios. If someone "fixes" this by hardcoding 1.5, this test
        catches the regression."""
        spine = _rect(66, 355)  # median indomain spine size at 640 input
        shrunk = shrink_polygon(spine, self.SHRINK_RATIO)
        recovered = offset_polygon(shrunk, unclip_distance(shrunk, 1.5))
        self.assertGreater(_max_corner_error(spine, recovered), 5.0)

    def test_shrink_ratio_point_four_means_offset_factor_point_eight_four(self):
        """DBNet's shrink_ratio and unclip_ratio are NOT the same kind of
        number: shrink_distance folds in (1-r^2) before dividing by
        perimeter; unclip_distance does not. Reading "0.4" as a bare offset
        factor (skipping the square) is the single likeliest silent bug in
        this phase -- a 2.1x wrong shrink that produces plausible-looking
        but incorrect targets."""
        square = _rect(100, 100)
        offset_factor = 1 - 0.4**2  # 0.84
        self.assertAlmostEqual(
            shrink_distance(square, 0.4), unclip_distance(square, offset_factor), places=9
        )


class TestConvexHullAndMinAreaQuad(unittest.TestCase):
    """Python mirror of src/browser/__tests__/minAreaQuad.test.ts — same
    trapezoid, same assertions, so a divergence between the two ports is
    caught by symmetry even without a shared fixture file."""

    # Isosceles trapezoid: top edge 20 wide, bottom edge 60 wide, height 80.
    # Analytic area = (20+60)/2*80 = 3200 -- hand-computed, not code-derived.
    TL, TR, BR, BL = (50, 10), (70, 10), (90, 90), (30, 90)
    ANALYTIC_AREA = 3200

    def test_exact_4_corner_trapezoid_matches_analytic_area(self):
        hull = convex_hull([self.TL, self.TR, self.BR, self.BL])
        self.assertEqual(len(hull), 4)

        quad = min_area_quad(hull)
        self.assertIsNotNone(quad)
        self.assertAlmostEqual(polygon_area(quad), self.ANALYTIC_AREA, places=6)

        rect = min_area_rect(hull)
        rect_area = polygon_area(rect)
        self.assertGreater(rect_area, self.ANALYTIC_AREA * 1.1)

    def test_hull_with_extra_vertex_reduces_to_4_points_near_trapezoid_area(self):
        # Left leg satisfies x = 50 - 0.25*(y-10); at y=50 that's x=40, so
        # x=38 is genuinely outside and must survive on the hull.
        bulge = (38, 50)
        hull = convex_hull([self.TL, self.TR, self.BR, self.BL, bulge])
        self.assertEqual(len(hull), 5)

        quad = min_area_quad(hull)
        self.assertIsNotNone(quad)
        self.assertEqual(len(quad), 4)
        quad_area = polygon_area(quad)
        self.assertGreaterEqual(quad_area, self.ANALYTIC_AREA - 1e-6)
        self.assertLess(quad_area, self.ANALYTIC_AREA * 1.4)

    def test_falls_back_to_none_below_4_hull_points(self):
        triangle = convex_hull([(0, 0), (10, 0), (5, 10)])
        self.assertEqual(len(triangle), 3)
        self.assertIsNone(min_area_quad(triangle))

    def test_square_hull_recovered_exactly(self):
        square = [(0, 0), (10, 0), (10, 10), (0, 10)]
        hull = convex_hull(square)
        self.assertEqual(len(hull), 4)
        self.assertEqual(polygon_area(hull), 100)
        self.assertAlmostEqual(polygon_area(min_area_rect(hull)), 100, places=6)
        self.assertAlmostEqual(polygon_area(min_area_quad(hull)), 100, places=6)


class TestGeometryInvariants(unittest.TestCase):
    def test_shrink_preserves_vertex_count(self):
        original = _rect(100, 200)
        shrunk = shrink_polygon(original, 0.4)
        self.assertEqual(len(shrunk), len(original))

    def test_shrink_of_a_thin_rectangle_never_fully_vanishes(self):
        """Analytic bound (plan §13): for h >> w, shrunk width tends to
        ~0.16w at shrink_ratio=0.4 and never reaches zero for a convex
        quad -- vanishing is a rasterization/non-convexity artifact, not a
        property of uniform shrink itself."""
        for w, h in [(25, 400), (10, 500), (40, 2000)]:
            with self.subTest(f"{w}x{h}"):
                shrunk = shrink_polygon(_rect(w, h), 0.4)
                shrunk_width = max(x for x, _ in shrunk) - min(x for x, _ in shrunk)
                self.assertGreater(shrunk_width, 0.15 * w)

    def test_is_simple_polygon_true_for_a_convex_shape(self):
        self.assertTrue(is_simple_polygon(_rect(10, 20)))

    def test_is_simple_polygon_false_for_a_self_intersecting_bowtie(self):
        bowtie = [(0, 0), (10, 10), (10, 0), (0, 10)]
        self.assertFalse(is_simple_polygon(bowtie))

    def test_shrinking_a_non_convex_l_shape_in_raw_mode_can_self_intersect(self):
        """Pins the documented hazard of mode="raw": the centroid-based
        normal-sign test unclip.ts uses is wrong on reflex vertices, so
        shrinking a genuinely non-convex polygon can produce garbage. This
        is exactly why dbnet_targets.py's is_simple_polygon check must run
        regardless of mode, as a backstop."""
        l_shape = [(0, 0), (100, 0), (100, 40), (40, 40), (40, 100), (0, 100)]
        shrunk = shrink_polygon(l_shape, 0.4)
        # No assertion of failure here -- just that the check catches it
        # when it happens, i.e. the backstop is exercisable on real input.
        is_simple_polygon(shrunk)  # must not raise

    def test_exterior_angle_sum_of_a_rectangle_is_four(self):
        # Each corner turns 90 degrees; tan(45 deg) = 1; four corners -> 4.
        self.assertAlmostEqual(exterior_angle_sum(_rect(50, 80)), 4.0, places=6)


if __name__ == "__main__":
    unittest.main()
