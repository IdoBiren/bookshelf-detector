"""
Tests for dbnet_targets.py — polygon preparation, rasterization, and the two
shrink-ratio validation metrics (plan §13 phase B). Scope note: this covers
what the go/no-go measurement needs (letterboxing, polygon-mode prep,
rasterized masks, the merged-pairs and vanishing-spines metrics). The full
DBNet loss-target dataclass (threshold map, binary map for actual training)
is deliberately deferred past this phase — it's a training-time concern,
not part of the shrink_ratio decision this phase exists to make.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from dbnet_targets import (  # noqa: E402
    LetterboxInfo,
    compute_letterbox,
    letterbox_polygon,
    merged_adjacent_pairs_metric,
    prepare_polygon,
    rasterize_polygon,
    vanishing_spines_metric,
)


def _rect(x: float, y: float, w: float, h: float) -> list[tuple[float, float]]:
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


class TestComputeLetterbox(unittest.TestCase):
    """Mirrors src/browser/letterbox.ts's computeLetterbox exactly — a
    sign/pad error here silently offsets every target and is invisible in
    aggregate metrics."""

    def test_wide_image_scales_to_width_and_pads_vertically(self):
        info = compute_letterbox(orig_width=2000, orig_height=1000, target_size=640)
        self.assertAlmostEqual(info.scale, 640 / 2000, places=9)
        self.assertAlmostEqual(info.pad_x, 0.0, places=9)
        expected_scaled_h = 1000 * (640 / 2000)
        self.assertAlmostEqual(info.pad_y, (640 - expected_scaled_h) / 2, places=9)

    def test_tall_image_scales_to_height_and_pads_horizontally(self):
        info = compute_letterbox(orig_width=1000, orig_height=2000, target_size=640)
        self.assertAlmostEqual(info.scale, 640 / 2000, places=9)
        self.assertAlmostEqual(info.pad_y, 0.0, places=9)

    def test_square_image_has_no_padding(self):
        info = compute_letterbox(orig_width=800, orig_height=800, target_size=640)
        self.assertAlmostEqual(info.pad_x, 0.0, places=9)
        self.assertAlmostEqual(info.pad_y, 0.0, places=9)

    def test_raises_on_non_positive_dimensions(self):
        with self.assertRaises(ValueError):
            compute_letterbox(orig_width=0, orig_height=100, target_size=640)


class TestLetterboxPolygon(unittest.TestCase):
    def test_maps_original_corner_to_expected_target_space(self):
        info = compute_letterbox(orig_width=2000, orig_height=1000, target_size=640)
        # (0,0) in original space -> (padX, padY) in target space.
        mapped = letterbox_polygon([(0, 0)], info, stride=1)
        self.assertAlmostEqual(mapped[0][0], info.pad_x, places=6)
        self.assertAlmostEqual(mapped[0][1], info.pad_y, places=6)

    def test_maps_far_corner_consistently_with_scale(self):
        info = compute_letterbox(orig_width=2000, orig_height=1000, target_size=640)
        mapped = letterbox_polygon([(2000, 1000)], info, stride=1)
        self.assertAlmostEqual(mapped[0][0], info.pad_x + 2000 * info.scale, places=6)
        self.assertAlmostEqual(mapped[0][1], info.pad_y + 1000 * info.scale, places=6)

    def test_stride_divides_the_target_space_coordinates(self):
        info = compute_letterbox(orig_width=640, orig_height=640, target_size=640)
        at_stride_1 = letterbox_polygon([(100, 200)], info, stride=1)
        at_stride_4 = letterbox_polygon([(100, 200)], info, stride=4)
        self.assertAlmostEqual(at_stride_4[0][0], at_stride_1[0][0] / 4, places=6)
        self.assertAlmostEqual(at_stride_4[0][1], at_stride_1[0][1] / 4, places=6)


class TestPreparePolygon(unittest.TestCase):
    def test_hull_mode_reduces_a_redundant_collinear_point(self):
        # midpoint of the top edge is redundant on the hull
        points = [(0, 0), (50, 0), (100, 0), (100, 100), (0, 100)]
        prepared = prepare_polygon(points, mode="hull")
        self.assertEqual(len(prepared), 4)

    def test_quad_mode_always_returns_4_points_for_a_valid_hull(self):
        points = [(0, 0), (50, 0), (100, 0), (100, 100), (0, 100)]
        prepared = prepare_polygon(points, mode="quad")
        self.assertEqual(len(prepared), 4)

    def test_raw_mode_preserves_every_vertex(self):
        points = [(0, 0), (50, 0), (100, 0), (100, 100), (0, 100)]
        prepared = prepare_polygon(points, mode="raw")
        self.assertEqual(len(prepared), 5)

    def test_quad_mode_on_a_triangle_falls_back_to_min_area_rect(self):
        triangle = [(0, 0), (10, 0), (5, 10)]
        prepared = prepare_polygon(triangle, mode="quad")
        self.assertEqual(len(prepared), 4)


class TestRasterizePolygon(unittest.TestCase):
    def test_rectangle_rasterizes_to_approximately_its_area(self):
        mask = rasterize_polygon(_rect(10, 10, 20, 30), width=100, height=100)
        self.assertEqual(mask.dtype, np.uint8)
        # cv2.fillPoly is boundary-inclusive (fills both the 0 and the W
        # edge), so a w x h rect at integer coordinates fills (w+1) x (h+1)
        # pixels -- verified directly, not a fuzzy "close enough" slack.
        self.assertEqual(int(mask.sum()), 21 * 31)

    def test_empty_polygon_rasterizes_to_nothing(self):
        mask = rasterize_polygon([], width=50, height=50)
        self.assertEqual(mask.sum(), 0)

    def test_polygon_entirely_outside_the_canvas_rasterizes_to_nothing(self):
        mask = rasterize_polygon(_rect(1000, 1000, 10, 10), width=50, height=50)
        self.assertEqual(mask.sum(), 0)


class TestMergedAdjacentPairsMetric(unittest.TestCase):
    """The decisive metric: what fraction of touching original spines are
    still connected after shrinking. Threshold from plan §13: >5% = NO-GO."""

    def test_two_touching_rectangles_that_stay_separated_after_shrink(self):
        polygons = [_rect(0, 0, 200, 20), _rect(200, 0, 200, 20)]  # share an edge
        result = merged_adjacent_pairs_metric(
            [polygons], shrink_ratio=0.4, canvas_size=640
        )
        self.assertEqual(result["adjacent_pairs"], 1)
        self.assertEqual(result["merged_pairs"], 0)

    def test_a_pair_still_touching_after_shrink_is_counted_as_merged(self):
        # Two 100x100 squares overlapping by 50px (not merely touching):
        # shrink_ratio=0.4 pulls each boundary in by 21px (area*0.84/perimeter
        # for a 100x100 square), leaving an 8px overlap after shrinking --
        # the metric's own true-positive case for "still merged".
        polygons = [_rect(0, 0, 100, 100), _rect(50, 0, 100, 100)]
        result = merged_adjacent_pairs_metric(
            [polygons], shrink_ratio=0.4, canvas_size=640
        )
        self.assertEqual(result["adjacent_pairs"], 1)
        self.assertEqual(result["merged_pairs"], 1)

    def test_non_adjacent_polygons_are_not_counted_as_a_pair_at_all(self):
        far_apart = [_rect(0, 0, 20, 20), _rect(600, 600, 20, 20)]
        result = merged_adjacent_pairs_metric(
            [far_apart], shrink_ratio=0.4, canvas_size=640
        )
        self.assertEqual(result["adjacent_pairs"], 0)

    def test_diagonal_only_touch_is_not_counted_as_adjacent(self):
        """4-connectivity (matching connectedComponents.ts): two squares
        touching only at a single diagonal corner must NOT count as
        adjacent, or the metric would over-count merges that the browser's
        4-connected component labeling would never actually produce.

        cv2.fillPoly is boundary-inclusive (a 0..10 rect fills pixel column
        10 as well as 0..9), so squares placed at exactly (0,0,10,10) and
        (10,10,10,10) would share pixel (10,10) directly -- not a diagonal
        touch but a genuine 1px overlap. The 1px gap here (11,11 instead of
        10,10) is what makes the two masks purely diagonal NEIGHBORS with
        no shared pixel, which is the case 4-connectivity must reject."""
        squares = [_rect(0, 0, 10, 10), _rect(11, 11, 10, 10)]
        result = merged_adjacent_pairs_metric(
            [squares], shrink_ratio=0.4, canvas_size=640
        )
        self.assertEqual(result["adjacent_pairs"], 0)

    def test_order_of_polygons_within_an_image_does_not_change_the_result(self):
        forward = [_rect(0, 0, 200, 20), _rect(200, 0, 200, 20)]
        reversed_order = list(reversed(forward))
        r1 = merged_adjacent_pairs_metric([forward], shrink_ratio=0.4, canvas_size=640)
        r2 = merged_adjacent_pairs_metric([reversed_order], shrink_ratio=0.4, canvas_size=640)
        self.assertEqual(r1["adjacent_pairs"], r2["adjacent_pairs"])
        self.assertEqual(r1["merged_pairs"], r2["merged_pairs"])

    def test_aggregates_across_multiple_images(self):
        image_a = [_rect(0, 0, 100, 100), _rect(50, 0, 100, 100)]  # overlap -> stays merged
        image_b = [_rect(0, 100, 20, 20), _rect(300, 300, 20, 20)]  # far apart -> 0 pairs
        result = merged_adjacent_pairs_metric(
            [image_a, image_b], shrink_ratio=0.4, canvas_size=640
        )
        self.assertEqual(result["adjacent_pairs"], 1)
        self.assertEqual(result["merged_pairs"], 1)


class TestVanishingSpinesMetric(unittest.TestCase):
    """Threshold from plan §13: >1% = bug, not noise."""

    def test_a_normal_spine_does_not_vanish(self):
        polygons = [_rect(0, 0, 66, 355)]
        result = vanishing_spines_metric([polygons], shrink_ratio=0.4, canvas_size=640)
        self.assertEqual(result["vanished"], 0)
        self.assertEqual(result["total"], 1)

    def test_a_small_polygon_falls_below_the_pixel_count_threshold(self):
        # MIN_SHRUNK_PIXELS=20 is a TOTAL pixel count, not a width check --
        # a small square (not a long thin sliver, which keeps plenty of
        # pixels even at 1px wide) is what actually falls under it.
        polygons = [_rect(0, 0, 4, 4)]  # area 16; shrinks to ~9px, well below the 20px floor
        result = vanishing_spines_metric([polygons], shrink_ratio=0.4, canvas_size=640)
        self.assertEqual(result["vanished"], 1)
        self.assertEqual(result["tiny_pixels"], 1)

    def test_a_long_thin_sliver_does_not_vanish_by_pixel_count_alone(self):
        """Documents a real property of the metric as defined (plan §13):
        a 1px-wide, 300px-tall shrunk region has ~300 total pixels, well
        above MIN_SHRUNK_PIXELS -- it does NOT count as vanished even though
        its width collapsed almost to nothing. Width-based thinness is a
        different failure mode than this metric measures."""
        polygons = [_rect(0, 0, 3, 300)]
        result = vanishing_spines_metric([polygons], shrink_ratio=0.4, canvas_size=640)
        self.assertEqual(result["vanished"], 0)

    def test_vanished_count_breaks_down_zero_vs_tiny(self):
        zero_width = [(10, 10), (10, 10), (10, 200), (10, 200)]  # degenerate: zero area
        result = vanishing_spines_metric([[zero_width]], shrink_ratio=0.4, canvas_size=640)
        self.assertEqual(result["vanished"], 1)
        self.assertGreaterEqual(result["zero_area_or_self_intersecting"], 1)


if __name__ == "__main__":
    unittest.main()
