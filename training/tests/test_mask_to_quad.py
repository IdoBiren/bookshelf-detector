"""
Tests for mask_to_quad.py — the bridge from a Mask R-CNN per-instance mask to
the quad the rest of the pipeline already speaks (plan §1: the output is a
quad, not a box and not an OBB).

This is the piece that replaces DBNet's shrink/connected-components/unclip
chain. Mask R-CNN gives one mask per instance, so touching spines are already
separate and none of that machinery is needed — but the mask still has to
become 4 corners, and that path must agree with what postprocess.ts does in
the browser (convexHull -> minAreaQuad, with minAreaRect as the fallback).
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from mask_to_quad import mask_to_quad, masks_to_quads  # noqa: E402
from polygon_offset import polygon_area  # noqa: E402


def _rect_mask(height: int, width: int, x: int, y: int, w: int, h: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y : y + h, x : x + w] = 1
    return mask


class TestMaskToQuad(unittest.TestCase):
    def test_axis_aligned_rectangle_recovers_its_own_corners(self):
        mask = _rect_mask(100, 100, x=10, y=20, w=30, h=50)
        quad = mask_to_quad(mask)
        self.assertIsNotNone(quad)
        self.assertEqual(len(quad), 4)
        xs = sorted(p[0] for p in quad)
        ys = sorted(p[1] for p in quad)
        # cv2 contours are pixel-index based, so the far edge lands on the
        # last filled pixel (x+w-1), not x+w.
        self.assertAlmostEqual(xs[0], 10, delta=1.5)
        self.assertAlmostEqual(xs[-1], 39, delta=1.5)
        self.assertAlmostEqual(ys[0], 20, delta=1.5)
        self.assertAlmostEqual(ys[-1], 69, delta=1.5)

    def test_area_is_close_to_the_mask_pixel_count(self):
        mask = _rect_mask(200, 200, x=20, y=30, w=40, h=90)
        quad = mask_to_quad(mask)
        # 40x90 = 3600 filled pixels; the quad traces pixel centres so it is
        # slightly smaller. Within 10% is the honest tolerance here.
        self.assertAlmostEqual(polygon_area(quad), 3600, delta=360)

    def test_float_probability_mask_is_thresholded(self):
        """Mask R-CNN emits float probabilities in 0..1, not a binary mask —
        the default 0.5 threshold is what turns one into the other."""
        mask = np.zeros((60, 60), dtype=np.float32)
        mask[10:40, 10:30] = 0.9
        mask[40:50, 10:30] = 0.2  # below threshold, must not be included
        quad = mask_to_quad(mask, threshold=0.5)
        ys = sorted(p[1] for p in quad)
        self.assertLess(ys[-1], 42)

    def test_empty_mask_returns_none_rather_than_raising(self):
        self.assertIsNone(mask_to_quad(np.zeros((50, 50), dtype=np.uint8)))

    def test_mask_below_min_area_returns_none(self):
        tiny = _rect_mask(50, 50, x=5, y=5, w=2, h=2)
        self.assertIsNone(mask_to_quad(tiny, min_area_px=16))

    def test_a_rotated_bar_produces_a_tilted_quad_not_an_axis_aligned_box(self):
        """The whole §1 argument: a tilted spine's axis-aligned box is mostly
        its neighbours' text. The quad must actually follow the tilt."""
        import cv2

        mask = np.zeros((200, 200), dtype=np.uint8)
        box = cv2.boxPoints(((100, 100), (30, 120), 30.0))
        cv2.fillPoly(mask, [np.round(box).astype(np.int32)], 1)

        quad = mask_to_quad(mask)
        self.assertIsNotNone(quad)
        # A 30x120 rectangle has area 3600 whatever its rotation; its
        # axis-aligned bounding box at 30 degrees is much larger.
        self.assertAlmostEqual(polygon_area(quad), 3600, delta=600)
        xs = [p[0] for p in quad]
        ys = [p[1] for p in quad]
        aabb_area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        self.assertGreater(aabb_area, polygon_area(quad) * 1.3)

    def test_a_perspective_trapezoid_is_not_forced_into_a_rectangle(self):
        """minAreaRect would overshoot a genuine trapezoid; minAreaQuad (which
        mask_to_quad uses first) should track it much more tightly. Same
        assertion the browser's minAreaQuad.test.ts makes."""
        import cv2

        mask = np.zeros((160, 160), dtype=np.uint8)
        trapezoid = np.array([[70, 20], [90, 20], [110, 140], [50, 140]], dtype=np.int32)
        cv2.fillPoly(mask, [trapezoid], 1)

        quad = mask_to_quad(mask)
        analytic_area = (20 + 60) / 2 * 120  # 4800
        self.assertAlmostEqual(polygon_area(quad), analytic_area, delta=analytic_area * 0.12)


class TestMasksToQuads(unittest.TestCase):
    def test_two_touching_instances_stay_two_quads(self):
        """The reason Mask R-CNN replaces DBNet: per-instance masks mean
        touching spines are separate by construction. No shrink mask, no
        connected components, no unclip -- and none of phase B's tapering
        self-intersection failure."""
        left = _rect_mask(100, 200, x=10, y=10, w=50, h=80)
        right = _rect_mask(100, 200, x=60, y=10, w=50, h=80)  # shares an edge
        quads = masks_to_quads(np.stack([left, right]))
        self.assertEqual(len(quads), 2)

    def test_degenerate_instances_are_dropped_not_returned_as_none(self):
        good = _rect_mask(100, 100, x=10, y=10, w=40, h=40)
        empty = np.zeros((100, 100), dtype=np.uint8)
        quads = masks_to_quads(np.stack([good, empty]))
        self.assertEqual(len(quads), 1)

    def test_accepts_the_extra_channel_dim_mask_rcnn_actually_emits(self):
        """torchvision's Mask R-CNN returns masks shaped (N, 1, H, W), not
        (N, H, W) -- feeding it in raw must work, not silently produce
        garbage."""
        mask = _rect_mask(80, 80, x=10, y=10, w=30, h=40)
        with_channel = mask[None, None, :, :]  # (1, 1, H, W)
        quads = masks_to_quads(with_channel)
        self.assertEqual(len(quads), 1)
        self.assertEqual(len(quads[0]), 4)


if __name__ == "__main__":
    unittest.main()
