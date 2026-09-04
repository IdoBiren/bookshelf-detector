"""
Tests for crop_quad.py — rectifying a detected spine for the VLM.

The VLM reads a horizontal line of text. A detected spine is a quad that is
usually near-vertical and often a perspective trapezoid, so "crop it" means
warp + rotate, not slice a rectangle out of the image.

The corner convention is NOT invented here: it mirrors
`src/browser/quad.ts`'s `canonicalizeQuad`, where the SHORT edge pair is the
spine's physical top/bottom and TL->TR spans it. If the two sides of the
codebase disagree about corner order, quads drawn in the browser and crops
fed to the VLM silently stop corresponding to each other.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from crop_quad import canonicalize_quad, rectify_spine  # noqa: E402


def _vertical_spine_quad(x: float, y: float, width: float, height: float):
    """A spine standing upright: `width` is the short side."""
    return [(x, y), (x + width, y), (x + width, y + height), (x, y + height)]


class TestCanonicalizeQuad(unittest.TestCase):
    def test_tl_to_tr_spans_the_short_edge(self):
        """The load-bearing half of the convention. TL->TR must be the
        spine's physical top (short), not its side (long) -- rectify_spine
        derives the output rectangle's width from it."""
        quad = _vertical_spine_quad(10, 20, 30, 200)
        tl, tr, br, bl = canonicalize_quad(quad)

        short = np.hypot(tr[0] - tl[0], tr[1] - tl[1])
        long_ = np.hypot(br[0] - tr[0], br[1] - tr[1])
        self.assertAlmostEqual(short, 30, delta=1)
        self.assertAlmostEqual(long_, 200, delta=1)
        self.assertLess(short, long_)

    def test_is_independent_of_input_corner_order(self):
        """Detections arrive from mask_to_quad in no guaranteed order."""
        quad = _vertical_spine_quad(10, 20, 30, 200)
        expected = canonicalize_quad(quad)

        for roll in range(4):
            rotated = quad[roll:] + quad[:roll]
            self.assertEqual(canonicalize_quad(rotated), expected)
        self.assertEqual(canonicalize_quad(list(reversed(quad))), expected)

    def test_picks_the_short_edge_nearer_the_image_top_as_tl_tr(self):
        quad = _vertical_spine_quad(10, 20, 30, 200)
        tl, tr, br, bl = canonicalize_quad(quad)
        self.assertLess(tl[1], bl[1])
        self.assertLess(tr[1], br[1])

    def test_tl_is_left_of_tr(self):
        quad = _vertical_spine_quad(10, 20, 30, 200)
        tl, tr, _, _ = canonicalize_quad(quad)
        self.assertLess(tl[0], tr[0])

    def test_handles_a_spine_that_is_already_horizontal(self):
        """A book lying on its side: the short edges are now left/right, so
        the ordering must still key off edge LENGTH, never off x/y extent."""
        quad = [(10, 50), (210, 50), (210, 80), (10, 80)]  # 200 wide, 30 tall
        tl, tr, br, bl = canonicalize_quad(quad)
        short = np.hypot(tr[0] - tl[0], tr[1] - tl[1])
        long_ = np.hypot(br[0] - tr[0], br[1] - tr[1])
        self.assertAlmostEqual(short, 30, delta=1)
        self.assertAlmostEqual(long_, 200, delta=1)


class TestRectifySpine(unittest.TestCase):
    def _image_with_marked_spine(self):
        """Black image, white spine standing upright at x=100..130,
        y=50..250, with a distinctive grey band across its TOP quarter."""
        image = np.zeros((300, 400, 3), dtype=np.uint8)
        image[50:250, 100:130] = 255
        image[50:100, 100:130] = 128  # the marker
        return image, _vertical_spine_quad(100, 50, 30, 200)

    def test_output_is_horizontal_so_the_vlm_reads_a_text_line(self):
        image, quad = self._image_with_marked_spine()
        crop = rectify_spine(image, quad)
        self.assertGreater(crop.shape[1], crop.shape[0], "crop must be wider than tall")

    def test_preserves_the_spine_aspect_ratio(self):
        image, quad = self._image_with_marked_spine()
        crop = rectify_spine(image, quad)
        # 200 long / 30 short, now horizontal -> width/height
        self.assertAlmostEqual(crop.shape[1] / crop.shape[0], 200 / 30, delta=0.3)

    def test_the_spines_top_ends_up_on_the_left(self):
        """A spine's text reads top-to-bottom when the book stands upright,
        so rotating counter-clockwise puts it left-to-right for the VLM.
        The grey marker was at the spine's top; it must land at the left."""
        image, quad = self._image_with_marked_spine()
        crop = rectify_spine(image, quad)

        width = crop.shape[1]
        left_mean = float(crop[:, : width // 4].mean())
        right_mean = float(crop[:, -width // 4 :].mean())
        self.assertLess(left_mean, right_mean, "grey marker should be on the left")

    def test_flip_reverses_the_reading_direction(self):
        """The 180 degree ambiguity is real -- a spine's text can run either
        way -- so the caller must be able to ask for the other orientation
        without re-deriving the quad."""
        image, quad = self._image_with_marked_spine()
        normal = rectify_spine(image, quad)
        flipped = rectify_spine(image, quad, flip=True)

        width = normal.shape[1]
        self.assertLess(float(normal[:, : width // 4].mean()), float(normal[:, -width // 4 :].mean()))
        self.assertGreater(float(flipped[:, : width // 4].mean()), float(flipped[:, -width // 4 :].mean()))

    def test_recovers_content_from_a_perspective_trapezoid(self):
        """The reason this is warpPerspective and not a rotated crop: shot
        at an angle, a spine is a trapezoid (plan section 1), and slicing its
        bounding box would include the neighbours."""
        image = np.zeros((300, 400, 3), dtype=np.uint8)
        trapezoid = [(100, 50), (130, 55), (135, 245), (95, 250)]
        cv_fill = np.array(trapezoid, dtype=np.int32)
        import cv2

        cv2.fillPoly(image, [cv_fill], (255, 255, 255))

        crop = rectify_spine(image, trapezoid)
        self.assertGreater(crop.shape[1], crop.shape[0])
        # The warp should be mostly filled with the white spine, not black
        # background bled in from outside the trapezoid.
        self.assertGreater(float(crop.mean()), 200)

    def test_caps_the_long_side_to_keep_the_payload_small(self):
        """Measured on a real 4080x3072 shelf photo: uncapped crops came out
        ~2000x220 each, 24 of them, 18.6MB of PNG -- unusable over a tunnel
        from a laptop. A book title does not need 2000px to be read."""
        image = np.zeros((3000, 4000, 3), dtype=np.uint8)
        image[500:2500, 1000:1300] = 255
        quad = _vertical_spine_quad(1000, 500, 300, 2000)

        crop = rectify_spine(image, quad, max_long_side=1024)
        self.assertLessEqual(max(crop.shape[0], crop.shape[1]), 1024)

    def test_capping_preserves_the_aspect_ratio(self):
        image = np.zeros((3000, 4000, 3), dtype=np.uint8)
        image[500:2500, 1000:1300] = 255
        quad = _vertical_spine_quad(1000, 500, 300, 2000)

        uncapped = rectify_spine(image, quad)
        capped = rectify_spine(image, quad, max_long_side=1024)
        self.assertAlmostEqual(
            capped.shape[1] / capped.shape[0],
            uncapped.shape[1] / uncapped.shape[0],
            delta=0.05,
        )

    def test_a_crop_already_under_the_cap_is_not_upscaled(self):
        """Upscaling would inflate the payload for no added detail."""
        image, quad = self._image_with_marked_spine()
        uncapped = rectify_spine(image, quad)
        capped = rectify_spine(image, quad, max_long_side=4096)
        self.assertEqual(capped.shape, uncapped.shape)

    def test_a_degenerate_quad_returns_none_rather_than_raising(self):
        """mask_to_quad can emit a near-zero-area quad on a bad mask; a
        server must skip it, not 500."""
        image = np.zeros((300, 400, 3), dtype=np.uint8)
        self.assertIsNone(rectify_spine(image, [(10, 10), (10, 10), (10, 10), (10, 10)]))


if __name__ == "__main__":
    unittest.main()
