"""
Tests for evaluate.py — plan §8א's geometric metrics.

The two error-prone pieces are matching and AP integration, so those get the
most attention. A wrong AP does not crash; it just reports a number that
looks plausible and is wrong, which is the worst possible failure for the
one measurement this whole phase exists to produce.

§8א is explicit that IoU must be measured on the mask/quad, NOT on an
axis-aligned box — "otherwise you're measuring the wrong shape". A tilted
spine's AABB is mostly its neighbours (§1), so AABB IoU would flatter the
model exactly where it matters most.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate import (  # noqa: E402
    average_precision,
    evaluate,
    match_predictions,
    quad_iou,
    spine_width,
    tercile_edges,
)


def _rect(x: float, y: float, w: float, h: float) -> list[tuple[float, float]]:
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


class TestQuadIou(unittest.TestCase):
    def test_identical_quads_score_one(self):
        quad = _rect(10, 10, 40, 80)
        self.assertAlmostEqual(quad_iou(quad, quad), 1.0, places=2)

    def test_disjoint_quads_score_zero(self):
        self.assertEqual(quad_iou(_rect(0, 0, 20, 20), _rect(100, 100, 20, 20)), 0.0)

    def test_half_overlap_scores_about_one_third(self):
        # Two equal boxes overlapping on half their area:
        # intersection = 0.5A, union = 1.5A -> IoU = 1/3.
        a = _rect(0, 0, 40, 40)
        b = _rect(20, 0, 40, 40)
        self.assertAlmostEqual(quad_iou(a, b), 1 / 3, delta=0.05)

    def test_measures_the_quad_not_its_bounding_box(self):
        """Two thin bars crossing in an X share very little area, but their
        axis-aligned boxes are nearly identical. AABB IoU would say ~1.0;
        quad IoU must say almost nothing. This is the §8א requirement."""
        import math

        def rotated_bar(angle_deg: float) -> list[tuple[float, float]]:
            cx = cy = 50.0
            half_len, half_wid = 40.0, 4.0
            a = math.radians(angle_deg)
            ca, sa = math.cos(a), math.sin(a)
            corners = []
            for dx, dy in ((-half_len, -half_wid), (half_len, -half_wid), (half_len, half_wid), (-half_len, half_wid)):
                corners.append((cx + dx * ca - dy * sa, cy + dx * sa + dy * ca))
            return corners

        iou = quad_iou(rotated_bar(45), rotated_bar(-45))
        self.assertLess(iou, 0.2)


class TestMatchPredictions(unittest.TestCase):
    def test_a_perfect_prediction_is_a_true_positive(self):
        gt = [_rect(10, 10, 40, 60)]
        preds = [(_rect(10, 10, 40, 60), 0.9)]
        flags, matched = match_predictions(preds, gt, iou_threshold=0.5)
        self.assertEqual(flags, [True])
        self.assertEqual(matched, 1)

    def test_a_far_off_prediction_is_a_false_positive(self):
        gt = [_rect(10, 10, 40, 60)]
        preds = [(_rect(300, 300, 40, 60), 0.9)]
        flags, matched = match_predictions(preds, gt, iou_threshold=0.5)
        self.assertEqual(flags, [False])
        self.assertEqual(matched, 0)

    def test_each_ground_truth_can_only_be_matched_once(self):
        """Two predictions on the same book: one TP, one FP. Without this,
        a model that fires ten boxes per spine would score perfectly."""
        gt = [_rect(10, 10, 40, 60)]
        preds = [(_rect(10, 10, 40, 60), 0.9), (_rect(11, 11, 40, 60), 0.8)]
        flags, matched = match_predictions(preds, gt, iou_threshold=0.5)
        self.assertEqual(flags, [True, False])
        self.assertEqual(matched, 1)

    def test_higher_scoring_prediction_claims_the_match_first(self):
        gt = [_rect(10, 10, 40, 60)]
        exact = _rect(10, 10, 40, 60)
        sloppy = _rect(16, 16, 40, 60)
        # Deliberately passed lower-score-first to prove it sorts.
        flags, _ = match_predictions([(sloppy, 0.4), (exact, 0.95)], gt, iou_threshold=0.5)
        self.assertEqual(flags, [True, False], "predictions must be ranked by score before matching")

    def test_two_spines_two_predictions_both_match(self):
        gt = [_rect(10, 10, 30, 60), _rect(50, 10, 30, 60)]
        preds = [(_rect(50, 10, 30, 60), 0.9), (_rect(10, 10, 30, 60), 0.8)]
        flags, matched = match_predictions(preds, gt, iou_threshold=0.5)
        self.assertEqual(flags, [True, True])
        self.assertEqual(matched, 2)


class TestAveragePrecision(unittest.TestCase):
    def test_all_true_positives_with_full_recall_is_one(self):
        self.assertAlmostEqual(average_precision([True, True, True], [0.9, 0.8, 0.7], 3), 1.0, places=2)

    def test_no_predictions_is_zero(self):
        self.assertEqual(average_precision([], [], 5), 0.0)

    def test_no_ground_truth_is_zero(self):
        self.assertEqual(average_precision([True], [0.9], 0), 0.0)

    def test_missing_half_the_ground_truth_caps_ap_near_half(self):
        # 2 perfect detections, but 4 objects exist -> recall tops out at 0.5.
        ap = average_precision([True, True], [0.9, 0.8], 4)
        self.assertAlmostEqual(ap, 0.5, delta=0.05)

    def test_false_positives_ranked_above_true_positives_lower_ap(self):
        good = average_precision([True, True, False], [0.9, 0.8, 0.7], 2)
        bad = average_precision([False, True, True], [0.9, 0.8, 0.7], 2)
        self.assertGreater(good, bad)


class TestWidthTerciles(unittest.TestCase):
    """§8א: mAP broken down by spine-width tercile. A good overall mAP can
    hide total failure on pocket paperbacks -- that's trap #4."""

    def test_spine_width_is_the_short_side_not_the_long_one(self):
        self.assertAlmostEqual(spine_width(_rect(0, 0, 25, 400)), 25, delta=1)
        self.assertAlmostEqual(spine_width(_rect(0, 0, 400, 25)), 25, delta=1)

    def test_tercile_edges_split_the_population_into_three(self):
        widths = [float(w) for w in range(1, 100)]
        low, high = tercile_edges(widths)
        self.assertLess(low, high)
        thin = [w for w in widths if w <= low]
        wide = [w for w in widths if w > high]
        self.assertAlmostEqual(len(thin), len(widths) / 3, delta=3)
        self.assertAlmostEqual(len(wide), len(widths) / 3, delta=3)

    def test_tercile_edges_on_an_empty_population_does_not_crash(self):
        low, high = tercile_edges([])
        self.assertEqual((low, high), (0.0, 0.0))


class TestEvaluateEndToEnd(unittest.TestCase):
    """The single most important sanity check in this file: a predictor that
    returns the ground truth exactly must score 1.0 -- overall AND in every
    width band. Anything less means the metric itself is broken, and a
    broken metric is worse than no metric because it looks like a result.

    This caught a real bug: restricting ground truth to one width band while
    keeping ALL predictions made the other bands' correct detections count
    as false positives, dragging per-band AP to ~0.43 on perfect input.
    COCO's answer is to IGNORE detections that match out-of-band ground
    truth rather than penalise them."""

    def _perfect(self, ground_truth_per_image):
        return [([(gt, 1.0) for gt in gts], gts) for gts in ground_truth_per_image]

    def test_perfect_predictions_score_one_overall(self):
        gts = [[_rect(0, 0, 10, 100), _rect(20, 0, 30, 100), _rect(60, 0, 50, 100)]]
        results = evaluate(self._perfect(gts))
        self.assertAlmostEqual(results["mAP@50"], 1.0, places=3)
        self.assertAlmostEqual(results["mAP@50:95"], 1.0, places=3)

    def test_perfect_predictions_score_one_in_every_width_band(self):
        gts = [
            [_rect(0, 0, 10, 100), _rect(20, 0, 30, 100), _rect(60, 0, 50, 100)],
            [_rect(0, 0, 12, 90), _rect(30, 0, 28, 90), _rect(70, 0, 55, 90)],
        ]
        results = evaluate(self._perfect(gts))
        for band in ("thin", "medium", "wide"):
            self.assertAlmostEqual(
                results["by_width"][band]["AP@50"], 1.0, places=3,
                msg=f"{band} band scored below 1.0 on perfect predictions",
            )

    def test_every_ground_truth_object_lands_in_exactly_one_band(self):
        gts = [[_rect(0, 0, 10, 100), _rect(20, 0, 30, 100), _rect(60, 0, 50, 100)]]
        results = evaluate(self._perfect(gts))
        counted = sum(results["by_width"][b]["ground_truth"] for b in ("thin", "medium", "wide"))
        self.assertEqual(counted, 3)

    def test_missing_every_thin_spine_shows_up_as_a_thin_band_failure(self):
        """The whole point of the breakdown (trap #4): overall mAP stays
        respectable while the thin band collapses."""
        thin = _rect(0, 0, 8, 100)
        medium = _rect(20, 0, 30, 100)
        wide = _rect(60, 0, 60, 100)
        ground_truth = [thin, medium, wide]
        predictions = [(medium, 0.9), (wide, 0.9)]  # thin one never detected

        results = evaluate([(predictions, ground_truth)])
        self.assertAlmostEqual(results["by_width"]["thin"]["AP@50"], 0.0, places=3)
        self.assertGreater(results["by_width"]["medium"]["AP@50"], 0.9)
        self.assertGreater(results["by_width"]["wide"]["AP@50"], 0.9)


if __name__ == "__main__":
    unittest.main()
