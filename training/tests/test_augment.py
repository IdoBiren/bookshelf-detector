import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from augment import (  # noqa: E402
    build_augmentation_pipeline,
    flatten_polygons_to_keypoints,
    regroup_keypoints_to_polygons,
)


class TestFlattenRegroupRoundTrip(unittest.TestCase):
    def test_round_trip_preserves_points_and_grouping(self):
        annotations = [
            {"segmentation": [[0, 0, 10, 0, 10, 10, 0, 10]]},  # 4 points
            {"segmentation": [[1, 1, 2, 2, 3, 3, 4, 4, 5, 5]]},  # 5 points
        ]
        keypoints, counts = flatten_polygons_to_keypoints(annotations)
        self.assertEqual(counts, [4, 5])
        self.assertEqual(len(keypoints), 9)
        self.assertEqual(keypoints[0], (0, 0))
        self.assertEqual(keypoints[4], (1, 1))  # first point of the second polygon

        regrouped = regroup_keypoints_to_polygons(keypoints, counts)
        self.assertEqual(len(regrouped), 2)
        self.assertEqual(regrouped[0], [(0, 0), (10, 0), (10, 10), (0, 10)])
        self.assertEqual(regrouped[1], [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)])

    def test_handles_variable_vertex_counts_seen_in_real_data(self):
        # Real datasets have polygons from 5 to 9+ points, not a fixed 4 —
        # this is exactly what a bbox-based augmentation pipeline would lose.
        annotations = [
            {"segmentation": [[0] * 14]},  # 7 points
            {"segmentation": [[0] * 4]},  # 2 points (degenerate, but shouldn't crash bookkeeping)
        ]
        keypoints, counts = flatten_polygons_to_keypoints(annotations)
        self.assertEqual(counts, [7, 2])
        self.assertEqual(len(keypoints), 9)


class TestAugmentationPipelineIntegration(unittest.TestCase):
    """Runs the REAL albumentations pipeline (no mocking) on a synthetic
    image+polygon, per plan §3's core concern: a polygon-transform bug is a
    classic silent failure that only shows up when you actually run the
    transform, not when you read the code."""

    def test_polygon_survives_pipeline_with_all_points_preserved(self):
        annotations = [
            {"segmentation": [[100, 100, 200, 100, 200, 300, 100, 300]]},  # a spine-like rect
            {"segmentation": [[50, 50, 90, 55, 95, 150, 45, 145, 40, 90]]},  # 5-point polygon
        ]
        keypoints, counts = flatten_polygons_to_keypoints(annotations)

        image = np.zeros((640, 480, 3), dtype=np.uint8)
        pipeline = build_augmentation_pipeline(seed=42)
        result = pipeline(image=image, keypoints=keypoints)

        # remove_invisible=False guarantees no silent point-dropping even
        # when a crop/perspective pushes a vertex outside the frame.
        self.assertEqual(len(result["keypoints"]), len(keypoints))
        self.assertEqual(result["image"].shape[:2], (640, 640))  # RandomCrop target size

        regrouped = regroup_keypoints_to_polygons(result["keypoints"], counts)
        self.assertEqual(len(regrouped), 2)
        self.assertEqual(len(regrouped[0]), 4)
        self.assertEqual(len(regrouped[1]), 5)

    def test_same_seed_is_reproducible(self):
        annotations = [{"segmentation": [[10, 10, 50, 10, 50, 50, 10, 50]]}]
        keypoints, _ = flatten_polygons_to_keypoints(annotations)
        image = np.random.default_rng(0).integers(0, 255, (300, 300, 3), dtype=np.uint8)

        result_a = build_augmentation_pipeline(seed=123)(image=image.copy(), keypoints=keypoints)
        result_b = build_augmentation_pipeline(seed=123)(image=image.copy(), keypoints=keypoints)

        np.testing.assert_array_equal(result_a["image"], result_b["image"])
        self.assertEqual(result_a["keypoints"], result_b["keypoints"])


if __name__ == "__main__":
    unittest.main()
