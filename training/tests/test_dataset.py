"""
Tests for dataset.py — COCO polygons -> the (image, target) pairs
torchvision's Mask R-CNN expects.

The failure this guards against is the expensive kind: a target-format bug
does not crash, it trains, and you find out six GPU-hours later that the
masks were empty or the boxes were in the wrong convention. Every assertion
here is about the contract with torchvision, checked on synthetic data that
runs in milliseconds.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from dataset import SpineDataset, collate_fn  # noqa: E402


def _write_dataset(tmpdir: Path, polygons_per_image: list[list[list[float]]]) -> Path:
    """Builds a minimal COCO file plus real image files on disk."""
    images_dir = tmpdir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    images, annotations = [], []
    ann_id = 1
    for image_index, polygons in enumerate(polygons_per_image, start=1):
        name = f"img{image_index}.jpg"
        Image.new("RGB", (200, 150), (128, 128, 128)).save(images_dir / name)
        images.append({"id": image_index, "file_name": name, "width": 200, "height": 150})
        for polygon in polygons:
            xs, ys = polygon[0::2], polygon[1::2]
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_index,
                    "category_id": 1,
                    "segmentation": [polygon],
                    "bbox": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
                    "area": 1.0,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    coco_path = tmpdir / "data.json"
    coco_path.write_text(
        json.dumps({"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "spine"}]}),
        encoding="utf-8",
    )
    return coco_path


def _square(x: float, y: float, w: float, h: float) -> list[float]:
    return [x, y, x + w, y, x + w, y + h, x, y + h]


class TestSpineDataset(unittest.TestCase):
    def test_image_is_a_float_chw_tensor_in_0_to_1(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            coco = _write_dataset(tmpdir, [[_square(10, 10, 50, 60)]])
            dataset = SpineDataset(coco, tmpdir / "images")
            image, _ = dataset[0]

            self.assertEqual(image.dtype, torch.float32)
            self.assertEqual(image.shape[0], 3)  # CHW, not HWC
            self.assertGreaterEqual(float(image.min()), 0.0)
            self.assertLessEqual(float(image.max()), 1.0)

    def test_target_has_the_keys_and_dtypes_torchvision_requires(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            coco = _write_dataset(tmpdir, [[_square(10, 10, 50, 60)]])
            dataset = SpineDataset(coco, tmpdir / "images")
            _, target = dataset[0]

            self.assertEqual(target["boxes"].dtype, torch.float32)
            self.assertEqual(target["labels"].dtype, torch.int64)
            self.assertEqual(target["masks"].dtype, torch.uint8)

    def test_boxes_are_xyxy_not_the_xywh_coco_stores(self):
        """COCO's bbox is [x, y, width, height]; torchvision wants
        [x1, y1, x2, y2]. Passing COCO's format straight through trains
        without complaint and learns nonsense."""
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            coco = _write_dataset(tmpdir, [[_square(10, 20, 50, 60)]])
            dataset = SpineDataset(coco, tmpdir / "images")
            _, target = dataset[0]

            box = target["boxes"][0].tolist()
            self.assertAlmostEqual(box[0], 10, delta=1)
            self.assertAlmostEqual(box[1], 20, delta=1)
            self.assertAlmostEqual(box[2], 60, delta=1)   # x + w, not w
            self.assertAlmostEqual(box[3], 80, delta=1)   # y + h, not h

    def test_every_label_is_the_single_spine_class(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            coco = _write_dataset(tmpdir, [[_square(10, 10, 40, 40), _square(60, 10, 40, 40)]])
            dataset = SpineDataset(coco, tmpdir / "images")
            _, target = dataset[0]
            self.assertTrue(bool((target["labels"] == 1).all()))

    def test_masks_are_one_per_instance_and_actually_filled(self):
        """An all-zero mask is the classic silent failure: the shapes are
        right, the loss is finite, and the model learns nothing."""
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            coco = _write_dataset(tmpdir, [[_square(10, 10, 40, 50), _square(80, 20, 30, 60)]])
            dataset = SpineDataset(coco, tmpdir / "images")
            _, target = dataset[0]

            masks = target["masks"]
            self.assertEqual(masks.shape[0], 2)
            self.assertEqual(tuple(masks.shape[1:]), (150, 200))  # (H, W)
            for i in range(2):
                self.assertGreater(int(masks[i].sum()), 100, f"mask {i} is empty or near-empty")

    def test_touching_instances_stay_separate_masks(self):
        """The property that made Mask R-CNN the right replacement for
        DBNet: adjacency is not a problem the target format has to solve."""
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            coco = _write_dataset(tmpdir, [[_square(10, 10, 50, 80), _square(60, 10, 50, 80)]])
            dataset = SpineDataset(coco, tmpdir / "images")
            _, target = dataset[0]

            overlap = int((target["masks"][0] & target["masks"][1]).sum())
            total = int(target["masks"][0].sum())
            self.assertLess(overlap, total * 0.1)

    def test_images_with_no_annotations_are_skipped_entirely(self):
        """torchvision raises on a target with zero boxes. Dropping those
        images at load time is cheaper than special-casing the training
        loop -- and merge_datasets.py already drops empty images upstream,
        so this is a belt-and-braces guard, counted not silent."""
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            coco = _write_dataset(tmpdir, [[_square(10, 10, 40, 40)], []])
            dataset = SpineDataset(coco, tmpdir / "images")
            self.assertEqual(len(dataset), 1)

    def test_missing_image_file_is_skipped_with_a_count_not_a_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            coco = _write_dataset(tmpdir, [[_square(10, 10, 40, 40)], [_square(20, 20, 40, 40)]])
            (tmpdir / "images" / "img2.jpg").unlink()
            dataset = SpineDataset(coco, tmpdir / "images")
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset.stats["missing_image_file"], 1)

    def test_a_degenerate_zero_area_polygon_is_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            flat = [10.0, 10.0, 60.0, 10.0, 60.0, 10.0, 10.0, 10.0]  # zero height
            coco = _write_dataset(tmpdir, [[_square(10, 30, 40, 40), flat]])
            dataset = SpineDataset(coco, tmpdir / "images")
            _, target = dataset[0]
            self.assertEqual(target["masks"].shape[0], 1)
            self.assertEqual(dataset.stats["degenerate_annotation"], 1)


class TestCollateFn(unittest.TestCase):
    def test_keeps_images_as_a_list_rather_than_stacking_them(self):
        """Detection models take a LIST of images of differing sizes --
        default_collate would try to stack them and fail."""
        images = [torch.rand(3, 100, 120), torch.rand(3, 150, 90)]
        targets = [{"boxes": torch.zeros(1, 4)}, {"boxes": torch.zeros(2, 4)}]
        batched_images, batched_targets = collate_fn(list(zip(images, targets)))
        self.assertIsInstance(batched_images, list)
        self.assertEqual(len(batched_images), 2)
        self.assertEqual(batched_images[1].shape, (3, 150, 90))
        self.assertEqual(len(batched_targets), 2)


if __name__ == "__main__":
    unittest.main()
