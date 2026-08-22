import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from merge_datasets import (  # noqa: E402
    is_valid_annotation,
    load_dataset,
    merge_all,
    polygon_area,
    scene_id_from_filename,
    should_keep_category,
    split_scenes_by_id,
    write_merged_dataset,
)


class TestSceneIdFromFilename(unittest.TestCase):
    def test_strips_roboflow_hash_suffix(self):
        # Real example from harald-varner's export.
        name = "FullSizeRender-2_JPEG_jpg.rf.b0b0ccf9a4df205fae247402aca05a68.jpg"
        self.assertEqual(
            scene_id_from_filename("harald-varner", name),
            "harald-varner::FullSizeRender-2_JPEG_jpg",
        )

    def test_namespaces_by_dataset_to_avoid_collisions(self):
        name = "img.rf.abcdef0123456789.jpg"
        a = scene_id_from_filename("dataset-a", name)
        b = scene_id_from_filename("dataset-b", name)
        self.assertNotEqual(a, b)

    def test_filename_without_rf_suffix_is_unchanged(self):
        self.assertEqual(
            scene_id_from_filename("ds", "plain_photo.jpg"),
            "ds::plain_photo.jpg",
        )


class TestPolygonArea(unittest.TestCase):
    def test_square(self):
        self.assertAlmostEqual(polygon_area([0, 0, 10, 0, 10, 10, 0, 10]), 100.0)

    def test_triangle(self):
        # base=10 along x-axis, apex at (5,10) -> area = 0.5*base*height = 50
        self.assertAlmostEqual(polygon_area([0, 0, 10, 0, 5, 10]), 50.0)

    def test_degenerate_returns_zero(self):
        self.assertEqual(polygon_area([0, 0, 10, 10]), 0.0)  # only 2 points


class TestIsValidAnnotation(unittest.TestCase):
    def test_valid_polygon_kept(self):
        ann = {"segmentation": [[0, 0, 100, 0, 100, 100, 0, 100]]}
        ok, reason = is_valid_annotation(ann, 1000, 1000)
        self.assertTrue(ok)
        self.assertEqual(reason, "")

    def test_missing_segmentation_dropped(self):
        ok, reason = is_valid_annotation({"segmentation": []}, 1000, 1000)
        self.assertFalse(ok)
        self.assertEqual(reason, "no_segmentation")

    def test_degenerate_polygon_dropped(self):
        ann = {"segmentation": [[0, 0, 10, 10]]}
        ok, reason = is_valid_annotation(ann, 1000, 1000)
        self.assertFalse(ok)
        self.assertEqual(reason, "degenerate_polygon")

    def test_zero_area_dropped(self):
        # Three collinear points -> zero area.
        ann = {"segmentation": [[0, 0, 5, 0, 10, 0]]}
        ok, reason = is_valid_annotation(ann, 1000, 1000)
        self.assertFalse(ok)
        self.assertEqual(reason, "zero_area")

    def test_too_small_dropped(self):
        # image is 1000x1000 = 1,000,000px; threshold is 0.1% = 1,000px.
        # a 10x10 square (area 100) is well under that.
        ann = {"segmentation": [[0, 0, 10, 0, 10, 10, 0, 10]]}
        ok, reason = is_valid_annotation(ann, 1000, 1000)
        self.assertFalse(ok)
        self.assertEqual(reason, "too_small")

    def test_just_above_threshold_kept(self):
        # 1,000,000 * 0.001 = 1000px threshold; a 40x40 square (1600px) clears it.
        ann = {"segmentation": [[0, 0, 40, 0, 40, 40, 0, 40]]}
        ok, _ = is_valid_annotation(ann, 1000, 1000)
        self.assertTrue(ok)


class TestShouldKeepCategory(unittest.TestCase):
    def test_dvd_excluded_case_insensitive(self):
        self.assertFalse(should_keep_category("dvd_spine"))
        self.assertFalse(should_keep_category("DVD_SPINE"))

    def test_book_variants_kept(self):
        for name in ("book spine", "book_spine", "book-spines", "books"):
            self.assertTrue(should_keep_category(name))


class TestSplitScenesById(unittest.TestCase):
    def test_ratio_is_approximately_respected(self):
        scenes = [f"scene-{i}" for i in range(100)]
        assignment = split_scenes_by_id(scenes, val_fraction=0.1, seed=42)
        val_count = sum(1 for v in assignment.values() if v == "val")
        self.assertEqual(val_count, 10)  # round(100 * 0.1)

    def test_same_seed_is_reproducible(self):
        scenes = [f"scene-{i}" for i in range(50)]
        a = split_scenes_by_id(scenes, val_fraction=0.2, seed=7)
        b = split_scenes_by_id(scenes, val_fraction=0.2, seed=7)
        self.assertEqual(a, b)

    def test_repeated_scene_ids_get_one_consistent_assignment(self):
        # Simulates multiple augmented copies of the same scene: the
        # function only ever sees the deduplicated set, so every occurrence
        # necessarily maps to the same split by construction of the dict.
        scenes = ["a", "a", "a", "b", "c"]
        assignment = split_scenes_by_id(scenes, val_fraction=0.34, seed=1)
        self.assertEqual(len(assignment), 3)  # one entry per unique scene


def _write_fake_dataset(root: Path, name: str, splits: dict) -> None:
    """splits: {split_name: {"images": [...], "annotations": [...], "categories": [...]}}"""
    for split_name, data in splits.items():
        split_dir = root / name / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "_annotations.coco.json").write_text(json.dumps(data), encoding="utf-8")
        for img in data["images"]:
            (split_dir / img["file_name"]).write_bytes(b"fake-image-bytes")


VALID_SEG = [[0, 0, 100, 0, 100, 100, 0, 100]]  # 100x100 square, well above threshold


class TestMergeAllIntegration(unittest.TestCase):
    def test_reproduces_and_fixes_the_real_cross_split_leak(self):
        """Mirrors the actual harald-varner bug: the SAME source photo
        (same base filename, different .rf.<hash> suffix) appears in both
        train and valid in the raw export. merge_all must consolidate all
        its copies into a single split."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_fake_dataset(
                root,
                "ds-a",
                {
                    "train": {
                        "categories": [{"id": 1, "name": "book spine"}],
                        "images": [
                            {"id": 1, "file_name": "photo1.rf.aaaa1111.jpg", "width": 1000, "height": 1000},
                            {"id": 2, "file_name": "photo2.rf.bbbb2222.jpg", "width": 1000, "height": 1000},
                        ],
                        "annotations": [
                            {"id": 1, "image_id": 1, "category_id": 1, "segmentation": VALID_SEG},
                            {"id": 2, "image_id": 2, "category_id": 1, "segmentation": VALID_SEG},
                        ],
                    },
                    "valid": {
                        "categories": [{"id": 1, "name": "book spine"}],
                        "images": [
                            # SAME scene as train's photo1 (different rf hash) — this
                            # is the leak.
                            {"id": 1, "file_name": "photo1.rf.cccc3333.jpg", "width": 1000, "height": 1000},
                        ],
                        "annotations": [
                            {"id": 1, "image_id": 1, "category_id": 1, "segmentation": VALID_SEG},
                        ],
                    },
                    "test": {"categories": [{"id": 1, "name": "book spine"}], "images": [], "annotations": []},
                },
            )

            kept_images, all_annotations, stats, dropped_empty = merge_all(
                root, ["ds-a"], val_fraction=0.5, seed=42
            )

            scene_to_splits = {}
            for img in kept_images:
                scene_to_splits.setdefault(img["_scene_id"], set()).add(img["_split"])

            leaked = {s: v for s, v in scene_to_splits.items() if len(v) > 1}
            self.assertEqual(leaked, {}, "photo1's two copies must land in the same split")
            self.assertEqual(len(scene_to_splits), 2)  # photo1, photo2 (photo1 has 2 copies)
            self.assertEqual(len(kept_images), 3)  # 2 train copies of photo-scenes + 1 valid copy
            self.assertEqual(dropped_empty, 0)
            self.assertEqual(stats, {})

    def test_dvd_category_and_bad_geometry_are_filtered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_fake_dataset(
                root,
                "ds-b",
                {
                    "train": {
                        "categories": [
                            {"id": 1, "name": "book_spine"},
                            {"id": 2, "name": "dvd_spine"},
                        ],
                        "images": [
                            {"id": 1, "file_name": "a.rf.1111.jpg", "width": 1000, "height": 1000},
                            {"id": 2, "file_name": "b.rf.2222.jpg", "width": 1000, "height": 1000},
                            {"id": 3, "file_name": "c.rf.3333.jpg", "width": 1000, "height": 1000},
                        ],
                        "annotations": [
                            {"id": 1, "image_id": 1, "category_id": 1, "segmentation": VALID_SEG},
                            # dvd -> excluded by category
                            {"id": 2, "image_id": 2, "category_id": 2, "segmentation": VALID_SEG},
                            # too small -> excluded by geometry; image 3 ends up empty
                            {"id": 3, "image_id": 3, "category_id": 1, "segmentation": [[0, 0, 5, 0, 5, 5, 0, 5]]},
                        ],
                    },
                    "valid": {"categories": [{"id": 1, "name": "book_spine"}, {"id": 2, "name": "dvd_spine"}], "images": [], "annotations": []},
                    "test": {"categories": [{"id": 1, "name": "book_spine"}, {"id": 2, "name": "dvd_spine"}], "images": [], "annotations": []},
                },
            )

            kept_images, all_annotations, stats, dropped_empty = merge_all(
                root, ["ds-b"], val_fraction=0.0, seed=1
            )

            self.assertEqual(len(kept_images), 1)  # only image "a" survives
            self.assertEqual(len(all_annotations), 1)
            self.assertEqual(stats.get("dropped_category"), 1)
            self.assertEqual(stats.get("dropped_too_small"), 1)
            # images "b" (dvd-only) and "c" (too-small-only) both end up
            # with zero valid annotations.
            self.assertEqual(dropped_empty, 2)

    def test_write_merged_dataset_produces_valid_coco_with_remapped_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_fake_dataset(
                root,
                "ds-c",
                {
                    "train": {
                        "categories": [{"id": 1, "name": "book"}],
                        "images": [
                            {"id": 5, "file_name": "x.rf.9999.jpg", "width": 500, "height": 500},
                        ],
                        "annotations": [
                            {"id": 9, "image_id": 5, "category_id": 1, "segmentation": VALID_SEG},
                        ],
                    },
                    "valid": {"categories": [{"id": 1, "name": "book"}], "images": [], "annotations": []},
                    "test": {"categories": [{"id": 1, "name": "book"}], "images": [], "annotations": []},
                },
            )
            kept_images, all_annotations, _, _ = merge_all(root, ["ds-c"], val_fraction=0.0, seed=1)

            out_dir = root / "merged"
            coco = write_merged_dataset(kept_images, all_annotations, out_dir)

            self.assertEqual(len(coco["train"]["images"]), 1)
            self.assertEqual(coco["train"]["categories"], [{"id": 1, "name": "spine", "supercategory": "none"}])
            img_out = coco["train"]["images"][0]
            self.assertEqual(img_out["file_name"], "ds-c__x.rf.9999.jpg")
            self.assertTrue((out_dir / "images" / "ds-c__x.rf.9999.jpg").exists())

            ann_out = coco["train"]["annotations"][0]
            self.assertEqual(ann_out["image_id"], img_out["id"])
            self.assertEqual(ann_out["bbox"], [0, 0, 100, 100])
            self.assertAlmostEqual(ann_out["area"], 10000.0)

            # Files written to disk are also valid, loadable JSON.
            with (out_dir / "pretrain_train.json").open() as f:
                reloaded = json.load(f)
            self.assertEqual(reloaded["images"][0]["file_name"], "ds-c__x.rf.9999.jpg")


class TestLoadDataset(unittest.TestCase):
    def test_pools_all_three_splits_ignoring_roboflow_split_assignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_fake_dataset(
                root,
                "ds-d",
                {
                    "train": {
                        "categories": [{"id": 1, "name": "book"}],
                        "images": [{"id": 1, "file_name": "t.rf.1.jpg", "width": 10, "height": 10}],
                        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "segmentation": VALID_SEG}],
                    },
                    "valid": {
                        "categories": [{"id": 1, "name": "book"}],
                        "images": [{"id": 1, "file_name": "v.rf.2.jpg", "width": 10, "height": 10}],
                        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "segmentation": VALID_SEG}],
                    },
                    "test": {
                        "categories": [{"id": 1, "name": "book"}],
                        "images": [{"id": 1, "file_name": "e.rf.3.jpg", "width": 10, "height": 10}],
                        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "segmentation": VALID_SEG}],
                    },
                },
            )
            images, annotations, categories, missing_files = load_dataset(root / "ds-d")
            self.assertEqual(len(images), 3)  # one from each of train/valid/test
            self.assertEqual(len(annotations), 3)
            self.assertEqual(categories, {1: "book"})
            self.assertEqual(missing_files, 0)

    def test_skips_missing_files_without_crashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_fake_dataset(
                root,
                "ds-e",
                {
                    "train": {
                        "categories": [{"id": 1, "name": "book"}],
                        "images": [
                            {"id": 1, "file_name": "present.rf.1.jpg", "width": 10, "height": 10},
                        ],
                        "annotations": [
                            {"id": 1, "image_id": 1, "category_id": 1, "segmentation": VALID_SEG},
                        ],
                    },
                    "valid": {"categories": [{"id": 1, "name": "book"}], "images": [], "annotations": []},
                    "test": {"categories": [{"id": 1, "name": "book"}], "images": [], "annotations": []},
                },
            )
            # Add a second json-referenced image/annotation whose file is
            # never written to disk, mirroring the real gap found in
            # harald-varner's train split.
            ann_path = root / "ds-e" / "train" / "_annotations.coco.json"
            data = json.loads(ann_path.read_text(encoding="utf-8"))
            data["images"].append({"id": 2, "file_name": "ghost.rf.2.jpg", "width": 10, "height": 10})
            data["annotations"].append(
                {"id": 2, "image_id": 2, "category_id": 1, "segmentation": VALID_SEG}
            )
            ann_path.write_text(json.dumps(data), encoding="utf-8")

            images, annotations, categories, missing_files = load_dataset(root / "ds-e")
            self.assertEqual(len(images), 1)
            self.assertEqual(len(annotations), 1)
            self.assertEqual(missing_files, 1)


if __name__ == "__main__":
    unittest.main()
