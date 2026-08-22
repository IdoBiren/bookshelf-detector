import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from convert_labelstudio_export import (  # noqa: E402
    convert,
    resolve_local_file_path,
    scene_id_from_basename,
    three_way_split_scenes,
)
from merge_datasets import write_merged_dataset  # noqa: E402

VALID_SEG = [[0, 0, 100, 0, 100, 100, 0, 100]]  # 100x100 square, well above threshold


class TestSceneIdFromBasename(unittest.TestCase):
    def test_plain_filename(self):
        self.assertEqual(scene_id_from_basename("scene001.jpg"), "scene001")

    def test_explorer_bulk_rename_suffix_stripped(self):
        # Windows Explorer's multi-select-rename produces these for the
        # 2nd+ photo of the same shelf (plan LABELING.md).
        self.assertEqual(scene_id_from_basename("scene001 (2).jpg"), "scene001")
        self.assertEqual(scene_id_from_basename("scene001 (13).jpg"), "scene001")

    def test_no_false_positive_on_unrelated_parenthesized_name(self):
        # Only a trailing " (<digits>)" right before the extension counts.
        self.assertEqual(scene_id_from_basename("scene (final) v2.jpg"), "scene (final) v2")


class TestResolveLocalFilePath(unittest.TestCase):
    def test_decodes_label_studio_local_files_url(self):
        photos_dir = Path("C:/fake/photos")
        result = resolve_local_file_path("/data/local-files/?d=photos%5Cscene001.jpg", photos_dir)
        self.assertEqual(result, photos_dir / "scene001.jpg")

    def test_handles_forward_slash_variant(self):
        photos_dir = Path("C:/fake/photos")
        result = resolve_local_file_path("/data/local-files/?d=photos/scene002.jpg", photos_dir)
        self.assertEqual(result, photos_dir / "scene002.jpg")


class TestThreeWaySplitScenes(unittest.TestCase):
    def test_ratio_is_approximately_respected(self):
        scenes = [f"scene{i:03d}" for i in range(100)]
        assignment = three_way_split_scenes(scenes, val_fraction=0.15, test_fraction=0.15, seed=42)
        counts = {"train": 0, "val": 0, "test": 0}
        for split in assignment.values():
            counts[split] += 1
        self.assertEqual(counts["train"], 70)
        self.assertEqual(counts["val"] + counts["test"], 30)
        # Composed from two 50/50-ish stages — both non-trivial, neither empty.
        self.assertGreater(counts["val"], 0)
        self.assertGreater(counts["test"], 0)

    def test_every_scene_gets_assigned_exactly_once(self):
        scenes = [f"s{i}" for i in range(37)]
        assignment = three_way_split_scenes(scenes, 0.15, 0.15, seed=1)
        self.assertEqual(set(assignment.keys()), set(scenes))
        self.assertTrue(all(v in ("train", "val", "test") for v in assignment.values()))


def _write_export(tmp: Path, images: list, annotations: list, categories: list) -> Path:
    export_path = tmp / "result.json"
    export_path.write_text(
        json.dumps({"images": images, "annotations": annotations, "categories": categories, "info": {}}),
        encoding="utf-8",
    )
    return export_path


class TestConvertIntegration(unittest.TestCase):
    def test_pairs_unreadable_marker_onto_matching_spine_annotation(self):
        """The real export we tested against had zero 'unreadable' markers
        — this synthetic case exercises that pairing logic directly, since
        plan §8's read-accuracy metric depends on it being correct whenever
        it DOES show up in a later batch."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            photos_dir = tmp_path / "photos"
            photos_dir.mkdir()
            (photos_dir / "scene001.jpg").write_bytes(b"fake")
            (photos_dir / "scene002.jpg").write_bytes(b"fake")

            export_path = _write_export(
                tmp_path,
                images=[
                    {"id": 0, "width": 1000, "height": 1000, "file_name": "/data/local-files/?d=photos%5Cscene001.jpg"},
                    {"id": 1, "width": 1000, "height": 1000, "file_name": "/data/local-files/?d=photos%5Cscene002.jpg"},
                ],
                annotations=[
                    # image 0: a spine marked unreadable (duplicate category+geometry).
                    {"id": 0, "image_id": 0, "category_id": 0, "segmentation": VALID_SEG},
                    {"id": 1, "image_id": 0, "category_id": 1, "segmentation": VALID_SEG},
                    # image 1: an ordinary readable spine, no duplicate.
                    {"id": 2, "image_id": 1, "category_id": 0, "segmentation": VALID_SEG},
                ],
                categories=[{"id": 0, "name": "spine"}, {"id": 1, "name": "unreadable"}],
            )

            kept_images, kept_annotations, stats = convert(
                export_path, photos_dir, val_fraction=0.0, test_fraction=0.0, seed=1
            )

            self.assertEqual(stats["unreadable_markers_found"], 1)
            self.assertEqual(len(kept_images), 2)
            # 2 real spine annotations: image 0's (marked unreadable) and
            # image 1's ordinary one. The unreadable-category duplicate
            # itself is not a spine annotation, so it isn't counted here.
            self.assertEqual(len(kept_annotations), 2)

            by_image = {a["_image_ref"]["id"]: a for a in kept_annotations}
            self.assertTrue(by_image[0]["_unreadable"])
            self.assertFalse(by_image[1]["_unreadable"])

    def test_missing_photo_file_is_skipped_not_crashed(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            photos_dir = tmp_path / "photos"
            photos_dir.mkdir()
            (photos_dir / "scene001.jpg").write_bytes(b"fake")
            # scene002.jpg deliberately never created.

            export_path = _write_export(
                tmp_path,
                images=[
                    {"id": 0, "width": 1000, "height": 1000, "file_name": "/data/local-files/?d=photos%5Cscene001.jpg"},
                    {"id": 1, "width": 1000, "height": 1000, "file_name": "/data/local-files/?d=photos%5Cscene002.jpg"},
                ],
                annotations=[
                    {"id": 0, "image_id": 0, "category_id": 0, "segmentation": VALID_SEG},
                    {"id": 1, "image_id": 1, "category_id": 0, "segmentation": VALID_SEG},
                ],
                categories=[{"id": 0, "name": "spine"}],
            )

            kept_images, kept_annotations, stats = convert(
                export_path, photos_dir, val_fraction=0.0, test_fraction=0.0, seed=1
            )
            self.assertEqual(len(kept_images), 1)
            self.assertEqual(stats["dropped_missing_file"], 1)

    def test_full_pipeline_writes_a_clean_output_filename(self):
        """Missed by every other test here: they only checked convert()'s
        in-memory data, never the actual write_merged_dataset step — which
        is exactly where a real run against real data broke, because
        img["file_name"] still held Label Studio's raw
        "/data/local-files/?d=..." URL instead of the real basename."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            photos_dir = tmp_path / "photos"
            photos_dir.mkdir()
            (photos_dir / "scene001.jpg").write_bytes(b"fake")

            export_path = _write_export(
                tmp_path,
                images=[{"id": 0, "width": 1000, "height": 1000, "file_name": "/data/local-files/?d=photos%5Cscene001.jpg"}],
                annotations=[{"id": 0, "image_id": 0, "category_id": 0, "segmentation": VALID_SEG}],
                categories=[{"id": 0, "name": "spine"}],
            )

            kept_images, kept_annotations, _ = convert(export_path, photos_dir, 0.0, 0.0, seed=1)
            out_dir = tmp_path / "merged"
            coco = write_merged_dataset(
                kept_images, kept_annotations, out_dir, splits=("train", "val", "test"), filename_prefix="indomain"
            )

            written_files = list((out_dir / "images").iterdir())
            self.assertEqual(len(written_files), 1)
            self.assertEqual(written_files[0].name, "indomain__scene001.jpg")
            self.assertEqual(coco["train"]["images"][0]["file_name"], "indomain__scene001.jpg")

    def test_raises_clear_error_when_no_spine_category_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            photos_dir = tmp_path / "photos"
            photos_dir.mkdir()
            export_path = _write_export(tmp_path, images=[], annotations=[], categories=[{"id": 0, "name": "something_else"}])
            with self.assertRaises(ValueError):
                convert(export_path, photos_dir, 0.15, 0.15, seed=1)

    def test_scene_split_has_zero_leakage_with_repeated_scene_ids(self):
        """Explorer-renamed duplicates of the same scene (e.g. two angles of
        one shelf) must always land in the same split."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            photos_dir = tmp_path / "photos"
            photos_dir.mkdir()
            filenames = [f"scene{i:03d}.jpg" for i in range(20)] + [f"scene{i:03d} (2).jpg" for i in range(20)]
            for fn in filenames:
                (photos_dir / fn).write_bytes(b"fake")

            images = [
                {"id": i, "width": 1000, "height": 1000, "file_name": f"/data/local-files/?d=photos%5C{fn}"}
                for i, fn in enumerate(filenames)
            ]
            annotations = [
                {"id": i, "image_id": i, "category_id": 0, "segmentation": VALID_SEG} for i in range(len(filenames))
            ]
            export_path = _write_export(tmp_path, images, annotations, [{"id": 0, "name": "spine"}])

            kept_images, _, _ = convert(export_path, photos_dir, val_fraction=0.2, test_fraction=0.2, seed=5)
            scene_to_splits = {}
            for img in kept_images:
                scene_to_splits.setdefault(img["_scene_id"], set()).add(img["_split"])
            leaked = {s: v for s, v in scene_to_splits.items() if len(v) > 1}
            self.assertEqual(leaked, {})
            self.assertEqual(len(scene_to_splits), 20)  # 20 unique scenes despite 40 images


if __name__ == "__main__":
    unittest.main()
