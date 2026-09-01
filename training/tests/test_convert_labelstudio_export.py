import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from convert_labelstudio_export import (  # noqa: E402
    convert,
    load_scene_groups,
    load_test_scenes,
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

    def test_decodes_the_coco_zip_export_format(self):
        """The COCO export downloaded from the UI (Export -> COCO -> zip) uses
        a DIFFERENT file_name shape than the raw API export the other two
        tests cover: "images\\<8-hex-hash>__<url-encoded-source-path>", no
        query string at all. First observed on a real export -- the smoke
        test this fixture is named after."""
        photos_dir = Path("C:/fake/photos")
        result = resolve_local_file_path(
            "images\\b358c755__photos%5Cscene001.jpg", photos_dir
        )
        self.assertEqual(result, photos_dir / "scene001.jpg")

    def test_coco_zip_export_format_with_forward_slashes(self):
        photos_dir = Path("C:/fake/photos")
        result = resolve_local_file_path("images/b358c755__photos%5Cscene001.jpg", photos_dir)
        self.assertEqual(result, photos_dir / "scene001.jpg")


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


class TestForcedTestScenes(unittest.TestCase):
    """plan §13 (מסלול מקביל): the ~40 stratified eval photos are chosen by
    hand, not by a random 15%. At 165 photos a random test split yields ~25
    images — short of the 40-60 §8's evaluation protocol requires — and there
    was no way to say "THESE scenes are the benchmark". Forcing them fixes
    both."""

    def test_forced_scenes_all_land_in_test(self):
        scenes = [f"scene{i:03d}" for i in range(20)]
        forced = {"scene003", "scene007", "scene011"}
        assignment = three_way_split_scenes(scenes, 0.15, 0.15, seed=42, forced_test_scenes=forced)
        for s in forced:
            self.assertEqual(assignment[s], "test")

    def test_remaining_scenes_split_train_val_only_never_test(self):
        """The point of forcing: test is EXACTLY the declared benchmark set.
        A stray random scene leaking into test would silently contaminate the
        frozen eval set."""
        scenes = [f"scene{i:03d}" for i in range(20)]
        forced = {"scene003", "scene007"}
        assignment = three_way_split_scenes(scenes, 0.15, 0.15, seed=42, forced_test_scenes=forced)
        for scene, split in assignment.items():
            if scene not in forced:
                self.assertIn(split, ("train", "val"), f"{scene} leaked into test")
        in_test = {s for s, split in assignment.items() if split == "test"}
        self.assertEqual(in_test, forced)

    def test_remainder_splits_85_15_not_70_15_15(self):
        """test_fraction is ignored when scenes are forced — the remainder is
        train/val only, so val_fraction applies to the REMAINDER."""
        scenes = [f"scene{i:03d}" for i in range(100)]
        forced = {f"scene{i:03d}" for i in range(20)}
        assignment = three_way_split_scenes(scenes, 0.15, 0.15, seed=42, forced_test_scenes=forced)
        counts = {"train": 0, "val": 0, "test": 0}
        for split in assignment.values():
            counts[split] += 1
        self.assertEqual(counts["test"], 20)
        self.assertEqual(counts["val"], 12)   # 15% of the 80 remaining
        self.assertEqual(counts["train"], 68)

    def test_every_scene_still_assigned_exactly_once(self):
        scenes = [f"scene{i:03d}" for i in range(20)]
        forced = {"scene003"}
        assignment = three_way_split_scenes(scenes, 0.15, 0.15, seed=3, forced_test_scenes=forced)
        self.assertEqual(set(assignment.keys()), set(scenes))

    def test_unknown_scene_in_file_raises_rather_than_silently_shrinking_the_eval_set(self):
        """A typo'd scene id must be loud. Silently dropping it would shrink
        the benchmark below the 40-60 §8 requires while still reporting
        'PASSED' — exactly the failure this flag exists to prevent."""
        scenes = [f"scene{i:03d}" for i in range(5)]
        with self.assertRaises(ValueError) as ctx:
            three_way_split_scenes(scenes, 0.15, 0.15, seed=1, forced_test_scenes={"scene002", "scene999"})
        self.assertIn("scene999", str(ctx.exception))
        self.assertNotIn("scene002", str(ctx.exception))

    def test_empty_forced_set_behaves_exactly_like_before(self):
        scenes = [f"scene{i:03d}" for i in range(50)]
        baseline = three_way_split_scenes(scenes, 0.15, 0.15, seed=42)
        for forced in (None, set()):
            self.assertEqual(
                three_way_split_scenes(scenes, 0.15, 0.15, seed=42, forced_test_scenes=forced),
                baseline,
            )


class TestLoadTestScenes(unittest.TestCase):
    def test_reads_one_scene_per_line_ignoring_blanks_and_comments(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "eval_scenes.txt"
            f.write_text(
                "\n".join(["# the frozen eval set (plan section 8)", "scene001", "", "scene002", "  scene003  "]),
                encoding="utf-8",
            )
            self.assertEqual(load_test_scenes(f), {"scene001", "scene002", "scene003"})

    def test_accepts_pasted_filenames_not_just_bare_scene_ids(self):
        """Normalised through scene_id_from_basename, because the natural way
        to build this list is to copy filenames out of the photos folder —
        including Explorer's "scene008 (2).jpg" auto-numbering."""
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "eval_scenes.txt"
            f.write_text("\n".join(["scene007.jpg", "scene008 (2).jpg", "scene009"]), encoding="utf-8")
            self.assertEqual(load_test_scenes(f), {"scene007", "scene008", "scene009"})

    def test_empty_file_raises_rather_than_silently_producing_no_test_set(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "eval_scenes.txt"
            f.write_text("# nothing but a comment", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_test_scenes(f)


class TestLoadSceneGroups(unittest.TestCase):
    """plan §3 / LABELING.md. Every in-domain photo arrived named sceneNNN.jpg,
    one id each — but EXIF shows all 165 were shot in one 30-minute session with
    139 of 164 consecutive gaps under 5 seconds, i.e. bursts of the same shelf.
    Without regrouping, near-duplicate shots land on both sides of the split:
    the leakage that made harald-varner's own public split unusable."""

    def _write(self, tmpdir, text):
        f = Path(tmpdir) / "scene_groups.txt"
        f.write_text(text, encoding="utf-8")
        return f

    def test_first_entry_names_the_scene(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = self._write(tmpdir, "\n".join(["scene001, scene002, scene003", "scene004, scene005"]))
            self.assertEqual(
                load_scene_groups(f),
                {
                    "scene001": "scene001",
                    "scene002": "scene001",
                    "scene003": "scene001",
                    "scene004": "scene004",
                    "scene005": "scene004",
                },
            )

    def test_ignores_comments_and_blank_lines_and_accepts_filenames(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = self._write(tmpdir, "\n".join(["# proposed from EXIF", "", "scene007.jpg, scene008 (2).jpg"]))
            self.assertEqual(load_scene_groups(f), {"scene007": "scene007", "scene008": "scene007"})

    def test_a_photo_in_two_groups_raises(self):
        """Hand-edited file, so a photo left in its old group after being moved
        is the likely mistake. Silently picking one would put it in a split the
        other group's photos are not in — leakage, reported as success."""
        with tempfile.TemporaryDirectory() as tmpdir:
            f = self._write(tmpdir, "\n".join(["scene001, scene002", "scene003, scene002"]))
            with self.assertRaises(ValueError) as ctx:
                load_scene_groups(f)
            self.assertIn("scene002", str(ctx.exception))


class TestGroupingAppliedInConvert(unittest.TestCase):
    def _export_with_two_photos(self, tmpdir):
        photos_dir = Path(tmpdir) / "photos"
        photos_dir.mkdir()
        images, annotations = [], []
        for index, name in enumerate(("scene001.jpg", "scene002.jpg"), start=1):
            (photos_dir / name).write_bytes(b"not a real jpeg, only existence is checked")
            images.append(
                {
                    "id": index,
                    "file_name": "/data/local-files/?d=photos/" + name,
                    "width": 800,
                    "height": 600,
                }
            )
            annotations.append(
                {
                    "id": index,
                    "image_id": index,
                    "category_id": 1,
                    "segmentation": VALID_SEG,
                    "bbox": [0, 0, 100, 100],
                    "area": 10000,
                    "iscrowd": 0,
                }
            )
        export = _write_export(Path(tmpdir), images, annotations, [{"id": 1, "name": "spine"}])
        return export, photos_dir

    def test_grouped_photos_share_a_split(self):
        """The whole point: two shots of one shelf must never straddle the
        train/val/test boundary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            export, photos_dir = self._export_with_two_photos(tmpdir)
            groups_file = Path(tmpdir) / "scene_groups.txt"
            groups_file.write_text("scene001, scene002", encoding="utf-8")

            kept_images, _, _ = convert(
                export, photos_dir, 0.5, 0.0, seed=1, scene_groups=load_scene_groups(groups_file)
            )
            splits = {img["file_name"]: img["_split"] for img in kept_images}
            self.assertEqual(len(splits), 2)
            self.assertEqual(len(set(splits.values())), 1, f"grouped photos were split apart: {splits}")

    def test_ungrouped_run_can_split_the_same_two_photos_apart(self):
        """Guards the test above from being vacuous — without grouping, this
        exact seed and fraction does separate them."""
        with tempfile.TemporaryDirectory() as tmpdir:
            export, photos_dir = self._export_with_two_photos(tmpdir)
            kept_images, _, _ = convert(export, photos_dir, 0.5, 0.0, seed=1)
            splits = {img["_split"] for img in kept_images}
            self.assertEqual(len(splits), 2)

    def test_forced_test_scene_may_be_named_by_any_member_of_its_group(self):
        """You pick eval photos by looking at them, so you name whichever one
        you were looking at — not necessarily the one that happens to be its
        group's canonical id."""
        with tempfile.TemporaryDirectory() as tmpdir:
            export, photos_dir = self._export_with_two_photos(tmpdir)
            groups = load_scene_groups(
                self._write_groups(Path(tmpdir), "scene001, scene002")
            )
            kept_images, _, _ = convert(
                export,
                photos_dir,
                0.0,
                0.0,
                seed=1,
                scene_groups=groups,
                forced_test_scenes={"scene002"},
            )
            self.assertEqual({img["_split"] for img in kept_images}, {"test"})

    def _write_groups(self, tmpdir, text):
        f = tmpdir / "scene_groups.txt"
        f.write_text(text, encoding="utf-8")
        return f


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
