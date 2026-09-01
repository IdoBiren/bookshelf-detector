import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from group_scenes import (  # noqa: E402
    assert_safe_to_write,
    group_by_time_gap,
    groups_from_file,
)


def _touch_photos(photos_dir: Path, stems: list[str]) -> None:
    photos_dir.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        (photos_dir / f"{stem}.jpg").write_bytes(b"not a real jpeg; only existence is checked")


class TestWriteProtection(unittest.TestCase):
    """The groups file holds hours of hand review. group_scenes.py's automatic
    proposal is explicitly documented as "a starting point, NOT a verdict" --
    so it must not silently delete the verdict once one has been made.

    This is not hypothetical: the automatic proposal produced 26 scenes for
    this dataset and the hand review produced 58, so an accidental re-run
    would replace correct groupings with ones already known to be wrong."""

    def test_refuses_to_overwrite_an_existing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "scene_groups.txt"
            out.write_text("scene001, scene002", encoding="utf-8")
            with self.assertRaises(SystemExit) as ctx:
                assert_safe_to_write(out, force=False)
            self.assertIn("--force", str(ctx.exception))

    def test_force_allows_overwrite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "scene_groups.txt"
            out.write_text("scene001, scene002", encoding="utf-8")
            assert_safe_to_write(out, force=True)  # must not raise

    def test_writing_a_new_file_is_always_allowed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert_safe_to_write(Path(tmpdir) / "does_not_exist.txt", force=False)


class TestGroupsFromFile(unittest.TestCase):
    """--from-groups renders contact sheets from an EXISTING grouping instead
    of re-proposing one from EXIF. Without it there is no way to look at the
    hand-reviewed grouping, which is exactly what picking a stratified eval
    set requires."""

    def _write_groups(self, tmpdir: Path, text: str) -> Path:
        f = tmpdir / "scene_groups.txt"
        f.write_text(text, encoding="utf-8")
        return f

    def test_resolves_scene_ids_to_photo_paths_in_file_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            photos = tmpdir / "photos"
            _touch_photos(photos, ["scene001", "scene002", "scene003"])
            groups_file = self._write_groups(tmpdir, "scene001, scene002\nscene003")

            groups, missing = groups_from_file(groups_file, photos)

            self.assertEqual(missing, [])
            self.assertEqual(
                [[p.stem for p in g] for g in groups],
                [["scene001", "scene002"], ["scene003"]],
            )

    def test_photos_absent_from_the_file_become_their_own_scene(self):
        """A photo nobody grouped is an independent scene -- the same rule the
        converter applies. This is the scene166-169 case: shelves added later
        that need no grouping line but still need a sheet."""
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            photos = tmpdir / "photos"
            _touch_photos(photos, ["scene001", "scene002", "scene166"])
            groups_file = self._write_groups(tmpdir, "scene001, scene002")

            groups, missing = groups_from_file(groups_file, photos)

            self.assertEqual(missing, [])
            self.assertIn(["scene166"], [[p.stem for p in g] for g in groups])

    def test_scene_ids_with_no_photo_are_reported_not_crashed_on(self):
        """Quarantining a photo (blurry/unrelated) leaves its id behind in the
        file. Rendering sheets must survive that and say which ones, rather
        than dying on a missing file or silently drawing a smaller sheet."""
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            photos = tmpdir / "photos"
            _touch_photos(photos, ["scene001"])
            groups_file = self._write_groups(tmpdir, "scene001, scene002")

            groups, missing = groups_from_file(groups_file, photos)

            self.assertEqual(missing, ["scene002"])
            self.assertEqual([[p.stem for p in g] for g in groups], [["scene001"]])

    def test_a_group_whose_photos_are_all_missing_is_dropped_entirely(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            photos = tmpdir / "photos"
            _touch_photos(photos, ["scene001"])
            groups_file = self._write_groups(tmpdir, "scene001\nscene900, scene901")

            groups, missing = groups_from_file(groups_file, photos)

            self.assertEqual([[p.stem for p in g] for g in groups], [["scene001"]])
            self.assertEqual(missing, ["scene900", "scene901"])

    def test_comments_and_blank_lines_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            photos = tmpdir / "photos"
            _touch_photos(photos, ["scene001", "scene002"])
            groups_file = self._write_groups(
                tmpdir, "\n".join(["# hand reviewed", "", "scene001, scene002"])
            )

            groups, missing = groups_from_file(groups_file, photos)

            self.assertEqual([[p.stem for p in g] for g in groups], [["scene001", "scene002"]])


class TestGroupByTimeGap(unittest.TestCase):
    """The existing automatic proposal had no tests at all."""

    def test_splits_on_a_gap_larger_than_the_threshold(self):
        from datetime import datetime, timedelta

        base = datetime(2026, 8, 22, 16, 0, 0)
        photos = [
            (Path("a.jpg"), base),
            (Path("b.jpg"), base + timedelta(seconds=2)),
            (Path("c.jpg"), base + timedelta(seconds=30)),
        ]
        groups = group_by_time_gap(photos, gap_seconds=5)
        self.assertEqual([[p.name for p in g] for g in groups], [["a.jpg", "b.jpg"], ["c.jpg"]])

    def test_orders_by_capture_time_not_input_order(self):
        from datetime import datetime, timedelta

        base = datetime(2026, 8, 22, 16, 0, 0)
        photos = [
            (Path("late.jpg"), base + timedelta(seconds=1)),
            (Path("early.jpg"), base),
        ]
        groups = group_by_time_gap(photos, gap_seconds=5)
        self.assertEqual([[p.name for p in g] for g in groups], [["early.jpg", "late.jpg"]])


if __name__ == "__main__":
    unittest.main()
