"""
Converts a Label Studio COCO export (plan §3, the in-domain photos) into
the same shape merge_datasets.py already produces for pretrain — but as
its own "indomain" set, split 70/15/15 by scene (plan §4) rather than
90/10, and never mixed into the pretrain files (plan §3: pretrain and
indomain "stay marked separately, even after merging").

Label Studio's per-region "unreadable" Choice (plan §3/§8) exports as a
SEPARATE annotation with its own category rather than a field on the
spine annotation — this script pairs each one back onto its matching
spine annotation by (image_id, exact segmentation match) and folds it
into a single `unreadable: true/false` field instead.

We used Local Files storage (plan LABELING.md — the browser-upload path
has an unresolved Django/DRF bug), so file_name comes back as
"/data/local-files/?d=<url-encoded-relative-path>" rather than a real
path or a bundled image; this resolves it back to the real file already
sitting in the local photos directory.

Reuses merge_datasets.py's already-tested geometry validation, scene
split, and COCO-assembly logic rather than duplicating it.
"""

from __future__ import annotations

import argparse
import re
import json
import urllib.parse
from pathlib import Path

from merge_datasets import (
    is_valid_annotation,
    split_scenes_by_id,
    write_merged_dataset,
)

# Windows Explorer's bulk-rename auto-numbering produces "scene001 (2).jpg",
# "scene001 (3).jpg", ... for the 2nd+ file in a multi-select rename
# (plan LABELING.md) — strip that suffix to recover the shared scene id.
_EXPLORER_SUFFIX_RE = re.compile(r" \(\d+\)$")


def scene_id_from_basename(basename: str) -> str:
    return _EXPLORER_SUFFIX_RE.sub("", Path(basename).stem)


def load_test_scenes(path: Path) -> set[str]:
    """Reads the frozen eval set's scene ids, one per line ('#' comments and
    blank lines ignored).

    Entries run through scene_id_from_basename, so a list built the natural
    way — copying filenames out of the photos folder, Explorer's
    "scene008 (2).jpg" numbering included — works as well as bare scene ids.

    An effectively empty file raises: silently producing no forced test set
    would fall back to the random split this flag exists to replace.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    scenes = {
        scene_id_from_basename(stripped)
        for line in lines
        if (stripped := line.strip()) and not stripped.startswith("#")
    }
    if not scenes:
        raise ValueError(f"No scene ids found in {path} (only blank lines and comments?)")
    return scenes


def parse_groups_file(path: Path) -> list[list[str]]:
    """Reads the scene-grouping file into ordered groups of raw file stems.

    Shared deliberately with group_scenes.py rather than parsed twice: the two
    scripts write and read the same file, and a format drift between them
    would be silent. (Same reasoning as write_merged_dataset being shared
    between the pretrain merge and this converter.)

    Returns stems, NOT scene ids — the extension is dropped but Explorer's
    " (2)" bulk-rename suffix is preserved, because group_scenes.py needs the
    real filename back to find the photo on disk. load_scene_groups narrows
    these to scene ids for its own purposes.
    """
    groups: list[list[str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # Comma-separated, not whitespace: Explorer's bulk-rename numbering
        # puts a space inside the filename ("scene008 (2).jpg"), so splitting
        # on whitespace invents a bogus "(2)" scene. Caught by a test.
        members = [Path(part.strip()).stem for part in stripped.split(",") if part.strip()]
        if members:
            groups.append(members)
    return groups


def load_scene_groups(path: Path) -> dict[str, str]:
    """Reads the scene-grouping file written by group_scenes.py and returns
    {member scene id -> canonical scene id}. One comma-separated line per scene, the
    first entry naming it; '#' comments and blank lines ignored.

    Needed because every in-domain photo arrived as its own sceneNNN.jpg even
    though EXIF shows they were shot in bursts — several photos per shelf.
    Ungrouped scene ids simply stay themselves, so a partial file is valid.

    A scene id listed in two groups raises: after a hand edit that is almost
    certainly a photo moved but not deleted from its old line, and picking
    either silently would put it in a split its group-mates are not in.
    """
    canonical_by_member: dict[str, str] = {}
    for raw_members in parse_groups_file(path):
        members = [scene_id_from_basename(m) for m in raw_members]
        for member in members:
            if member in canonical_by_member and canonical_by_member[member] != members[0]:
                raise ValueError(
                    f"Scene {member!r} appears in two different groups in {path} "
                    f"({canonical_by_member[member]!r} and {members[0]!r}). "
                    "Remove it from one of them."
                )
            canonical_by_member[member] = members[0]
    if not canonical_by_member:
        raise ValueError(f"No scene groups found in {path} (only blank lines and comments?)")
    return canonical_by_member


def resolve_local_file_path(file_name: str, photos_dir: Path) -> Path:
    """Label Studio hands back file_name in two DIFFERENT shapes depending on
    export path, both observed on real exports:

    - the raw API export: "/data/local-files/?d=<url-encoded-relative-path>"
    - the COCO zip export (Export -> COCO in the UI): a query-string-free
      "images\\<8-hex-hash>__<url-encoded-source-path>", e.g.
      "images\\b358c755__photos%5Cscene001.jpg"

    We already know exactly where the real files live, so only the basename
    matters in either case -- unquote first (the COCO shape's separator is
    itself URL-encoded), normalise backslashes, then take the last segment.
    """
    parsed = urllib.parse.urlparse(file_name)
    query = urllib.parse.parse_qs(parsed.query)
    relative = query.get("d", [file_name])[0]
    unquoted = urllib.parse.unquote(relative)
    basename = Path(unquoted.replace("\\", "/")).name
    return photos_dir / basename


def three_way_split_scenes(
    scene_ids: list[str],
    val_fraction: float,
    test_fraction: float,
    seed: int,
    forced_test_scenes: set[str] | None = None,
) -> dict[str, str]:
    """Composes split_scenes_by_id twice to get train/val/test (70/15/15
    by default) instead of writing separate 3-way split logic.

    With `forced_test_scenes`, test stops being random: exactly those scenes
    become the test split and everything else splits train/val by
    val_fraction (plan §13). `test_fraction` is then unused — the benchmark's
    size is whatever was declared, not a percentage.

    Why this exists: at 165 photos a random 15% test split is ~25 images,
    below the 40-60 §8's evaluation protocol requires, and the eval set has
    to be a *stratified* hand-pick across §3's six scenarios rather than a
    random draw, or the benchmark misses the hard cases and the numbers come
    out optimistically wrong.
    """
    if forced_test_scenes:
        unknown = forced_test_scenes - set(scene_ids)
        if unknown:
            raise ValueError(
                f"Test scenes not present in the export: {sorted(unknown)}. "
                "Either they are not labeled yet, or the id is a typo. Refusing "
                "to continue: silently dropping them would shrink the frozen "
                "eval set below what plan §8 requires while still reporting success."
            )
        remainder = [s for s in scene_ids if s not in forced_test_scenes]
        result = dict(split_scenes_by_id(remainder, val_fraction, seed))
        result.update({s: "test" for s in forced_test_scenes})
        return result

    holdout_fraction = val_fraction + test_fraction
    stage1 = split_scenes_by_id(scene_ids, holdout_fraction, seed)
    train_scenes = [s for s, split in stage1.items() if split == "train"]
    holdout_scenes = [s for s, split in stage1.items() if split == "val"]

    test_fraction_of_holdout = test_fraction / holdout_fraction if holdout_fraction > 0 else 0
    stage2 = split_scenes_by_id(holdout_scenes, test_fraction_of_holdout, seed + 1)

    result = {s: "train" for s in train_scenes}
    for s, split in stage2.items():
        result[s] = "test" if split == "val" else "val"
    return result


def convert(
    export_path: Path,
    photos_dir: Path,
    val_fraction: float,
    test_fraction: float,
    seed: int,
    forced_test_scenes: set[str] | None = None,
    scene_groups: dict[str, str] | None = None,
) -> tuple[list[dict], list[dict], dict[str, int]]:
    with export_path.open(encoding="utf-8") as f:
        data = json.load(f)

    category_names = {c["id"]: c["name"] for c in data["categories"]}
    spine_cat_ids = {cid for cid, name in category_names.items() if name.lower() == "spine"}
    unreadable_cat_ids = {cid for cid, name in category_names.items() if name.lower() == "unreadable"}
    if not spine_cat_ids:
        raise ValueError(f"No 'spine' category found in export categories: {category_names}")

    images_by_id = {img["id"]: img for img in data["images"]}
    stats: dict[str, int] = {}
    for img in images_by_id.values():
        resolved = resolve_local_file_path(img["file_name"], photos_dir)
        if not resolved.exists():
            # Counted once, below, per orphaned annotation — not here too
            # (an earlier version double-counted: once per missing image,
            # again per annotation on it).
            continue
        img["_dataset"] = "indomain"
        img["_source_path"] = str(resolved)
        raw_scene_id = scene_id_from_basename(resolved.name)
        # Several photos of one shelf share a scene so they cannot straddle
        # the split; without a groups file each photo stays its own scene.
        img["_scene_id"] = (scene_groups or {}).get(raw_scene_id, raw_scene_id)
        # write_merged_dataset builds the output filename from file_name —
        # overwrite Label Studio's raw "/data/local-files/?d=..." URL with
        # the real basename, or it ends up as a literal (invalid) filename.
        img["file_name"] = resolved.name

    spine_anns = [a for a in data["annotations"] if a["category_id"] in spine_cat_ids]
    unreadable_anns = [a for a in data["annotations"] if a["category_id"] in unreadable_cat_ids]
    unreadable_keys = {(a["image_id"], tuple(a["segmentation"][0])) for a in unreadable_anns}
    stats["unreadable_markers_found"] = len(unreadable_anns)

    kept_annotations = []
    for ann in spine_anns:
        img = images_by_id.get(ann["image_id"])
        if img is None or "_source_path" not in img:
            stats["dropped_missing_file"] = stats.get("dropped_missing_file", 0) + 1
            continue
        ok, reason = is_valid_annotation(ann, img["width"], img["height"])
        if not ok:
            key = f"dropped_{reason}"
            stats[key] = stats.get(key, 0) + 1
            continue
        is_unreadable = (ann["image_id"], tuple(ann["segmentation"][0])) in unreadable_keys
        ann_copy = dict(ann)
        ann_copy["_image_ref"] = img
        ann_copy["_unreadable"] = is_unreadable
        kept_annotations.append(ann_copy)

    images_with_anns = {a["_image_ref"]["id"] for a in kept_annotations}
    kept_images = [img for img in images_by_id.values() if img.get("id") in images_with_anns]
    stats["dropped_empty_images"] = len([i for i in images_by_id.values() if "_source_path" in i]) - len(kept_images)

    scene_ids = [img["_scene_id"] for img in kept_images]
    if forced_test_scenes and scene_groups:
        # You pick eval photos by looking at them, so you name whichever photo
        # you were looking at — not necessarily its group's canonical id.
        forced_test_scenes = {scene_groups.get(s, s) for s in forced_test_scenes}
    split_assignment = three_way_split_scenes(
        scene_ids, val_fraction, test_fraction, seed, forced_test_scenes
    )
    stats["scenes_after_grouping"] = len(set(scene_ids))
    for img in kept_images:
        img["_split"] = split_assignment[img["_scene_id"]]

    return kept_images, kept_annotations, stats


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True, help="Path to Label Studio's COCO export result.json")
    parser.add_argument("--photos-dir", type=Path, default=repo_root / "data" / "indomain" / "photos")
    parser.add_argument("--out-dir", type=Path, default=repo_root / "data" / "merged")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--test-scenes",
        type=Path,
        default=None,
        help="File of scene ids (one per line) to force into the test split — the frozen, "
        "hand-stratified eval set (plan §8/§13). The remainder then splits train/val by "
        "--val-fraction and --test-fraction is ignored. Without it, test is a random 15%%.",
    )
    parser.add_argument(
        "--scene-groups",
        type=Path,
        default=None,
        help="File grouping photos of the same physical shelf into one scene, as written by "
        "group_scenes.py. Without it every photo is treated as an independent scene — which is "
        "wrong for burst shots and silently inflates every validation number.",
    )
    args = parser.parse_args()

    forced_test_scenes = load_test_scenes(args.test_scenes.resolve()) if args.test_scenes else None
    scene_groups = load_scene_groups(args.scene_groups.resolve()) if args.scene_groups else None

    kept_images, kept_annotations, stats = convert(
        args.export.resolve(),
        args.photos_dir.resolve(),
        args.val_fraction,
        args.test_fraction,
        args.seed,
        forced_test_scenes,
        scene_groups,
    )

    scene_to_splits: dict[str, set] = {}
    for img in kept_images:
        scene_to_splits.setdefault(img["_scene_id"], set()).add(img["_split"])
    leaked = {s: v for s, v in scene_to_splits.items() if len(v) > 1}
    if leaked:
        raise RuntimeError(f"Scene leakage across splits detected: {leaked}")

    coco_by_split = write_merged_dataset(
        kept_images,
        kept_annotations,
        args.out_dir.resolve(),
        splits=("train", "val", "test"),
        filename_prefix="indomain",
    )

    print(f"Kept images: {len(kept_images)}")
    print(f"Kept annotations: {len(kept_annotations)}")
    print(f"Stats: {stats}")
    print(f"Unique scenes: {len(scene_to_splits)}")
    if scene_groups:
        print(f"Scene grouping: {args.scene_groups}")
    else:
        print("Scene grouping: NONE — every photo treated as its own scene")
    if forced_test_scenes:
        print(f"Test split: FORCED from {args.test_scenes} ({len(forced_test_scenes)} scenes)")
    else:
        print("Test split: random by scene (no --test-scenes given)")
    for split in ("train", "val", "test"):
        print(
            f"{split.capitalize()}: {len(coco_by_split[split]['images'])} images, "
            f"{len(coco_by_split[split]['annotations'])} annotations"
        )
    print("Scene leakage check: PASSED (0 scenes span multiple splits)")


if __name__ == "__main__":
    main()
