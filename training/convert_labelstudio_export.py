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


def resolve_local_file_path(file_name: str, photos_dir: Path) -> Path:
    """Label Studio Local Files storage exports file_name as
    "/data/local-files/?d=<url-encoded-path-relative-to-document-root>".
    We already know exactly where the real files live, so only the
    basename matters here."""
    parsed = urllib.parse.urlparse(file_name)
    query = urllib.parse.parse_qs(parsed.query)
    relative = query.get("d", [file_name])[0]
    basename = Path(relative.replace("\\", "/")).name
    return photos_dir / basename


def three_way_split_scenes(
    scene_ids: list[str], val_fraction: float, test_fraction: float, seed: int
) -> dict[str, str]:
    """Composes split_scenes_by_id twice to get train/val/test (70/15/15
    by default) instead of writing separate 3-way split logic."""
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
        img["_scene_id"] = scene_id_from_basename(resolved.name)
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
    split_assignment = three_way_split_scenes(scene_ids, val_fraction, test_fraction, seed)
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
    args = parser.parse_args()

    kept_images, kept_annotations, stats = convert(
        args.export.resolve(), args.photos_dir.resolve(), args.val_fraction, args.test_fraction, args.seed
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
    for split in ("train", "val", "test"):
        print(
            f"{split.capitalize()}: {len(coco_by_split[split]['images'])} images, "
            f"{len(coco_by_split[split]['annotations'])} annotations"
        )
    print("Scene leakage check: PASSED (0 scenes span multiple splits)")


if __name__ == "__main__":
    main()
