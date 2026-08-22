"""
Merges the approved public book-spine datasets (plan §0/§3) into one COCO
instance-segmentation corpus, split by SCENE rather than by image.

This matters because it's not hypothetical: harald-varner's own Roboflow
train/valid/test split turned out NOT to be scene-aware — 43% of its unique
source photos had augmented copies (Roboflow appends `.rf.<hash>` per
copy) scattered across train AND valid/test. Re-deriving scene identity
from the base filename and re-splitting by that (ignoring Roboflow's given
split entirely) is the fix; it also means we don't need to discard the
pre-baked augmented copies, just keep every copy of a scene on one side.

Zero third-party dependencies on purpose — training/inference will need
PyTorch/onnxruntime later, which may not have wheels yet for this
environment's Python 3.14 (plan §12); this script has no reason to share
that risk.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from pathlib import Path

SPINE_CATEGORY_ID = 1
SPINE_CATEGORY_NAME = "spine"
EXCLUDE_CATEGORY_SUBSTRINGS = ("dvd",)  # woody-willis' dvd_spine (plan §3)
MIN_AREA_FRACTION = 0.001  # plan §3: drop instances <~0.1% of image area
MIN_POLYGON_POINTS = 3

DEFAULT_DATASETS = [
    "harald-varner-book-spine-instance-segmentation",
    "leo-ueno-book-spine-segmentation",
    "woody-willis-dahl-s-book-spine-detection",
]

# Roboflow appends ".rf.<hex-hash>.<ext>" to every exported filename,
# regardless of whether the image is an augmented copy or the original —
# stripping it recovers the underlying source-photo identity.
_RF_SUFFIX_RE = re.compile(r"\.rf\.[0-9a-f]+\.(jpg|jpeg|png)$", re.IGNORECASE)


def scene_id_from_filename(dataset_name: str, file_name: str) -> str:
    """Derives a scene identity from a Roboflow export filename, namespaced
    by dataset so two datasets can never collide on a coincidentally-shared
    base name."""
    base = _RF_SUFFIX_RE.sub("", file_name)
    return f"{dataset_name}::{base}"


def polygon_area(coords: list[float]) -> float:
    """Shoelace formula on a flat [x0,y0,x1,y1,...] segmentation polygon."""
    n = len(coords) // 2
    if n < 3:
        return 0.0
    total = 0.0
    for i in range(n):
        x0, y0 = coords[2 * i], coords[2 * i + 1]
        j = (i + 1) % n
        x1, y1 = coords[2 * j], coords[2 * j + 1]
        total += x0 * y1 - x1 * y0
    return abs(total) / 2.0


def should_keep_category(category_name: str) -> bool:
    name_lower = category_name.lower()
    return not any(bad in name_lower for bad in EXCLUDE_CATEGORY_SUBSTRINGS)


def is_valid_annotation(ann: dict, image_width: float, image_height: float) -> tuple[bool, str]:
    """Returns (keep, reason_if_dropped)."""
    seg = ann.get("segmentation")
    if not seg or not seg[0]:
        return False, "no_segmentation"
    coords = seg[0]
    if len(coords) < MIN_POLYGON_POINTS * 2:
        return False, "degenerate_polygon"
    area = polygon_area(coords)
    if area <= 0:
        return False, "zero_area"
    image_area = image_width * image_height
    if image_area > 0 and area < MIN_AREA_FRACTION * image_area:
        return False, "too_small"
    return True, ""


def split_scenes_by_id(scene_ids: list[str], val_fraction: float, seed: int) -> dict[str, str]:
    """Assigns each UNIQUE scene_id to 'train' or 'val', deterministically.
    Every image sharing a scene_id always lands in the same split — this is
    the fix for the cross-split leakage found in harald-varner's data.

    sorted() before shuffling is not cosmetic: Python randomizes string hash
    seeds per process by default, so set/dict iteration order for strings
    varies run to run — without sorting first, the same `seed` would not
    reproduce the same split.
    """
    unique_scenes = sorted(set(scene_ids))
    rng = random.Random(seed)
    rng.shuffle(unique_scenes)
    val_count = round(len(unique_scenes) * val_fraction)
    val_scenes = set(unique_scenes[:val_count])
    return {s: ("val" if s in val_scenes else "train") for s in unique_scenes}


def load_dataset(
    dataset_dir: Path,
) -> tuple[list[dict], list[dict], dict[int, str], int]:
    """Pools all images+annotations from a Roboflow COCO export's
    train/valid/test folders, deliberately IGNORING Roboflow's own split
    assignment (see module docstring). Returns (images, annotations,
    category_id -> name, missing_file_count), each annotation carrying an
    `_image_ref` pointer to its (mutable) image dict so later steps don't
    need a second lookup.

    A real download can have a few files the json references but that
    never made it to disk (observed: 2/1024 in harald-varner's train split,
    one with a base64-style filename over 200 chars — plausibly a Windows
    MAX_PATH truncation during zip extraction). Skip them and report the
    count rather than crashing on the first missing file.
    """
    images: list[dict] = []
    annotations: list[dict] = []
    categories: dict[int, str] = {}
    missing_files = 0

    for split_dir_name in ("train", "valid", "test"):
        split_dir = dataset_dir / split_dir_name
        ann_path = split_dir / "_annotations.coco.json"
        if not ann_path.exists():
            continue
        with ann_path.open(encoding="utf-8") as f:
            data = json.load(f)

        for cat in data["categories"]:
            categories[cat["id"]] = cat["name"]

        # image ids are only unique WITHIN one split's json, so annotations
        # are linked by object reference rather than by a shared id space.
        id_map: dict[int, dict] = {}
        for img in data["images"]:
            source_path = split_dir / img["file_name"]
            if not source_path.exists():
                missing_files += 1
                continue
            img_copy = dict(img)
            img_copy["_source_path"] = str(source_path)
            images.append(img_copy)
            id_map[img["id"]] = img_copy
        for ann in data["annotations"]:
            if ann["image_id"] not in id_map:
                continue  # image was missing on disk, already skipped above
            ann_copy = dict(ann)
            ann_copy["_image_ref"] = id_map[ann["image_id"]]
            annotations.append(ann_copy)

    return images, annotations, categories, missing_files


def merge_all(
    raw_root: Path, datasets: list[str], val_fraction: float, seed: int
) -> tuple[list[dict], list[dict], dict[str, int], int]:
    all_images: list[dict] = []
    all_annotations: list[dict] = []
    stats: dict[str, int] = {}
    temp_id_counter = 0

    for dataset_name in datasets:
        images, annotations, categories, missing_files = load_dataset(raw_root / dataset_name)
        if missing_files:
            stats["dropped_missing_file"] = stats.get("dropped_missing_file", 0) + missing_files
        keep_category_ids = {
            cid for cid, name in categories.items() if should_keep_category(name)
        }

        for img in images:
            img["_dataset"] = dataset_name
            img["_scene_id"] = scene_id_from_filename(dataset_name, img["file_name"])
            img["_temp_id"] = temp_id_counter
            temp_id_counter += 1

        for ann in annotations:
            if ann["category_id"] not in keep_category_ids:
                stats["dropped_category"] = stats.get("dropped_category", 0) + 1
                continue
            img = ann["_image_ref"]
            ok, reason = is_valid_annotation(ann, img["width"], img["height"])
            if not ok:
                key = f"dropped_{reason}"
                stats[key] = stats.get(key, 0) + 1
                continue
            ann["_keep"] = True

        all_images.extend(images)
        all_annotations.extend(a for a in annotations if a.get("_keep"))

    images_with_anns = {a["_image_ref"]["_temp_id"] for a in all_annotations}
    kept_images = [img for img in all_images if img["_temp_id"] in images_with_anns]
    dropped_empty_images = len(all_images) - len(kept_images)

    scene_ids = [img["_scene_id"] for img in kept_images]
    split_assignment = split_scenes_by_id(scene_ids, val_fraction, seed)
    for img in kept_images:
        img["_split"] = split_assignment[img["_scene_id"]]

    return kept_images, all_annotations, stats, dropped_empty_images


def write_merged_dataset(
    kept_images: list[dict], all_annotations: list[dict], out_dir: Path
) -> dict[str, dict]:
    images_out_dir = out_dir / "images"
    images_out_dir.mkdir(parents=True, exist_ok=True)

    coco_by_split = {
        split: {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": SPINE_CATEGORY_ID, "name": SPINE_CATEGORY_NAME, "supercategory": "none"}
            ],
        }
        for split in ("train", "val")
    }

    for i, img in enumerate(kept_images, start=1):
        img["_global_id"] = i
        new_filename = f"{img['_dataset']}__{Path(img['file_name']).name}"
        dest_path = images_out_dir / new_filename
        if not dest_path.exists():
            shutil.copyfile(img["_source_path"], dest_path)
        coco_by_split[img["_split"]]["images"].append(
            {
                "id": img["_global_id"],
                "file_name": new_filename,
                "width": img["width"],
                "height": img["height"],
            }
        )

    ann_id_counter = 1
    for ann in all_annotations:
        img = ann["_image_ref"]
        coords = ann["segmentation"][0]
        xs = coords[0::2]
        ys = coords[1::2]
        bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
        coco_by_split[img["_split"]]["annotations"].append(
            {
                "id": ann_id_counter,
                "image_id": img["_global_id"],
                "category_id": SPINE_CATEGORY_ID,
                "segmentation": ann["segmentation"],
                "bbox": bbox,
                "area": polygon_area(coords),
                "iscrowd": 0,
            }
        )
        ann_id_counter += 1

    for split in ("train", "val"):
        with (out_dir / f"pretrain_{split}.json").open("w", encoding="utf-8") as f:
            json.dump(coco_by_split[split], f)

    return coco_by_split


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=repo_root / "data" / "raw")
    parser.add_argument("--out-dir", type=Path, default=repo_root / "data" / "merged")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    args = parser.parse_args()

    raw_root = args.raw_root.resolve()
    out_dir = args.out_dir.resolve()

    kept_images, all_annotations, stats, dropped_empty_images = merge_all(
        raw_root, args.datasets, args.val_fraction, args.seed
    )

    # Leak check BEFORE writing anything — fail loudly, don't silently ship
    # a contaminated split.
    scene_to_splits: dict[str, set] = {}
    for img in kept_images:
        scene_to_splits.setdefault(img["_scene_id"], set()).add(img["_split"])
    leaked = {s: v for s, v in scene_to_splits.items() if len(v) > 1}
    if leaked:
        raise RuntimeError(f"Scene leakage across splits detected: {leaked}")

    coco_by_split = write_merged_dataset(kept_images, all_annotations, out_dir)

    print(f"Kept images: {len(kept_images)} (dropped {dropped_empty_images} with no valid annotation)")
    print(f"Kept annotations: {len(all_annotations)}")
    print(f"Dropped annotations by reason: {stats}")
    print(f"Unique scenes: {len(scene_to_splits)}")
    print(
        f"Train: {len(coco_by_split['train']['images'])} images, "
        f"{len(coco_by_split['train']['annotations'])} annotations"
    )
    print(
        f"Val:   {len(coco_by_split['val']['images'])} images, "
        f"{len(coco_by_split['val']['annotations'])} annotations"
    )
    print("Scene leakage check: PASSED (0 scenes span both splits)")
    print(f"Written to: {out_dir}")


if __name__ == "__main__":
    main()
