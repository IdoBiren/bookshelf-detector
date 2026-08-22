"""
Applies the real augmentation pipeline (augment.py) to a sample of merged
training images and draws the AUGMENTED polygons on the AUGMENTED image
(plan §9, Stage 2 deliverable: "20 augmented images with labels drawn on
top, checked by eye"). This is what catches a polygon-transform bug that
the unit tests' synthetic cases might not — real polygons, real images,
the exact pipeline that will run at training time.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from augment import (
    build_augmentation_pipeline,
    flatten_polygons_to_keypoints,
    regroup_keypoints_to_polygons,
)

OUTLINE_COLOR = (255, 0, 0)
OUTLINE_WIDTH = 3


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-dir", type=Path, default=repo_root / "data" / "merged")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    merged_dir = args.merged_dir.resolve()
    out_dir = (args.out_dir or (merged_dir / "preview_augmented")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    coco_path = merged_dir / f"pretrain_{args.split}.json"
    with coco_path.open(encoding="utf-8") as f:
        coco = json.load(f)

    anns_by_image: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    # Only sample images that actually have annotations (a merged image is
    # always guaranteed at least one, per merge_datasets.py's own
    # dropped_empty_images filtering — this guard is just defensive).
    images = [img for img in coco["images"] if anns_by_image.get(img["id"])]

    rng = random.Random(args.seed)
    sample = rng.sample(images, min(args.count, len(images)))
    pipeline = build_augmentation_pipeline(seed=args.seed)

    mismatch_count = 0
    for img_meta in sample:
        annotations = anns_by_image[img_meta["id"]]
        keypoints, vertex_counts = flatten_polygons_to_keypoints(annotations)

        src_path = merged_dir / "images" / img_meta["file_name"]
        image = cv2.imread(str(src_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        result = pipeline(image=image, keypoints=keypoints)

        if len(result["keypoints"]) != len(keypoints):
            # Should be impossible with remove_invisible=False — if this
            # ever fires, it means a point got silently dropped, exactly
            # the class of bug this whole script exists to catch.
            mismatch_count += 1
            print(f"  MISMATCH on {img_meta['file_name']}: "
                  f"{len(keypoints)} in, {len(result['keypoints'])} out")
            continue

        polygons = regroup_keypoints_to_polygons(result["keypoints"], vertex_counts)

        out_img = Image.fromarray(result["image"])
        draw = ImageDraw.Draw(out_img)
        for poly in polygons:
            draw.polygon(poly, outline=OUTLINE_COLOR, width=OUTLINE_WIDTH)

        out_path = out_dir / img_meta["file_name"]
        out_img.save(out_path)

    print(f"Wrote {len(sample) - mismatch_count} augmented preview images to {out_dir}")
    if mismatch_count:
        print(f"WARNING: {mismatch_count} images had a keypoint-count mismatch — investigate.")
    print("Check by eye: polygons should still hug spine edges after rotation/perspective/crop.")


if __name__ == "__main__":
    main()
