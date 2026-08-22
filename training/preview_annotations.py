"""
Draws segmentation polygons over a sample of merged images and saves them
to disk for a manual eye-check (plan §9, Stage 2 deliverable). This is the
check that catches merge-time coordinate bugs specifically — a polygon
that shifted, scaled wrong, or got attached to the wrong image would be
obvious by eye but invisible to the unit tests, which only exercise
synthetic fixtures.

Deliberately separate from data augmentation (Albumentations, plan §3):
that's a later, training-time concern with its own dependency footprint.
This script only needs Pillow, which — unlike PyTorch/onnxruntime — is
confirmed to install cleanly on this environment's Python 3.14.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

OUTLINE_COLOR = (255, 0, 0)
OUTLINE_WIDTH = 3


def draw_annotations(image_path: Path, annotations: list[dict], out_path: Path) -> None:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for ann in annotations:
        coords = ann["segmentation"][0]
        points = list(zip(coords[0::2], coords[1::2]))
        draw.polygon(points, outline=OUTLINE_COLOR, width=OUTLINE_WIDTH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


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
    out_dir = (args.out_dir or (merged_dir / "preview")).resolve()

    coco_path = merged_dir / f"pretrain_{args.split}.json"
    with coco_path.open(encoding="utf-8") as f:
        coco = json.load(f)

    anns_by_image: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    images = coco["images"]
    rng = random.Random(args.seed)
    sample = rng.sample(images, min(args.count, len(images)))

    for img in sample:
        src = merged_dir / "images" / img["file_name"]
        dst = out_dir / img["file_name"]
        draw_annotations(src, anns_by_image.get(img["id"], []), dst)

    print(f"Wrote {len(sample)} preview images to {out_dir}")
    print("Check by eye: polygons should hug spine edges, not be offset/scaled/on the wrong book.")


if __name__ == "__main__":
    main()
