"""
COCO -> torchvision Mask R-CNN (image, target) pairs.

Reads the merged COCO files merge_datasets.py / convert_labelstudio_export.py
already produce (`data/merged/pretrain_{train,val}.json`,
`indomain_{train,val,test}.json`), so the whole scene-aware splitting and
the hand-reviewed grouping carry over untouched.

Two conversions here are silent-failure territory and both have tests:
COCO stores bbox as [x, y, w, h] while torchvision wants [x1, y1, x2, y2],
and every polygon has to become its own filled mask — an all-zero mask
trains happily and teaches nothing.

Augmentation (`augment.py`'s pipeline, carrying polygons through as
keypoints) is wired in behind `augment=False` by default, so every existing
call site is unaffected unless it opts in. When it runs, masks are
RE-RASTERIZED from the transformed polygons rather than warping the mask
images directly -- the same reasoning as the unaugmented path: a polygon is
the source of truth, and the mask is derived from it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from augment import build_augmentation_pipeline, flatten_polygons_to_keypoints, regroup_keypoints_to_polygons

# rasterize_polygon lives in dbnet_targets.py but is generic raster geometry,
# not DBNet-specific -- reused rather than duplicated, on the same reasoning
# that write_merged_dataset is shared between the two dataset builders.
from dbnet_targets import rasterize_polygon
from model import SPINE_LABEL

MIN_MASK_PIXELS = 4


class SpineDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        coco_path: Path,
        images_dir: Path,
        augment: bool = False,
        augment_seed: int | None = None,
    ):
        self.images_dir = Path(images_dir)
        self.stats: dict[str, int] = {"missing_image_file": 0, "degenerate_annotation": 0}
        # One pipeline instance per dataset, not per __getitem__ call: an
        # Albumentations Compose seeded once at construction gives a
        # reproducible SEQUENCE of augmentations across an epoch, not an
        # identical result on every call -- which is what "same seed,
        # reproducible run" is supposed to mean. None when augment=False, so
        # the unaugmented path below never touches augment.py at all.
        self._augment_pipeline = build_augmentation_pipeline(seed=augment_seed) if augment else None

        with Path(coco_path).open(encoding="utf-8") as f:
            coco = json.load(f)

        polygons_by_image: dict[int, list[list[float]]] = {}
        for ann in coco["annotations"]:
            segmentation = ann.get("segmentation")
            if segmentation and segmentation[0]:
                polygons_by_image.setdefault(ann["image_id"], []).append(segmentation[0])

        self.entries: list[dict] = []
        for image in coco["images"]:
            polygons = polygons_by_image.get(image["id"])
            if not polygons:
                continue  # torchvision raises on a target with zero boxes
            if not (self.images_dir / image["file_name"]).exists():
                self.stats["missing_image_file"] += 1
                continue
            self.entries.append(
                {
                    "file_name": image["file_name"],
                    "width": image["width"],
                    "height": image["height"],
                    "polygons": polygons,
                }
            )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int):
        entry = self.entries[index]
        path = self.images_dir / entry["file_name"]

        with Image.open(path) as pil_image:
            array = np.asarray(pil_image.convert("RGB"))  # uint8 HWC

        polygons = entry["polygons"]

        if self._augment_pipeline is not None:
            # Flatten every polygon into one keypoints list (variable vertex
            # counts per annotation -- see augment.py), transform, regroup.
            # Any resulting polygon (e.g. cropped fully out of frame) that
            # rasterizes to nothing is caught by the SAME degenerate-mask
            # check below that already handles the unaugmented path -- no
            # separate "did the crop miss this spine" logic needed.
            annotations = [{"segmentation": [coords]} for coords in polygons]
            keypoints, vertex_counts = flatten_polygons_to_keypoints(annotations)
            transformed = self._augment_pipeline(image=array, keypoints=keypoints)
            array = transformed["image"]
            regrouped = regroup_keypoints_to_polygons(transformed["keypoints"], vertex_counts)
            polygons = [[coordinate for point in polygon for coordinate in point] for polygon in regrouped]

        image = torch.from_numpy(array.astype(np.float32) / 255.0).permute(2, 0, 1)  # HWC -> CHW

        height, width = image.shape[1], image.shape[2]

        masks, boxes = [], []
        for coords in polygons:
            points = list(zip(coords[0::2], coords[1::2]))
            mask = rasterize_polygon(points, width, height)
            if int(mask.sum()) < MIN_MASK_PIXELS:
                self.stats["degenerate_annotation"] += 1
                continue

            ys, xs = np.nonzero(mask)
            # Box derived from the RASTERIZED mask, not from COCO's stored
            # bbox: the two can disagree after any coordinate change, and a
            # box that doesn't contain its mask is a subtle training bug.
            x1, x2 = float(xs.min()), float(xs.max())
            y1, y2 = float(ys.min()), float(ys.max())
            if x2 <= x1 or y2 <= y1:
                self.stats["degenerate_annotation"] += 1
                continue

            masks.append(mask)
            boxes.append([x1, y1, x2, y2])

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.full((len(boxes),), SPINE_LABEL, dtype=torch.int64),
            "masks": torch.as_tensor(np.array(masks), dtype=torch.uint8)
            if masks
            else torch.zeros((0, height, width), dtype=torch.uint8),
            "image_id": torch.tensor([index]),
        }
        return image, target


def collate_fn(batch):
    """Detection models take a LIST of variable-sized images, so the default
    collate (which stacks into one tensor) cannot be used."""
    images, targets = zip(*batch)
    return list(images), list(targets)
