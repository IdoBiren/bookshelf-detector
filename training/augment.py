"""
Augmentation pipeline for spine training (plan §3). Only the transforms
marked "OK" in the plan's table are here — no vertical flip, no 90/180
rotation, no mosaic/cutmix: those destroy the "spine is roughly vertical"
prior that a bounding-shape model relies on.

Ranges here are STARTING values, not calibrated ones — same caveat the
plan gives its own shrink/unclip hyperparameters (§4/§10): real tuning
happens once training is actually running, not at this stage.

Polygons vary in vertex count per annotation (real data: 5-9+ points, not
a fixed 4), so they're carried through Albumentations as one flat
keypoints list per image plus a per-annotation point-count, then
regrouped after the transform — NOT as a bbox, which would silently
throw away the shape information the whole quad/OBB argument (plan §1)
depends on.
"""

from __future__ import annotations

import cv2
import albumentations as A

TARGET_SIZE = 640


def build_augmentation_pipeline(seed: int | None = None) -> A.Compose:
    return A.Compose(
        [
            # Geometric — the ones that matter for domain gap #1 (angled photos).
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=12, border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.8),
            A.Perspective(scale=(0.02, 0.08), border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.7),
            # Bidirectional scale (0.5x-1.6x, plan §3 trap #4: thin pocket
            # books next to wide albums) via RandomScale (resizes the whole
            # image, can go either direction) — RandomResizedCrop alone
            # can only zoom IN, never out, since it can't crop beyond the
            # original frame.
            A.RandomScale(scale_limit=(-0.5, 0.6), p=0.7),
            # Fixed-size crop after the scale jump. pad_if_needed handles
            # the case where RandomScale shrank the image below 640px.
            # This is also what deliberately produces "spine cut at the
            # frame edge" (plan §3's labeling rule exists for this).
            A.RandomCrop(
                height=TARGET_SIZE,
                width=TARGET_SIZE,
                pad_if_needed=True,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                p=1.0,
            ),
            # Photometric — aggressive on purpose, including a warm shift
            # to simulate tungsten home lighting the public data lacks.
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=35, val_shift_limit=20, p=0.7),
            A.RandomGamma(p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.GaussNoise(p=0.2),
            A.ImageCompression(quality_range=(40, 90), p=0.3),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        seed=seed,
    )


def flatten_polygons_to_keypoints(
    annotations: list[dict],
) -> tuple[list[tuple[float, float]], list[int]]:
    """Flattens every annotation's polygon into one keypoints list, plus
    the vertex count per annotation needed to regroup them afterward."""
    keypoints: list[tuple[float, float]] = []
    vertex_counts: list[int] = []
    for ann in annotations:
        coords = ann["segmentation"][0]
        pts = list(zip(coords[0::2], coords[1::2]))
        keypoints.extend(pts)
        vertex_counts.append(len(pts))
    return keypoints, vertex_counts


def regroup_keypoints_to_polygons(
    keypoints: list[tuple[float, float]], vertex_counts: list[int]
) -> list[list[tuple[float, float]]]:
    polygons: list[list[tuple[float, float]]] = []
    i = 0
    for n in vertex_counts:
        polygons.append(list(keypoints[i : i + n]))
        i += n
    return polygons
