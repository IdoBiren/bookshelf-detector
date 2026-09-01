"""
Polygon preparation, rasterization, and the two shrink-ratio validation
metrics for DBNet targets (plan §13 phase B).

Scope: this covers what the go/no-go measurement in measure_shrink_ratio.py
needs. The full DBNet training-loss target maps (threshold map, binary map)
are a training-time concern deferred past this phase — the question this
phase answers is narrower: is shrink_ratio=0.4 valid for book spines, and
at what target-map resolution?

Depends on numpy + opencv (both already arrive transitively via
albumentations; pinned explicitly in requirements.txt now that this module
uses them directly). merge_datasets.py stays stdlib-only — this module is
NOT imported by it and never will be.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from polygon_offset import (
    Point,
    convex_hull,
    is_simple_polygon,
    min_area_quad,
    min_area_rect,
    polygon_area,
    shrink_polygon,
)

TARGET_SIZE = 640
MIN_SHRUNK_PIXELS = 20


@dataclass(frozen=True)
class LetterboxInfo:
    scale: float
    pad_x: float
    pad_y: float
    target_size: int


def compute_letterbox(orig_width: float, orig_height: float, target_size: int) -> LetterboxInfo:
    """Exact mirror of src/browser/letterbox.ts's computeLetterbox. A
    sign/pad error here silently offsets every target and is invisible in
    any aggregate metric — hence the dedicated tests against its own
    numbers rather than trusting "it looks about right"."""
    if orig_width <= 0 or orig_height <= 0:
        raise ValueError(f"Invalid image dimensions: {orig_width}x{orig_height}")
    scale = min(target_size / orig_width, target_size / orig_height)
    scaled_width = orig_width * scale
    scaled_height = orig_height * scale
    pad_x = (target_size - scaled_width) / 2
    pad_y = (target_size - scaled_height) / 2
    return LetterboxInfo(scale=scale, pad_x=pad_x, pad_y=pad_y, target_size=target_size)


def letterbox_polygon(points: list[Point], info: LetterboxInfo, stride: int = 1) -> list[Point]:
    """Original-image-space polygon -> target-map space (letterboxed to
    `info.target_size`, then divided by `stride` for the DBNet head's
    output resolution). The inverse of letterbox.ts's unletterboxPoint."""
    return [
        ((x * info.scale + info.pad_x) / stride, (y * info.scale + info.pad_y) / stride)
        for x, y in points
    ]


def prepare_polygon(points: list[Point], mode: str = "quad") -> list[Point]:
    """Reduces a raw annotated polygon to the shape used for shrink-target
    generation. Three modes, each independently measurable so the cost of
    each stage is a number rather than an assumption:

    - "raw":  as annotated. UNSAFE for ~22.6% of pretrain polygons, whose
      reflex (non-convex) vertices flip offset_polygon's centroid-based
      normal sign (see polygon_offset.py's module docstring).
    - "hull": convex_hull(points). Safe signs; can lose a small sliver of
      area and, on many vertices, degrades exact_unclip_distance's inverse
      (short edges collapse when offset inward).
    - "quad": min_area_quad(hull), falling back to min_area_rect when the
      hull has fewer than 4 vertices or reduction gets stuck. RECOMMENDED
      and the default — it is also what postprocess.ts:64 does before
      unclipping, so training targets match what inference can express,
      and it is the mode that makes the shrink/unclip inverse exact.
    """
    if mode == "raw":
        return points
    hull = convex_hull(points)
    if mode == "hull":
        return hull
    if mode == "quad":
        if len(hull) < 3:
            # Degenerate input (a point or a zero-area line) — nothing
            # sensible to reduce to; the caller's area/shrink checks handle
            # this as a vanished polygon rather than this function raising.
            return hull
        quad = min_area_quad(hull)
        return quad if quad is not None else min_area_rect(hull)
    raise ValueError(f"Unknown mode {mode!r}: expected 'raw', 'hull', or 'quad'")


def rasterize_polygon(polygon: list[Point], width: int, height: int) -> np.ndarray:
    """Boolean-as-uint8 mask, via cv2.fillPoly. Deliberately kept SEPARATE
    from the analytic (polygon_area-based) failure checks: a polygon can be
    analytically non-vanishing and still rasterize to zero pixels (integer
    truncation), or vice versa near a boundary — collapsing the two into
    one number would hide which failure mode is actually occurring."""
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(polygon) < 3:
        return mask
    int_points = np.round(np.array(polygon, dtype=np.float64)).astype(np.int32)
    cv2.fillPoly(mask, [int_points], 1)
    return mask


_ADJACENCY_KERNEL = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)  # 4-connected cross


def _touches(mask_a: np.ndarray, mask_b: np.ndarray) -> bool:
    """4-connectivity adjacency test: dilate one mask by a single cross-
    shaped step and check overlap with the other. The cross (not a 3x3
    square) matters — connectedComponents.ts is 4-connected, so a
    diagonal-only touch does NOT fuse two spines in the browser and must
    not be counted as adjacent here either."""
    dilated = cv2.dilate(mask_a, _ADJACENCY_KERNEL, iterations=1)
    return bool(np.any(dilated & mask_b))


def merged_adjacent_pairs_metric(
    polygons_per_image: list[list[list[Point]]],
    shrink_ratio: float,
    canvas_size: int,
    mode: str = "quad",
) -> dict:
    """Metric 1 (plan §13, the decisive one): of every pair of polygons
    that touch/overlap in their ORIGINAL form, what fraction still touch
    after shrinking? >5% => shrink_ratio is wrong for spines at this
    canvas_size (which stands in for target-map resolution — pass
    TARGET_SIZE/stride as canvas_size for a given stride).

    `polygons_per_image` is already in target-map coordinate space (i.e.
    already letterboxed and stride-divided by the caller) — this function
    is deliberately coordinate-space-agnostic so it is testable with plain
    pixel rectangles.
    """
    adjacent_pairs = 0
    merged_pairs = 0
    per_image: list[dict] = []

    for polygons in polygons_per_image:
        prepared = [prepare_polygon(p, mode) for p in polygons]
        original_masks = [rasterize_polygon(p, canvas_size, canvas_size) for p in prepared]
        shrunk_masks = [
            rasterize_polygon(shrink_polygon(p, shrink_ratio), canvas_size, canvas_size)
            for p in prepared
        ]

        image_adjacent = 0
        image_merged = 0
        n = len(prepared)
        for i in range(n):
            for j in range(i + 1, n):
                if not _touches(original_masks[i], original_masks[j]):
                    continue
                image_adjacent += 1
                if _touches(shrunk_masks[i], shrunk_masks[j]):
                    image_merged += 1

        adjacent_pairs += image_adjacent
        merged_pairs += image_merged
        if image_adjacent:
            per_image.append({"adjacent": image_adjacent, "merged": image_merged})

    return {
        "adjacent_pairs": adjacent_pairs,
        "merged_pairs": merged_pairs,
        "merged_fraction": merged_pairs / adjacent_pairs if adjacent_pairs else 0.0,
        "per_image": per_image,
    }


def vanishing_spines_metric(
    polygons_per_image: list[list[list[Point]]],
    shrink_ratio: float,
    canvas_size: int,
    mode: str = "quad",
) -> dict:
    """Metric 2 (plan §13): fraction of polygons that vanish after
    shrinking, checked BOTH analytically (area<=0 or self-intersecting)
    and by rasterization (<MIN_SHRUNK_PIXELS px) — kept as separate
    counters because they are different failure modes with different
    fixes (geometry vs. resolution)."""
    total = 0
    vanished = 0
    zero_area_or_self_intersecting = 0
    zero_pixels = 0
    tiny_pixels = 0

    for polygons in polygons_per_image:
        for raw in polygons:
            total += 1
            prepared = prepare_polygon(raw, mode)
            shrunk = shrink_polygon(prepared, shrink_ratio)

            analytic_bad = polygon_area(shrunk) <= 0 or not is_simple_polygon(shrunk)
            if analytic_bad:
                zero_area_or_self_intersecting += 1
                vanished += 1
                continue

            pixel_count = int(rasterize_polygon(shrunk, canvas_size, canvas_size).sum())
            if pixel_count == 0:
                zero_pixels += 1
                vanished += 1
            elif pixel_count < MIN_SHRUNK_PIXELS:
                tiny_pixels += 1
                vanished += 1

    return {
        "total": total,
        "vanished": vanished,
        "vanished_fraction": vanished / total if total else 0.0,
        "zero_area_or_self_intersecting": zero_area_or_self_intersecting,
        "zero_pixels": zero_pixels,
        "tiny_pixels": tiny_pixels,
    }
