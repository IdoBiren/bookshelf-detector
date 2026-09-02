"""
Mask R-CNN per-instance mask -> quad (plan §1: the output is four free
corners, not a box and not an OBB).

This is the training-side counterpart to what `src/browser/postprocess.ts`
does, minus the two steps Mask R-CNN makes unnecessary:

    DBNet:       probMap -> threshold -> connected components -> hull
                         -> minAreaQuad -> unclip -> quad
    Mask R-CNN:  mask    -> threshold -> contour              -> hull
                         -> minAreaQuad                       -> quad

`connected components` is gone because each instance already arrives as its
own mask — "books touch and a binary mask glues them into one blob" (plan §1)
simply cannot happen. `unclip` is gone with it: unclip exists only to undo
DBNet's shrink, and there is no shrink here. That also removes the failure
phase B measured — uniform per-edge shrink self-intersecting on tapering
quads (see HANDOFF.md "Open questions" §1).

The hull -> minAreaQuad half is deliberately the SAME code the browser runs
(`polygon_offset.py`, itself a tested mirror of convexHull.ts/minAreaQuad.ts),
so a quad fitted here during evaluation and a quad fitted in the browser
agree rather than drifting.
"""

from __future__ import annotations

import cv2
import numpy as np

from polygon_offset import Point, convex_hull, min_area_quad, min_area_rect

DEFAULT_MASK_THRESHOLD = 0.5
DEFAULT_MIN_AREA_PX = 16  # matches postprocess.ts's DEFAULT_OPTIONS.minAreaPx


def mask_to_quad(
    mask: np.ndarray,
    threshold: float = DEFAULT_MASK_THRESHOLD,
    min_area_px: int = DEFAULT_MIN_AREA_PX,
) -> list[Point] | None:
    """One instance mask -> its 4 corners, or None if the mask is empty,
    too small, or too degenerate to have a meaningful quad.

    Accepts either a binary mask or Mask R-CNN's raw float probabilities
    (which is what it actually emits) — `threshold` is what turns the
    latter into the former.

    Returns None rather than raising: a detection whose mask is junk should
    be dropped by the caller, exactly as postprocess.ts skips a component
    below `minAreaPx` or a hull with fewer than 3 points.
    """
    binary = (np.asarray(mask) >= threshold).astype(np.uint8)
    if binary.ndim != 2:
        raise ValueError(f"expected a 2D mask, got shape {binary.shape}")
    if int(binary.sum()) < min_area_px:
        return None

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Largest contour only. A Mask R-CNN mask can have speckle outside the
    # main blob; the instance is the biggest connected piece, and taking all
    # contours would let a stray pixel drag a corner far out of place.
    largest = max(contours, key=cv2.contourArea)
    points: list[Point] = [(float(x), float(y)) for x, y in largest.reshape(-1, 2)]

    hull = convex_hull(points)
    if len(hull) < 3:
        return None  # a line or a point, not a shape

    # Same fallback chain as postprocess.ts:64.
    return min_area_quad(hull) or min_area_rect(hull)


def masks_to_quads(
    masks: np.ndarray,
    threshold: float = DEFAULT_MASK_THRESHOLD,
    min_area_px: int = DEFAULT_MIN_AREA_PX,
) -> list[list[Point]]:
    """A batch of instance masks -> one quad each, dropping the ones that
    fail. Accepts (N, H, W) or torchvision's actual (N, 1, H, W) output
    shape — the extra channel dim is squeezed rather than being a silent
    source of wrong results.
    """
    array = np.asarray(masks)
    if array.ndim == 4 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 3:
        raise ValueError(f"expected (N,H,W) or (N,1,H,W) masks, got shape {array.shape}")

    quads = []
    for mask in array:
        quad = mask_to_quad(mask, threshold, min_area_px)
        if quad is not None:
            quads.append(quad)
    return quads
