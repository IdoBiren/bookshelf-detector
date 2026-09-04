"""
Detected spine quad -> a horizontal image strip the VLM can read.

The detector's output is four free corners (plan §1), and a spine is usually
near-vertical and, shot at an angle, a perspective trapezoid. Feeding the
VLM the quad's bounding box would hand it ~64% neighbours' text — the exact
argument §1 makes for why the output is a quad in the first place. So the
crop is a perspective warp, then a rotation to put the text horizontal.

The corner convention is deliberately NOT re-invented here: it mirrors
`src/browser/quad.ts`'s `canonicalizeQuad`, where the two SHORT edges are
the spine's physical top and bottom and TL->TR spans one of them. The
browser draws quads with that convention and this module crops with it; if
they diverge, the boxes a user sees stop corresponding to the titles that
come back, and nothing raises.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from polygon_offset import Point

# Below this, warpPerspective's output would be empty or numerically
# meaningless. mask_to_quad can emit a degenerate quad from a bad mask, and a
# server has to skip it rather than fail the whole request.
MIN_SIDE_PX = 2.0


def canonicalize_quad(points: list[Point]) -> list[Point]:
    """Reorder four arbitrary corners into (TL, TR, BR, BL), where TL->TR is
    the spine's SHORT edge — a direct mirror of `canonicalizeQuad` in
    `src/browser/quad.ts`.

    Detections arrive from `mask_to_quad` in no guaranteed order, so this
    makes no assumption about the input's winding or starting corner."""
    centre_x = sum(p[0] for p in points) / 4
    centre_y = sum(p[1] for p in points) / 4

    # Sort by angle around the centroid -> clockwise cyclic order in image
    # coordinates (y-down), since increasing atan2 sweeps right->down->left->up.
    ordered = sorted(points, key=lambda p: math.atan2(p[1] - centre_y, p[0] - centre_x))

    def edge_length(i: int) -> float:
        a, b = ordered[i], ordered[(i + 1) % 4]
        return math.hypot(a[0] - b[0], a[1] - b[1])

    # Edges 0&2 are opposite each other, as are 1&3. The shorter pair is the
    # spine's top and bottom -- keyed off LENGTH, never off x/y extent, so a
    # book lying on its side still works.
    short_pair = (0, 2) if edge_length(0) + edge_length(2) < edge_length(1) + edge_length(3) else (1, 3)

    def mid_y(i: int) -> float:
        return (ordered[i][1] + ordered[(i + 1) % 4][1]) / 2

    top_edge = short_pair[0] if mid_y(short_pair[0]) <= mid_y(short_pair[1]) else short_pair[1]

    a = ordered[top_edge]
    b = ordered[(top_edge + 1) % 4]

    if a[0] <= b[0]:
        return [a, b, ordered[(top_edge + 2) % 4], ordered[(top_edge + 3) % 4]]
    # 'a' is actually the top-right corner: walk a's OTHER neighbour to reach
    # BR, keeping the result a simple clockwise polygon.
    return [b, a, ordered[(top_edge + 3) % 4], ordered[(top_edge + 2) % 4]]


def rectify_spine(
    image: np.ndarray,
    quad: list[Point],
    flip: bool = False,
    max_long_side: int | None = None,
) -> np.ndarray | None:
    """Warp one spine out of `image` and rotate it so its text runs
    horizontally. Returns None for a degenerate quad.

    `canonicalize_quad` leaves the spine VERTICAL (TL->TR is its short top
    edge), so the warp target is `short x long` and a 90 degree rotation
    follows. Counter-clockwise by default: a spine's title reads top-to-
    bottom when the book stands upright, so rotating that way puts it
    left-to-right.

    `flip=True` gives the other 180 degree orientation. The ambiguity is
    real — spine text runs both ways in practice, and nothing in the
    geometry says which — so this is a caller's choice, not something this
    function can decide. Measured on real shelves: **Hebrew titles need
    flip=True** (they read bottom-to-top), which is why `serve.py` defaults
    to it; English spines conventionally read top-to-bottom and want the
    default here.

    `max_long_side` downscales the result to fit, preserving aspect ratio
    and never upscaling. Measured need: a 4080x3072 shelf photo produced 24
    crops of roughly 2000x220, 18.6MB of PNG in one response — far more
    resolution than reading a title requires, and far too much to move over
    a tunnel from a laptop."""
    tl, tr, br, bl = canonicalize_quad(quad)

    short_side = (math.hypot(tr[0] - tl[0], tr[1] - tl[1]) + math.hypot(br[0] - bl[0], br[1] - bl[1])) / 2
    long_side = (math.hypot(br[0] - tr[0], br[1] - tr[1]) + math.hypot(bl[0] - tl[0], bl[1] - tl[1])) / 2

    if short_side < MIN_SIDE_PX or long_side < MIN_SIDE_PX:
        return None

    width, height = int(round(short_side)), int(round(long_side))

    if max_long_side is not None and height > max_long_side:
        # Scale before warping, not after: warping straight to the final
        # size is one interpolation instead of two, and cheaper.
        scale = max_long_side / height
        width = max(int(round(width * scale)), 1)
        height = max_long_side

    source = np.array([tl, tr, br, bl], dtype=np.float32)
    target = np.array(
        [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)],
        dtype=np.float32,
    )

    upright = cv2.warpPerspective(image, cv2.getPerspectiveTransform(source, target), (width, height))

    rotation = cv2.ROTATE_90_CLOCKWISE if flip else cv2.ROTATE_90_COUNTERCLOCKWISE
    return cv2.rotate(upright, rotation)
