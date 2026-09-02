"""
Geometric evaluation, plan §8א: mAP@50, mAP@50:95, and mAP broken down by
spine-width tercile.

Two deliberate choices from §8א, both load-bearing:

**IoU is measured on the quad, not on an axis-aligned box.** §1's whole
argument is that a 200x30px spine tilted 15 degrees has an AABB that is ~64%
neighbours' text — so AABB IoU scores a detector well exactly where it is
failing worst. IoU here is computed by rasterizing both shapes, which is
both simple and literally the mask IoU §8א asks for.

**The width-tercile breakdown is not optional.** A healthy overall mAP can
hide complete failure on thin pocket paperbacks (trap #4), and thin spines
are the case the whole quad-not-box argument is about.

Scope: this is §8א, the SECONDARY measurement. §8ב (read accuracy through
the VLM, against the existing cloud model) is what actually decides
shipping, and it needs the hand-typed eval-set ground truth. §8א on
pretrain_val is what answers the cheaper, earlier question this phase
exists for: is the task learnable at all?
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from dbnet_targets import rasterize_polygon
from polygon_offset import Point

# COCO's IoU sweep, and its 101-point interpolated AP.
IOU_THRESHOLDS = [round(0.5 + 0.05 * i, 2) for i in range(10)]
RECALL_POINTS = np.linspace(0.0, 1.0, 101)


def _raster_canvas(*quads: list[Point]) -> tuple[int, int, float, float]:
    xs = [p[0] for quad in quads for p in quad]
    ys = [p[1] for quad in quads for p in quad]
    min_x, min_y = min(xs), min(ys)
    width = int(np.ceil(max(xs) - min_x)) + 2
    height = int(np.ceil(max(ys) - min_y)) + 2
    return width, height, min_x, min_y


def quad_iou(quad_a: list[Point], quad_b: list[Point]) -> float:
    """Rasterized intersection-over-union of two polygons.

    Rasterizing rather than doing exact polygon clipping is deliberate: it
    is the mask IoU §8א specifies, it needs no extra dependency, and at the
    resolutions involved the discretization error is far below the
    threshold spacing (0.05) that any of it feeds into.
    """
    width, height, min_x, min_y = _raster_canvas(quad_a, quad_b)
    shifted_a = [(x - min_x, y - min_y) for x, y in quad_a]
    shifted_b = [(x - min_x, y - min_y) for x, y in quad_b]

    mask_a = rasterize_polygon(shifted_a, width, height).astype(bool)
    mask_b = rasterize_polygon(shifted_b, width, height).astype(bool)

    union = int((mask_a | mask_b).sum())
    if union == 0:
        return 0.0
    return int((mask_a & mask_b).sum()) / union


def match_predictions(
    predictions: list[tuple[list[Point], float]],
    ground_truth: list[list[Point]],
    iou_threshold: float,
) -> tuple[list[bool], int]:
    """Greedy score-ranked matching, COCO-style.

    Returns (is_true_positive per prediction IN SCORE ORDER, matched GT
    count). Each ground-truth object can be claimed only once — without
    that, a model that fires ten overlapping boxes per spine would score
    perfectly, and merge/split behaviour would be invisible.
    """
    ranked = sorted(predictions, key=lambda pair: pair[1], reverse=True)
    claimed = [False] * len(ground_truth)
    flags: list[bool] = []

    for quad, _score in ranked:
        best_iou, best_index = 0.0, -1
        for index, gt_quad in enumerate(ground_truth):
            if claimed[index]:
                continue
            iou = quad_iou(quad, gt_quad)
            if iou > best_iou:
                best_iou, best_index = iou, index

        if best_index >= 0 and best_iou >= iou_threshold:
            claimed[best_index] = True
            flags.append(True)
        else:
            flags.append(False)

    return flags, sum(claimed)


def average_precision(flags: list[bool], scores: list[float], num_ground_truth: int) -> float:
    """101-point interpolated AP over the precision-recall curve (COCO's
    definition), from per-prediction TP/FP flags."""
    if num_ground_truth == 0 or not flags:
        return 0.0

    order = np.argsort(-np.asarray(scores, dtype=np.float64))
    ordered = np.asarray(flags, dtype=bool)[order]

    true_positives = np.cumsum(ordered)
    false_positives = np.cumsum(~ordered)
    recalls = true_positives / num_ground_truth
    precisions = true_positives / np.maximum(true_positives + false_positives, 1e-9)

    # Make precision monotonically decreasing (standard: at each recall the
    # best precision achievable at that recall or beyond).
    precisions = np.maximum.accumulate(precisions[::-1])[::-1]

    interpolated = np.zeros_like(RECALL_POINTS)
    for i, recall_point in enumerate(RECALL_POINTS):
        candidates = precisions[recalls >= recall_point]
        interpolated[i] = candidates.max() if candidates.size else 0.0
    return float(interpolated.mean())


def spine_width(quad: list[Point]) -> float:
    """The SHORT side of a spine quad — the dimension trap #4 is about.
    Measured from the quad's own edges, so a rotated spine reports its true
    width rather than its bounding box's."""
    n = len(quad)
    edges = [
        float(np.hypot(quad[(i + 1) % n][0] - quad[i][0], quad[(i + 1) % n][1] - quad[i][1]))
        for i in range(n)
    ]
    edges.sort()
    # Opposite sides of a quad pair up; the two shortest are the width pair.
    return (edges[0] + edges[1]) / 2


def tercile_edges(widths: list[float]) -> tuple[float, float]:
    """The two cut points that split spine widths into thin/medium/wide."""
    if not widths:
        return 0.0, 0.0
    return float(np.percentile(widths, 100 / 3)), float(np.percentile(widths, 200 / 3))


def evaluate(
    per_image: list[tuple[list[tuple[list[Point], float]], list[list[Point]]]],
) -> dict:
    """`per_image` is [(predictions, ground_truth), ...] where predictions
    are (quad, score) pairs. Returns mAP@50, mAP@50:95, and the per-tercile
    breakdown."""
    all_widths = [spine_width(gt) for _, gts in per_image for gt in gts]
    low_edge, high_edge = tercile_edges(all_widths)

    def bucket(quad: list[Point]) -> str:
        width = spine_width(quad)
        if width <= low_edge:
            return "thin"
        return "medium" if width <= high_edge else "wide"

    results: dict = {"by_iou": {}, "by_width": {}}

    ap_per_threshold = []
    for threshold in IOU_THRESHOLDS:
        flags, scores, total_gt = [], [], 0
        for predictions, ground_truth in per_image:
            image_flags, _ = match_predictions(predictions, ground_truth, threshold)
            ranked_scores = sorted((s for _, s in predictions), reverse=True)
            flags.extend(image_flags)
            scores.extend(ranked_scores)
            total_gt += len(ground_truth)
        ap = average_precision(flags, scores, total_gt)
        ap_per_threshold.append(ap)
        results["by_iou"][f"AP@{threshold:.2f}"] = ap

    results["mAP@50"] = ap_per_threshold[0]
    results["mAP@50:95"] = float(np.mean(ap_per_threshold))

    # Per-tercile AP@50: restrict ground truth to one width band at a time.
    #
    # Detections that match an OUT-OF-BAND ground-truth object are IGNORED,
    # not counted as false positives -- COCO's semantics for its own
    # size-based AP. Without this, correctly detecting a wide spine would be
    # punished when scoring the thin band, and a perfect predictor scores
    # ~0.43 per band instead of 1.0. There is a test pinning exactly that.
    for band in ("thin", "medium", "wide"):
        flags, scores, total_gt = [], [], 0
        for predictions, ground_truth in per_image:
            band_gt = [gt for gt in ground_truth if bucket(gt) == band]
            other_gt = [gt for gt in ground_truth if bucket(gt) != band]
            if not band_gt:
                continue

            ranked = sorted(predictions, key=lambda pair: pair[1], reverse=True)
            image_flags, _ = match_predictions(ranked, band_gt, IOU_THRESHOLDS[0])

            for (quad, score), is_true_positive in zip(ranked, image_flags):
                if not is_true_positive and other_gt:
                    if max(quad_iou(quad, gt) for gt in other_gt) >= IOU_THRESHOLDS[0]:
                        continue  # belongs to another band -- neither TP nor FP here
                flags.append(is_true_positive)
                scores.append(score)
            total_gt += len(band_gt)

        results["by_width"][band] = {
            "AP@50": average_precision(flags, scores, total_gt),
            "ground_truth": total_gt,
        }

    results["width_tercile_edges"] = {"thin_max": low_edge, "medium_max": high_edge}
    return results


def evaluate_checkpoint(
    checkpoint: str,
    coco_path: str,
    images_dir: str,
    score_threshold: float = 0.5,
    limit: int | None = None,
) -> dict:
    """Loads a trained checkpoint, runs it over a COCO split, and returns
    §8א's numbers. Ground truth quads come from the SAME mask_to_quad path
    the predictions do, so the comparison isn't confounded by two different
    polygon-fitting routes.

    torch is imported lazily here on purpose: every metric function above is
    torch-free and works on plain quads, so all of §8א stays unit-testable
    without an ML stack. Only this function needs the model."""
    try:
        import torch
    except ModuleNotFoundError as error:  # pragma: no cover - environment guard
        # Same trap as train.py: torch is in training/.venv, not in the
        # system Python that also runs Label Studio.
        raise SystemExit(
            f"{error}\n\n"
            "torch is installed in training/.venv, not in this interpreter\n"
            f"  (currently running: {sys.executable})\n\n"
            "Use the venv's Python instead:\n"
            "  training/.venv/Scripts/python.exe training/evaluate.py --checkpoint <path>\n\n"
            "In Colab this does not apply -- plain `python` is correct there."
        ) from error

    from dataset import SpineDataset
    from mask_to_quad import mask_to_quad
    from model import build_model
    from train import load_checkpoint

    model = build_model(pretrained=False)
    load_checkpoint(Path(checkpoint), model)
    model.eval()

    dataset = SpineDataset(Path(coco_path), Path(images_dir))
    if limit:
        dataset.entries = dataset.entries[:limit]

    per_image = []
    for index in range(len(dataset)):
        image, target = dataset[index]
        with torch.no_grad():
            output = model([image])[0]

        keep = output["scores"] >= score_threshold
        predictions = []
        for mask, score in zip(output["masks"][keep].numpy(), output["scores"][keep].tolist()):
            quad = mask_to_quad(mask[0])
            if quad is not None:
                predictions.append((quad, score))

        ground_truth = [
            quad
            for quad in (mask_to_quad(m.numpy()) for m in target["masks"])
            if quad is not None
        ]
        per_image.append((predictions, ground_truth))

    return evaluate(per_image)


def main() -> None:
    import argparse

    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--coco", default=str(repo_root / "data" / "merged" / "pretrain_val.json"))
    parser.add_argument("--images-dir", default=str(repo_root / "data" / "merged" / "images"))
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    results = evaluate_checkpoint(
        args.checkpoint, args.coco, args.images_dir, args.score_threshold, args.limit
    )

    print("=== §8א geometric evaluation (quad IoU, not AABB) ===")
    print(f"mAP@50    : {results['mAP@50']:.4f}")
    print(f"mAP@50:95 : {results['mAP@50:95']:.4f}")
    print("\nby spine-width tercile (trap #4 -- a good overall mAP can hide this):")
    edges = results["width_tercile_edges"]
    print(f"  thin   (<= {edges['thin_max']:.1f}px) AP@50={results['by_width']['thin']['AP@50']:.4f}"
          f"  n={results['by_width']['thin']['ground_truth']}")
    print(f"  medium (<= {edges['medium_max']:.1f}px) AP@50={results['by_width']['medium']['AP@50']:.4f}"
          f"  n={results['by_width']['medium']['ground_truth']}")
    print(f"  wide            AP@50={results['by_width']['wide']['AP@50']:.4f}"
          f"  n={results['by_width']['wide']['ground_truth']}")

    if args.report:
        import json

        Path(args.report).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nreport -> {args.report}")


if __name__ == "__main__":
    main()
