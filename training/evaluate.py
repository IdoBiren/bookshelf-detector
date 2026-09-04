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


def iou_matrix(
    predictions: list[tuple[list[Point], float]], ground_truth: list[list[Point]]
) -> np.ndarray:
    """Every (prediction, ground-truth) IoU for one image, as a P x G array
    indexed by INPUT order.

    Exists because `evaluate` sweeps ten IoU thresholds and then walks three
    width bands, and quad_iou rasterizes two polygons on every call -- so
    without this the same pair is rasterized fourteen times over. That was
    affordable only while a 0.5 score threshold was keeping the prediction
    count down, which is exactly the thing we need to stop doing."""
    matrix = np.zeros((len(predictions), len(ground_truth)), dtype=np.float64)
    for i, (quad, _score) in enumerate(predictions):
        for j, gt_quad in enumerate(ground_truth):
            matrix[i, j] = quad_iou(quad, gt_quad)
    return matrix


def score_order(predictions: list[tuple[list[Point], float]]) -> list[int]:
    """Indices of `predictions` from highest score to lowest.

    Callers that pair a `match_predictions` result back up with the
    predictions it came from need this, because the flags come back in score
    order while a cached IoU matrix is indexed by input order. Stable, so
    equal scores keep their input order."""
    return sorted(range(len(predictions)), key=lambda i: predictions[i][1], reverse=True)


def match_predictions(
    predictions: list[tuple[list[Point], float]],
    ground_truth: list[list[Point]],
    iou_threshold: float,
    ious: np.ndarray | None = None,
) -> tuple[list[bool], int]:
    """Greedy score-ranked matching, COCO-style.

    Returns (is_true_positive per prediction IN SCORE ORDER, matched GT
    count). Each ground-truth object can be claimed only once — without
    that, a model that fires ten overlapping boxes per spine would score
    perfectly, and merge/split behaviour would be invisible.

    `ious` is an optional precomputed P x G matrix from `iou_matrix`,
    indexed by the INPUT order of `predictions` (not score order). Passing
    it changes nothing about the result — there is a test pinning that at
    every threshold — it only avoids re-rasterizing the same polygon pairs
    once per threshold.
    """
    if ious is None:
        ious = iou_matrix(predictions, ground_truth)

    claimed = [False] * len(ground_truth)
    flags: list[bool] = []

    for prediction_index in score_order(predictions):
        best_iou, best_index = 0.0, -1
        for index in range(len(ground_truth)):
            if claimed[index]:
                continue
            iou = float(ious[prediction_index, index])
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

    # Rasterize every (prediction, ground-truth) pair ONCE. The threshold
    # sweep and all three width bands then read from these matrices.
    cached = [
        (predictions, ground_truth, iou_matrix(predictions, ground_truth))
        for predictions, ground_truth in per_image
    ]

    ap_per_threshold = []
    for threshold in IOU_THRESHOLDS:
        flags, scores, total_gt, matched_gt = [], [], 0, 0
        for predictions, ground_truth, ious in cached:
            image_flags, matched = match_predictions(
                predictions, ground_truth, threshold, ious=ious
            )
            ranked_scores = sorted((s for _, s in predictions), reverse=True)
            flags.extend(image_flags)
            scores.extend(ranked_scores)
            total_gt += len(ground_truth)
            matched_gt += matched
        ap = average_precision(flags, scores, total_gt)
        ap_per_threshold.append(ap)
        results["by_iou"][f"AP@{threshold:.2f}"] = ap

        if threshold == IOU_THRESHOLDS[0]:
            # Recall at IoU 0.50, reported because AP alone cannot tell
            # "the model never produced this detection" apart from "we
            # filtered it out before scoring": average_precision divides by
            # the full ground-truth count, so a dropped detection caps
            # recall and every recall point above the cap contributes 0.0.
            # A low mAP with high recall is a precision problem; a low mAP
            # with low recall is a detections problem. Different fixes.
            results["recall@50"] = matched_gt / total_gt if total_gt else 0.0

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
        flags, scores, total_gt, matched_gt = [], [], 0, 0
        for predictions, ground_truth, ious in cached:
            band_columns = [j for j, gt in enumerate(ground_truth) if bucket(gt) == band]
            other_columns = [j for j, gt in enumerate(ground_truth) if bucket(gt) != band]
            if not band_columns:
                continue

            band_gt = [ground_truth[j] for j in band_columns]
            image_flags, matched = match_predictions(
                predictions, band_gt, IOU_THRESHOLDS[0], ious=ious[:, band_columns]
            )

            for position, prediction_index in enumerate(score_order(predictions)):
                is_true_positive = image_flags[position]
                if not is_true_positive and other_columns:
                    out_of_band = max(float(ious[prediction_index, j]) for j in other_columns)
                    if out_of_band >= IOU_THRESHOLDS[0]:
                        continue  # belongs to another band -- neither TP nor FP here
                flags.append(is_true_positive)
                scores.append(predictions[prediction_index][1])
            total_gt += len(band_gt)
            matched_gt += matched

        results["by_width"][band] = {
            "AP@50": average_precision(flags, scores, total_gt),
            "recall@50": matched_gt / total_gt if total_gt else 0.0,
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
    log_every: int = 10,
    nms_thresh: float | None = None,
    detections_per_img: int | None = None,
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
    from model import build_model, set_detection_thresholds
    from train import load_checkpoint

    # Same selection as train.py: eval was running Mask R-CNN on the CPU
    # while the GPU that just did the training sat idle, which is most of
    # why a 147-image run looked like a hang.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = build_model(pretrained=False)
    # Load onto the CPU first, then move -- load_checkpoint maps to CPU, and
    # this order never holds two copies of the weights on the GPU.
    load_checkpoint(Path(checkpoint), model)

    # Post-processing thresholds are inference-time only, so sweeping them
    # needs no retraining. Printed rather than assumed: a run's own output
    # has to say which configuration produced its numbers, or a sweep's
    # results cannot be told apart afterwards.
    overrides = set_detection_thresholds(
        model, nms_thresh=nms_thresh, detections_per_img=detections_per_img
    )
    heads = model.roi_heads
    print(
        f"nms_thresh: {heads.nms_thresh}  "
        f"score_thresh: {heads.score_thresh}  "
        f"detections_per_img: {heads.detections_per_img}"
        + (f"   (overridden: {overrides})" if overrides else "   (torchvision defaults)")
    )

    model.to(device)
    model.eval()

    dataset = SpineDataset(Path(coco_path), Path(images_dir))
    if limit:
        dataset.entries = dataset.entries[:limit]

    # Same reason train.py prints its batch count (d727acc): a slow run and
    # a hung run are indistinguishable from silence, and this loop used to
    # print nothing at all until it was completely finished.
    print(f"evaluating {len(dataset)} images  (a line every {log_every})")
    print(f"score threshold: {score_threshold}")

    per_image = []
    for index in range(len(dataset)):
        image, target = dataset[index]
        with torch.no_grad():
            output = model([image.to(device)])[0]

        keep = output["scores"] >= score_threshold
        predictions = []
        # .cpu() before .numpy(): on CUDA the bare .numpy() raises
        # "can't convert cuda:0 device type tensor to numpy". target["masks"]
        # below needs no such thing -- ground truth never leaves the CPU.
        masks = output["masks"][keep].cpu().numpy()
        for mask, score in zip(masks, output["scores"][keep].tolist()):
            quad = mask_to_quad(mask[0])
            if quad is not None:
                predictions.append((quad, score))

        ground_truth = [
            quad
            for quad in (mask_to_quad(m.numpy()) for m in target["masks"])
            if quad is not None
        ]
        per_image.append((predictions, ground_truth))

        if index % log_every == 0:
            # detections vs gt is the live version of the recall question:
            # detections sitting far below gt on dense shelves is the
            # suppression this eval exists to measure.
            print(
                f"  image {index:>4}/{len(dataset)}"
                f"  detections={len(predictions):>3}  gt={len(ground_truth):>3}",
                flush=True,
            )

    print("matching and integrating AP...", flush=True)
    return evaluate(per_image)


def main() -> None:
    import argparse

    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--coco", default=str(repo_root / "data" / "merged" / "pretrain_val.json"))
    parser.add_argument("--images-dir", default=str(repo_root / "data" / "merged" / "images"))
    parser.add_argument(
        "--score-threshold", type=float, default=0.5,
        help="Drop detections below this score BEFORE scoring. This caps recall "
             "and therefore caps AP -- COCO applies no such filter, because "
             "ranking by score IS the threshold sweep. Use 0.05 (the model's own "
             "box_score_thresh) for a comparable mAP; the 0.5 default is kept "
             "only so earlier numbers stay reproducible.",
    )
    parser.add_argument("--log-every", type=int, default=10,
                        help="Print a progress line every N images.")
    parser.add_argument(
        "--box-nms-thresh", type=float, default=None,
        help="Override the detection head's NMS IoU threshold (torchvision "
             "default 0.5). Inference-time only, so no retraining. Raising it "
             "suppresses less: if recall climbs, the detections existed and "
             "axis-aligned NMS was discarding them, which is the case for "
             "mask-based NMS. Expect AP to fall as precision drops -- that "
             "cost is the point, not a side effect.",
    )
    parser.add_argument(
        "--detections-per-img", type=int, default=None,
        help="Override the cap on detections per image (default 100). Only "
             "matters if a dense shelf is actually hitting it.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    results = evaluate_checkpoint(
        args.checkpoint, args.coco, args.images_dir, args.score_threshold,
        args.limit, args.log_every, args.box_nms_thresh, args.detections_per_img,
    )

    print("=== §8א geometric evaluation (quad IoU, not AABB) ===")
    print(f"mAP@50    : {results['mAP@50']:.4f}")
    print(f"mAP@50:95 : {results['mAP@50:95']:.4f}")
    print(f"recall@50 : {results['recall@50']:.4f}")
    print(f"  (at score threshold {args.score_threshold} -- AP cannot exceed recall,")
    print("   so a low mAP with high recall is a precision problem and a low mAP")
    print("   with low recall is a missing-detections problem)")
    print("\nby spine-width tercile (trap #4 -- a good overall mAP can hide this):")
    edges = results["width_tercile_edges"]
    labels = {
        "thin": f"thin   (<= {edges['thin_max']:.1f}px)",
        "medium": f"medium (<= {edges['medium_max']:.1f}px)",
        "wide": "wide             ",
    }
    for band, label in labels.items():
        band_results = results["by_width"][band]
        print(f"  {label} AP@50={band_results['AP@50']:.4f}"
              f"  recall@50={band_results['recall@50']:.4f}"
              f"  n={band_results['ground_truth']}")

    if args.report:
        import json

        Path(args.report).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nreport -> {args.report}")


if __name__ == "__main__":
    main()
