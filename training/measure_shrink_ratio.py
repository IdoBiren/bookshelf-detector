"""
The go/no-go measurement (plan §13 phase B): is DBNet's standard
shrink_ratio=0.4 valid for book spines, and at what target-map resolution?

Runs merged_adjacent_pairs_metric and vanishing_spines_metric from
dbnet_targets.py across a sweep of (dataset x stride x mode x shrink_ratio),
on REAL annotations rather than synthetic shapes — the two thresholds this
is judged against are plan §13's own:
  - merged adjacent pairs > 5%  -> shrink_ratio is wrong at this resolution
  - vanished spines       > 1%  -> bug, not noise

Usage:
    python training/measure_shrink_ratio.py
    python training/measure_shrink_ratio.py --stride 1 4 --shrink-ratio 0.4
    python training/measure_shrink_ratio.py --coco data/merged/indomain_train.json --limit 50

Scope note: --augment (pushing polygons through augment.py's pipeline
before measuring, plan §13's metric 2b) and a --preview-dir hookup to
preview_shrink_masks.py are follow-ups, not implemented here — this script
answers the primary shrink_ratio/stride question first.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dbnet_targets import (
    TARGET_SIZE,
    compute_letterbox,
    letterbox_polygon,
    merged_adjacent_pairs_metric,
    vanishing_spines_metric,
)

DEFAULT_COCO_FILES = [
    "data/merged/indomain_train.json",
    "data/merged/pretrain_train.json",
]
DEFAULT_STRIDES = [1, 4]
DEFAULT_MODES = ["quad"]
DEFAULT_SHRINK_RATIOS = [0.4]

MERGED_PAIRS_THRESHOLD = 0.05
VANISHED_THRESHOLD = 0.01


def load_polygons_by_image(coco_path: Path) -> list[tuple[int, int, list[list[tuple[float, float]]]]]:
    """Returns [(width, height, polygons), ...] — one entry per image that
    has at least one annotation. Segmentation is taken as-is (real polygons
    have 5-9+ vertices, not a fixed 4 — see training/augment.py's own
    docstring for the same point) and prepare_polygon (called inside the
    metric functions) is what reduces vertex count, not this loader.
    """
    with coco_path.open(encoding="utf-8") as f:
        coco = json.load(f)

    polygons_by_image_id: dict[int, list[list[tuple[float, float]]]] = {}
    for ann in coco["annotations"]:
        coords = ann["segmentation"][0]
        points = list(zip(coords[0::2], coords[1::2]))
        polygons_by_image_id.setdefault(ann["image_id"], []).append(points)

    result = []
    for img in coco["images"]:
        polys = polygons_by_image_id.get(img["id"])
        if polys:
            result.append((img["width"], img["height"], polys))
    return result


def letterboxed_polygons_by_image(
    images: list[tuple[int, int, list[list[tuple[float, float]]]]],
    stride: int,
    target_size: int,
) -> list[list[list[tuple[float, float]]]]:
    result = []
    for width, height, polygons in images:
        info = compute_letterbox(width, height, target_size)
        result.append([letterbox_polygon(p, info, stride) for p in polygons])
    return result


def run_one_cell(
    images: list[tuple[int, int, list[list[tuple[float, float]]]]],
    stride: int,
    mode: str,
    shrink_ratio: float,
    target_size: int,
) -> dict:
    canvas_size = target_size // stride
    letterboxed = letterboxed_polygons_by_image(images, stride, target_size)

    merged = merged_adjacent_pairs_metric(letterboxed, shrink_ratio, canvas_size, mode)
    vanished = vanishing_spines_metric(letterboxed, shrink_ratio, canvas_size, mode)

    merged_ok = merged["merged_fraction"] <= MERGED_PAIRS_THRESHOLD
    vanished_ok = vanished["vanished_fraction"] <= VANISHED_THRESHOLD

    return {
        "stride": stride,
        "mode": mode,
        "shrink_ratio": shrink_ratio,
        "canvas_size": canvas_size,
        "images": len(images),
        "merged": merged,
        "vanished": vanished,
        "verdict": "GO" if (merged_ok and vanished_ok) else "NO-GO",
    }


def print_cell(dataset_name: str, cell: dict) -> None:
    m, v = cell["merged"], cell["vanished"]
    print(
        f"dataset={dataset_name:<20} stride={cell['stride']:<3} mode={cell['mode']:<5} "
        f"shrink_ratio={cell['shrink_ratio']:<4} canvas={cell['canvas_size']}x{cell['canvas_size']}"
    )
    print(
        f"  merged adjacent pairs: {m['merged_pairs']:>5}/{m['adjacent_pairs']:<5} "
        f"= {m['merged_fraction']*100:6.2f}%  [threshold {MERGED_PAIRS_THRESHOLD*100:.1f}%]"
    )
    print(
        f"  vanished spines:       {v['vanished']:>5}/{v['total']:<5} "
        f"= {v['vanished_fraction']*100:6.2f}%  [threshold {VANISHED_THRESHOLD*100:.1f}%]  "
        f"(zero_area/self-x={v['zero_area_or_self_intersecting']} "
        f"zero_px={v['zero_pixels']} tiny_px={v['tiny_pixels']})"
    )
    print(f"  VERDICT: {cell['verdict']}")
    print()


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco", type=Path, nargs="+", default=None)
    parser.add_argument("--stride", type=int, nargs="+", default=DEFAULT_STRIDES)
    parser.add_argument("--mode", choices=["raw", "hull", "quad"], nargs="+", default=DEFAULT_MODES)
    parser.add_argument("--shrink-ratio", type=float, nargs="+", default=DEFAULT_SHRINK_RATIOS)
    parser.add_argument("--target-size", type=int, default=TARGET_SIZE)
    parser.add_argument("--limit", type=int, default=None, help="Sample at most N images per dataset.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", type=Path, default=repo_root / "data" / "merged" / "shrink_ratio_report.json")
    parser.add_argument("--fail-on-nogo", action="store_true")
    args = parser.parse_args()

    coco_paths = args.coco or [repo_root / p for p in DEFAULT_COCO_FILES]

    all_results = []
    any_nogo = False

    print("=== DBNet shrink target validation (plan §13 phase B) ===\n")

    for coco_path in coco_paths:
        coco_path = coco_path.resolve()
        if not coco_path.exists():
            print(f"SKIP: {coco_path} does not exist")
            continue

        images = load_polygons_by_image(coco_path)
        total_anns = sum(len(p) for _, _, p in images)
        if args.limit and len(images) > args.limit:
            import random

            rng = random.Random(args.seed)
            images = rng.sample(images, args.limit)

        dataset_name = coco_path.stem
        print(f"--- {dataset_name}: {len(images)} images (of {total_anns} total annotations) ---\n")

        for stride in args.stride:
            for mode in args.mode:
                for shrink_ratio in args.shrink_ratio:
                    cell = run_one_cell(images, stride, mode, shrink_ratio, args.target_size)
                    cell["dataset"] = dataset_name
                    print_cell(dataset_name, cell)
                    all_results.append(cell)
                    if cell["verdict"] == "NO-GO":
                        any_nogo = True

    print("=== Summary ===")
    print(f"{'dataset':<20}{'stride':>7}{'mode':>7}{'ratio':>7}{'merged%':>10}{'vanished%':>11}  verdict")
    for cell in all_results:
        print(
            f"{cell['dataset']:<20}{cell['stride']:>7}{cell['mode']:>7}{cell['shrink_ratio']:>7}"
            f"{cell['merged']['merged_fraction']*100:>9.2f}%{cell['vanished']['vanished_fraction']*100:>10.2f}%"
            f"  {cell['verdict']}"
        )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull report written to {args.report}")
    print(
        "NOTE: data/merged/ is gitignored -- paste the verdict block into HANDOFF.md "
        "or this result disappears on the next clone."
    )

    if any_nogo and args.fail_on_nogo:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
