"""
Single-image inference — the serving path.

`evaluate.py`'s `evaluate_checkpoint` does inference too, but it is built
around a COCO dataset: it loads a split, loops it, and integrates mAP. A
server needs one image in and quads out, with the model loaded once rather
than per request (the checkpoint is ~350MB; reloading it per call would put
~30s on every request).

This module is the extracted core, not a second implementation. It reuses
`build_model`, `set_detection_thresholds`, `load_checkpoint`,
`read_checkpoint_mask_resolution` and `mask_to_quad` — the same functions on
the same path — so a quad served here and a quad scored during evaluation
come from identical code.

**Preprocessing is duplicated from `SpineDataset.__getitem__` deliberately**
and pinned by a parity test: the dataset couples image loading to polygon
rasterization and COCO entries, none of which exist at serving time. The
duplication is three lines; importing the dataset to get them would drag a
COCO file along with it.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mask_to_quad import mask_to_quad
from model import build_model, set_detection_thresholds
from polygon_offset import Point
from train import load_checkpoint, read_checkpoint_mask_resolution

# 0.05 is torchvision's own box_score_thresh and the value §8א measures at,
# but it emits 70-100 detections on a dense shelf -- unusable in a UI. 0.5
# costs only ~1.4pp of recall (measured: 0.6412 -> 0.6277) while removing
# most of the false positives, so serving defaults there and evaluation
# keeps its own default.
DEFAULT_SERVING_SCORE_THRESHOLD = 0.5


@dataclass
class Detector:
    """A loaded model plus the settings it will be served with. Built once
    at process start and reused for every request."""

    model: torch.nn.Module
    device: torch.device
    score_threshold: float
    mask_resolution: int


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """PIL image -> the exact CHW float tensor the model was trained on.

    Must stay identical to `SpineDataset.__getitem__`
    (`training/dataset.py`): RGB, float32, /255, HWC->CHW, and **no resize
    and no normalization** — torchvision's `GeneralizedRCNNTransform` does
    both of those inside the model. Adding either here would silently halve
    accuracy rather than fail. There is a parity test."""
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def load_detector(
    checkpoint: str | Path,
    score_threshold: float = DEFAULT_SERVING_SCORE_THRESHOLD,
    device: str | None = None,
) -> Detector:
    """Build the model, load the checkpoint's weights, and return it ready
    to serve. Call once per process."""
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint}")

    resolved_device = torch.device(device) if device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # The checkpoint records what it was trained with. Reading it back rather
    # than defaulting matters because mask_head/mask_predictor weight shapes
    # do NOT depend on mask_resolution -- building at the wrong one loads
    # cleanly and then produces masks the weights have never seen.
    mask_resolution = read_checkpoint_mask_resolution(checkpoint)

    model = build_model(pretrained=False, mask_resolution=mask_resolution)
    load_checkpoint(checkpoint, model)
    model.to(resolved_device)
    model.eval()

    return Detector(
        model=model,
        device=resolved_device,
        score_threshold=score_threshold,
        mask_resolution=mask_resolution,
    )


def detect_spines(detector: Detector, image: Image.Image) -> list[dict]:
    """One image -> [{"quad": [(x, y) x4], "score": float}], in the ORIGINAL
    image's pixel coordinates.

    Detections whose mask does not yield a usable quad are dropped rather
    than returned as None, so a caller never has to filter."""
    tensor = preprocess_image(image).to(detector.device)

    with torch.no_grad():
        output = detector.model([tensor])[0]

    keep = output["scores"] >= detector.score_threshold
    # .cpu() before .numpy(): on CUDA the bare call raises "can't convert
    # cuda:0 device type tensor to numpy".
    masks = output["masks"][keep].cpu().numpy()
    scores = output["scores"][keep].tolist()

    spines: list[dict] = []
    for mask, score in zip(masks, scores):
        quad: list[Point] | None = mask_to_quad(mask[0])
        if quad is not None:
            spines.append({"quad": [(float(x), float(y)) for x, y in quad], "score": float(score)})
    return spines


def main() -> None:
    """Ad-hoc check against a real photo, without starting the server."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--score-threshold", type=float, default=DEFAULT_SERVING_SCORE_THRESHOLD)
    parser.add_argument("--crops-dir", default=None,
                        help="Also write each rectified spine crop here, to eyeball what "
                             "the VLM would actually receive.")
    args = parser.parse_args()

    detector = load_detector(args.checkpoint, args.score_threshold)
    print(f"device: {detector.device}  mask_resolution: {detector.mask_resolution}"
          f"  score_threshold: {detector.score_threshold}")

    image = Image.open(args.image)
    spines = detect_spines(detector, image)
    print(f"{args.image}: {len(spines)} spines")
    for index, spine in enumerate(spines):
        print(f"  {index:>3}  score={spine['score']:.3f}  quad={spine['quad']}")

    if args.crops_dir:
        import cv2

        from crop_quad import rectify_spine

        crops_dir = Path(args.crops_dir)
        crops_dir.mkdir(parents=True, exist_ok=True)
        bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        written = 0
        for index, spine in enumerate(spines):
            crop = rectify_spine(bgr, spine["quad"])
            if crop is not None:
                cv2.imwrite(str(crops_dir / f"spine_{index:03d}.png"), crop)
                written += 1
        print(f"wrote {written} crops -> {crops_dir}")


if __name__ == "__main__":
    main()
