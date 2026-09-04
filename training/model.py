"""
Mask R-CNN for book-spine instance segmentation.

Why this and not the DBNet-lite of plan §1: DBNet's shrink-mask machinery
existed for exactly one reason — to separate touching instances inside a
~2.5M parameter budget. Mask R-CNN produces one mask per RoI, so touching
spines are separate by construction: no shrink map, no unclip, and none of
the tapering-quad self-intersection phase B measured (HANDOFF.md "Open
questions" §1). The size budget is suspended while the quality ceiling is
being measured, which is what makes this affordable.

`torchvision` is BSD-3-Clause (plan §0 already approved it as a source, and
§1 already accepted its ImageNet-pretrained weights for the MobileNetV3
backbone). Note for the record: the COCO-pretrained detection weights carry
no explicit `license` field in torchvision's own metadata — they are
produced by torchvision's training recipe and distributed with the library.
That is the same level of certainty §1's ImageNet weights have, not a new
exposure, but it is not a formal grant either.

The notebook stays thin (plan §13 phase D): clone -> pip install -> call
into this module. No model development inside a notebook cell.
"""

from __future__ import annotations

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# Background + spine. One real class on purpose (plan §3): a `non_book`
# class only gets added if evaluation shows false positives are an actual
# problem, and torchvision counts background as class 0.
NUM_CLASSES = 2
SPINE_LABEL = 1

DEFAULT_HIDDEN_LAYER = 256  # torchvision's own default for the mask head


def build_model(
    pretrained: bool = True,
    trainable_backbone_layers: int | None = None,
    nms_thresh: float | None = 0.6,
):
    """Mask R-CNN ResNet50-FPN with both predictor heads resized to
    NUM_CLASSES.

    `pretrained=False` is for tests — it keeps them offline and fast. Real
    training wants the COCO weights: 1,440 pretrain images is far too few to
    learn detection from scratch, and the whole point of this phase is to
    measure the ceiling, not to handicap it.

    Both heads must be replaced. Swapping only the box predictor leaves the
    mask head predicting COCO's 91 classes — the model still trains, and the
    bug is invisible until the masks come out wrong. There's a test for it.

    `nms_thresh` overrides `roi_heads.nms_thresh` (torchvision's raw default
    is 0.5) and defaults to **0.6**, not to torchvision's own default: 0.6
    measured +1.0pp mAP@50 and +3.1pp recall@50 on pretrain_val over 0.5, at
    no retraining cost, because NMS is a post-processing step, applied only
    when the model is in eval() mode. 0.7 was also measured and rejected --
    recall kept climbing but the precision cost overtook it, netting a lower
    mAP than 0.6. Pass `nms_thresh=None` for torchvision's untouched 0.5,
    e.g. to reproduce the pre-measurement baseline."""
    # weights_backbone must be pinned off too. torchvision defaults it to
    # ImageNet weights independently of `weights`, so `weights=None` alone
    # still triggers a ~100MB ResNet50 download -- which quietly made the
    # "offline" tests not offline until this was caught.
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights="DEFAULT" if pretrained else None,
        weights_backbone="DEFAULT" if pretrained else None,
        trainable_backbone_layers=trainable_backbone_layers,
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, DEFAULT_HIDDEN_LAYER, NUM_CLASSES
    )

    if nms_thresh is not None:
        set_detection_thresholds(model, nms_thresh=nms_thresh)

    return model


def set_detection_thresholds(
    model: torch.nn.Module,
    nms_thresh: float | None = None,
    score_thresh: float | None = None,
    detections_per_img: int | None = None,
    rpn_nms_thresh: float | None = None,
) -> dict:
    """Override post-processing thresholds on a model that is already built.
    Returns only what it changed, so a caller can print it and have the
    run's output record its own configuration.

    `build_model` sets none of these, so torchvision's defaults apply:
    roi_heads.nms_thresh=0.5, roi_heads.score_thresh=0.05,
    roi_heads.detections_per_img=100, rpn.nms_thresh=0.7.

    `rpn_nms_thresh` is a SEPARATE, upstream stage from `nms_thresh`: the RPN
    suppresses region proposals before the detection head ever sees them,
    while `nms_thresh` suppresses the head's final boxes. Sweeping one tells
    you nothing about the other -- keep them as distinct attributes on
    distinct modules (`model.rpn` vs `model.roi_heads`) rather than
    collapsing them into one name.

    Exists as a tested function because the names do not match the
    constructor and the mismatch fails silently. The MaskRCNN CONSTRUCTOR
    takes `box_nms_thresh`; RoIHeads STORES it as `nms_thresh`. Assigning
    `roi_heads.box_nms_thresh = 0.7` raises nothing, changes nothing, and an
    NMS sweep built that way comes back flat -- which reads exactly like
    "NMS is not the bottleneck" and would retire a live hypothesis on a
    typo.

    The range checks are for the same reason. `nms_thresh=7` (a dropped
    decimal point) disables suppression completely rather than loosening it,
    and the result is a plausible-looking recall jump caused by returning
    every proposal."""
    changed: dict = {}

    for name, value in (("nms_thresh", nms_thresh), ("score_thresh", score_thresh)):
        if value is None:
            continue
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(
                f"{name} must be in [0.0, 1.0], got {value!r}. "
                "These are IoU/probability thresholds -- a value above 1 "
                "disables the filter instead of tightening it."
            )
        setattr(model.roi_heads, name, float(value))
        changed[name] = float(value)

    if rpn_nms_thresh is not None:
        if not 0.0 <= float(rpn_nms_thresh) <= 1.0:
            raise ValueError(
                f"rpn_nms_thresh must be in [0.0, 1.0], got {rpn_nms_thresh!r}"
            )
        model.rpn.nms_thresh = float(rpn_nms_thresh)
        changed["rpn_nms_thresh"] = float(rpn_nms_thresh)

    if detections_per_img is not None:
        if int(detections_per_img) < 1:
            raise ValueError(
                f"detections_per_img must be >= 1, got {detections_per_img!r}"
            )
        model.roi_heads.detections_per_img = int(detections_per_img)
        changed["detections_per_img"] = int(detections_per_img)

    return changed


def describe_model(model: torch.nn.Module) -> dict:
    """Parameter count and fp32 footprint, for the record.

    Reported, not asserted: the size budget is deliberately suspended for
    this phase (see the module docstring). Keeping the number visible means
    the later "how much do we lose going small?" decision starts from a
    measured baseline instead of an estimate.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "size_mb_fp32": total * 4 / (1024 * 1024),
    }


if __name__ == "__main__":
    model = build_model(pretrained=False)
    info = describe_model(model)
    print(f"classes:          {NUM_CLASSES} (background + spine)")
    print(f"total params:     {info['total_params']:,}")
    print(f"trainable params: {info['trainable_params']:,}")
    print(f"fp32 size:        {info['size_mb_fp32']:.1f} MB")
