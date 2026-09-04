"""
Tests for model.py — the Mask R-CNN wrapper (plan: "answer the quality
question before the deployment question").

These run on CPU with random-ish input and no training. They check the
things that are cheap to get wrong and expensive to discover in Colab six
hours in: head shape, output structure, and that train-mode actually
produces the loss keys the training loop will read.

Deliberately NOT here: any parameter-count budget assertion. The whole point
of this architecture change is that the size constraint is suspended until a
quality ceiling exists; re-imposing it in a test would defeat that. The count
is reported by `describe_model` instead, so it stays visible without being a
gate.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from model import (  # noqa: E402
    NUM_CLASSES,
    build_model,
    describe_model,
    set_detection_thresholds,
)


class TestBuildModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # weights=None keeps the test offline and fast -- pretrained COCO
        # weights are a ~170MB download and are not what these assertions
        # are about.
        cls.model = build_model(pretrained=False)

    def test_predicts_two_classes_background_plus_spine(self):
        """Single class by design (plan §3: a non_book class only gets added
        if evaluation shows false positives are an actual problem) -- but
        torchvision counts background as class 0, so the head must be 2."""
        self.assertEqual(NUM_CLASSES, 2)
        box_predictor = self.model.roi_heads.box_predictor
        self.assertEqual(box_predictor.cls_score.out_features, 2)
        # 4 box coordinates per class
        self.assertEqual(box_predictor.bbox_pred.out_features, 8)

    def test_mask_head_also_predicts_two_classes(self):
        """Easy to replace the box predictor and forget the mask predictor --
        the model then trains but the mask head still has COCO's 91 classes."""
        mask_predictor = self.model.roi_heads.mask_predictor
        self.assertEqual(mask_predictor.mask_fcn_logits.out_channels, 2)

    def test_eval_mode_returns_boxes_labels_scores_and_masks(self):
        self.model.eval()
        with torch.no_grad():
            output = self.model([torch.rand(3, 256, 256)])
        self.assertEqual(len(output), 1)
        for key in ("boxes", "labels", "scores", "masks"):
            self.assertIn(key, output[0])

    def test_masks_come_back_with_the_channel_dim_mask_to_quad_expects(self):
        """torchvision emits (N, 1, H, W). masks_to_quads handles that shape
        explicitly; this pins the contract between the two."""
        self.model.eval()
        with torch.no_grad():
            output = self.model([torch.rand(3, 256, 256)])
        masks = output[0]["masks"]
        self.assertEqual(masks.ndim, 4)
        if masks.shape[0] > 0:
            self.assertEqual(masks.shape[1], 1)

    def test_defaults_to_the_measured_nms_thresh(self):
        """0.6 measured +1.0pp mAP@50 and +3.1pp recall@50 over torchvision's
        raw 0.5, at no retraining cost -- so it is the default rather than
        something every caller has to remember to pass. 0.7 was also
        measured and rejected: recall kept rising but precision loss caught
        up, netting a LOWER mAP than 0.6."""
        self.assertAlmostEqual(self.model.roi_heads.nms_thresh, 0.6)

    def test_nms_thresh_none_keeps_torchvisions_raw_default(self):
        """The escape hatch a sweep needs: comparing against the untouched
        baseline requires being able to ask for it."""
        raw = build_model(pretrained=False, nms_thresh=None)
        self.assertAlmostEqual(raw.roi_heads.nms_thresh, 0.5)

    def test_defaults_to_torchvisions_mask_resolution_of_14(self):
        self.assertEqual(self.model.roi_heads.mask_roi_pool.output_size, (14, 14))

    def test_train_mode_returns_the_four_loss_terms_the_loop_will_sum(self):
        self.model.train()
        image = torch.rand(3, 256, 256)
        target = {
            "boxes": torch.tensor([[10.0, 10.0, 100.0, 200.0]]),
            "labels": torch.tensor([1]),
            "masks": torch.zeros((1, 256, 256), dtype=torch.uint8),
        }
        target["masks"][0, 10:200, 10:100] = 1
        losses = self.model([image], [target])
        for key in ("loss_classifier", "loss_box_reg", "loss_mask", "loss_objectness"):
            self.assertIn(key, losses)
        total = sum(losses.values())
        self.assertTrue(torch.isfinite(total), f"non-finite loss: {losses}")


class TestMaskResolution(unittest.TestCase):
    """A 13.8:1 median spine, boxed with near-perfect accuracy (measured
    recall 0.99+), still loses to quad recall of ~0.67 -- because the fixed
    14x14 mask_roi_pool grid gives its WIDTH ~2px (p90 spine: ~1px) to work
    with, and mask_to_quad then fits a quad to that. Raising this resolution
    is the fix under test; these pins exist because a wrong output_size or a
    hidden shape mismatch would be a silent failure, not a crash -- exactly
    the kind of bug the checkpoint-compatibility claim below depends on
    being false."""

    def test_raises_the_mask_roi_pool_output_size(self):
        model = build_model(pretrained=False, mask_resolution=28)
        self.assertEqual(model.roi_heads.mask_roi_pool.output_size, (28, 28))

    def test_mask_head_and_predictor_weight_shapes_are_unchanged(self):
        """The finding this whole plan is built on: mask_head/mask_predictor
        are conv/deconv layers, so their weight shapes depend on CHANNEL
        counts, not on mask_roi_pool's spatial output size. If a future
        torchvision version breaks this, checkpoint_epoch_009.pt would fail
        to load into a higher-resolution model with a shape-mismatch error
        -- this test catches that before a wasted Colab run, not after."""
        default = build_model(pretrained=False)
        raised = build_model(pretrained=False, mask_resolution=28)

        default_head = {k: v.shape for k, v in default.roi_heads.mask_head.state_dict().items()}
        raised_head = {k: v.shape for k, v in raised.roi_heads.mask_head.state_dict().items()}
        self.assertEqual(default_head, raised_head)

        default_pred = {k: v.shape for k, v in default.roi_heads.mask_predictor.state_dict().items()}
        raised_pred = {k: v.shape for k, v in raised.roi_heads.mask_predictor.state_dict().items()}
        self.assertEqual(default_pred, raised_pred)

    def test_a_checkpoint_trained_at_14_loads_into_a_28_model(self):
        """The actual compatibility claim, not just matching shapes in the
        abstract: load_state_dict must not raise."""
        source = build_model(pretrained=False)
        target = build_model(pretrained=False, mask_resolution=28)
        target.load_state_dict(source.state_dict())  # raises on any mismatch

    def test_higher_resolution_model_still_runs_end_to_end(self):
        model = build_model(pretrained=False, mask_resolution=28)
        model.eval()
        with torch.no_grad():
            output = model([torch.rand(3, 256, 256)])
        for key in ("boxes", "labels", "scores", "masks"):
            self.assertIn(key, output[0])

    def test_default_resolution_is_still_14_when_not_asked_for(self):
        """The other half of the compatibility claim: this is an opt-in
        change, not a silent default flip that would make every EXISTING
        caller's checkpoints subtly wrong."""
        model = build_model(pretrained=False)
        self.assertEqual(model.roi_heads.mask_roi_pool.output_size, (14, 14))


class TestDescribeModel(unittest.TestCase):
    def test_reports_a_plausible_parameter_count(self):
        model = build_model(pretrained=False)
        info = describe_model(model)
        # ResNet50-FPN Mask R-CNN is ~44M params. Assert only a broad range:
        # this is a report, not a budget gate.
        self.assertGreater(info["total_params"], 10_000_000)
        self.assertLess(info["total_params"], 100_000_000)
        self.assertGreater(info["size_mb_fp32"], 0)


class TestSetDetectionThresholds(unittest.TestCase):
    """The constructor names these `box_nms_thresh` / `box_score_thresh`,
    but RoIHeads stores them as `nms_thresh` / `score_thresh`. Assigning the
    constructor name to a built model does not fail -- it silently creates an
    unused attribute, inference is completely unaffected, and an NMS sweep
    would come back flat and be read as "NMS is not the problem".

    That wrong conclusion is the reason this seam is a function with tests
    rather than two lines in evaluate.py."""

    def setUp(self):
        # nms_thresh=None: these tests are about the setter itself, not
        # build_model's baked-in 0.6 -- starting from torchvision's raw
        # defaults keeps "untouched" assertions meaningful.
        self.model = build_model(pretrained=False, nms_thresh=None)

    def test_torchvision_defaults_are_what_we_think_they_are(self):
        """Pinned because the whole recall investigation is reasoned against
        these numbers. If a torchvision upgrade moves them, the measured
        history stops being comparable and we want to be told.

        Built with nms_thresh=None here specifically -- build_model's OWN
        default is no longer torchvision's raw 0.5, see
        TestBuildModel.test_defaults_to_the_measured_nms_thresh below."""
        raw = build_model(pretrained=False, nms_thresh=None)
        heads = raw.roi_heads
        self.assertAlmostEqual(heads.nms_thresh, 0.5)
        self.assertAlmostEqual(heads.score_thresh, 0.05)
        self.assertEqual(heads.detections_per_img, 100)
        self.assertAlmostEqual(raw.rpn.nms_thresh, 0.7)

    def test_sets_rpn_nms_thresh_not_roi_heads_nms_thresh(self):
        """rpn.nms_thresh suppresses PROPOSALS, upstream of roi_heads.nms_thresh
        which suppresses final detections. They are two separate stages, and
        this test exists so a future refactor cannot quietly collapse them
        into one attribute -- that would make an RPN-stage sweep silently
        change detection-stage behaviour instead, or vice versa."""
        set_detection_thresholds(self.model, rpn_nms_thresh=0.8)
        self.assertAlmostEqual(self.model.rpn.nms_thresh, 0.8)
        self.assertAlmostEqual(self.model.roi_heads.nms_thresh, 0.5)  # untouched

    def test_rejects_rpn_nms_thresh_outside_zero_to_one(self):
        with self.assertRaises(ValueError):
            set_detection_thresholds(self.model, rpn_nms_thresh=7)

    def test_rpn_nms_thresh_is_reported_in_the_changed_dict(self):
        changed = set_detection_thresholds(self.model, rpn_nms_thresh=0.8)
        self.assertEqual(changed, {"rpn_nms_thresh": 0.8})

    def test_sets_the_attribute_inference_actually_reads(self):
        set_detection_thresholds(self.model, nms_thresh=0.7)
        self.assertAlmostEqual(self.model.roi_heads.nms_thresh, 0.7)

    def test_does_not_create_the_constructor_spelling(self):
        """The bug this function exists to prevent."""
        set_detection_thresholds(self.model, nms_thresh=0.7)
        self.assertFalse(hasattr(self.model.roi_heads, "box_nms_thresh"))

    def test_none_leaves_a_threshold_untouched(self):
        set_detection_thresholds(self.model, nms_thresh=0.7)
        set_detection_thresholds(self.model, score_thresh=0.01)
        self.assertAlmostEqual(self.model.roi_heads.nms_thresh, 0.7)
        self.assertAlmostEqual(self.model.roi_heads.score_thresh, 0.01)

    def test_sets_all_three_together(self):
        set_detection_thresholds(
            self.model, nms_thresh=0.8, score_thresh=0.02, detections_per_img=300
        )
        heads = self.model.roi_heads
        self.assertAlmostEqual(heads.nms_thresh, 0.8)
        self.assertAlmostEqual(heads.score_thresh, 0.02)
        self.assertEqual(heads.detections_per_img, 300)

    def test_returns_what_it_changed_for_the_record(self):
        """evaluate.py prints this, so a run's output says which thresholds
        produced its numbers instead of leaving it to the shell history."""
        changed = set_detection_thresholds(self.model, nms_thresh=0.7)
        self.assertEqual(changed, {"nms_thresh": 0.7})
        self.assertEqual(set_detection_thresholds(self.model), {})

    def test_rejects_a_threshold_outside_zero_to_one(self):
        """A typo'd 7 instead of 0.7 disables NMS entirely and silently
        returns up to detections_per_img boxes per spine."""
        for bad in (-0.1, 1.5, 7):
            with self.assertRaises(ValueError):
                set_detection_thresholds(self.model, nms_thresh=bad)

    def test_rejects_non_positive_detection_cap(self):
        with self.assertRaises(ValueError):
            set_detection_thresholds(self.model, detections_per_img=0)


if __name__ == "__main__":
    unittest.main()
