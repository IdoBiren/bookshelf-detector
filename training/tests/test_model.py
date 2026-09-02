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

from model import NUM_CLASSES, build_model, describe_model  # noqa: E402


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


class TestDescribeModel(unittest.TestCase):
    def test_reports_a_plausible_parameter_count(self):
        model = build_model(pretrained=False)
        info = describe_model(model)
        # ResNet50-FPN Mask R-CNN is ~44M params. Assert only a broad range:
        # this is a report, not a budget gate.
        self.assertGreater(info["total_params"], 10_000_000)
        self.assertLess(info["total_params"], 100_000_000)
        self.assertGreater(info["size_mb_fp32"], 0)


if __name__ == "__main__":
    unittest.main()
