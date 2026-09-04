"""
Tests for detect.py — single-image inference for the serving path.

detect.py exists because the inference core was buried inside
`evaluate_checkpoint`, which loads a COCO dataset, loops it, and computes
mAP. A server needs "one image in, quads out" without any of that.

The risk in extracting it is silent divergence: preprocessing that differs
from `SpineDataset` by a normalization step, or a model built at the wrong
mask_resolution, degrades quality without raising anything. Both are pinned
here.
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from detect import load_detector, preprocess_image  # noqa: E402
from model import build_model  # noqa: E402
from train import save_checkpoint  # noqa: E402


def _write_checkpoint(directory: Path, mask_resolution: int = 14) -> Path:
    """A real (untrained) checkpoint. Untrained is fine — these tests are
    about plumbing, not detection quality, and it keeps them offline."""
    model = build_model(pretrained=False, mask_resolution=mask_resolution)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    return save_checkpoint(
        directory, model, optimizer, epoch=0, history=[], mask_resolution=mask_resolution
    )


def _write_image(path: Path, width: int = 64, height: int = 48) -> Path:
    rng = np.random.default_rng(0)
    array = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    Image.fromarray(array).save(path)
    return path


class TestPreprocessingParity(unittest.TestCase):
    """The server's tensor must be element-wise identical to the one
    SpineDataset builds during training and evaluation. A stray /255, a
    missing convert("RGB"), or a channel-order slip here costs accuracy
    silently rather than crashing."""

    def test_matches_spine_dataset_exactly(self):
        from dataset import SpineDataset

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            images_dir = tmpdir / "images"
            images_dir.mkdir()
            _write_image(images_dir / "img.png")

            # A minimal COCO file pointing at that one image, with one
            # polygon so SpineDataset keeps the entry.
            import json

            coco = {
                "images": [{"id": 1, "file_name": "img.png", "width": 64, "height": 48}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "segmentation": [[5, 5, 25, 5, 25, 40, 5, 40]],
                        "bbox": [5, 5, 20, 35],
                        "area": 700,
                        "iscrowd": 0,
                    }
                ],
                "categories": [{"id": 1, "name": "spine"}],
            }
            coco_path = tmpdir / "coco.json"
            coco_path.write_text(json.dumps(coco), encoding="utf-8")

            dataset_tensor, _ = SpineDataset(coco_path, images_dir)[0]
            server_tensor = preprocess_image(Image.open(images_dir / "img.png"))

            self.assertEqual(server_tensor.shape, dataset_tensor.shape)
            self.assertEqual(server_tensor.dtype, dataset_tensor.dtype)
            self.assertTrue(torch.equal(server_tensor, dataset_tensor))

    def test_produces_chw_float_in_zero_to_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_image(Path(tmp) / "img.png")
            tensor = preprocess_image(Image.open(path))
            self.assertEqual(tensor.shape[0], 3)
            self.assertEqual(tensor.dtype, torch.float32)
            self.assertGreaterEqual(float(tensor.min()), 0.0)
            self.assertLessEqual(float(tensor.max()), 1.0)

    def test_converts_a_grayscale_image_to_three_channels(self):
        """A user's phone photo is RGB, but a stray grayscale or RGBA upload
        must not reach the model with the wrong channel count."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gray.png"
            Image.fromarray(np.zeros((20, 30), dtype=np.uint8)).save(path)
            self.assertEqual(preprocess_image(Image.open(path)).shape[0], 3)


class TestLoadDetector(unittest.TestCase):
    def test_honors_the_checkpoints_own_mask_resolution(self):
        """Shapes are compatible across resolutions, so building at the
        wrong one succeeds silently and produces garbage masks. The
        checkpoint records its own value; this must read it back."""
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_checkpoint(Path(tmp), mask_resolution=28)
            detector = load_detector(path)
            self.assertEqual(detector.model.roi_heads.mask_roi_pool.output_size, (28, 28))

    def test_applies_the_serving_score_threshold(self):
        """0.05 (the model's own floor) emits 70-100 detections per image and
        is unusable in a UI; serving defaults higher."""
        with tempfile.TemporaryDirectory() as tmp:
            detector = load_detector(_write_checkpoint(Path(tmp)), score_threshold=0.5)
            self.assertAlmostEqual(detector.score_threshold, 0.5)

    def test_model_is_in_eval_mode(self):
        """In train mode the model returns losses, not detections, and would
        also update BatchNorm statistics on user photos."""
        with tempfile.TemporaryDirectory() as tmp:
            detector = load_detector(_write_checkpoint(Path(tmp)))
            self.assertFalse(detector.model.training)


class TestDetectSpines(unittest.TestCase):
    def test_returns_quads_and_scores_in_the_serving_shape(self):
        from detect import detect_spines

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            detector = load_detector(_write_checkpoint(tmpdir), score_threshold=0.0)
            image = Image.open(_write_image(tmpdir / "img.png"))

            spines = detect_spines(detector, image)

            self.assertIsInstance(spines, list)
            for spine in spines:
                self.assertIn("quad", spine)
                self.assertIn("score", spine)
                self.assertEqual(len(spine["quad"]), 4)
                for point in spine["quad"]:
                    self.assertEqual(len(point), 2)

    def test_an_untrained_model_finding_nothing_is_an_empty_list_not_an_error(self):
        """A photo with no books must return [], not raise -- the server
        has to answer that case cleanly."""
        from detect import detect_spines

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            detector = load_detector(_write_checkpoint(tmpdir), score_threshold=0.999)
            image = Image.open(_write_image(tmpdir / "img.png"))
            self.assertEqual(detect_spines(detector, image), [])


if __name__ == "__main__":
    unittest.main()
