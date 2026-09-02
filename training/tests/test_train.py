"""
Tests for train.py's checkpoint/resume machinery.

Colab's free tier caps a session at ~4 hours and the pretrain run is
expected to need ~6 (plan §13 phase D), so resume is not a nicety — it is
the only way the run finishes at all. It has to be built up front, and it
has to be right: a resume that silently restarts from epoch 0 wastes hours
and looks exactly like normal training.

The training loop itself isn't tested here (it needs a GPU and real data to
mean anything); what's tested is everything that decides WHERE the loop
picks up.
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from train import find_latest_checkpoint, load_checkpoint, save_checkpoint  # noqa: E402


class _TinyModel(torch.nn.Module):
    def __init__(self, value: float = 0.0):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([value]))

    def forward(self, x):
        return x * self.weight


class TestCheckpointRoundTrip(unittest.TestCase):
    def test_saved_state_is_restored_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = _TinyModel(3.5)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            path = save_checkpoint(Path(tmp), model, optimizer, epoch=2, history=[{"loss": 1.0}])

            restored = _TinyModel(0.0)
            restored_optimizer = torch.optim.SGD(restored.parameters(), lr=0.1)
            epoch, history = load_checkpoint(path, restored, restored_optimizer)

            self.assertEqual(epoch, 2)
            self.assertEqual(history, [{"loss": 1.0}])
            self.assertAlmostEqual(float(restored.weight.item()), 3.5, places=6)

    def test_optimizer_state_survives_too(self):
        """Restoring weights but not optimizer momentum is a subtle resume
        bug: training continues but the first steps after each resume are
        effectively from a cold optimizer."""
        with tempfile.TemporaryDirectory() as tmp:
            model = _TinyModel(1.0)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
            model(torch.tensor([2.0])).backward()
            optimizer.step()

            path = save_checkpoint(Path(tmp), model, optimizer, epoch=0, history=[])

            restored = _TinyModel(0.0)
            restored_optimizer = torch.optim.SGD(restored.parameters(), lr=0.1, momentum=0.9)
            load_checkpoint(path, restored, restored_optimizer)

            saved_state = optimizer.state_dict()["state"]
            restored_state = restored_optimizer.state_dict()["state"]
            self.assertEqual(len(saved_state), len(restored_state))


class TestFindLatestCheckpoint(unittest.TestCase):
    def test_returns_none_when_the_directory_is_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(find_latest_checkpoint(Path(tmp)))

    def test_returns_none_when_the_directory_does_not_exist(self):
        self.assertIsNone(find_latest_checkpoint(Path("/definitely/not/here")))

    def test_picks_the_highest_epoch_not_the_newest_mtime(self):
        """Sorting by modification time breaks the moment Drive re-syncs a
        file. Sorting by the epoch number in the filename doesn't."""
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            for epoch in (0, 1, 2, 10):
                model = _TinyModel(float(epoch))
                optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
                save_checkpoint(tmpdir, model, optimizer, epoch=epoch, history=[])
            # Touch an early one so it is the most recently modified.
            (tmpdir / "checkpoint_epoch_001.pt").touch()

            latest = find_latest_checkpoint(tmpdir)
            self.assertEqual(latest.name, "checkpoint_epoch_010.pt")

    def test_ignores_unrelated_files_in_the_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            (tmpdir / "notes.txt").write_text("hello", encoding="utf-8")
            (tmpdir / "model_final.pt").write_text("not a checkpoint", encoding="utf-8")
            model = _TinyModel(1.0)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            save_checkpoint(tmpdir, model, optimizer, epoch=3, history=[])

            self.assertEqual(find_latest_checkpoint(tmpdir).name, "checkpoint_epoch_003.pt")

    def test_epoch_numbers_are_zero_padded_so_ten_sorts_after_nine(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            model = _TinyModel(1.0)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            path = save_checkpoint(tmpdir, model, optimizer, epoch=9, history=[])
            self.assertEqual(path.name, "checkpoint_epoch_009.pt")


if __name__ == "__main__":
    unittest.main()
