"""
Training loop for the Mask R-CNN spine detector (plan §13 phase D).

Lives in the repo, not in a notebook cell: the notebook stays thin —
clone, pip install, call `main()`. Model development inside a Colab cell is
how logic ends up unversioned and untested.

**Resume is built in from the start, not added later.** Colab's free tier
caps a session at ~4 hours and the pretrain run is expected to need ~6, so
the run only finishes if it can pick up where it stopped. Checkpoints go to
a directory that should be on mounted Drive; the latest is found by the
epoch number in the filename rather than by mtime, because Drive re-syncing
a file rewrites its mtime and would otherwise resume from the wrong epoch.

Usage (Colab):
    from train import main
    main(coco_train="data/merged/pretrain_train.json",
         images_dir="data/merged/images",
         checkpoint_dir="/content/drive/MyDrive/bookshelf-detector/checkpoints",
         epochs=10)

Stage 5 first, always (plan §9): overfit ~20 images until the loss goes to
~0. It costs minutes and catches target/loss bugs that would otherwise cost
six GPU-hours to discover.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch

from dataset import SpineDataset, collate_fn
from model import build_model, describe_model

CHECKPOINT_PATTERN = re.compile(r"^checkpoint_epoch_(\d+)\.pt$")


def save_checkpoint(
    checkpoint_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: list[dict],
) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # Zero-padded so plain lexicographic order agrees with numeric order --
    # otherwise checkpoint_epoch_10 sorts before checkpoint_epoch_9.
    path = checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
        },
        path,
    )
    return path


def load_checkpoint(
    path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None
) -> tuple[int, list[dict]]:
    """Returns (epoch of the saved checkpoint, history). Restores optimizer
    state too when one is given — resuming with a cold optimizer silently
    throws away momentum at every resume boundary."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint.get("history", [])


def find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Highest EPOCH NUMBER, not newest mtime — Drive re-syncs rewrite
    mtimes and would resume from the wrong place."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        return None
    best_epoch, best_path = -1, None
    for path in checkpoint_dir.iterdir():
        match = CHECKPOINT_PATTERN.match(path.name)
        if match and int(match.group(1)) > best_epoch:
            best_epoch, best_path = int(match.group(1)), path
    return best_path


def train_one_epoch(model, optimizer, loader, device, log_every: int = 20) -> float:
    model.train()
    running_loss, batches = 0.0, 0
    for index, (images, targets) in enumerate(loader):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        losses = model(images, targets)
        total = sum(losses.values())

        optimizer.zero_grad()
        total.backward()
        optimizer.step()

        running_loss += float(total)
        batches += 1
        if index % log_every == 0:
            parts = "  ".join(f"{k}={float(v):.3f}" for k, v in losses.items())
            print(f"  batch {index:>5}  total={float(total):.4f}  {parts}", flush=True)

    return running_loss / max(batches, 1)


def main(
    coco_train: str,
    images_dir: str,
    checkpoint_dir: str,
    epochs: int = 10,
    batch_size: int = 2,
    learning_rate: float = 0.005,
    limit: int | None = None,
    num_workers: int = 2,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    dataset = SpineDataset(Path(coco_train), Path(images_dir))
    if limit:
        # Stage 5's overfit check (plan §9): a handful of images, loss -> ~0.
        dataset.entries = dataset.entries[:limit]
    print(f"dataset: {len(dataset)} images  stats={dataset.stats}")

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    model = build_model(pretrained=True).to(device)
    print(f"model: {describe_model(model)}")

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(parameters, lr=learning_rate, momentum=0.9, weight_decay=0.0005)

    start_epoch, history = 0, []
    latest = find_latest_checkpoint(Path(checkpoint_dir))
    if latest:
        saved_epoch, history = load_checkpoint(latest, model, optimizer)
        start_epoch = saved_epoch + 1
        print(f"resuming from {latest.name} -> starting at epoch {start_epoch}")
    else:
        print("no checkpoint found -- starting from scratch")

    for epoch in range(start_epoch, epochs):
        started = time.time()
        print(f"\n=== epoch {epoch}/{epochs - 1} ===", flush=True)
        mean_loss = train_one_epoch(model, optimizer, loader, device)
        elapsed = time.time() - started

        history.append({"epoch": epoch, "mean_loss": mean_loss, "seconds": elapsed})
        path = save_checkpoint(Path(checkpoint_dir), model, optimizer, epoch, history)
        print(f"epoch {epoch}: mean_loss={mean_loss:.4f}  {elapsed:.0f}s  -> {path.name}", flush=True)

    history_path = Path(checkpoint_dir) / "history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"\ndone. history -> {history_path}")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-train", default=str(repo_root / "data" / "merged" / "pretrain_train.json"))
    parser.add_argument("--images-dir", default=str(repo_root / "data" / "merged" / "images"))
    parser.add_argument("--checkpoint-dir", default=str(repo_root / "data" / "merged" / "checkpoints"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--limit", type=int, default=None, help="Stage 5 overfit check: use only N images.")
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    main(
        coco_train=args.coco_train,
        images_dir=args.images_dir,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        limit=args.limit,
        num_workers=args.num_workers,
    )
