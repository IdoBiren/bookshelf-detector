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
import sys
import time
from pathlib import Path

try:
    import torch
except ModuleNotFoundError as error:  # pragma: no cover - environment guard
    # torch lives in training/.venv, NOT in the system Python (which is 3.14
    # and also runs Label Studio -- the two were deliberately kept apart, see
    # HANDOFF.md). Running `python training/train.py` with the system
    # interpreter otherwise dies on a bare ModuleNotFoundError that says
    # nothing about which interpreter to use.
    raise SystemExit(
        f"{error}\n\n"
        "torch is installed in training/.venv, not in this interpreter\n"
        f"  (currently running: {sys.executable})\n\n"
        "Use the venv's Python instead:\n"
        "  training/.venv/Scripts/python.exe training/train.py --limit 20 --epochs 30\n\n"
        "In Colab this does not apply -- torch is already present, so plain\n"
        "`python training/train.py ...` is correct there."
    ) from error

from dataset import SpineDataset, collate_fn
from model import build_model, describe_model

CHECKPOINT_PATTERN = re.compile(r"^checkpoint_epoch_(\d+)\.pt$")


def save_checkpoint(
    checkpoint_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: list[dict],
    mask_resolution: int = 14,
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
            "mask_resolution": mask_resolution,
        },
        path,
    )
    return path


def read_checkpoint_mask_resolution(path: Path) -> int:
    """The mask_resolution a checkpoint's weights were built with, recorded
    at save time so a checkpoint is self-describing. Shapes are compatible
    across resolutions (mask_head/mask_predictor are conv/deconv layers, so
    load_state_dict does not raise on a mismatch) -- which means evaluating
    at the WRONG resolution fails silently, producing garbage instead of an
    error. Defaults to 14 for a checkpoint saved before this field existed,
    e.g. checkpoint_epoch_009.pt from the original pretrain run."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint.get("mask_resolution", 14)


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


def load_model_weights_only(path: Path, model: torch.nn.Module) -> None:
    """For --init-from: warm-start a NEW run from another run's weights,
    with a fresh optimizer and epoch counter reset to 0. Unlike
    load_checkpoint, deliberately ignores that checkpoint's
    optimizer_state_dict/epoch/history -- this is not resuming that run, it
    is starting a different one from its weights. Works across different
    mask_resolution values: mask_head/mask_predictor are conv/deconv layers,
    so their weight shapes depend on channel counts, not on the spatial size
    mask_roi_pool produces."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])


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

        # detach() before reading any of these as plain numbers. Without it
        # torch warns on every batch ("Converting a tensor with
        # requires_grad=True to a scalar"), and holding a grad-tracking
        # tensor in a running total would keep its graph alive for the whole
        # epoch. Safe here because backward() has already run.
        total_value = float(total.detach())
        running_loss += total_value
        batches += 1
        if index % log_every == 0:
            parts = "  ".join(f"{k}={float(v.detach()):.3f}" for k, v in losses.items())
            print(f"  batch {index:>5}  total={total_value:.4f}  {parts}", flush=True)

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
    log_every: int = 20,
    mask_resolution: int = 14,
    init_from: str | None = None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    dataset = SpineDataset(Path(coco_train), Path(images_dir))
    if limit:
        # Stage 5's overfit check (plan §9): a handful of images, loss -> ~0.
        dataset.entries = dataset.entries[:limit]
    print(f"dataset: {len(dataset)} images  stats={dataset.stats}")
    batches_per_epoch = (len(dataset) + batch_size - 1) // batch_size
    # Printed so a quiet stretch is readable as "still working" rather
    # than "hung" -- with 720 batches and log_every=20 there are ~36
    # lines per epoch, i.e. tens of seconds of silence between them.
    print(f"batches/epoch: {batches_per_epoch}  (a log line every {log_every} batches)")

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    # pretrained=False when warm-starting: init_from's state_dict overwrites
    # the whole model anyway, so downloading ~170MB of COCO weights first
    # just to discard them is wasted bandwidth.
    model = build_model(pretrained=init_from is None, mask_resolution=mask_resolution)
    print(f"model: {describe_model(model)}")
    if init_from:
        print(f"initializing weights from {init_from}  "
              "(fresh optimizer, epoch 0 -- this is NOT a resume)")
        load_model_weights_only(Path(init_from), model)
    model = model.to(device)

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(parameters, lr=learning_rate, momentum=0.9, weight_decay=0.0005)

    start_epoch, history = 0, []
    latest = find_latest_checkpoint(Path(checkpoint_dir))
    if latest:
        saved_epoch, history = load_checkpoint(latest, model, optimizer)
        start_epoch = saved_epoch + 1
        print(f"resuming from {latest.name} -> starting at epoch {start_epoch}")
    else:
        print("no checkpoint found -- starting from scratch"
              + (" (after --init-from)" if init_from else ""))

    for epoch in range(start_epoch, epochs):
        started = time.time()
        print(f"\n=== epoch {epoch}/{epochs - 1} ===", flush=True)
        mean_loss = train_one_epoch(model, optimizer, loader, device, log_every)
        elapsed = time.time() - started

        history.append({"epoch": epoch, "mean_loss": mean_loss, "seconds": elapsed})
        path = save_checkpoint(
            Path(checkpoint_dir), model, optimizer, epoch, history,
            mask_resolution=mask_resolution,
        )
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
    parser.add_argument("--num-workers", type=int, default=2,
                        help="0 is the safe value if the loader appears to hang (notebook "
                             "multiprocessing is the usual cause).")
    parser.add_argument("--log-every", type=int, default=20,
                        help="Batches between progress lines. Lower it to tell a slow "
                             "epoch apart from a hang.")
    parser.add_argument(
        "--mask-resolution", type=int, default=14,
        help="mask_roi_pool's output grid size (torchvision default 14). Raising it "
             "gives a spine's width more pixels to work with before mask_to_quad -- "
             "see model.build_model's docstring for the measurement this responds to. "
             "Recorded in every checkpoint this run saves, so evaluate.py can read it "
             "back instead of needing to be told correctly by hand.",
    )
    parser.add_argument(
        "--init-from", default=None,
        help="Warm-start from another checkpoint's weights: fresh optimizer, epoch 0 "
             "-- this is NOT --checkpoint-dir resume, it starts a different run. Works "
             "across a different --mask-resolution than that checkpoint was saved "
             "with (mask_head/mask_predictor weight shapes don't depend on it).",
    )
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
        log_every=args.log_every,
        mask_resolution=args.mask_resolution,
        init_from=args.init_from,
    )
