"""
Flags likely-blurry photos in the in-domain set so a human can decide what to
do with them (plan §3). Does NOT delete anything — see the module's caller
in HANDOFF/the assistant's own safety rules: permanent deletion is something
the user does themselves. This script's job stops at proposing a list and,
optionally, moving candidates to a separate quarantine folder the user can
inspect and empty by hand.

Method: variance of the Laplacian (standard, cheap, no ML). A sharp photo has
strong edges everywhere, so the second derivative has high variance; a blurry
one is smooth almost everywhere, so the variance collapses. There is no
universal "blurry" threshold — it depends on scene content and resolution —
so this reports a sorted list with scores rather than a hard cutoff, and
lets a human pick where to draw the line by looking at a contact sheet of
the lowest-scoring photos.

Cross-checks against Label Studio's export (if given) so an already-labeled
photo is called out explicitly rather than silently quarantined out from
under its annotations — LS tasks key on filename.
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
from pathlib import Path

import cv2


def blur_score(path: Path, max_dim: int = 1024) -> float:
    """Higher = sharper. Downscales first so a photo's raw megapixel count
    doesn't dominate the score — variance of the Laplacian grows with detail
    density, and a bigger image simply has more edges to find."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"OpenCV could not read {path}")
    scale = max_dim / max(image.shape)
    if scale < 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


def labeled_filenames(export_path: Path) -> set[str]:
    """Basenames that already have at least one annotation in a Label Studio
    COCO export — the set a quarantine move should warn about, not silently
    include."""
    data = json.loads(export_path.read_text(encoding="utf-8"))
    annotated_image_ids = {a["image_id"] for a in data["annotations"]}
    names = set()
    for img in data["images"]:
        if img["id"] not in annotated_image_ids:
            continue
        parsed = urllib.parse.urlparse(img["file_name"])
        relative = urllib.parse.parse_qs(parsed.query).get("d", [img["file_name"]])[0]
        names.add(Path(relative.replace("\\", "/")).name)
    return names


def render_contact_sheet(paths: list[Path], scores: list[float], out_path: Path, thumb_width: int = 320) -> None:
    from PIL import Image, ImageDraw

    tiles = []
    for path, score in zip(paths, scores):
        with Image.open(path) as im:
            im = im.convert("RGB")
            height = max(1, round(im.height * thumb_width / im.width))
            tile = im.resize((thumb_width, height))
        draw = ImageDraw.Draw(tile)
        label = f"{path.name}  score={score:.1f}"
        draw.rectangle([0, 0, thumb_width, 22], fill=(0, 0, 0))
        draw.text((4, 4), label, fill=(255, 255, 0))
        tiles.append(tile)

    columns = min(4, len(tiles))
    rows = (len(tiles) + columns - 1) // columns
    cell_height = max(t.height for t in tiles)
    sheet = Image.new("RGB", (columns * thumb_width, rows * cell_height), (24, 24, 24))
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        sheet.paste(tile, (column * thumb_width, row * cell_height))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=85)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photos-dir", type=Path, default=repo_root / "data" / "indomain" / "photos")
    parser.add_argument(
        "--export",
        type=Path,
        default=None,
        help="Label Studio COCO export (result.json) — if given, already-labeled photos are flagged.",
    )
    parser.add_argument("--count", type=int, default=20, help="How many lowest-scoring photos to report.")
    parser.add_argument(
        "--sheet-out",
        type=Path,
        default=repo_root / "data" / "merged" / "preview_blurry" / "candidates.jpg",
        help="Contact sheet of the lowest-scoring photos, for a quick visual check.",
    )
    parser.add_argument(
        "--quarantine-dir",
        type=Path,
        default=None,
        help="If given, MOVE (not delete) the --count lowest-scoring photos here after you confirm the "
        "list. Nothing is ever deleted by this script.",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Required together with --quarantine-dir to actually move files, after reviewing the printed "
        "list and contact sheet on a prior dry run.",
    )
    args = parser.parse_args()

    photos = sorted(args.photos_dir.resolve().glob("*.jpg"))
    if not photos:
        raise SystemExit(f"No .jpg files in {args.photos_dir}")

    scored = sorted(((blur_score(p), p) for p in photos), key=lambda pair: pair[0])

    labeled = labeled_filenames(args.export.resolve()) if args.export else set()

    print(f"Photos: {len(photos)}")
    print(f"\n{args.count} lowest sharpness scores (lower = more likely blurry):")
    candidates = scored[: args.count]
    for score, path in candidates:
        flag = "  [ALREADY LABELED]" if path.name in labeled else ""
        print(f"  {score:>10.1f}  {path.name}{flag}")

    already_labeled_candidates = [p for _, p in candidates if p.name in labeled]
    if already_labeled_candidates:
        print(
            f"\nWARNING: {len(already_labeled_candidates)} candidate(s) already have Label Studio "
            "annotations. Moving them will orphan those annotations (LS tasks key on filename)."
        )

    render_contact_sheet([p for _, p in candidates], [s for s, _ in candidates], args.sheet_out.resolve())
    print(f"\nContact sheet: {args.sheet_out.resolve()}")
    print("Look at it before deciding — a low score can also mean a plain/out-of-focus-background")
    print("shot that is otherwise fine, not just genuine blur.")

    if args.quarantine_dir:
        if not args.confirm:
            print(
                f"\n--quarantine-dir given without --confirm: NOT moving anything. Review the list "
                f"and sheet above, then re-run with --confirm to move these {len(candidates)} files to "
                f"{args.quarantine_dir}."
            )
            return
        quarantine_dir = args.quarantine_dir.resolve()
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        for _, path in candidates:
            path.rename(quarantine_dir / path.name)
        print(f"\nMoved {len(candidates)} files to {quarantine_dir} (not deleted).")
        print("Delete them yourself from there if you're sure, or move them back to undo.")


if __name__ == "__main__":
    main()
