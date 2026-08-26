"""
Proposes scene groupings for the in-domain photos, and renders a contact
sheet per proposed group so a human can confirm or correct them (plan §3,
LABELING.md "name files by scene").

Why this exists: every photo arrived named sceneNNN.jpg, one id each — so
165 photos looked like 165 independent scenes. They are not. EXIF says all
165 were shot in a single 30-minute session and 139 of the 164 gaps between
consecutive photos are <=5 seconds, i.e. burst shots of the same shelf.
Left alone, near-duplicate shots of one shelf land on opposite sides of the
train/val/test split, which is exactly the leakage that made harald-varner's
own public split unusable (43% of its scenes spanned splits).

Why it PROPOSES rather than decides: time is a strong signal but not proof —
you can turn to the next shelf in five seconds. Visual similarity was
measured as a possible tie-breaker and is not discriminative enough to
automate: downscaled-thumbnail correlation on same-shelf pairs has a median
of only 0.11, with plenty of negatives, because a large viewpoint change
destroys pixel correspondence. Colour histograms separate a little better
(0.96 vs 0.87 median) but nowhere near cleanly. So the machine does the
tedious part — reading timestamps, clustering, laying out the sheets — and
the eye makes the call.

Nothing is renamed. Renaming would orphan the annotations already made in
Label Studio, whose tasks key on filename. The output is a mapping file that
convert_labelstudio_export.py reads via --scene-groups.

Usage:
    python training/group_scenes.py                 # propose + contact sheets
    python training/group_scenes.py --gap-seconds 10
    # then edit training/scene_groups.txt by hand, and:
    python training/convert_labelstudio_export.py --export result.json \
        --scene-groups training/scene_groups.txt
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from PIL import Image, ExifTags

_DATETIME_ORIGINAL = {v: k for k, v in ExifTags.TAGS.items()}["DateTimeOriginal"]
_EXIF_IFD = 0x8769


def read_capture_time(path: Path) -> datetime | None:
    """EXIF DateTimeOriginal, or None if the photo carries no timestamp.
    Lives in the Exif sub-IFD, not the top-level tags — getexif() alone
    returns nothing useful here."""
    with Image.open(path) as im:
        exif = im.getexif()
        if not exif:
            return None
        sub_ifd = exif.get_ifd(_EXIF_IFD)
        raw = sub_ifd.get(_DATETIME_ORIGINAL) if sub_ifd else None
    if not raw:
        return None
    return datetime.strptime(raw, "%Y:%m:%d %H:%M:%S")


def group_by_time_gap(
    timed_photos: list[tuple[Path, datetime]], gap_seconds: float
) -> list[list[Path]]:
    """Splits a chronologically sorted photo list wherever the gap to the
    previous shot exceeds gap_seconds. Consecutive-only by design: a shelf
    revisited later in the session is a genuinely separate opportunity to
    reshoot it, and merging across a long gap on a hunch is the kind of
    silent over-merge that costs training data for nothing."""
    groups: list[list[Path]] = []
    previous: datetime | None = None
    for path, taken in sorted(timed_photos, key=lambda pair: pair[1]):
        if previous is None or (taken - previous).total_seconds() > gap_seconds:
            groups.append([])
        groups[-1].append(path)
        previous = taken
    return groups


def write_groups_file(groups: list[list[Path]], out_path: Path, gap_seconds: float) -> None:
    lines = [
        "# Scene groups for the in-domain photos (plan §3).",
        "# One line per scene = the photos of ONE physical shelf.",
        "# The first entry names the scene; the rest join it. Comma-separated.",
        "#",
        f"# PROPOSED automatically from EXIF capture times, gap > {gap_seconds:g}s = new scene.",
        "# This is a starting point, NOT a verdict — check the contact sheets and",
        "# edit freely. Merging too much only costs a few independent scenes;",
        "# merging too little puts near-duplicate shots on both sides of the",
        "# split, which silently inflates every validation number.",
        "",
    ]
    for group in groups:
        # Comma-separated: Explorer's bulk-rename numbering puts a space
        # inside the stem ("scene008 (2)"), so whitespace cannot separate them.
        lines.append(", ".join(p.stem for p in group))
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_contact_sheet(
    group: list[Path], out_path: Path, thumb_width: int = 320, columns: int = 4
) -> None:
    """One image per proposed scene, so the question 'is this all the same
    shelf?' can be answered at a glance instead of by opening N files."""
    thumbs = []
    for path in group:
        with Image.open(path) as im:
            im.draft("RGB", (thumb_width * 2, thumb_width * 2))  # DCT-scaled decode, ~10x faster
            im = im.convert("RGB")
            height = max(1, round(im.height * thumb_width / im.width))
            thumbs.append(im.resize((thumb_width, height)))

    columns = min(columns, len(thumbs))
    rows = (len(thumbs) + columns - 1) // columns
    cell_height = max(t.height for t in thumbs)
    sheet = Image.new("RGB", (columns * thumb_width, rows * cell_height), (24, 24, 24))
    for index, thumb in enumerate(thumbs):
        row, column = divmod(index, columns)
        sheet.paste(thumb, (column * thumb_width, row * cell_height))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=85)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photos-dir", type=Path, default=repo_root / "data" / "indomain" / "photos")
    parser.add_argument(
        "--out-file",
        type=Path,
        # training/, not data/indomain/: this is a decision about the dataset,
        # not the dataset itself, and it belongs in version control. It holds
        # scene ids only — no photo content, nothing identifying — so the
        # gitignore rule protecting the photos stays untouched.
        default=repo_root / "training" / "scene_groups.txt",
    )
    parser.add_argument(
        "--sheets-dir",
        type=Path,
        # Under data/ on purpose: these are crops of photos of real homes and
        # data/ is gitignored. Never write them anywhere tracked.
        default=repo_root / "data" / "merged" / "preview_scene_groups",
        help="Where to write one contact sheet per proposed scene.",
    )
    parser.add_argument(
        "--gap-seconds",
        type=float,
        default=5.0,
        help="A gap larger than this between consecutive shots starts a new scene (default: 5).",
    )
    parser.add_argument("--no-sheets", action="store_true", help="Write the groups file only.")
    args = parser.parse_args()

    photos = sorted(args.photos_dir.resolve().glob("*.jpg"))
    if not photos:
        raise SystemExit(f"No .jpg files in {args.photos_dir}")

    timed, undated = [], []
    for path in photos:
        taken = read_capture_time(path)
        (timed.append((path, taken)) if taken else undated.append(path))

    if undated:
        # Loud, not silent: an undated photo cannot be placed and would
        # otherwise quietly become its own scene.
        print(f"WARNING: {len(undated)} photo(s) have no EXIF timestamp and were left ungrouped:")
        for path in undated[:10]:
            print(f"  {path.name}")

    groups = group_by_time_gap(timed, args.gap_seconds)
    groups.extend([path] for path in undated)

    write_groups_file(groups, args.out_file.resolve(), args.gap_seconds)

    sizes = [len(g) for g in groups]
    print(f"Photos: {len(photos)}")
    print(f"Proposed scenes: {len(groups)}  (gap > {args.gap_seconds:g}s)")
    print(f"Photos per scene: min={min(sizes)} max={max(sizes)} mean={sum(sizes)/len(sizes):.1f}")
    print(f"Wrote {args.out_file}")

    if not args.no_sheets:
        sheets_dir = args.sheets_dir.resolve()
        for group in groups:
            render_contact_sheet(group, sheets_dir / f"{group[0].stem}.jpg")
        print(f"Wrote {len(groups)} contact sheets to {sheets_dir}")
        print("\nOpen that folder and check each sheet: every photo on one sheet should be")
        print("the SAME shelf. Split a sheet that mixes shelves, merge two that don't.")


if __name__ == "__main__":
    main()
