# Labeling the in-domain dataset (plan §3, Stage 3)

## Setup

```bash
pip install label-studio
label-studio start
```

Create a new project, import the images, then go to
**Settings → Labeling Interface → Code** and paste in
[`labelstudio_config.xml`](labelstudio_config.xml).

## Before you upload: name files by scene

This is not cosmetic. It's the exact same fix that was needed for the
public data (`training/merge_datasets.py`): if two photos of the *same*
physical shelf end up in different splits, validation numbers stop
meaning anything, because the model can partly recognize the shelf it
already saw in training. One scene = the same physical shelf setup, even
across multiple shots (different angle, retake, etc.) — every photo of
that shelf needs to share the same `sceneNN` prefix.

**Easiest way — Windows Explorer's bulk-rename, not manual typing per file:**

1. Put all your photos in one folder (copy them together first if they
   came from different phones).
2. Select all the photos of one shelf (Ctrl+click each one).
3. Press `F2`, type `scene01`, press Enter. Explorer renames them to
   `scene01.jpg`, `scene01 (2).jpg`, `scene01 (3).jpg`, ... automatically.
4. Select the next shelf's photos, repeat with `scene02`, then `scene03`,
   and so on.

The exact suffix format doesn't matter (`scene01 (2).jpg` from Explorer's
auto-numbering is fine) — what matters is that every photo of the *same*
shelf starts with the same `sceneNN`.

**"Same shelf" means "same shelf location," but that's not quite the right
test if you rearranged books between shots.** What actually matters is how
*similar the photo's content* is — a model can only leak information from
a shot it could partially recognize:

- Shelf reshot with a **few** books added/removed/rearranged, still mostly
  the same spines → **same** scene number.
- Shelf reshot with **most** books swapped out, so it looks substantially
  different → fine to use a **new** scene number, even at the same location.
- **Not sure which side it's on? Use the same number.** Grouping too much
  costs nothing (just slightly fewer independent scenes). Grouping too
  little is the actual bug — that's exactly how a near-duplicate shot ends
  up in both train and validation.

## What to shoot — in this order (plan §3)

Prioritize failure modes over "normal shelf" shots. ~25 images per row:

| # | Scenario | Why |
|---|---|---|
| 1 | **Angled photos** (30–45° off perpendicular) | The default for a real user, and what the public data (shot straight-on) doesn't cover. |
| 2 | **Leaning / tilted books** | The entire justification for using quads instead of boxes. |
| 3 | **Thin + wide spines in the same frame** | Pocket book next to an album — must be in one photo, not separate ones. |
| 4 | **Warm/tungsten lighting, partial shadow** | Public data is shot in good library lighting. |
| 5 | **Books lying flat on others, mixed heights** | Common at home, rare in libraries. |
| 6 | **Non-book objects on the shelf** (plant, frame, box) | Source of false positives. |

Use multiple phones, not just one. Include some shelves that aren't yours.

## Labeling rules (already decided, plan §3 — don't re-derive per image)

- **Trace the polygon exactly to the spine's edges** — not generous, not
  tight. The edge you draw is what determines whether a letter at the
  edge of the title gets cut off later. Padding is added later as a
  calibrated post-processing parameter (plan §7), not here.
- A book **lying flat on others** → still labeled, its quad is just
  horizontal.
- A spine **cut off at the frame edge** → labeled by its visible portion.
- A **partially occluded** spine → labeled by its visible portion.
- **Non-book objects** → not labeled. (Single class for now; a `non_book`
  class only gets added if evaluation later shows false positives are an
  actual problem.)
- A spine with **no readable text** (very worn, or a blank/plain spine) →
  still labeled as `spine`, but check the **`unreadable`** box for it.
  This flag is what keeps the downstream read-accuracy metric (plan §8)
  from penalizing the detector for a spine no model could ever read.

## Export

Export from Label Studio (JSON, or COCO if your Label Studio version
offers it directly). Don't build the export→merge converter yet — that's
easier to get right against a real batch of your actual annotations than
against a guess, so it comes next once you have some labeled data to test
it on. Even 10–20 labeled images checked in is enough to build and verify
that converter against.
