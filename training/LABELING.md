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

## Speed tip: use the keyboard, not the mouse, to select the label

Press **`1`** to select the "spine" label instead of clicking it — it's
bound as a hotkey in the config. Across ~1,200 expected spine instances
(200 images x ~6 spines), pressing one key between polygons instead of
reaching for the label pill every time adds up.

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

## First: confirm the scene grouping (do this before labeling anything else)

Every photo arrived as its own `sceneNNN.jpg`, which made 165 photos look
like 165 independent scenes. **They are not.** EXIF says all 165 were shot
in one 30-minute session, and 139 of the 164 gaps between consecutive shots
are 5 seconds or less — bursts of the same shelf. Grouped at 5s, there are
about **26 real scenes, not 165**.

This matters because near-duplicate shots of one shelf that land on opposite
sides of the split let the model half-recognise a shelf it already trained
on. Every validation number then reads high, and nothing in the output looks
wrong. It is the same failure that made harald-varner's public split
unusable (43% of its scenes spanned splits).

```bash
python training/group_scenes.py
```

That writes a proposal to `training/scene_groups.txt` and one **contact
sheet per proposed scene** to `data/merged/preview_scene_groups/`. Open that
folder and look at each sheet: every photo on one sheet should be the same
shelf.

- **Sheet mixes two shelves?** Split its line in `scene_groups.txt` in two.
- **Two sheets show the same shelf?** Join their lines into one.
- **Unsure?** Merge them. Over-merging costs a couple of independent scenes;
  under-merging is the actual bug.

Time is a strong hint, not proof — you can turn to the next shelf in five
seconds, and one proposed group here holds 43 photos, which is almost
certainly several shelves. Visual similarity was measured as an automatic
tie-breaker and is not good enough to trust: same-shelf pairs shot from
different angles have a median thumbnail correlation of only 0.11. Hence the
eye.

Nothing gets renamed — renaming would orphan the annotations already in
Label Studio, whose tasks key on filename. Pass the file to the converter
instead:

```bash
python training/convert_labelstudio_export.py --export <result.json> --scene-groups training/scene_groups.txt
```

## The eval set comes first, and it is chosen by hand (plan §8/§13)

Label these ~40 photos **before** the rest. Without an eval set there is no
way to tell whether the model is good, so it — not the fine-tune data — is
what blocks the ship decision. Fine-tune data can arrive in batches and the
model improves with each one.

Two things make an eval photo more expensive than a training photo:

1. **Stratified, not 40 random.** ~7 from each of the six scenarios in the
   table above. A random draw under-samples the hard cases and the benchmark
   comes out optimistically wrong.
2. **Type the title and author for every book**, by hand. That is the ground
   truth for §8's read-accuracy metric — the measurement that actually
   decides shipping.

### Declaring them: `--test-scenes`

Random splitting cannot express "these are the benchmark". Write the chosen
scene ids into a file, one per line:

```
# data/indomain/eval_scenes.txt — the frozen eval set (plan §8)
scene004
scene017
scene031
```

Pasted filenames work too (`scene004.jpg`, or Explorer's `scene004 (2).jpg`)
— they get normalised to the scene id. Then:

```bash
python training/convert_labelstudio_export.py --export <path-to-result.json> --test-scenes data/indomain/eval_scenes.txt
```

Those scenes become the test split **exactly**; everything else splits
85/15 train/val and `--test-fraction` is ignored. A scene id that is not in
the export — a typo, or simply not labeled yet — **raises** rather than being
skipped, because silently dropping one shrinks the benchmark below the 40–60
§8 requires while still printing success.

> **Freeze it.** Once the file is written and those scenes are labeled, don't
> add to it or swap entries based on how the model scores. A benchmark you
> edit after seeing results stops being a benchmark.

### Flag `unreadable` at least once, deliberately

The pairing logic behind the `unreadable` checkbox is unit-tested against a
synthetic case but **has never been verified on a real export** — the first
batch contained zero flagged spines. A stratified eval set will almost
certainly contain a worn or blank spine. The first time you check that box,
re-run the converter and confirm `unreadable: true` actually reached the
JSON.

## Export

Export from Label Studio (JSON, or COCO if your Label Studio version
offers it directly). Don't build the export→merge converter yet — that's
easier to get right against a real batch of your actual annotations than
against a guess, so it comes next once you have some labeled data to test
it on. Even 10–20 labeled images checked in is enough to build and verify
that converter against.
