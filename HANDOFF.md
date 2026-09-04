# Handoff — browser book-spine detector

**Repo:** https://github.com/IdoBiren/bookshelf-detector · **Branch:** `main` · **Last commit:** `8ff3781`

Full design spec (architecture rationale, licensing verdict, evaluation
protocol, ship thresholds) lives in the plan file at
`C:\Users\User\.claude\plans\inherited-jumping-sunrise.md`. Section refs
below (§0–§12) point there. **This doc is the "where are we / what will
bite you" companion, not a substitute.**

---

## What this is

A book-spine detector for a neighborhood-library app. User photographs a
home shelf (~5–8 books, Hebrew titles); the detector returns **where each
spine is** so each one can be cropped, rotated to horizontal, and read by
a cloud VLM. **The detector does positions only** — text reading is a
separate, already-working cloud call.

Hard constraints that drove every decision (§0–§1):
- **Runs in the browser**, not a server. React + Vite (Lovable) + Supabase.
  Train in Colab → ONNX → onnxruntime-web (WebGPU, WASM fallback).
- **≤5MB** model after quantization.
- **Permissive license only** — must ship in a closed-source app.
- One-shot, ~2s on a mid-range phone is acceptable.

### The three decisions worth not re-litigating

1. **No Ultralytics, ever** (AGPL-3.0 — even "just for a test", since
   weights trained in an AGPL pipeline are a derivative). **RF-DETR is
   out too — on size, not license**: nano is 30.5M params ≈ 30MB int8,
   6× over budget. Verified-permissive and small enough: YOLOX
   (Apache-2.0, Nano = 0.91M), MMRotate/RTMDet-R (Apache-2.0), or own
   code on a torchvision backbone (BSD-3).
2. **Output is a quad (4 free corners), not a box and not an OBB.** A
   200×30px spine tilted 15° has an axis-aligned box that's ~82px wide —
   **~64% of it is the neighbours' text**, so the VLM reads a blended
   title. And a *rotated rect* isn't enough either: shot at an angle, a
   spine is a perspective **trapezoid**. This is now measured, not just
   argued — see "minAreaQuad" below.
3. **DBNet-style segmentation → quad**, ~2.5M params, self-implemented in
   PyTorch. Books on a shelf are the same problem as tightly-packed text
   lines: naive binary masks fuse touching instances, and DBNet's
   shrink-mask + unclip is the direct fix. (§1 has the full "why not
   MMRotate / PaddleOCR" reasoning.)

---

## Status

| Stage (§9) | State |
|---|---|
| 0 · Dataset verification | ✅ Done |
| 1 · Contract + browser geometry | ✅ Done, 22/22 tests |
| 2 · Merge + augmentation | ✅ Done, 38/38 tests, verified on real data |
| 3 · Label 200 own photos | 🟡 **In progress — 17 of 165 photos labeled** |
| 4 · Track-A baseline (YOLOX-Nano end-to-end) | ⬜ Not started — needs Colab GPU |
| 5 · Overfit 20 images (Track B) | ⬜ Not started — needs GPU |
| 6 · Pretrain + fine-tune | ⬜ Not started — needs GPU |
| 7 · Calibrate shrink/unclip | ⬜ Blocked on 6 |
| 8 · ONNX export + parity | ⬜ Blocked on 6 |
| 9 · Worker integration (+9a/9b/9c gates) | ⬜ Blocked on 8 |
| 10 · Crop-padding calibration | ⬜ Blocked on 9 |
| 11 · Full evaluation vs ship thresholds | ⬜ Blocked on 10 |

**This table predates plan §13's phase A-E restructure — phase B (target
generation + shrink_ratio validation, see "Open questions" §1 above) is
done and landed a NO-GO finding that phase C/D should read before
proceeding. Stage 7's calibration is no longer a blind sweep — the
`unclip_ratio` finding above feeds it directly.**

**Tests, both green as of this handoff:**
```bash
npx vitest run                                         # 34/34 (browser/TS)
cd training && python -m unittest discover -s tests    # 176/176 (pipeline/Python)
#   ^ needs training/.venv/Scripts/python.exe -- the system 3.14 has no torch,
#     so test_dataset/test_model/test_train fail to import and you see 146.
```

### ⚠️ Two Pythons on this machine — pick the right one

- **System Python 3.14** runs Label Studio and the stdlib-only data
  pipeline. It has no `torch`.
- **`training/.venv`** has torch/torchvision/onnx/onnxruntime. Everything
  that touches the model needs it:

```bash
training/.venv/Scripts/python.exe training/train.py --limit 20 --epochs 30
training/.venv/Scripts/python.exe training/evaluate.py --checkpoint <path>
```

They were split deliberately, to keep torch away from the working
label-studio install. `train.py` and `evaluate.py` now fail with an
explicit message naming the right interpreter rather than a bare
`ModuleNotFoundError: No module named 'torch'`, which is what actually
happened the first time. **In Colab none of this applies** — torch is
already there, so plain `python training/train.py ...` is correct.

### Data on disk (all gitignored — never commit)

| Path | Contents |
|---|---|
| `data/raw/` | 3 public Roboflow datasets, as downloaded |
| `data/merged/pretrain_{train,val}.json` | 1440 / 147 images · 27109 / 2294 annotations |
| `data/merged/indomain_{train,val,test}.json` | 12 / 3 / 2 images · 131 / 36 / 13 annotations |
| `data/indomain/photos/` | **165 own photos**, 17 labeled so far |

⚠️ **All 165 photos are `sceneNNN.jpg`, one photo per scene id** — 165
distinct scenes, zero multi-shot groups. LABELING.md's scene-grouping rule
assumed several shots would share a prefix. If every photo really is a
different shelf, nothing is wrong and splitting is trivially leak-free. If
any two of them are the same shelf from different angles, they are currently
in *different* scenes and can land on opposite sides of the split — which is
precisely the leakage the mechanism exists to stop (it made 43% of
harald-varner's public split unusable). **Cheap to fix by renaming now,
painful after labeling**, since Label Studio tasks key on filename.

---

## The immediate next step

**Label the ~40-photo eval set first** — hand-picked and stratified across
§3's six scenarios, not the next 40 in order. It is what blocks the ship
decision; the remaining ~125 fine-tune photos get **pre-annotated by the
pretrained model** from phase D and only reviewed by hand (decided
2026-08-26). Declare the chosen scenes with `--test-scenes` — see
LABELING.md, "The eval set comes first".

**Do not pre-annotate the eval set itself.** A wrong polygon is easy to
correct; a *missing* one is nearly invisible to a reviewer, so the model's
own blind spots get baked into the ground truth and recall reads higher
than it is — in the exact metric that decides shipping.
Photos are already shot and scene-named; only annotation remains.
`training/LABELING.md` has the rules (trace tight to spine edges; label
books lying flat and edge-cropped spines; skip non-books; flag genuinely
unreadable spines). Press `1` to reselect the spine label between polygons.

Then re-export and re-run — the converter is idempotent and picks up
however many scenes exist:

```bash
python training/convert_labelstudio_export.py --export <path-to-result.json>
python training/preview_annotations.py --prefix indomain --split train --count 5 --out-dir data/merged/preview_indomain
```

**Everything past Stage 3 needs a Colab GPU**, which no one has run yet.
Stage 4 (YOLOX-Nano, axis-aligned, ~5h on free T4) exists specifically to
prove the *whole chain* — export → quantize → browser → VLM — on a
trivial model before betting days on the real one. Don't skip it.

---

## Gotchas — the expensive-to-rediscover part

### Label Studio

- **Browser file upload is broken. Don't debug it — route around it.**
  `POST /api/projects/N/import` dies with Django's
  `RawPostDataException: You cannot access body after reading from
  request's data stream`. It's an [open upstream bug](https://github.com/HumanSignal/label-studio/issues/6794),
  reproduced here on Django 5.1.15 / DRF 3.15.2. **Disabling Sentry did
  not fix it** (a plausible-looking hypothesis that turned out wrong —
  Sentry's Django integration reads the body early, but the crash persists
  without it). The likely real culprit is CSRF middleware doing the same.
  **Use Local Files Storage instead** — it reads from disk and never
  touches the HTTP-body path.
- **Start it like this** (the `django-environ` upgrade is required —
  bundled 0.10.0 calls `pkgutil.find_loader`, removed in Python 3.12+):
  ```powershell
  python -m pip install --upgrade django-environ
  $env:LOCAL_FILES_SERVING_ENABLED = "true"
  $env:LOCAL_FILES_DOCUMENT_ROOT = "C:\Users\User\OneDrive\Documents\Code\bookshelf-detector\data\indomain"
  & "$env:LOCALAPPDATA\Python\pythoncore-3.14-64\Scripts\label-studio.exe" start
  ```
  pip warns that label-studio pins `django-environ==0.10.0`; the newer
  version works anyway — verified by actually booting the server.
  `label-studio` isn't on PATH, hence the full path.
- **`LOCAL_FILES_DOCUMENT_ROOT` must be the *parent*** of the folder you
  point a storage at — Label Studio refuses a storage path equal to the
  document root. Hence root = `data/indomain`, storage path =
  `data/indomain/photos`.
- **Tick the "interpret objects as BLOBs / generate URLs" checkbox** on
  the storage (field `use_blob_urls`, default off). Without it, sync
  rejects `.jpg` with "only .json/.jsonl/.parquet can be processed".
- The **config editor's live preview always shows a generic sample image
  with red rectangles**, regardless of your config. It's not rendering
  your polygon setup and it is not a signal. Also expect a harmless
  `cannot pickle 'itertools.count'` in the log — it fires for Label
  Studio's own built-in templates too.
- **`unreadable` exports as a separate annotation with its own category**,
  geometry-identical to the spine it belongs to — *not* as a field.
  `convert_labelstudio_export.py` pairs them back by
  `(image_id, exact segmentation)` into one `unreadable: bool`.
  ⚠️ **Zero unreadable markers have been observed in a real export yet**
  (the first batch had none). The pairing is unit-tested against a
  synthetic case but **never validated end-to-end on real output** — check
  it explicitly the first time a spine actually gets flagged.

### Python 3.14 (this machine)

- **Working, verified by running them:** Pillow 12.3.0, Albumentations
  2.0.8, numpy 2.5.2, opencv 5.0.0, scipy, label-studio 1.23.0.
- **PyTorch and onnxruntime DO have Python 3.14 wheels** — the earlier
  "may need a 3.11/3.12 env" worry is resolved, verified against the index:
  `torch 2.13.0` (cp314), `torchvision 0.28.0` (cp314),
  `onnxruntime 1.29.0` (cp314), `onnx 1.22.0` (cp312-abi3, so it loads on
  3.14). No downgrade needed anywhere, including for the §5 parity check.
- **They still live in `training/.venv`, not the system Python** — for a
  different reason than wheels: keeping torch away from the working
  label-studio install, which was expensive to get running and pins its own
  dependencies. `training/requirements-train.txt` is separate from
  `requirements.txt`; `merge_datasets.py` stays stdlib-only.
- **This machine's network drops large downloads.** pip died repeatedly
  mid-wheel with `Failed to resolve files.pythonhosted.org` after ~3MB of
  122MB, and albumentations' version check times out its SSL handshake.
  DNS and ranged GETs both work — it is flakiness, not a block. For big
  wheels, `curl -L -C - --retry 15 --retry-all-errors` to disk and then
  `pip install <file>` rather than fighting pip's own retry logic.
- `merge_datasets.py` is deliberately **stdlib-only** so the data pipeline
  never shares that risk. Keep it that way.

### Data findings that contradict the original notes

- **The Cowork dataset report was wrong on counts**, in both directions:
  leo-ueno is 100 images (reported 193), woody-willis is 72 (reported 28),
  harald-varner is 1,461 (reported 1,463 ✓). Total landed at 1,633 —
  coincidentally near the original ~1,600 estimate, for the wrong reasons.
- **Real scene diversity is ~696, not 1,633.** All three datasets ship
  pre-baked augmented duplicates (Roboflow's `.rf.<hash>` suffix). If
  Stage 6 mAP disappoints, *this* is the likelier explanation than "200
  own photos isn't enough".
- **harald-varner's own train/valid/test split leaks: 43% of its scenes
  had augmented copies on both sides.** Both pipeline scripts therefore
  ignore the provided splits entirely and re-split by scene id derived
  from the filename. Preserve that behaviour in anything new.
- **90% of pretrain data is one dataset** (harald-varner). A ~20-image
  manual spot-check looked clean, but that's a small sample of 1,461 — a
  systematic bias there propagates everywhere.
- woody-willis carries a stray `dvd_spine` category (9 annotations),
  filtered out at merge.

### Code specifics

- **`minAreaQuad` greedy reduction must reject any candidate that
  *decreases* area** (`increase < -1e-6`). Extending two hull edges the
  wrong way produces a self-intersecting bowtie whose shoelace area comes
  out *smaller* than the hull's, so a naive "pick the minimum increase"
  search actively prefers the broken candidate. Caught by the trapezoid
  test (720 vs a hull area of 3280) before it ever saw real data.
- **Connected components uses 4-connectivity, not 8.** 8-connectivity
  bridges single diagonal pixel touches and re-fuses spines that the
  shrink-mask deliberately separated — defeating the whole §1 mechanism.
  There's a regression test for exactly this.
- **§1's quad-vs-rect claim is now quantified**, not just asserted: on an
  exact trapezoid, `minAreaQuad` reproduces the analytic area (3200) to 6
  decimals while `minAreaRect` necessarily overshoots by ≥10%.
- **Albumentations `RandomResizedCrop` can only zoom *in*.** For the
  bidirectional 0.5–1.6× scale jitter §3 requires (pocket paperback beside
  a wide album), use `RandomScale` + `RandomCrop(pad_if_needed=True)`.
- **Polygons have 5–9+ vertices in real data, not a fixed 4.** They travel
  through Albumentations as flattened keypoints plus a per-annotation
  vertex count, with `remove_invisible=False` so a crop can't silently
  drop a vertex.
- `write_merged_dataset` / `build_coco_annotation` are **shared** by both
  the pretrain merge and the in-domain converter, on purpose — bbox/area
  computation must not drift between them.
- `detectSpines.ts` takes an injectable `ModelRunner`, so the real ONNX
  session drops in behind the same interface with zero geometry changes.
  `contract.test.ts` must keep passing untouched when it does.

---

## Stage 5 (overfit check) — run, and what it found

Ran 2026-09-01: 20 images from `pretrain_train`, 30 epochs, CPU, no
augmentation. **Loss 3.67 → 0.638.** Evaluated back on those same 20 images
(§9's actual requirement is "the model reproduces the labels of those 20",
not merely "the loss went down"):

```
mAP@50 = 0.5642   mAP@50:95 = 0.3424
thin AP@50=0.359 (n=96) · medium=0.324 (n=95) · wide=0.709 (n=96)
```

**The chain works.** Per-instance mask IoU against ground truth is
**0.86–0.96**, and quad IoU tracks it within ~3% — so `mask_to_quad.py` is
not where quality is lost, and train→checkpoint→inference→quads→mAP runs
end to end.

**The mAP gap is axis-aligned NMS suppressing adjacent spines.** The densest
image (25 spines) returned only 12 detections at the default
`box_nms_thresh=0.5`; raising it to 0.7 returned 29. The detections existed
and were being discarded.

This is §1's own argument surfacing somewhere new. §1 says a spine tilted
15° has an AABB that is ~64% its neighbours' content — which is exactly why
the *output* is a quad. But **torchvision's Mask R-CNN runs its NMS on
axis-aligned boxes**, so for tightly-packed tilted spines it sees heavy box
overlap between genuinely distinct books and suppresses the neighbour.

Not a hard ceiling, and **not yet a verdict**:
- `nms_thresh=0.7` recovers the missing ones but over-detects elsewhere
  (14 GT → 28 predictions), so the threshold is a trade, not a fix.
- The real fix if it persists is **mask-based NMS** — suppress on mask/quad
  IoU rather than box IoU, the same substitution `evaluate.py` already
  makes for its metrics.
- A model trained for 30 epochs on 20 CPU images produces sloppy boxes, and
  sloppy boxes make NMS behave worse than it will after real training. Check
  whether this survives the full Colab run before building anything.

Width breakdown is consistent with the same cause: thin and medium spines
pack closer together than wide ones, so they lose more to NMS.

### Correction, 2026-09-04: the attribution above is confounded

Read this before acting on the NMS conclusion. Two parts of it have held up
differently.

**Still solid:** raising `box_nms_thresh` 0.5 → 0.7 took the densest image
from 12 detections to 29. Both runs applied the same score filter, so that
delta is purely an NMS effect. NMS *is* suppressing real detections.

**Confounded:** the claim that NMS explains the *size* of the mAP gap.
`evaluate.py` was discarding every detection below `--score-threshold 0.5`
before AP was computed, while `average_precision` divides by the full
ground-truth count — so dropped detections cap recall, and each of the 101
recall points above the cap contributes 0.0 to the mean. AP was bounded by
achieved recall. The model emits down to `box_score_thresh=0.05`, so
everything from 0.05 to 0.5 was produced and then binned by our own eval.

Both causes are recall losses, so they are **additive, not rival** — the
open question is the split between them, not which one is real. Stage 5's
0.5642 and the full run's 0.5585 are therefore both understated by an
unknown amount, and neither is a quality ceiling.

`9cd50dc` reports `recall@50` overall and per band to separate them, and
caches the per-image IoU matrix so running at 0.05 is affordable (the
threshold sweep alone was re-rasterizing every polygon pair ten times;
measured 9.8x). The two-point experiment — same checkpoint, same data,
`--score-threshold 0.5` then `0.05` — is what settles it. The 0.5 run must
reproduce 0.5585 as a control.

**Do not build mask-based NMS until that split is known.** The gate in the
previous session's handoff ("check whether this survives the full Colab
run") was passed — 0.5585 held up on held-out data — but passing it no
longer means what it was taken to mean.

---

## Open questions for whoever picks this up

1. ~~**`shrink_ratio` / `unclip_ratio` are tuned for text lines, not books**~~
   **— MEASURED, plan §13 phase B. Verdict: NO-GO on `shrink_ratio=0.4` as
   shipped, at every stride tested.** `training/measure_shrink_ratio.py`
   (`training/dbnet_targets.py` + `training/polygon_offset.py`, 112 tests)
   ran the two defined metrics on real annotations, quad-mode (the shape
   `postprocess.ts:64` actually expands):

   | dataset | stride | canvas | merged pairs | vanished | verdict |
   |---|---|---|---|---|---|
   | indomain_train (12 img/131 ann) | 1 | 640² | 0.00% | **1.53%** | NO-GO |
   | indomain_train | 4 | 160² | 0.00% | **1.53%** | NO-GO |
   | pretrain_train (1440 img/27109 ann) | 1 | 640² | 0.56% | **1.59%** | NO-GO |
   | pretrain_train | 4 | 160² | **6.56%** | **5.56%** | NO-GO |

   Thresholds: merged pairs >5%, vanished >1%.

   **Two separate, independent problems, not one:**
   - **A real geometric bug, present even at full 640×640 resolution
     (stride 1).** Uniform per-edge shrink (the DBNet formula, mirrored
     exactly from `unclip.ts` — not a training-side reimplementation bug)
     self-intersects on *tapering* quads: a trapezoid whose two ends have
     different widths, where the narrower end's width is less than
     roughly `2 × shrink_distance`. Confirmed on real spines in both
     datasets, e.g. an indomain quad with edges `[365, 14, 362, 35]` —
     the 14px end inverts under a ~3px shrink applied from both long
     sides. This is a genuine property of the shared shrink/unclip
     algorithm on non-parallel-sided quads, not a resolution artifact —
     it will not go away by picking a different stride. An earlier
     informal exploration during planning reported 0.00% vanished at
     stride 1 for `pretrain`; that number is superseded — it did not
     include a self-intersection check (`is_simple_polygon`), only area.
   - **A resolution problem, additional to the above, at stride 4.**
     Merged pairs jump from 0.56%→6.56% and vanished from 1.59%→5.56%
     between stride 1 and stride 4 on `pretrain` — DBNet's usual
     1/4-resolution head is not viable for book spines; **stride 1 (full
     640×640 target maps) is required**, confirming §13's original
     resolution concern.

   **Not yet resolved, needs a decision before Phase C:** since the
   vanish/merge rate is small (~1.5%) and driven by shape (tapering),
   not systemic geometry breakage, the pragmatic options are (a) accept
   it — the code already turns a failed polygon into an ignore region,
   not a crash or corrupted target, so ~1.5% fewer supervised pixels per
   epoch may simply be acceptable; or (b) a shape-aware shrink (smaller
   effective ratio, or width-relative rather than uniform) for quads
   whose two end-widths differ significantly. Whoever picks up Phase C
   should decide which, and should NOT change `polygon_offset.py`'s core
   offset algorithm to "fix" this — it is a deliberate mirror of the
   shipped `unclip.ts`, and diverging breaks the shrink/unclip inverse
   property the whole module exists to guarantee.

   Also measured in the same pass: `postprocess.ts`'s shipped
   `unclip_ratio=1.5` is **not** the inverse of `shrink_ratio=0.4` for
   spine aspect ratios (a 66×355 quad — the median indomain spine size at
   640 input — round-trips to >5px corner error). The exact inverse is
   aspect-ratio-dependent (`training/polygon_offset.py`'s
   `exact_unclip_distance`, closed-form, 0.0 error on real quads) — a
   single scalar `unclip_ratio` cannot be exactly correct for both a
   200×200 and a 25×400 spine simultaneously. Relevant to Stage 7
   calibration, not yet acted on.
2. **int8 on WebGPU is unverified** (§5). If it silently falls back to CPU,
   p95 blows past 2.5s and we're stuck with fp16 at ~5MB — which turns the
   2.5M-param budget from a guideline into a hard ceiling.
3. **Is 200 own photos enough?** Judgment, not arithmetic. Expected weakest
   cases are sharp-angle shots and books lying flat. Remedy is cheap and
   targeted: +100 photos *of the specific failing mode*, not 100 random.
4. **The privacy claim must stay accurate.** Local inference keeps the
   *full frame* on-device, but crops still go to the cloud VLM, and the
   fallback path uploads the whole photo. Never phrase this to users as
   "photos never leave your device" (§Context, §6.4, §11).
5. **Whoever owns the Python retraining loop owns the improvement loop.**
   Lovable can't touch it; if nobody runs it, the model freezes at
   whatever the first round produces (§10.7).

### Not yet tried, possibly worth 2–3 hours

A classical baseline — Hough on vertical gradients + column segmentation,
0MB, no training. It will fail on flat-lying books, non-book objects, and
hard lighting, so it's **not** a product candidate. But it would give
§8's metrics a floor to beat, and it's cheap next to 6 hours of GPU.

---

## Environment notes

- Label Studio is **not currently running** — restart with the command
  above when resuming labeling.
- Node 24.13.0, npm 11.6.2, Python 3.14.6.
- Commits in this repo were authored as
  `Ido Biren <ido.birenboim@students.binahbalev.ai>` via `git -c` (no
  global git identity is configured on this machine).
- `.gitignore` covers `node_modules/`, `dist/`, `__pycache__/`,
  `data/raw/*/*`, `data/merged/`, and `data/indomain/`. **`data/indomain/`
  is photos of real homes — that rule was committed before any photo
  landed, and must stay.**

### Uploading `data/indomain/` — decision, 2026-09-01

An earlier note here said "never upload it anywhere". **The owner has since
explicitly approved uploading these photos to Google (Drive/Colab)** for
fine-tuning, so that blanket rule no longer holds and should not be treated
as blocking. Two things it does NOT change:

- **`.gitignore` stays exactly as it is.** "Don't commit" and "don't upload
  to Drive" are separate rules; only the second was lifted. These photos
  must still never enter git history, where they would be public and
  permanent.
- **`scene166`–`scene169` may not be the owner's own home.** They arrive
  with no EXIF at 2048×1152 (vs 4080×3072 for the rest) — the signature of
  an image that came through a messaging app, i.e. plausibly shot by
  someone else and forwarded. Whoever took them has not been asked. Worth
  raising again if the dataset grows by more third-party photos.

For the quality-ceiling run this is moot: that measurement uses only the
**public** Roboflow data (`data/merged/images/`, 141MB, 1604 files).
`data/indomain/photos/` (332MB, 103 files) is needed for fine-tune and for
the §8ב ship decision, not before.
