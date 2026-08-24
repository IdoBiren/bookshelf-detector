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

**Tests, both green as of this handoff:**
```bash
npx vitest run                                    # 22/22 (browser/TS)
cd training && python -m unittest discover -s tests   # 38/38 (pipeline/Python)
```

### Data on disk (all gitignored — never commit)

| Path | Contents |
|---|---|
| `data/raw/` | 3 public Roboflow datasets, as downloaded |
| `data/merged/pretrain_{train,val}.json` | 1440 / 147 images · 27109 / 2294 annotations |
| `data/merged/indomain_{train,val,test}.json` | 12 / 3 / 2 images · 131 / 36 / 13 annotations |
| `data/indomain/photos/` | **165 own photos**, 17 labeled so far |

---

## The immediate next step

**Keep labeling** `data/indomain/photos/` — 148 of 165 photos still to go.
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
- **PyTorch and onnxruntime are UNVERIFIED** and may not have 3.14 wheels.
  Colab has its own Python so training is unaffected — but the §5 parity
  check needs onnxruntime *locally in Python* to diff against JS, so
  budget for a separate 3.11/3.12 env before Stage 8.
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

## Open questions for whoever picks this up

1. **`shrink_ratio` / `unclip_ratio` are tuned for text lines, not books**
   — DBNet expects many *small* instances; we have 5–8 *large* ones.
   `shrink_ratio=0.4` on a 400px-tall spine shrinks far more aggressively
   than on a 20px text line. This is the biggest unknown in the whole
   architecture recommendation (§10.2) and may need a change to
   target *generation*, not just a parameter sweep.
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
