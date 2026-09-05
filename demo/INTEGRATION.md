# Integration brief — spine detector → the app

Hand this to whoever (or whatever) is building the app. It is written to be
executable without access to this repo.

The goal it serves: a screen recording that shows a user photographing a
shelf and the books being added to their library, with titles read
automatically.

---

## What the detector does, and does not

It returns **where each book spine is**, plus that spine cropped and
rotated flat. It does **not** read text. Title reading stays in the app's
existing cloud VLM call — unchanged, same credentials, same code path. The
only difference is what you feed it: one image per spine instead of one
image of the whole shelf.

That substitution is the entire product claim. A whole-shelf photo sent to
a VLM returns a blur of mixed-up titles; a single rectified spine returns
one clean title.

---

## The API

One endpoint. Base URL is a Cloudflare tunnel address that **changes every
time the server restarts**, so read it from config, never hardcode it.

### `POST {BASE}/detect`

`multipart/form-data`, one field:

| field | type | notes |
|---|---|---|
| `image` | file | JPEG. See "Downscale first" below — this matters. |

Optional query params:

| param | default | notes |
|---|---|---|
| `flip` | `true` | `true` = Hebrew spines (text reads bottom-to-top). Pass `false` for a shelf of English books. |

### Response

```json
{
  "width": 2040,
  "height": 1536,
  "seconds": 2.4,
  "spines": [
    {
      "quad": [[x, y], [x, y], [x, y], [x, y]],
      "score": 0.99,
      "crop_jpeg_b64": "/9j/4AAQSkZJRg..."
    }
  ]
}
```

- **`quad`** — four corners, clockwise, in the pixel coordinates of **the
  image you uploaded**, not the original camera file. Draw with these.
- **`score`** — 0..1 confidence. The server already filters below 0.8, so
  everything returned is worth showing. No need to filter again.
- **`crop_jpeg_b64`** — the spine, cropped out and rotated so its text runs
  horizontally. Render with
  `src={`data:image/jpeg;base64,${spine.crop_jpeg_b64}`}` and send the same
  string to the VLM.

### `GET {BASE}/health`

```json
{"status": "ok", "device": "cpu", "score_threshold": 0.8}
```

Use it for a "detector offline" state in the UI, and to check the tunnel is
alive before recording.

---

## Downscale first — this is the one thing that is easy to get wrong

Resize the photo so its **long side is 2040px** before uploading. Measured
over the tunnel with a 4080×3072 phone photo:

| upload size | round trip | books found | crop readability |
|---|---|---|---|
| 4080px (original) | 14.1s | 24 | good |
| **2040px** | **4.9s** | 22 | **identical** |

Three times faster for no loss you can see. The model resizes internally to
800px regardless, so full resolution never reaches it; the only thing large
uploads buy is sharper crops, and those are capped server-side at 1024px
anyway.

```js
async function downscale(file, maxLongSide = 2040) {
  const bitmap = await createImageBitmap(file);
  const scale = Math.min(1, maxLongSide / Math.max(bitmap.width, bitmap.height));
  const w = Math.round(bitmap.width * scale);
  const h = Math.round(bitmap.height * scale);

  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  canvas.getContext("2d").drawImage(bitmap, 0, 0, w, h);

  const blob = await new Promise(r => canvas.toBlob(r, "image/jpeg", 0.92));
  return { blob, width: w, height: h };
}
```

**Keep the downscaled blob.** The returned `quad` coordinates are in its
coordinate space, so that is the image the overlay must be drawn on. Drawing
quads over the original camera file puts every box in the wrong place — and
it looks like a detector bug rather than a scaling bug.

---

## The call

```js
const { blob, width, height } = await downscale(file);

const body = new FormData();
body.append("image", blob, "photo.jpg");

const response = await fetch(`${BASE}/detect?flip=true`, { method: "POST", body });
if (!response.ok) throw new Error(`detector ${response.status}`);
const { spines } = await response.json();

// Each spine's crop goes to the EXISTING VLM call, one at a time.
const titles = await Promise.all(
  spines.map(s => readTitleWithExistingVlm(s.crop_jpeg_b64))
);
```

---

## The screen the recording needs

The moment that sells this is **titles appearing**. Everything else is
setup. Suggested flow, in the order it should happen on screen:

1. **Camera / file picker.** Portrait, mobile.
2. **The photo, with quads drawn over it.** This is the "it can see" beat.
   Draw them animated in if cheap — it reads as the app working, not as a
   static result.
3. **A row per detected spine**, each showing the rectified crop, then its
   title filling in as the VLM answers. Crops arrive together and titles
   arrive one by one, so this fills in progressively, which looks good on
   video for free.
4. **"Add all to my library"**, and the books appearing in the library
   list. This is the product, and it is the shot the recording exists for.

Loading states are not optional here: detection takes 2–7 seconds and the
VLM adds its own time. Dead air on a recording reads as a broken app. A
spinner with "מזהה ספרים…" then "קורא כותרות…" is enough.

---

## Failure states worth building, because they will show up

| case | what to show |
|---|---|
| `/health` fails | "הזיהוי לא זמין" — do not let the camera flow start |
| `spines` is empty | "לא זוהו ספרים — נסה מזווית ישרה יותר, מקביל למדף" |
| a title comes back garbled | let the user edit it before adding. Also try `flip=false` for that shelf |
| VLM fails on one crop | keep the other titles; that row shows a retry |

---

## Honest limits, so the demo is aimed at what works

Measured on 147 held-out photos, recall by how many books are in the shot:

| books in shot | found |
|---|---|
| **1–5** | **93%** |
| **6–8** | **97%** |
| 9–14 | 82% |
| 15–21 | 75% |
| 22+ | 59% |

**Point the demo at 5–8 books.** That is both the product's stated use case
and where the model is strongest. A packed 24-book bookcase finds under
60% and will look bad on camera.

Photograph parallel to the shelf rather than at a sharp angle, fill the
frame with the books, and avoid hard shadows across the spines.

Thin paperbacks are the weakest case (65% vs 86% for wide books), so a
shelf of slim volumes is the one to avoid for a recording.
