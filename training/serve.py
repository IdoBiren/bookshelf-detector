"""
Detector HTTP service — the demo path (plan: two-day investor prototype).

Serves what the browser cannot: the trained model is ~350MB on disk against
a ≤5MB browser budget, so this is a server-side model by construction, not
by preference. Everything the browser path would need — ONNX export, int8
quantization, onnxruntime-web, WebGPU/WASM fallback, distillation — is
skipped entirely by running the checkpoint directly.

Shape of the demo:

    Lovable app  --POST /detect-->  this service (laptop + tunnel)
                 <--quads + crops--
    app draws the quads, sends each crop to its EXISTING VLM call

Cropping happens here rather than in the app because cv2 is already present
and `warpPerspective` is three lines, and because it keeps the VLM
credentials where they already work.

Run:
    training/.venv/Scripts/python.exe -m uvicorn serve:app --host 0.0.0.0 --port 8000
    (from the training/ directory, with CHECKPOINT set)

Then expose it over HTTPS:
    cloudflared tunnel --url http://localhost:8000

HTTPS is not optional: the Lovable app is served over HTTPS, and a browser
blocks a plain-HTTP call from an HTTPS page as mixed content.
"""

from __future__ import annotations

import base64
import io
import os
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from crop_quad import rectify_spine
from detect import DEFAULT_SERVING_SCORE_THRESHOLD, detect_spines, load_detector

# Configured by environment so the same file serves a laptop demo and
# anything later, without editing code between them.
CHECKPOINT = os.environ.get("CHECKPOINT", "demo_checkpoint.pt")
SCORE_THRESHOLD = float(os.environ.get("SCORE_THRESHOLD", DEFAULT_SERVING_SCORE_THRESHOLD))
# Default open: a quick tunnel's hostname is random and unguessable, and the
# demo's Lovable origin is not known ahead of time. Narrow this to the real
# origin for anything beyond a demo.
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
# Measured: a 4080x3072 photo yielded 24 crops of ~2000x220 as PNG -- 18.6MB
# of image data, 26MB once base64'd, for one request. Capping the long side
# and encoding as JPEG (these are photographs, not line art) cuts that by
# well over an order of magnitude with no loss the VLM can notice.
MAX_CROP_LONG_SIDE = int(os.environ.get("MAX_CROP_LONG_SIDE", 1024))
CROP_JPEG_QUALITY = int(os.environ.get("CROP_JPEG_QUALITY", 90))
# Measured on real Hebrew shelf photos, not assumed: a Hebrew spine's title
# reads BOTTOM-TO-TOP, so the counter-clockwise rotation that suits
# top-to-bottom (English) spines delivers the text upside down. Verified by
# eye on scene001 -- flipped, the titles read cleanly ("פאולו קואלו ·
# האלכימאי"); unflipped they are inverted. The VLM would have returned
# nothing useful and the cause would have looked like a model problem.
DEFAULT_FLIP = os.environ.get("DEFAULT_FLIP", "true").lower() != "false"

app = FastAPI(title="bookshelf-detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

_detector = None


def get_detector():
    """Loaded once, on first use. Reloading a ~350MB checkpoint per request
    would add ~30s to every call."""
    global _detector
    if _detector is None:
        _detector = load_detector(CHECKPOINT, score_threshold=SCORE_THRESHOLD)
    return _detector


@app.on_event("startup")
def _warm_up() -> None:
    """Load at startup, not on the first user request -- otherwise the
    opening moment of the demo pays the entire model-load cost."""
    detector = get_detector()
    print(f"checkpoint: {CHECKPOINT}")
    print(f"device: {detector.device}  mask_resolution: {detector.mask_resolution}"
          f"  score_threshold: {detector.score_threshold}")
    # One throwaway forward pass: the first inference is markedly slower
    # than the rest (lazy CUDA/cuDNN init, allocator warm-up).
    detect_spines(detector, Image.new("RGB", (640, 480)))
    print("warm-up complete -- ready")


@app.get("/health")
def health() -> dict:
    """Smoke-test the tunnel before the demo, without uploading anything."""
    detector = get_detector()
    return {
        "status": "ok",
        "device": str(detector.device),
        "checkpoint": str(CHECKPOINT),
        "score_threshold": detector.score_threshold,
        "mask_resolution": detector.mask_resolution,
    }


@app.post("/detect")
async def detect(image: UploadFile = File(...), flip: bool | None = None) -> dict:
    """One shelf photo -> one entry per detected spine.

    Each entry carries the `quad` (original-image pixel coordinates, for
    drawing over the photo) and `crop_jpeg_b64`, a rectified horizontal
    strip ready to hand to the VLM.

    `flip` selects which of the two 180-degree orientations to return; it
    defaults to DEFAULT_FLIP, which is true because Hebrew spine text reads
    bottom-to-top (measured, see above). Pass flip=false for a shelf of
    English books, whose spines conventionally read top-to-bottom.
    """
    if flip is None:
        flip = DEFAULT_FLIP
    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty upload")

    try:
        pil_image = Image.open(io.BytesIO(raw))
        pil_image.load()
    except Exception as error:  # noqa: BLE001 - any decode failure is a 400
        raise HTTPException(status_code=400, detail=f"could not decode image: {error}") from error

    started = time.perf_counter()
    detector = get_detector()
    spines = detect_spines(detector, pil_image)

    # cv2 works in BGR; convert once here rather than per spine.
    bgr = cv2.cvtColor(np.asarray(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)

    results = []
    for spine in spines:
        crop = rectify_spine(bgr, spine["quad"], flip=flip, max_long_side=MAX_CROP_LONG_SIDE)
        if crop is None:
            continue  # degenerate quad -- skip it rather than fail the request
        ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, CROP_JPEG_QUALITY])
        if not ok:
            continue
        results.append(
            {
                "quad": spine["quad"],
                "score": spine["score"],
                "crop_jpeg_b64": base64.b64encode(encoded.tobytes()).decode("ascii"),
            }
        )

    elapsed = time.perf_counter() - started
    payload_kb = sum(len(r["crop_jpeg_b64"]) for r in results) / 1024
    print(f"/detect  {pil_image.width}x{pil_image.height}  "
          f"{len(results)} spines  {elapsed:.2f}s  {payload_kb:.0f}KB crops", flush=True)

    return {
        "width": pil_image.width,
        "height": pil_image.height,
        "seconds": round(elapsed, 3),
        "spines": results,
    }
