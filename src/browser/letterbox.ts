/**
 * Letterbox math (plan §6 step 6). Kept as pure, independently-testable
 * functions because a silent bug here — normalizing against the model's
 * 640x640 input instead of the original image — produces boxes that are
 * consistently offset in a way that's easy to miss by eye.
 */

export interface LetterboxInfo {
  scale: number;
  padX: number;
  padY: number;
  targetSize: number;
}

export function computeLetterbox(
  origWidth: number,
  origHeight: number,
  targetSize: number,
): LetterboxInfo {
  if (origWidth <= 0 || origHeight <= 0) {
    throw new Error(`Invalid image dimensions: ${origWidth}x${origHeight}`);
  }
  const scale = Math.min(targetSize / origWidth, targetSize / origHeight);
  const scaledWidth = origWidth * scale;
  const scaledHeight = origHeight * scale;
  const padX = (targetSize - scaledWidth) / 2;
  const padY = (targetSize - scaledHeight) / 2;
  return { scale, padX, padY, targetSize };
}

/** Model-input-space (e.g. 0..640) point -> original-image-space point. */
export function unletterboxPoint(
  x: number,
  y: number,
  info: LetterboxInfo,
): [number, number] {
  return [(x - info.padX) / info.scale, (y - info.padY) / info.scale];
}

/** Original-image-space point -> normalized 0..1000 space. */
export function normalizeTo1000(
  x: number,
  y: number,
  origWidth: number,
  origHeight: number,
): [number, number] {
  return [(x / origWidth) * 1000, (y / origHeight) * 1000];
}

/** Composition of the two above — what the post-processing pipeline actually calls. */
export function modelPointToNormalized(
  x: number,
  y: number,
  info: LetterboxInfo,
  origWidth: number,
  origHeight: number,
): [number, number] {
  const [ox, oy] = unletterboxPoint(x, y, info);
  return normalizeTo1000(ox, oy, origWidth, origHeight);
}
