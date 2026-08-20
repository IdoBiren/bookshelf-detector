import { computeLetterbox, modelPointToNormalized } from "./letterbox.js";
import { canonicalizeQuad, quadToAABB } from "./quad.js";
import type { Box, QuadCoords, RawQuad } from "./types.js";

/** One raw detection, in MODEL-INPUT pixel space (0..targetSize), corners in any order. */
export interface RawDetection {
  quad: RawQuad;
  score: number;
}

/**
 * Abstraction over "run the model". Stage 1 (this module) is tested against
 * a fixed mock; later stages swap in a real onnxruntime-web session behind
 * the same interface without touching any of the geometry below.
 */
export interface ModelRunner {
  run(targetSize: number): Promise<RawDetection[]>;
}

/**
 * Full geometric pipeline from plan §6, minus the actual neural net and
 * minus connected-components/minAreaRect (that's Stage 9 — this stage takes
 * already-formed quads and proves the coordinate math end to end).
 */
export async function detectSpines(
  origWidth: number,
  origHeight: number,
  model: ModelRunner,
  targetSize = 640,
): Promise<Box[]> {
  const letterbox = computeLetterbox(origWidth, origHeight, targetSize);
  const detections = await model.run(targetSize);

  return detections.map(({ quad, score }): Box => {
    const canonical = canonicalizeQuad(quad);
    const normalized = mapQuadToNormalized(canonical, letterbox, origWidth, origHeight);
    const aabb = quadToAABB(normalized);
    return { ...aabb, quad: normalized, score };
  });
}

function mapQuadToNormalized(
  quad: QuadCoords,
  letterbox: ReturnType<typeof computeLetterbox>,
  origWidth: number,
  origHeight: number,
): QuadCoords {
  const out: number[] = [];
  for (let i = 0; i < 8; i += 2) {
    const [nx, ny] = modelPointToNormalized(
      quad[i]!,
      quad[i + 1]!,
      letterbox,
      origWidth,
      origHeight,
    );
    out.push(nx, ny);
  }
  return out as QuadCoords;
}
