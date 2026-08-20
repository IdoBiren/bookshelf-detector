import { findConnectedComponents } from "./connectedComponents.js";
import { convexHull } from "./convexHull.js";
import type { ModelRunner, RawDetection } from "./detectSpines.js";
import { minAreaQuad } from "./minAreaQuad.js";
import { minAreaRect } from "./minAreaRect.js";
import type { Point, RawQuad } from "./types.js";
import { unclipQuad } from "./unclip.js";

export interface PostprocessOptions {
  /** Probability-map threshold that separates foreground from background. */
  probThreshold: number;
  /** DBNet unclip ratio (plan §6/§7 — not the final calibrated value). */
  unclipRatio: number;
  /** Components smaller than this many pixels are discarded as noise. */
  minAreaPx: number;
}

const DEFAULT_OPTIONS: PostprocessOptions = {
  probThreshold: 0.3,
  unclipRatio: 1.5,
  minAreaPx: 16,
};

function pointsToRawQuad(points: Point[]): RawQuad {
  return [
    [points[0]!.x, points[0]!.y],
    [points[1]!.x, points[1]!.y],
    [points[2]!.x, points[2]!.y],
    [points[3]!.x, points[3]!.y],
  ];
}

/**
 * probMap -> raw (uncanonicalized) quad detections in MODEL-INPUT pixel
 * space. Deliberately does NOT canonicalize corner order or map through
 * letterbox/normalization — `detectSpines()` (plan §2, Stage 1) already
 * does both, and re-doing it here would duplicate already-tested logic.
 *
 * No NMS: shrink-mask + connected-components (§6) already separates
 * touching spines, so there's nothing to suppress.
 */
export function postprocess(
  probMap: Float32Array,
  width: number,
  height: number,
  options: Partial<PostprocessOptions> = {},
): RawDetection[] {
  const opts = { ...DEFAULT_OPTIONS, ...options };

  const mask = new Uint8Array(width * height);
  for (let i = 0; i < probMap.length; i++) {
    mask[i] = probMap[i]! >= opts.probThreshold ? 1 : 0;
  }

  const components = findConnectedComponents(mask, width, height);
  const detections: RawDetection[] = [];

  for (const component of components) {
    if (component.length < opts.minAreaPx) continue;

    const hull = convexHull(component);
    if (hull.length < 3) continue; // a line/point — not a shape

    const quadPoints = minAreaQuad(hull) ?? minAreaRect(hull).corners;
    const expanded = unclipQuad(quadPoints, opts.unclipRatio);

    let probSum = 0;
    for (const p of component) probSum += probMap[p.y * width + p.x]!;
    const score = probSum / component.length;

    detections.push({ quad: pointsToRawQuad(expanded), score });
  }

  return detections;
}

/** Wraps a fixed probMap as a ModelRunner, for feeding into detectSpines(). */
export function postprocessModelRunner(
  probMap: Float32Array,
  width: number,
  height: number,
  options?: Partial<PostprocessOptions>,
): ModelRunner {
  return {
    run: async () => postprocess(probMap, width, height, options),
  };
}
