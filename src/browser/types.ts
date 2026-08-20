/**
 * Integration contract (plan §2). All coordinates normalized to 0..1000
 * against the ORIGINAL image dimensions (not the model's input size).
 */

/** Eight numbers: x0,y0, x1,y1, x2,y2, x3,y3 — corners of a spine quad. */
export type QuadCoords = [number, number, number, number, number, number, number, number];

export interface Box {
  ymin: number;
  xmin: number;
  ymax: number;
  xmax: number;
  /**
   * Canonical order: corner 0 is the one that becomes top-left once the
   * spine is rotated to horizontal, then clockwise (0=TL, 1=TR, 2=BR, 3=BL
   * in the spine's own frame). Required for warpPerspective to produce a
   * consistently-oriented crop — without this the crop rotates 90°/180°
   * unpredictably per spine.
   */
  quad?: QuadCoords;
  score?: number;
}

/** A raw (unordered) quad as it might come out of a contour/minAreaRect step. */
export type RawQuad = [
  [number, number],
  [number, number],
  [number, number],
  [number, number],
];

/** A single 2D point, used throughout the connected-components/hull/quad-fit pipeline. */
export interface Point {
  x: number;
  y: number;
}
