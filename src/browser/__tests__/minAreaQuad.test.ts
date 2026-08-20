import { describe, expect, it } from "vitest";
import { convexHull } from "../convexHull.js";
import { minAreaQuad } from "../minAreaQuad.js";
import { minAreaRect } from "../minAreaRect.js";
import { polygonArea } from "../polygonMath.js";
import type { Point } from "../types.js";

// Plan §12 test scenario 4 — the §1 test: a genuine (non-rectangular)
// trapezoid must be fit tightly by minAreaQuad, while minAreaRect (which
// can only produce a rotated RECTANGLE) necessarily has measurable excess
// area, since a rectangle can never equal a trapezoid's true area unless
// the trapezoid degenerates into a rectangle.
describe("minAreaQuad vs minAreaRect on a perspective trapezoid", () => {
  // Isosceles trapezoid: top edge 20 wide, bottom edge 60 wide, height 80.
  // Analytic area = (20 + 60) / 2 * 80 = 3200 — hand-computed, not code-derived.
  const TL: Point = { x: 50, y: 10 };
  const TR: Point = { x: 70, y: 10 };
  const BR: Point = { x: 90, y: 90 };
  const BL: Point = { x: 30, y: 90 };
  const ANALYTIC_AREA = 3200;

  it("on an exact 4-corner trapezoid: minAreaQuad matches the analytic area exactly; minAreaRect is measurably larger", () => {
    const hull = convexHull([TL, TR, BR, BL]);
    expect(hull).toHaveLength(4); // already a quad — the reduction loop shouldn't even run

    const quad = minAreaQuad(hull);
    expect(quad).not.toBeNull();
    expect(polygonArea(quad!)).toBeCloseTo(ANALYTIC_AREA, 6);

    const rect = minAreaRect(hull);
    // A rectangle can only equal a trapezoid's area if the trapezoid IS a
    // rectangle. This one clearly isn't (top width 20 vs bottom width 60),
    // so the excess must be substantial — 10% is a conservative floor.
    expect(rect.area).toBeGreaterThan(ANALYTIC_AREA * 1.1);
  });

  it("on a hull with an extra (non-redundant) vertex: reduces to 4 points, area stays close to the trapezoid and well below minAreaRect's", () => {
    // A point bulging slightly outside the left leg (TL->BL). The left leg
    // satisfies x = 50 - 0.25*(y-10); at y=50 that's x=40, so x=38 is
    // genuinely outside the trapezoid and must survive on the hull.
    const bulge: Point = { x: 38, y: 50 };
    const hull = convexHull([TL, TR, BR, BL, bulge]);
    expect(hull).toHaveLength(5); // bulge point must survive as a hull vertex

    const quad = minAreaQuad(hull);
    expect(quad).not.toBeNull();
    expect(quad).toHaveLength(4);

    const quadArea = polygonArea(quad!);
    // Must cover at least the original trapezoid...
    expect(quadArea).toBeGreaterThanOrEqual(ANALYTIC_AREA - 1e-6);
    // ...but the reduction shouldn't wildly overshoot it either.
    expect(quadArea).toBeLessThan(ANALYTIC_AREA * 1.4);

    const rectArea = minAreaRect(hull).area;
    expect(rectArea).toBeGreaterThan(quadArea * 1.1);
  });

  it("falls back to null on a hull with fewer than 4 points (caller uses minAreaRect instead)", () => {
    const triangle = convexHull([{ x: 0, y: 0 }, { x: 10, y: 0 }, { x: 5, y: 10 }]);
    expect(triangle).toHaveLength(3);
    expect(minAreaQuad(triangle)).toBeNull();
  });
});

describe("convexHull + minAreaRect: basic sanity on an axis-aligned square", () => {
  it("recovers the exact square unchanged", () => {
    const square: Point[] = [
      { x: 0, y: 0 },
      { x: 10, y: 0 },
      { x: 10, y: 10 },
      { x: 0, y: 10 },
    ];
    const hull = convexHull(square);
    expect(hull).toHaveLength(4);
    expect(polygonArea(hull)).toBe(100);

    const rect = minAreaRect(hull);
    expect(rect.area).toBeCloseTo(100, 6);

    const quad = minAreaQuad(hull);
    expect(polygonArea(quad!)).toBeCloseTo(100, 6);
  });
});
