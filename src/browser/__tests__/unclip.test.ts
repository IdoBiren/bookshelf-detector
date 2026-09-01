import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { polygonArea, polygonPerimeter } from "../polygonMath.js";
import { unclipQuad } from "../unclip.js";
import type { Point } from "../types.js";

// unclip.ts had zero direct tests before this file — everything exercising
// it went through postprocess.test.ts's end-to-end path instead. This is
// also the golden-master fixture the Python training-side mirror
// (training/polygon_offset.py, plan §13 phase B) is checked against, so a
// regression here is a regression in BOTH languages at once, not just this
// one. See fixtures/_generate_unclip_parity.ts for how the expected values
// were produced (unclipQuad's own output, captured once and pinned).
interface ParityCase {
  name: string;
  polygon: [number, number][];
  unclipRatio: number;
  expected: [number, number][];
}

const fixture: ParityCase[] = JSON.parse(
  readFileSync(new URL("./fixtures/unclipParity.json", import.meta.url), "utf-8"),
);

function toPoints(coords: [number, number][]): Point[] {
  return coords.map(([x, y]) => ({ x, y }));
}

describe("unclipQuad — golden fixture (shared with the Python mirror)", () => {
  for (const { name, polygon, unclipRatio, expected } of fixture) {
    it(`${name}: is a pinned regression, not just internally consistent`, () => {
      const result = unclipQuad(toPoints(polygon), unclipRatio);
      expect(result).toHaveLength(expected.length);
      for (let i = 0; i < result.length; i++) {
        expect(result[i]!.x).toBeCloseTo(expected[i]![0]!, 9);
        expect(result[i]!.y).toBeCloseTo(expected[i]![1]!, 9);
      }
    });
  }
});

describe("unclipQuad — direct behavioural tests", () => {
  it("expands a unit square outward by area*ratio/perimeter on every side", () => {
    const square: Point[] = [
      { x: 0, y: 0 },
      { x: 1, y: 0 },
      { x: 1, y: 1 },
      { x: 0, y: 1 },
    ];
    // area=1, perimeter=4, ratio=1 -> distance=0.25
    const result = unclipQuad(square, 1.0);
    expect(result[0]).toEqual({ x: -0.25, y: -0.25 });
    expect(result[2]).toEqual({ x: 1.25, y: 1.25 });
  });

  it("returns the input unchanged when perimeter is zero (all points coincide)", () => {
    const point: Point[] = [
      { x: 5, y: 5 },
      { x: 5, y: 5 },
      { x: 5, y: 5 },
      { x: 5, y: 5 },
    ];
    expect(unclipQuad(point, 1.5)).toEqual(point);
  });

  it("a zero-length edge is pushed through unmodified rather than throwing", () => {
    const withRepeatedPoint: Point[] = [
      { x: 0, y: 0 },
      { x: 0, y: 0 },
      { x: 10, y: 0 },
      { x: 10, y: 10 },
      { x: 0, y: 10 },
    ];
    expect(() => unclipQuad(withRepeatedPoint, 1.0)).not.toThrow();
  });

  it("falls back to the un-intersected offset point when two neighboring edges are exactly parallel", () => {
    // (50,0) lies exactly on the line from (0,0) to (100,0).
    const withCollinearVertex: Point[] = [
      { x: 0, y: 0 },
      { x: 50, y: 0 },
      { x: 100, y: 0 },
      { x: 100, y: 100 },
      { x: 0, y: 100 },
    ];
    const result = unclipQuad(withCollinearVertex, 1.0);
    // Must not throw and must not silently drop the vertex.
    expect(result).toHaveLength(5);
    // The fallback vertex is pushed straight along its own edge's normal,
    // not intersected with its (parallel) neighbor.
    expect(result[1]).toEqual({ x: 50, y: -25 });
  });

  it("larger unclipRatio expands more (monotonic in the ratio, for a fixed polygon)", () => {
    const trapezoid: Point[] = [
      { x: 50, y: 10 },
      { x: 70, y: 10 },
      { x: 90, y: 90 },
      { x: 30, y: 90 },
    ];
    const small = polygonArea(unclipQuad(trapezoid, 0.5));
    const large = polygonArea(unclipQuad(trapezoid, 2.0));
    expect(large).toBeGreaterThan(small);
  });

  it("the expanded polygon's own area exceeds the original by roughly perimeter*distance (first-order)", () => {
    // Sanity check independent of the fixture: for a convex polygon expanded
    // outward by a small uniform distance d, area grows by approximately
    // perimeter*d (ignoring the second-order corner-bulge term), which is a
    // useful cross-check that the offset direction is truly outward, not
    // inward or sideways.
    const square: Point[] = [
      { x: 0, y: 0 },
      { x: 100, y: 0 },
      { x: 100, y: 100 },
      { x: 0, y: 100 },
    ];
    const originalArea = polygonArea(square);
    const perimeter = polygonPerimeter(square);
    const ratio = 0.1; // distance = area*ratio/perimeter = 250
    const distance = (originalArea * ratio) / perimeter;
    const expandedArea = polygonArea(unclipQuad(square, ratio));
    const firstOrderPrediction = originalArea + perimeter * distance;
    // Second-order term is the 4 corner squares (distance^2 each for a
    // right-angle corner), so allow that much slack.
    expect(expandedArea).toBeGreaterThan(firstOrderPrediction);
    expect(expandedArea).toBeLessThan(firstOrderPrediction + 4 * distance * distance * 1.5);
  });
});
