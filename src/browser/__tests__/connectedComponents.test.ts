import { describe, expect, it } from "vitest";
import { findConnectedComponents } from "../connectedComponents.js";
import { rasterizeRect } from "./testFixtures.js";

// Plan §12 test scenario 1: 5 vertical rectangles with gaps -> exactly 5
// components, and each component's AABB matches its rectangle exactly
// (axis-aligned rectangles have no hull/quad-fit ambiguity).
describe("findConnectedComponents: 5 separated rectangles (scenario 1)", () => {
  it("finds exactly 5 components with exact bounding boxes", () => {
    const width = 110;
    const height = 100;
    const mask = new Uint8Array(width * height);

    // Rect i occupies columns [10+i*20, 19+i*20], rows [10, 89].
    for (let i = 0; i < 5; i++) {
      const x0 = 10 + i * 20;
      for (let y = 10; y <= 89; y++) {
        for (let x = x0; x <= x0 + 9; x++) {
          mask[y * width + x] = 1;
        }
      }
    }

    const components = findConnectedComponents(mask, width, height);
    expect(components).toHaveLength(5);

    // Sort by min-x so component order matches rectangle order.
    const sorted = [...components].sort(
      (a, b) => Math.min(...a.map((p) => p.x)) - Math.min(...b.map((p) => p.x)),
    );

    for (let i = 0; i < 5; i++) {
      const xs = sorted[i]!.map((p) => p.x);
      const ys = sorted[i]!.map((p) => p.y);
      expect(Math.min(...xs)).toBe(10 + i * 20);
      expect(Math.max(...xs)).toBe(19 + i * 20);
      expect(Math.min(...ys)).toBe(10);
      expect(Math.max(...ys)).toBe(89);
      expect(sorted[i]!.length).toBe(10 * 80); // exact pixel count
    }
  });

  it("uses rasterizeRect fixture consistently (sanity on the test helper itself)", () => {
    const mask = rasterizeRect(20, 20, 5, 5, 10, 10);
    let count = 0;
    for (const v of mask) count += v;
    expect(count).toBe(5 * 5); // [5,10) x [5,10) = 5x5
  });
});

// Plan §12 test scenario 2 (corrected — see plan note): two blobs touching
// ONLY at a single diagonal pixel corner. This is the case that actually
// distinguishes 4- from 8-connectivity: a straight N-pixel gap stays
// separated under EITHER rule, but a diagonal touch is bridged under
// 8-connectivity and must NOT be under 4-connectivity (plan §1's shrink-mask
// argument depends on this).
describe("findConnectedComponents: diagonal corner touch (scenario 2)", () => {
  it("keeps two diagonally-touching 2x2 blocks as 2 separate components", () => {
    const width = 20;
    const height = 20;
    const mask = new Uint8Array(width * height);

    // Block A: (10,10)-(11,11). Block B: (12,12)-(13,13).
    // A's (11,11) and B's (12,12) are diagonal neighbors (dx=1, dy=1);
    // (11,12) and (12,11) are both background, so there is NO 4-connected
    // path between the blocks.
    for (const [x, y] of [
      [10, 10], [11, 10], [10, 11], [11, 11],
      [12, 12], [13, 12], [12, 13], [13, 13],
    ] as const) {
      mask[y * width + x] = 1;
    }

    const components = findConnectedComponents(mask, width, height);
    // If this implementation used 8-connectivity instead, the diagonal
    // touch at (11,11)/(12,12) would merge these into a single component
    // of length 8 — that's exactly the bug this test guards against.
    expect(components).toHaveLength(2);
    expect(components[0]!.length).toBe(4);
    expect(components[1]!.length).toBe(4);
  });
});
