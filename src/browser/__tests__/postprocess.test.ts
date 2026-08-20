import { describe, expect, it } from "vitest";
import { detectSpines } from "../detectSpines.js";
import { postprocess, postprocessModelRunner } from "../postprocess.js";
import type { Point } from "../types.js";
import { rasterizePolygons, rectPolygon, rotatePolygon } from "./testFixtures.js";

describe("postprocess: 5 separated rectangles (scenario 1, full pipeline)", () => {
  it("produces exactly 5 detections from a probMap with 5 separated blobs", () => {
    const width = 110;
    const height = 100;
    const polygons: Point[][] = [];
    for (let i = 0; i < 5; i++) {
      const x0 = 10 + i * 20;
      polygons.push(rectPolygon(x0, 10, x0 + 10, 90));
    }
    const probMap = rasterizePolygons(width, height, polygons);

    const detections = postprocess(probMap, width, height, { unclipRatio: 0 });
    expect(detections).toHaveLength(5);
    for (const d of detections) {
      expect(d.score).toBeGreaterThan(0.9); // uniform prob=1 inside each blob
    }
  });
});

// Plan §12 test scenario 3: a spine tilted 12° must recover as a quad in
// the canonical corner order from plan §2 — this is the case the whole
// oriented-detection argument (§1) exists for.
describe("postprocess + detectSpines: tilted spine (scenario 3)", () => {
  it("recovers a canonically-ordered quad from a 12°-rotated rectangle", async () => {
    const size = 640;
    const cx = 320;
    const cy = 320;
    const rectWidth = 40;
    const rectHeight = 300;
    const upright = rectPolygon(cx - rectWidth / 2, cy - rectHeight / 2, cx + rectWidth / 2, cy + rectHeight / 2);
    const tilted = rotatePolygon(upright, 12, cx, cy);

    const probMap = rasterizePolygons(size, size, [tilted]);
    const runner = postprocessModelRunner(probMap, size, size, { unclipRatio: 0 });

    // origWidth=origHeight=size=targetSize -> trivial letterbox (scale=1,
    // no padding), isolating the tilt-recovery from the letterbox math
    // already covered by contract.test.ts.
    const boxes = await detectSpines(size, size, runner, size);
    expect(boxes).toHaveLength(1);

    const box = boxes[0]!;
    expect(box.quad).toBeDefined();
    const q = box.quad!;

    // Canonical order invariants (same as contract.test.ts's rotation test).
    expect(q[1]).toBeLessThan(q[5]!); // TL.y < BR.y
    expect(q[3]).toBeLessThan(q[7]!); // TR.y < BL.y
    expect(q[0]).toBeLessThan(q[2]!); // TL.x < TR.x
    expect(q[6]).toBeLessThan(q[4]!); // BL.x < BR.x

    // Area sanity: normalized-space area should be close to the original
    // rectangle's area scaled by (1000/size)^2 — hand-derived, not code-derived.
    const scale = 1000 / size;
    const expectedArea = rectWidth * rectHeight * scale * scale;
    const qPoints: Point[] = [
      { x: q[0]!, y: q[1]! },
      { x: q[2]!, y: q[3]! },
      { x: q[4]!, y: q[5]! },
      { x: q[6]!, y: q[7]! },
    ];
    let shoelace = 0;
    for (let i = 0; i < 4; i++) {
      const a = qPoints[i]!;
      const b = qPoints[(i + 1) % 4]!;
      shoelace += a.x * b.y - b.x * a.y;
    }
    const actualArea = Math.abs(shoelace) / 2;
    expect(actualArea).toBeGreaterThan(expectedArea * 0.8);
    expect(actualArea).toBeLessThan(expectedArea * 1.2);
  });
});

// Plan §12 test scenario 5: probMap -> detectSpines -> normalized Box[],
// through the SAME (unmodified) detectSpines()/letterbox pipeline that
// contract.test.ts already validates against hand-derived numbers. This is
// the test that closes the loop between the Stage-1 mock and real geometry.
describe("postprocess + detectSpines: full pipeline against hand-derived letterbox numbers (scenario 5)", () => {
  it("matches independently-computed normalized coordinates for a known rectangle", async () => {
    const modelSize = 640;
    // Mask pixel columns [220,259], rows [100,539] (half-open [220,260)/[100,540)).
    const probMap = new Float32Array(modelSize * modelSize);
    for (let y = 100; y < 540; y++) {
      for (let x = 220; x < 260; x++) {
        probMap[y * modelSize + x] = 1;
      }
    }
    // unclipRatio=0 isolates letterbox/normalization from the (separately
    // calibrated, plan §7) unclip expansion.
    const runner = postprocessModelRunner(probMap, modelSize, modelSize, { unclipRatio: 0 });

    // origWidth=1600, origHeight=1200 -> letterbox scale=0.4, padX=0, padY=80
    // (same landscape setup as contract.test.ts). Pixel-index hull corners
    // are (220,100)-(259,539), i.e. width 39/height 439 in COORDINATE terms
    // (not 40/440 — that's the pixel-index-vs-continuous-coordinate
    // discretization the plan §12 update calls out, not a bug).
    // Hand-derived: unletterbox(220,100)=(550,50); unletterbox(259,539)=(647.5,1147.5).
    // normalize against 1600x1200: (343.75, 41.6666..) .. (404.6875, 956.25).
    const boxes = await detectSpines(1600, 1200, runner, modelSize);
    expect(boxes).toHaveLength(1);
    const box = boxes[0]!;

    expect(box.xmin).toBeCloseTo(343.75, 3);
    expect(box.xmax).toBeCloseTo(404.6875, 3);
    expect(box.ymin).toBeCloseTo(1000 / 24, 3); // 41.6666..
    expect(box.ymax).toBeCloseTo(956.25, 3);
    expect(box.score).toBeCloseTo(1, 6);
  });
});
