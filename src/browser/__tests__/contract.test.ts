import { describe, expect, it } from "vitest";
import { detectSpines, type ModelRunner, type RawDetection } from "../detectSpines.js";
import { computeLetterbox, modelPointToNormalized, unletterboxPoint } from "../letterbox.js";
import { canonicalizeQuad, quadToAABB } from "../quad.js";
import type { QuadCoords, RawQuad } from "../types.js";

// Plan §9, Stage 1: "contract.test.ts עובר על מודל מדומה שמחזיר quads קבועים.
// AABB נגזר נכון, נרמול 0..1000 נכון, ביטול letterbox נכון. אפס ML."
// Every number below is hand-derived in the accompanying PR notes, not just
// asserted against whatever the code happens to produce.

describe("computeLetterbox", () => {
  it("pads top/bottom for a landscape image (wider than tall)", () => {
    const lb = computeLetterbox(1600, 1200, 640);
    expect(lb.scale).toBeCloseTo(0.4, 10);
    expect(lb.padX).toBeCloseTo(0, 10);
    expect(lb.padY).toBeCloseTo(80, 10);
  });

  it("pads left/right for a portrait image (taller than wide)", () => {
    const lb = computeLetterbox(1200, 1600, 640);
    expect(lb.scale).toBeCloseTo(0.4, 10);
    expect(lb.padX).toBeCloseTo(80, 10);
    expect(lb.padY).toBeCloseTo(0, 10);
  });

  it("rejects zero/negative dimensions instead of producing Infinity/NaN silently", () => {
    expect(() => computeLetterbox(0, 100, 640)).toThrow();
    expect(() => computeLetterbox(100, -1, 640)).toThrow();
  });
});

describe("unletterboxPoint + normalizeTo1000 (via modelPointToNormalized)", () => {
  it("maps the model-space origin correctly, accounting for padding", () => {
    const lb = computeLetterbox(1600, 1200, 640); // padX=0, padY=80, scale=0.4
    // Model-space (320, 80) sits at the very top edge of real (non-pad) content.
    const [ox, oy] = unletterboxPoint(320, 80, lb);
    expect(ox).toBeCloseTo(800, 10);
    expect(oy).toBeCloseTo(0, 10);

    const [nx, ny] = modelPointToNormalized(320, 80, lb, 1600, 1200);
    expect(nx).toBeCloseTo(500, 10);
    expect(ny).toBeCloseTo(0, 10);
  });

  it("maps the far corner of real content back to (1000, 1000)", () => {
    const lb = computeLetterbox(1600, 1200, 640);
    const [nx, ny] = modelPointToNormalized(640, 560, lb, 1600, 1200);
    expect(nx).toBeCloseTo(1000, 10);
    expect(ny).toBeCloseTo(1000, 10);
  });
});

describe("quadToAABB", () => {
  it("derives the bounding rect from an axis-aligned quad", () => {
    const quad: QuadCoords = [100, 200, 300, 200, 300, 400, 100, 400];
    expect(quadToAABB(quad)).toEqual({ xmin: 100, xmax: 300, ymin: 200, ymax: 400 });
  });
});

describe("canonicalizeQuad", () => {
  // A narrow, tall "spine": 40 wide, 200 tall, top-left at (100, 50).
  const TL: [number, number] = [100, 50];
  const TR: [number, number] = [140, 50];
  const BR: [number, number] = [140, 250];
  const BL: [number, number] = [100, 250];

  it("is deterministic regardless of the input's original point order", () => {
    const permutations: RawQuad[] = [
      [TL, TR, BR, BL],
      [BR, TL, BL, TR],
      [TR, BL, TL, BR],
      [BL, BR, TR, TL],
    ];
    const results = permutations.map(canonicalizeQuad);
    for (const r of results) {
      expect(r).toEqual([100, 50, 140, 50, 140, 250, 100, 250]);
    }
  });

  it("orders a moderately rotated spine (±12° augmentation range) consistently", () => {
    // Same rectangle rotated 12° about its centroid — the augmentation
    // range the plan actually trains on (§3), not an arbitrary angle.
    const angle = (12 * Math.PI) / 180;
    const cx = 120;
    const cy = 150;
    const rotate = ([x, y]: [number, number]): [number, number] => {
      const dx = x - cx;
      const dy = y - cy;
      return [
        cx + dx * Math.cos(angle) - dy * Math.sin(angle),
        cy + dx * Math.sin(angle) + dy * Math.cos(angle),
      ];
    };
    const rotated: RawQuad = [BL, TR, TL, BR].map(rotate) as RawQuad;
    const result = canonicalizeQuad(rotated);

    // Top corners must be above (smaller y than) bottom corners, and left
    // corners left of (smaller x than) right corners — the whole point of
    // canonical ordering being well-defined.
    expect(result[1]).toBeLessThan(result[5]); // TL.y < BR.y
    expect(result[3]).toBeLessThan(result[7]); // TR.y < BL.y
    expect(result[0]).toBeLessThan(result[2]); // TL.x < TR.x
    expect(result[6]).toBeLessThan(result[4]); // BL.x < BR.x

    // And re-running on a different permutation of the SAME rotated points
    // must agree exactly — this is what prevents a crop from randomly
    // rotating 90°/180° between otherwise-identical detections.
    const reordered: RawQuad = [BR, TL, BL, TR].map(rotate) as RawQuad;
    expect(canonicalizeQuad(reordered)).toEqual(result);
  });
});

describe("detectSpines end-to-end (mock model, zero ML)", () => {
  function mockModel(detections: RawDetection[]): ModelRunner {
    return { run: async () => detections };
  }

  it("produces a correctly-normalized Box from a single fixed detection (landscape image)", async () => {
    // Physical spine in the source photo, expressed as a scrambled-order
    // quad in MODEL-INPUT (640x640) pixel space — canonicalization must
    // fix the order before the letterbox math runs.
    const scrambledModelSpaceQuad: RawQuad = [
      [260, 540], // BR
      [220, 100], // TL
      [220, 540], // BL
      [260, 100], // TR
    ];
    const model = mockModel([{ quad: scrambledModelSpaceQuad, score: 0.91 }]);

    const [box] = await detectSpines(1600, 1200, model, 640);
    expect(box).toBeDefined();

    // Hand-derived: letterbox(1600,1200,640) => scale=0.4, padX=0, padY=80.
    // unletterbox(220,100)=(550,50); unletterbox(260,540)=(650,1150).
    // normalize against 1600x1200 => (343.75, 41.666..) .. (406.25, 958.333..).
    expect(box!.xmin).toBeCloseTo(343.75, 6);
    expect(box!.xmax).toBeCloseTo(406.25, 6);
    expect(box!.ymin).toBeCloseTo(1000 / 24, 6); // 41.666..
    expect(box!.ymax).toBeCloseTo(11500 / 12, 6); // 958.333..
    expect(box!.score).toBe(0.91);

    // quad must be present and in canonical TL,TR,BR,BL order.
    expect(box!.quad).toBeDefined();
    const q = box!.quad!;
    expect(q[0]).toBeCloseTo(343.75, 6); // TL.x
    expect(q[2]).toBeCloseTo(406.25, 6); // TR.x
    expect(q[1]).toBeLessThan(q[5]!); // TL.y < BR.y
  });

  it("handles a portrait-oriented source image (padding on the other axis)", async () => {
    const scrambledModelSpaceQuad: RawQuad = [
      [420, 300],
      [380, 40],
      [380, 300],
      [420, 40],
    ];
    const model = mockModel([{ quad: scrambledModelSpaceQuad, score: 0.5 }]);

    // 1200x1600 portrait => scale=0.4, padX=80, padY=0.
    const [box] = await detectSpines(1200, 1600, model, 640);
    // unletterbox(380,40) = ((380-80)/0.4, (40-0)/0.4) = (750, 100)
    // normalize against 1200x1600 => (750/1200*1000, 100/1600*1000) = (625, 62.5)
    expect(box!.xmin).toBeCloseTo(625, 6);
    expect(box!.ymin).toBeCloseTo(62.5, 6);
  });

  it("returns an empty array when the model finds nothing — not a crash", async () => {
    const model = mockModel([]);
    const boxes = await detectSpines(1600, 1200, model, 640);
    expect(boxes).toEqual([]);
  });

  it("preserves backward compatibility: ymin/xmin/ymax/xmax are always present even though quad is optional on the type", async () => {
    const model = mockModel([
      { quad: [[100, 100], [200, 100], [200, 300], [100, 300]], score: 0.7 },
    ]);
    const [box] = await detectSpines(640, 640, model, 640);
    expect(box).toMatchObject({
      ymin: expect.any(Number),
      xmin: expect.any(Number),
      ymax: expect.any(Number),
      xmax: expect.any(Number),
    });
  });
});
