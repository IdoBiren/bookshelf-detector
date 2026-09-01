// One-off generator for unclipParity.json — NOT part of the test suite.
// Dumps unclipQuad()'s own (trusted, already-shipping) output for a set of
// hand-chosen inputs, so both unclip.test.ts and the Python mirror's tests
// can assert against the SAME golden values instead of two independent,
// possibly-divergent hand computations.
import { writeFileSync } from "node:fs";
import { unclipQuad } from "../../unclip.js";
import type { Point } from "../../types.js";

const cases: Array<{ name: string; polygon: Point[]; unclipRatio: number }> = [
  {
    name: "unit_square",
    polygon: [
      { x: 0, y: 0 },
      { x: 1, y: 0 },
      { x: 1, y: 1 },
      { x: 0, y: 1 },
    ],
    unclipRatio: 1.0,
  },
  {
    name: "handoff_trapezoid",
    // Same trapezoid as minAreaQuad.test.ts: top 20 wide, bottom 60 wide,
    // height 80, analytic area 3200 — a documented, independently-verified
    // shape, not invented for this fixture.
    polygon: [
      { x: 50, y: 10 },
      { x: 70, y: 10 },
      { x: 90, y: 90 },
      { x: 30, y: 90 },
    ],
    unclipRatio: 1.5, // the shipped postprocess.ts default
  },
  {
    name: "thin_spine_25x400",
    polygon: [
      { x: 0, y: 0 },
      { x: 25, y: 0 },
      { x: 25, y: 400 },
      { x: 0, y: 400 },
    ],
    unclipRatio: 1.5,
  },
  {
    name: "real_11vertex_from_indomain",
    // Lifted from data/merged/indomain_train.json — a real hand-labeled
    // spine polygon, not synthetic. Native photo-pixel coordinates.
    polygon: [
      { x: 726.5753424657534, y: 369.49771689497715 },
      { x: 1297.8995433789953, y: 381.917808219178 },
      { x: 1422.1004566210047, y: 2372.237442922374 },
      { x: 1335.1598173515983, y: 2428.127853881278 },
      { x: 894.2465753424657, y: 2400.1826484018256 },
      { x: 1061.917808219178, y: 2285.296803652968 },
      { x: 1086.7579908675798, y: 2043.10502283105 },
      { x: 1074.3378995433789, y: 1912.6940639269405 },
      { x: 1009.1324200913242, y: 1850.5936073059358 },
      { x: 919.0867579908676, y: 1825.753424657534 },
      { x: 825.9360730593606, y: 1850.5936073059358 },
    ],
    unclipRatio: 1.5,
  },
  {
    name: "zero_length_edge",
    // A repeated point creates a zero-length edge -- exercises unclip.ts's
    // `len === 0` branch, which pushes the edge through unmodified.
    polygon: [
      { x: 0, y: 0 },
      { x: 0, y: 0 },
      { x: 10, y: 0 },
      { x: 10, y: 10 },
      { x: 0, y: 10 },
    ],
    unclipRatio: 1.0,
  },
  {
    name: "collinear_vertex_triggers_parallel_fallback",
    // (50,0) sits exactly on the line from (0,0) to (100,0): the two edges
    // meeting there are exactly parallel, so lineIntersection returns null
    // and unclipQuad falls back to `currEdge.p1` at that corner.
    polygon: [
      { x: 0, y: 0 },
      { x: 50, y: 0 },
      { x: 100, y: 0 },
      { x: 100, y: 100 },
      { x: 0, y: 100 },
    ],
    unclipRatio: 1.0,
  },
];

const fixture = cases.map(({ name, polygon, unclipRatio }) => ({
  name,
  polygon: polygon.map((p) => [p.x, p.y]),
  unclipRatio,
  expected: unclipQuad(polygon, unclipRatio).map((p) => [p.x, p.y]),
}));

writeFileSync(
  new URL("./unclipParity.json", import.meta.url),
  JSON.stringify(fixture, null, 2) + "\n",
);
console.log(`Wrote ${fixture.length} cases to unclipParity.json`);
