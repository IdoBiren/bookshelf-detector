import type { Point } from "./types.js";

/**
 * BFS flood fill over a binary mask, 4-connectivity only.
 *
 * 8-connectivity would bridge diagonal single-pixel gaps and re-glue
 * neighbors that the DBNet shrink-mask training deliberately separated
 * (plan §1/§6) — two spines touching at a single diagonal pixel corner
 * must stay two components, not merge into one.
 */
export function findConnectedComponents(
  mask: Uint8Array,
  width: number,
  height: number,
): Point[][] {
  if (mask.length !== width * height) {
    throw new Error(`mask length ${mask.length} does not match ${width}x${height}`);
  }

  const visited = new Uint8Array(width * height);
  const components: Point[][] = [];

  for (let start = 0; start < mask.length; start++) {
    if (mask[start] === 0 || visited[start] === 1) continue;

    const component: Point[] = [];
    const queue: number[] = [start];
    visited[start] = 1;
    let head = 0;

    while (head < queue.length) {
      const idx = queue[head++]!;
      const x = idx % width;
      const y = (idx - x) / width;
      component.push({ x, y });

      const left = x > 0 ? idx - 1 : -1;
      const right = x < width - 1 ? idx + 1 : -1;
      const up = y > 0 ? idx - width : -1;
      const down = y < height - 1 ? idx + width : -1;

      for (const n of [left, right, up, down]) {
        if (n >= 0 && mask[n] !== 0 && visited[n] === 0) {
          visited[n] = 1;
          queue.push(n);
        }
      }
    }

    components.push(component);
  }

  return components;
}
