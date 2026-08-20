import type { Point } from "../types.js";

/** Point-in-polygon test (ray casting), used to rasterize test shapes. */
function pointInPolygon(x: number, y: number, poly: Point[]): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i]!.x;
    const yi = poly[i]!.y;
    const xj = poly[j]!.x;
    const yj = poly[j]!.y;
    const intersects =
      yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi;
    if (intersects) inside = !inside;
  }
  return inside;
}

/** Rasterizes one or more polygons into a flat probMap (1.0 inside, 0.0 outside). */
export function rasterizePolygons(
  width: number,
  height: number,
  polygons: Point[][],
): Float32Array {
  const map = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      // Sample at pixel center for a stable, symmetric raster.
      const cx = x + 0.5;
      const cy = y + 0.5;
      for (const poly of polygons) {
        if (pointInPolygon(cx, cy, poly)) {
          map[y * width + x] = 1;
          break;
        }
      }
    }
  }
  return map;
}

/** Axis-aligned rectangle as a 4-point polygon (TL, TR, BR, BL). */
export function rectPolygon(x0: number, y0: number, x1: number, y1: number): Point[] {
  return [
    { x: x0, y: y0 },
    { x: x1, y: y0 },
    { x: x1, y: y1 },
    { x: x0, y: y1 },
  ];
}

/** Rotates a polygon by `angleDeg` about (cx, cy). */
export function rotatePolygon(poly: Point[], angleDeg: number, cx: number, cy: number): Point[] {
  const angle = (angleDeg * Math.PI) / 180;
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  return poly.map(({ x, y }) => {
    const dx = x - cx;
    const dy = y - cy;
    return { x: cx + dx * cos - dy * sin, y: cy + dx * sin + dy * cos };
  });
}

/** Rasterizes a single filled rectangle directly into a mask (exact, no sampling). */
export function rasterizeRect(
  width: number,
  height: number,
  x0: number,
  y0: number,
  x1: number,
  y1: number,
): Uint8Array {
  const mask = new Uint8Array(width * height);
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      mask[y * width + x] = 1;
    }
  }
  return mask;
}
