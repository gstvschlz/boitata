import type * as THREE from "three";
import type { Vec3 } from "./bounds";
import { bitReversed } from "./motion";

/** A cell spanning fewer CSS pixels than this draws as one element standing for all of its own. */
export const LOD_PIXELS = 3;
/** Layers with fewer elements always draw every one. */
export const LOD_MIN = 20_000;

/**
 * Elements sorted along a Morton curve over their bounds, grouped into octree cells: `levels[l]` holds the first
 * position in `order` of each cell of level l (the root first, each level halving the cell side), then n. Levels
 * stop once they hold a quarter as many cells as elements; a cell of the last level refines into its elements.
 */
export interface Octree {
  order: Uint32Array;
  levels: Uint32Array[];
  /** Per level, the index in the next level of each cell's first child. */
  children: Uint32Array[];
  /** Per level, a cell's side in finest cells, as a power of two. */
  shifts: number[];
  lo: Vec3;
  /** Side of a finest cell along each axis. */
  cell: Vec3;
  xyz: ArrayLike<number>;
  from: Uint32Array;
  to: Uint32Array;
}

export function octree(xyz: ArrayLike<number>, n: number): Octree {
  const lo: Vec3 = [Infinity, Infinity, Infinity];
  const hi: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < n; i++)
    for (let a = 0; a < 3; a++) {
      const v = xyz[3 * i + a];
      if (v < lo[a]) lo[a] = v;
      if (v > hi[a]) hi[a] = v;
    }
  const indexBits = Math.ceil(Math.log2(Math.max(n, 2)));
  const bits = Math.max(1, Math.min(10, Math.floor((52 - indexBits) / 3)));
  const size = 2 ** indexBits;
  const cell = [0, 1, 2].map((a) => Math.max(hi[a] - lo[a], 1e-9) / 2 ** bits) as Vec3;
  const keys = new Float64Array(n);
  const q = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    quantize(xyz, i, lo, cell, bits, q);
    let code = 0;
    for (let b = bits - 1; b >= 0; b--) code = code * 8 + ((q[0] >> b) & 1) * 4 + ((q[1] >> b) & 1) * 2 + ((q[2] >> b) & 1);
    keys[i] = code * size + i;
  }
  keys.sort();
  const order = new Uint32Array(n);
  const codes = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    order[i] = keys[i] % size;
    codes[i] = Math.floor(keys[i] / size);
  }
  const levels: Uint32Array[] = [];
  const shifts: number[] = [];
  for (let shift = bits; shift >= 0; shift--) {
    const unit = 8 ** shift;
    const starts: number[] = [];
    for (let i = 0; i < n; i++) if (i === 0 || Math.floor(codes[i] / unit) !== Math.floor(codes[i - 1] / unit)) starts.push(i);
    starts.push(n);
    levels.push(Uint32Array.from(starts));
    shifts.push(shift);
    if (starts.length - 1 > n / 4) break;
  }
  const children = levels.slice(0, -1).map((level, l) => {
    const next = levels[l + 1];
    const out = new Uint32Array(level.length);
    for (let a = 0, c = 0; a < level.length; a++) {
      while (next[c] < level[a]) c++;
      out[a] = c;
    }
    return out;
  });
  return { order, levels, children, shifts, lo, cell, xyz, from: new Uint32Array(n), to: new Uint32Array(n) };
}

function quantize(xyz: ArrayLike<number>, i: number, lo: Vec3, cell: Vec3, bits: number, out: number[]): void {
  const top = 2 ** bits - 1;
  for (let a = 0; a < 3; a++) out[a] = Math.max(0, Math.min(top, Math.floor((xyz[3 * i + a] - lo[a]) / cell[a])));
}

/** CSS pixels a length `size` spans `distance` in front of a camera of vertical field `fov` degrees. */
export function projectedPixels(size: number, distance: number, fov: number, height: number): number {
  if (distance <= size) return Infinity;
  return (size * height) / (2 * Math.tan((fov * Math.PI) / 360) * distance);
}

/** A camera that draws the scene, in perspective or orthographic. */
export type ViewCamera = THREE.PerspectiveCamera | THREE.OrthographicCamera;

/** CSS pixels a length `size` spans `distance` from a camera; an orthographic one draws it the same at any distance. */
export function screenPixels(camera: ViewCamera, height: number): (size: number, distance: number) => number {
  if ("isOrthographicCamera" in camera) {
    const perUnit = (height * camera.zoom) / (camera.top - camera.bottom);
    return (size) => size * perUnit;
  }
  return (size, distance) => projectedPixels(size, distance, camera.fov, height);
}

/** What a camera makes of a cell: its size on screen, from the cell's center and side. */
export function cameraPixels(camera: ViewCamera, height: number): (center: Vec3, size: number) => number {
  const e = camera.matrixWorld.elements;
  const pixels = screenPixels(camera, height);
  return (c, size) => pixels(size, Math.hypot(c[0] - e[12], c[1] - e[13], c[2] - e[14]));
}

/** What a cut depends on from the camera; the same text means the same cut. */
export const viewKey = (camera: ViewCamera, height: number) =>
  `${camera.matrixWorld.elements.join()},${camera.projectionMatrix.elements.join()},${height}`;

/** Ranges of `order` that draw, as `from`, `to` pairs in `t`: one element each, or a cell standing for its elements. */
export interface Items {
  from: Uint32Array;
  to: Uint32Array;
  count: number;
  /** Item indices in bit-reversed order, so a prefix of them is spread evenly over the layer. */
  ranks: Uint32Array;
}

/**
 * The cut through the octree a camera calls for: a cell smaller than `threshold` pixels on screen draws as one
 * item when it holds more than one element `use` keeps; larger cells draw their children, and the last level its
 * elements one by one. Only elements `use` keeps take part; deterministic for a camera.
 */
export function lodItems(
  t: Octree,
  use: Uint8Array,
  pixels: (center: Vec3, size: number) => number,
  threshold = LOD_PIXELS,
): Items {
  const { order, levels, children, shifts, from, to } = t;
  const bits = shifts[0];
  const q = [0, 0, 0];
  const center: Vec3 = [0, 0, 0];
  let count = 0;
  const single = (s: number, e: number) => {
    for (let p = s; p < e; p++)
      if (use[order[p]]) {
        from[count] = p;
        to[count++] = p + 1;
      }
  };
  const visit = (l: number, a: number) => {
    const s = levels[l][a];
    const e = levels[l][a + 1];
    const shift = shifts[l];
    quantize(t.xyz, order[s], t.lo, t.cell, bits, q);
    const side = 2 ** shift;
    for (let k = 0; k < 3; k++) center[k] = t.lo[k] + (Math.floor(q[k] / side) + 0.5) * side * t.cell[k];
    const size = side * Math.max(t.cell[0], t.cell[1], t.cell[2]);
    if (e - s > 1 && pixels(center, size) < threshold) {
      let first = -1;
      let used = 0;
      for (let p = s; p < e && used < 2; p++)
        if (use[order[p]]) {
          if (first < 0) first = p;
          used++;
        }
      if (used === 1) {
        from[count] = first;
        to[count++] = first + 1;
      } else if (used > 1) {
        from[count] = s;
        to[count++] = e;
      }
      return;
    }
    if (l === levels.length - 1) return single(s, e);
    for (let c = children[l][a]; c < children[l][a + 1]; c++) visit(l + 1, c);
  };
  for (let a = 0; a < levels[0].length - 1; a++) visit(0, a);
  return { from, to, count, ranks: bitReversed(count) };
}

/** Mean of the finite values of the kept elements of `order[from:to]`; NaN for none. */
export function meanOf(data: ArrayLike<number>, order: Uint32Array, from: number, to: number, use: Uint8Array): number {
  let sum = 0;
  let n = 0;
  for (let p = from; p < to; p++) {
    const v = data[order[p]];
    if (use[order[p]] && Number.isFinite(v)) {
      sum += v;
      n++;
    }
  }
  return n ? sum / n : NaN;
}

/** Most frequent category code of the kept elements of `order[from:to]`, the lowest on a tie; -1 for none. */
export function modeOf(codes: ArrayLike<number>, order: Uint32Array, from: number, to: number, use: Uint8Array, counts: Int32Array): number {
  let best = -1;
  for (let p = from; p < to; p++) {
    const c = codes[order[p]];
    if (use[order[p]] && c >= 0 && c < counts.length) counts[c]++;
  }
  for (let p = from; p < to; p++) {
    const c = codes[order[p]];
    if (!use[order[p]] || c < 0 || c >= counts.length) continue;
    if (best < 0 || counts[c] > counts[best] || (counts[c] === counts[best] && c < best)) best = c;
  }
  for (let p = from; p < to; p++) {
    const c = codes[order[p]];
    if (c >= 0 && c < counts.length) counts[c] = 0;
  }
  return best;
}
