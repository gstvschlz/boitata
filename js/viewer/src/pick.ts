import * as THREE from "three";

/** Rows a layer can address in the id pass: 24 bits, the top value meaning "no row". */
export const NO_ROW = 0xffffff;
/** Pixels around the click searched for the nearest element, so thin lines and small points are easy to hit. */
export const PICK_RADIUS = 3;

/** What the id pass wrote at a pixel: a layer index and one of its rows. */
export interface Hit {
  layer: number;
  row: number;
}

/** RGBA bytes the id pass writes for `row` of layer `layer`; the JS mirror of the shader's `pickColor`. */
export function encodePick(layer: number, row: number): [number, number, number, number] {
  const id = row < 0 ? NO_ROW : row;
  return [id & 255, (id >> 8) & 255, (id >> 16) & 255, layer + 1];
}

/** The element at RGBA bytes from the id pass; null for the background or an element with no row. */
export function decodePick(r: number, g: number, b: number, a: number): Hit | null {
  const row = r | (g << 8) | (b << 16);
  return a === 0 || row === NO_ROW ? null : { layer: a - 1, row };
}

/** The hit nearest pixel (cx, cy) in a `width` by `height` window of RGBA bytes from the id pass. */
export function nearestHit(pixels: Uint8Array, width: number, height: number, cx: number, cy: number): Hit | null {
  let best: Hit | null = null;
  let distance = Infinity;
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const o = 4 * (y * width + x);
      const hit = decodePick(pixels[o], pixels[o + 1], pixels[o + 2], pixels[o + 3]);
      const d = (x - cx) ** 2 + (y - cy) ** 2;
      if (hit && d < distance) {
        best = hit;
        distance = d;
      }
    }
  return best;
}

/** Whether a press and release `slop` pixels apart or closer make a click rather than the end of a drag. */
export function isClick(down: [number, number] | null, up: [number, number], slop: number): boolean {
  return !!down && Math.hypot(up[0] - down[0], up[1] - down[1]) < slop;
}

/** A value for the inspect card: a number to 7 significant digits, a category, or an em dash for null. */
export function formatValue(value: number | string | null | undefined): string {
  if (value === null || value === undefined || (typeof value === "number" && !Number.isFinite(value))) return "—";
  if (typeof value === "string") return value;
  return String(round7(value));
}

/** Drops the noise a float32 value carries past 7 significant digits. */
export function round7(value: number): number {
  return Number(value.toPrecision(7));
}

/** The geometry's row attribute, grown to `n` elements and flagged for upload; representations fill it as they paint. */
export function pickArray(geometry: THREE.BufferGeometry, n: number, instanced: boolean): Float32Array {
  const old = geometry.getAttribute("aPick") as THREE.BufferAttribute | undefined;
  if (old && old.count >= n) {
    old.needsUpdate = true;
    return old.array as Float32Array;
  }
  const array = new Float32Array(Math.max(n, 1));
  geometry.setAttribute("aPick", instanced ? new THREE.InstancedBufferAttribute(array, 1) : new THREE.BufferAttribute(array, 1));
  return array;
}

/** Vertex-shader declarations and the statement passing the row to the fragment shader. */
export const PICK_VERTEX = /* glsl */ `
attribute float aPick;
flat varying float vPick;
`;
export const PICK_PASS = "vPick = aPick;";

/** Fragment-shader declarations: while `uPicking` is set, materials write `pickColor()` instead of their color. */
export const PICK_FRAGMENT = /* glsl */ `
uniform float uPicking;
uniform float uLayer;
flat varying float vPick;
vec4 pickColor() {
  uint id = vPick < -0.5 ? ${NO_ROW}u : uint(vPick + 0.5);
  return vec4(float(id & 255u), float((id >> 8u) & 255u), float((id >> 16u) & 255u), uLayer + 1.0) / 255.0;
}
`;
