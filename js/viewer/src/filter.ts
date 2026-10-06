import * as THREE from "three";
import { niceStep } from "./ticks";
import type { ColumnSpec } from "./types";

/** Filtered columns a layer evaluates at once: one vec4 attribute per instance. */
export const SLOTS = 4;
/** What a null or missing number becomes in the filter attribute; no range admits it. */
export const NULL = 3e38;
const UNBOUNDED = 1e38;
/** Width of the category texture; a slot takes as many rows as its column's categories need. */
const CATEGORY_WIDTH = 1024;
export const BINS = 40;

/** One condition of a layer's filter: a numeric range, bounds null when open, or the categories kept. */
export interface Condition {
  column: ColumnSpec;
  lo: number | null;
  hi: number | null;
  categories: Set<string> | null;
}

/** Filter state as Python reads and writes it: column name to a range or a list of categories. */
export type FilterState = Record<string, { range: [number | null, number | null] } | { categories: string[] }>;

/** A filtered column's values, as the representations write them into the filter attribute. */
export interface Slot {
  on: ColumnSpec["on"];
  data: Float32Array | Int32Array;
  text: boolean;
}

/** The value element `i` carries for `slot`; -1 is no element (a face missing beside an edge). */
export function encoded(slot: Slot, i: number): number {
  if (i < 0) return slot.text ? -1 : NULL;
  const v = slot.data[i];
  return slot.text ? v : Number.isFinite(v) ? v : NULL;
}

/** Writes element indices' values for each slot into `out` at `at`; `index(on)` gives the element per association. */
export function writeFilter(slots: Slot[], out: Float32Array, at: number, index: (on: ColumnSpec["on"]) => number): void {
  for (let k = 0; k < slots.length; k++) out[at + k] = encoded(slots[k], index(slots[k].on));
}

/** Uniform values of a filter: per slot a mode (0 off, 1 range, 2 categories), bounds and kept category codes. */
export interface Compiled {
  mode: number[];
  lo: number[];
  hi: number[];
  allowed: (Uint8Array | null)[];
}

export function compile(conditions: Condition[]): Compiled {
  const out: Compiled = { mode: [0, 0, 0, 0], lo: [0, 0, 0, 0], hi: [0, 0, 0, 0], allowed: [null, null, null, null] };
  conditions.slice(0, SLOTS).forEach((c, k) => {
    if (c.column.type === "text") {
      const names = c.column.categories ?? [];
      out.mode[k] = 2;
      out.allowed[k] = Uint8Array.from(names, (name) => (c.categories?.has(name) ? 1 : 0));
    } else {
      out.mode[k] = 1;
      out.lo[k] = c.lo ?? -UNBOUNDED;
      out.hi[k] = c.hi ?? UNBOUNDED;
    }
  });
  return out;
}

/** What the shader decides for one slot; the reference the GPU mirrors. */
export function slotPasses(f: Compiled, k: number, v: number): boolean {
  if (f.mode[k] === 0) return true;
  if (f.mode[k] === 1) return v >= f.lo[k] && v <= f.hi[k];
  const allowed = f.allowed[k]!;
  return v >= 0 && v < allowed.length && allowed[v] === 1;
}

/**
 * 1 where element i of association `on` passes every slot on that association and `keep` (the color column's
 * non-null mask, when it is on the same association).
 */
export function passMask(f: Compiled, slots: Slot[], on: ColumnSpec["on"], n: number, keep: Uint8Array | null): Uint8Array {
  const out = new Uint8Array(n);
  if (keep) out.set(keep.subarray(0, n));
  else out.fill(1);
  slots.forEach((slot, k) => {
    if (slot.on !== on || f.mode[k] === 0) return;
    for (let i = 0; i < n; i++) if (out[i] && !slotPasses(f, k, encoded(slot, i))) out[i] = 0;
  });
  return out;
}

export const FILTER_GLSL = /* glsl */ `
uniform vec4 uFilterLo;
uniform vec4 uFilterHi;
uniform vec4 uFilterMode;
uniform vec4 uFilterJoin;
uniform int uFilterRows;
uniform highp sampler2D uFilterCats;
bool filterSlot(float v, int k) {
  float mode = uFilterMode[k];
  if (mode < 0.5) return true;
  if (mode < 1.5) return v >= uFilterLo[k] && v <= uFilterHi[k];
  if (v < -0.5) return false;
  int code = int(v + 0.5);
  return texelFetch(uFilterCats, ivec2(code % ${CATEGORY_WIDTH}, k * uFilterRows + code / ${CATEGORY_WIDTH}), 0).r > 0.5;
}
bool filterPass(vec4 v) {
  return filterSlot(v.x, 0) && filterSlot(v.y, 1) && filterSlot(v.z, 2) && filterSlot(v.w, 3);
}
bool filterPass2(vec4 a, vec4 b) {
  for (int k = 0; k < 4; k++) {
    bool pa = filterSlot(a[k], k);
    bool pb = filterSlot(b[k], k);
    if (uFilterJoin[k] > 0.5 ? !(pa || pb) : !(pa && pb)) return false;
  }
  return true;
}
`;

/** Vertex-shader statement that sends a failing instance outside the clip volume. */
export const COLLAPSE = "gl_Position = vec4(2.0, 2.0, 2.0, 1.0);";

export type FilterUniforms = LayerFilter["uniforms"];

/**
 * The geometry's filter attribute, grown to `n` elements and flagged for upload, or null while the layer has no
 * filter: without one bound, the shader reads a constant that every slot, all off, lets through.
 */
export function filterArray(
  geometry: THREE.BufferGeometry,
  n: number,
  slots: Slot[],
  instanced: boolean,
  name = "aFilter",
): Float32Array | null {
  if (!slots.length) return null;
  const old = geometry.getAttribute(name) as THREE.BufferAttribute | undefined;
  if (old && old.count >= n) {
    old.needsUpdate = true;
    return old.array as Float32Array;
  }
  const array = new Float32Array(4 * Math.max(n, 1));
  geometry.setAttribute(name, instanced ? new THREE.InstancedBufferAttribute(array, 4) : new THREE.BufferAttribute(array, 4));
  return array;
}

/**
 * A layer's filter: its conditions, the columns its representations write per instance (`slots`) and the uniforms
 * its materials share. Editing a bound or a category only sets uniforms; adding or removing a condition repaints.
 */
export class LayerFilter {
  conditions: Condition[] = [];
  slots: Slot[] = [];
  compiled: Compiled = compile([]);
  readonly uniforms = {
    uFilterLo: { value: new THREE.Vector4() },
    uFilterHi: { value: new THREE.Vector4() },
    uFilterMode: { value: new THREE.Vector4() },
    uFilterJoin: { value: new THREE.Vector4() },
    uFilterRows: { value: 1 },
    uFilterCats: { value: categoryTexture(1) },
  };

  get active(): boolean {
    return this.conditions.length > 0;
  }

  update(): void {
    const f = (this.compiled = compile(this.conditions));
    const u = this.uniforms;
    u.uFilterMode.value.fromArray(f.mode);
    u.uFilterLo.value.fromArray(f.lo);
    u.uFilterHi.value.fromArray(f.hi);
    u.uFilterJoin.value.fromArray([0, 1, 2, 3].map((k) => (this.slots[k]?.on === "face" ? 1 : 0)));
    const rows = Math.max(1, ...f.allowed.map((a) => Math.ceil((a?.length ?? 0) / CATEGORY_WIDTH)));
    if (rows !== u.uFilterRows.value) {
      u.uFilterCats.value.dispose();
      u.uFilterCats.value = categoryTexture(rows);
      u.uFilterRows.value = rows;
    }
    const texels = u.uFilterCats.value.image.data as Uint8Array;
    texels.fill(0);
    f.allowed.forEach((a, k) => a && texels.set(a.map((x) => 255 * x), k * rows * CATEGORY_WIDTH));
    u.uFilterCats.value.needsUpdate = true;
  }

  dispose(): void {
    this.uniforms.uFilterCats.value.dispose();
  }
}

function categoryTexture(rows: number): THREE.DataTexture {
  const texture = new THREE.DataTexture(
    new Uint8Array(CATEGORY_WIDTH * rows * SLOTS),
    CATEGORY_WIDTH,
    rows * SLOTS,
    THREE.RedFormat,
    THREE.UnsignedByteType,
  );
  texture.needsUpdate = true;
  return texture;
}

/** `BINS` counts of the finite values over `[lo, hi]`, the top bin closed. */
export function histogram(data: ArrayLike<number>, lo: number, hi: number, bins = BINS): number[] {
  const counts = new Array<number>(bins).fill(0);
  const width = (hi - lo) / bins;
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (!Number.isFinite(v) || v < lo || v > hi) continue;
    counts[width > 0 ? Math.min(bins - 1, Math.floor((v - lo) / width)) : 0]++;
  }
  return counts;
}

/**
 * Histogram bar heights in [0, 1] on a log scale, so a tail of a few rows stays visible beside a spike of
 * thousands; a bin with any row is at least `floor` high.
 */
export function barHeights(counts: number[], floor = 0.06): number[] {
  const top = Math.log1p(Math.max(0, ...counts));
  return counts.map((c) => (c > 0 && top > 0 ? Math.max(Math.log1p(c) / top, floor) : 0));
}

/** Rows per category code; nulls are not counted. */
export function categoryCounts(codes: Int32Array, n: number): number[] {
  const counts = new Array<number>(n).fill(0);
  for (let i = 0; i < codes.length; i++) if (codes[i] >= 0 && codes[i] < n) counts[codes[i]]++;
  return counts;
}

/** Resolution of a range slider over a column spanning `span`: a hundredth of a tick step. */
export function sliderStep(span: number): number {
  return niceStep(Math.abs(span) || 1) / 100;
}

export function snap(value: number, step: number): number {
  return Number((Math.round(value / step) * step).toPrecision(12));
}

/** A filter bound as text, with the decimals the slider's step carries. */
export function formatBound(value: number, span: number): string {
  const step = sliderStep(span);
  const decimals = Math.min(12, Math.max(0, -Math.floor(Math.log10(step) + 1e-9)));
  const text = snap(value, step).toFixed(decimals);
  return /^-0(\.0*)?$/.test(text) ? text.slice(1) : text;
}

/** Counts in the panel, with thousands separators. */
export const count = (n: number) => n.toLocaleString("en-US");

/** A layer's conditions in the form Python reads. */
export function toState(conditions: Condition[]): FilterState {
  const out: FilterState = {};
  for (const c of conditions)
    out[c.column.name] =
      c.column.type === "text" ? { categories: (c.column.categories ?? []).filter((x) => c.categories?.has(x)) } : { range: [c.lo, c.hi] };
  return out;
}

/** Conditions from a state, on the layer's columns; unknown columns and categories are ignored. */
export function fromState(state: FilterState | undefined, columns: ColumnSpec[]): Condition[] {
  const out: Condition[] = [];
  for (const [name, value] of Object.entries(state ?? {})) {
    const column = columns.find((c) => c.name === name);
    if (!column || out.length >= SLOTS) continue;
    if (column.type === "text" && "categories" in value)
      out.push({ column, lo: null, hi: null, categories: new Set(value.categories.filter((x) => column.categories?.includes(x))) });
    else if (column.type === "number" && "range" in value)
      out.push({ column, lo: value.range[0] ?? null, hi: value.range[1] ?? null, categories: null });
  }
  return out;
}

/** A new condition that keeps everything. */
export function openCondition(column: ColumnSpec): Condition {
  return { column, lo: null, hi: null, categories: column.type === "text" ? new Set(column.categories ?? []) : null };
}
