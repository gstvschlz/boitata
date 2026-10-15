import { categoryColor, hexToRgb, lut, type Mapping, mapIndex, type RGB } from "./color";
import { meanOf, modeOf } from "./lod";
import type { ColumnSpec } from "./types";

/** Colors of a layer's elements: per row, vertex or face, or one solid color. `keep[i] = 0` hides element i. */
export interface Paint {
  on: ColumnSpec["on"] | null;
  colors: Uint8Array | null;
  keep: Uint8Array | null;
  solid: RGB;
  /** The column the colors come from, and the color of any of its values, for elements standing for several. */
  values?: Float32Array | Int32Array;
  colorOf?: (value: number, out: Uint8Array, at: number) => void;
}

export interface Scale extends Mapping {
  cmap: string;
  lo: number;
  hi: number;
  /** Category list shared by every layer the variable colors. */
  categories: string[];
  /** Category colors, index-matched to `categories`; null to use the built-in palette. */
  colors: string[] | null;
}

export function solidPaint(rgb: RGB): Paint {
  return { on: null, colors: null, keep: null, solid: rgb };
}

/** Paint of a column on a scale; null values are hidden, never colored. */
export function columnPaint(column: ColumnSpec, data: Float32Array | Int32Array, scale: Scale, solid: RGB): Paint {
  const n = data.length;
  const colors = new Uint8Array(3 * n);
  const keep = new Uint8Array(n);
  let colorOf: (value: number, out: Uint8Array, at: number) => void;
  if (column.type === "text") {
    const index = new Map(scale.categories.map((c, i) => [c, i]));
    const palette = (column.categories ?? []).map((c) => {
      const i = index.get(c) ?? 0;
      return scale.colors ? hexToRgb(scale.colors[i] ?? "#808080") : categoryColor(i);
    });
    colorOf = (code, out, at) => out.set(palette[code] ?? solid, at);
    for (let i = 0; i < n; i++) {
      const code = data[i];
      if (code < 0 || code >= palette.length) continue;
      keep[i] = 1;
      colors.set(palette[code], 3 * i);
    }
  } else {
    const table = lut(scale.cmap);
    colorOf = (v, out, at) => {
      const j = 3 * mapIndex(v, scale);
      out[at] = table[j];
      out[at + 1] = table[j + 1];
      out[at + 2] = table[j + 2];
    };
    for (let i = 0; i < n; i++) {
      if (!Number.isFinite(data[i])) continue;
      keep[i] = 1;
      colorOf(data[i], colors, 3 * i);
    }
  }
  return { on: column.on, colors, keep, solid, values: data, colorOf };
}

/** Element i's color under `paint`, for representations that fill their own buffers. */
export function colorAt(paint: Paint, i: number, out: Uint8Array, at: number): void {
  if (paint.colors) {
    out[at] = paint.colors[3 * i];
    out[at + 1] = paint.colors[3 * i + 1];
    out[at + 2] = paint.colors[3 * i + 2];
  } else {
    out[at] = paint.solid[0];
    out[at + 1] = paint.solid[1];
    out[at + 2] = paint.solid[2];
  }
}

/**
 * The color of an element standing for the kept elements of `order[from:to]`: that of their mean value, or of
 * their most frequent category; the solid color without a column.
 */
export function mergedColor(
  paint: Paint,
  order: Uint32Array,
  from: number,
  to: number,
  use: Uint8Array,
  out: Uint8Array,
  at: number,
  counts: Int32Array,
): void {
  const { values, colorOf } = paint;
  if (!values || !colorOf) return colorAt(paint, order[from], out, at);
  colorOf(values instanceof Int32Array ? modeOf(values, order, from, to, use, counts) : meanOf(values, order, from, to, use), out, at);
}

/** Scratch `mergedColor` counts categories in. */
export function mergeCounts(paint: Paint): Int32Array {
  let top = -1;
  if (paint.values instanceof Int32Array) for (const c of paint.values) if (c > top) top = c;
  return new Int32Array(top + 1);
}

export const kept =(paint: Paint, i: number): boolean => !paint.keep || paint.keep[i] === 1;
