import { categoryColor, lut, lutIndex, type RGB } from "./color";
import type { ColumnSpec } from "./types";

/** Colors of a layer's elements: per row, vertex or face, or one solid color. `keep[i] = 0` hides element i. */
export interface Paint {
  on: ColumnSpec["on"] | null;
  colors: Uint8Array | null;
  keep: Uint8Array | null;
  solid: RGB;
}

export interface Scale {
  cmap: string;
  lo: number;
  hi: number;
  /** Category list shared by every layer the variable colors. */
  categories: string[];
}

export function solidPaint(rgb: RGB): Paint {
  return { on: null, colors: null, keep: null, solid: rgb };
}

/** Paint of a column on a scale; null values are hidden, never colored. */
export function columnPaint(column: ColumnSpec, data: Float32Array | Int32Array, scale: Scale, solid: RGB): Paint {
  const n = data.length;
  const colors = new Uint8Array(3 * n);
  const keep = new Uint8Array(n);
  if (column.type === "text") {
    const index = new Map(scale.categories.map((c, i) => [c, i]));
    const palette = (column.categories ?? []).map((c) => categoryColor(index.get(c) ?? 0));
    for (let i = 0; i < n; i++) {
      const code = data[i];
      if (code < 0 || code >= palette.length) continue;
      keep[i] = 1;
      colors.set(palette[code], 3 * i);
    }
  } else {
    const table = lut(scale.cmap);
    for (let i = 0; i < n; i++) {
      const v = data[i];
      if (!Number.isFinite(v)) continue;
      keep[i] = 1;
      const j = 3 * lutIndex(v, scale.lo, scale.hi);
      colors[3 * i] = table[j];
      colors[3 * i + 1] = table[j + 1];
      colors[3 * i + 2] = table[j + 2];
    }
  }
  return { on: column.on, colors, keep, solid };
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

export const kept = (paint: Paint, i: number): boolean => !paint.keep || paint.keep[i] === 1;
