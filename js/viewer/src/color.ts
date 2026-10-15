import { CATEGORICAL, SEQUENTIAL } from "./colormaps";

export const COLORMAPS = Object.keys(SEQUENTIAL);
export const DEFAULT_COLORMAP = "viridis";
export type RGB = [number, number, number];

const tables = new Map<string, Uint8Array>();

/** 256 x 3 bytes of a sequential colormap. */
export function lut(name: string): Uint8Array {
  let table = tables.get(name);
  if (!table) {
    const hex = SEQUENTIAL[name] ?? SEQUENTIAL[DEFAULT_COLORMAP];
    table = new Uint8Array(768);
    for (let i = 0; i < 768; i++) table[i] = parseInt(hex.slice(2 * i, 2 * i + 2), 16);
    tables.set(name, table);
  }
  return table;
}

/** LUT entry of `value` in `[lo, hi]`, clamped; the middle entry when the range is empty. */
export function lutIndex(value: number, lo: number, hi: number): number {
  if (!(hi > lo)) return 128;
  return Math.min(255, Math.max(0, Math.floor(((value - lo) / (hi - lo)) * 256)));
}

/** How numbers map onto a colormap: a range, linear or log, optionally stepped at `bins` cutoffs, optionally reversed. */
export interface Mapping {
  lo: number;
  hi: number;
  log?: boolean;
  bins?: number[] | null;
  reverse?: boolean;
}

/**
 * Where `value` falls along a mapping, 0 at its low end and 1 at its high end, clamped. On a log scale a value at
 * or below 0 is below the range. With cutoffs, the value takes the middle of its class: below the first cutoff,
 * between two, or from the last up.
 */
export function mapPosition(value: number, m: Mapping): number {
  let t: number;
  if (m.bins?.length) {
    let k = 0;
    while (k < m.bins.length && value >= m.bins[k]) k++;
    t = (k + 0.5) / (m.bins.length + 1);
  } else if (m.log) {
    const [lo, hi] = logRange(m.lo, m.hi);
    t = value > 0 && hi > lo ? (Math.log10(value) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo)) : 0;
  } else t = m.hi > m.lo ? (value - m.lo) / (m.hi - m.lo) : 0.5;
  t = Math.min(1, Math.max(0, t));
  return m.reverse ? 1 - t : t;
}

/** A range a log scale can take: a low end at or below 0 becomes a thousandth of the high end. */
export function logRange(lo: number, hi: number): [number, number] {
  return [lo > 0 ? lo : hi > 0 ? hi / 1000 : 1, hi > 0 ? hi : 10];
}

/** Cutoffs typed as numbers apart by commas or spaces: the finite ones, ascending and distinct; null for none. */
export function parseCutoffs(text: string): number[] | null {
  const out = [...new Set(text.split(/[\s,;]+/).filter(Boolean).map(Number).filter(Number.isFinite))].sort((a, b) => a - b);
  return out.length ? out : null;
}

/** LUT entry of `value` under a mapping. */
export function mapIndex(value: number, m: Mapping): number {
  return Math.min(255, Math.floor(mapPosition(value, m) * 256));
}

export function lutColor(name: string, value: number, lo: number, hi: number): RGB {
  const table = lut(name);
  const i = 3 * lutIndex(value, lo, hi);
  return [table[i], table[i + 1], table[i + 2]];
}

export function categoryColor(index: number): RGB {
  return hexToRgb(CATEGORICAL[((index % CATEGORICAL.length) + CATEGORICAL.length) % CATEGORICAL.length]);
}

export function hexToRgb(hex: string): RGB {
  let h = hex.replace("#", "");
  if (h.length === 3) h = [...h].map((c) => c + c).join("");
  const n = parseInt(h.slice(0, 6), 16);
  return Number.isNaN(n) ? [128, 128, 128] : [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

export function rgbToHex([r, g, b]: RGB): string {
  return "#" + [r, g, b].map((v) => v.toString(16).padStart(2, "0")).join("");
}

/** CSS color as RGB bytes, using the browser's parser when available. */
export function cssToRgb(css: string): RGB {
  if (/^#([0-9a-f]{3}|[0-9a-f]{6})$/i.test(css)) return hexToRgb(css);
  if (typeof document === "undefined") return [128, 128, 128];
  const ctx = document.createElement("canvas").getContext("2d");
  if (!ctx) return [128, 128, 128];
  ctx.fillStyle = "#808080";
  ctx.fillStyle = css;
  return hexToRgb(String(ctx.fillStyle));
}
