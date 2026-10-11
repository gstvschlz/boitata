import { describe, expect, it } from "vitest";
import { columnPaint, type Scale } from "../src/paint";
import type { ColumnSpec } from "../src/types";

describe("columnPaint", () => {
  const column: ColumnSpec = { name: "class", type: "text", buffer: "b", on: "row", categories: ["measured", "inferred"] };

  it("uses the scale's own colors, index-matched to its categories, when given", () => {
    const scale: Scale = { cmap: "viridis", lo: 0, hi: 1, categories: ["measured", "inferred"], colors: ["#ff0000", "#0000ff"] };
    const codes = new Int32Array([0, 1]);
    const paint = columnPaint(column, codes, scale, [0, 0, 0]);
    expect(Array.from(paint.colors!.subarray(0, 3))).toEqual([255, 0, 0]);
    expect(Array.from(paint.colors!.subarray(3, 6))).toEqual([0, 0, 255]);
  });

  it("falls back to the built-in palette without a scheme", () => {
    const scale: Scale = { cmap: "viridis", lo: 0, hi: 1, categories: ["measured", "inferred"], colors: null };
    const codes = new Int32Array([0, 1]);
    const paint = columnPaint(column, codes, scale, [0, 0, 0]);
    expect(Array.from(paint.colors!.subarray(0, 3))).not.toEqual([0, 0, 0]);
  });
});
