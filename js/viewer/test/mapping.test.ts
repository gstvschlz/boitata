import { describe, expect, it } from "vitest";
import { logRange, mapIndex, mapPosition, parseCutoffs } from "../src/color";
import { colorBarSvg } from "../src/colorbar";
import { percentileRange } from "../src/filter";
import { resolveTheme } from "../src/theme";
import { logTicks } from "../src/ticks";

describe("log mapping", () => {
  const log = { lo: 1, hi: 1000, log: true };
  it("spaces decades evenly", () => {
    expect(mapPosition(1, log)).toBe(0);
    expect(mapPosition(10, log)).toBeCloseTo(1 / 3);
    expect(mapPosition(100, log)).toBeCloseTo(2 / 3);
    expect(mapPosition(1000, log)).toBe(1);
  });
  it("puts values at or below 0 below the range", () => {
    expect(mapIndex(0, log)).toBe(0);
    expect(mapIndex(-5, log)).toBe(0);
    expect(mapIndex(0.5, log)).toBe(0);
  });
  it("takes a thousandth of the top for a low end at or below 0", () => {
    expect(logRange(0, 50)).toEqual([0.05, 50]);
    expect(mapPosition(0.5, { lo: -1, hi: 50, log: true })).toBeCloseTo(Math.log10(10) / Math.log10(1000));
  });
  it("ticks at decades, with 2 and 5 over a short span", () => {
    expect(logTicks(0.5, 2000)).toEqual([1, 10, 100, 1000]);
    expect(logTicks(1, 20)).toEqual([1, 2, 5, 10, 20]);
    expect(logTicks(0, 10)).toEqual([]);
  });
});

describe("bins", () => {
  const bins = { lo: 0, hi: 10, bins: [0.3, 0.5, 1] };
  it("steps values into the middle of their class", () => {
    expect(mapPosition(0.1, bins)).toBeCloseTo(1 / 8);
    expect(mapPosition(0.3, bins)).toBeCloseTo(3 / 8);
    expect(mapPosition(0.7, bins)).toBeCloseTo(5 / 8);
    expect(mapPosition(5, bins)).toBeCloseTo(7 / 8);
  });
  it("reverses", () => {
    expect(mapPosition(5, { ...bins, reverse: true })).toBeCloseTo(1 / 8);
    expect(mapPosition(0, { lo: 0, hi: 1, reverse: true })).toBe(1);
  });
  it("parses typed cutoffs", () => {
    expect(parseCutoffs("1, 0.3 0.5;0.5 x")).toEqual([0.3, 0.5, 1]);
    expect(parseCutoffs("  ")).toBeNull();
  });
  it("draws one swatch per class and labels the cutoffs", () => {
    const svg = colorBarSvg({ title: "cu", cmap: "viridis", lo: 0, hi: 1, bins: [0.3, 0.5], categories: null, colors: null }, resolveTheme("light", false), "x");
    expect(svg.match(/<rect x="[\d.]+" y="28"/g)).toHaveLength(3);
    expect(svg).toContain(">0.3<");
  });
});

describe("percentile range", () => {
  it("interpolates between ranks and skips nulls", () => {
    const data = Float32Array.from({ length: 101 }, (_, i) => i);
    expect(percentileRange(data)).toEqual([2, 98]);
    expect(percentileRange([NaN, 4, NaN], 2, 98)).toEqual([4, 4]);
    expect(percentileRange([NaN])).toBeNull();
    expect(percentileRange([0, 10], 25, 75)).toEqual([2.5, 7.5]);
  });
});
