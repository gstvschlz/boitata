import { describe, expect, it } from "vitest";
import { backWalls, boxOf, type Box, visibleBox } from "../src/bounds";
import { CATEGORICAL, SEQUENTIAL } from "../src/colormaps";
import { hexToRgb, lut, lutColor, lutIndex } from "../src/color";
import { columnPaint } from "../src/paint";
import { resolveTheme, THEMES, withBase } from "../src/theme";
import { formatTicks, niceStep, niceTicks } from "../src/ticks";

describe("nice ticks", () => {
  it("steps are 1, 2 or 5 times a power of ten", () => {
    for (const span of [0.0007, 0.3, 1, 7, 13, 99, 451, 12_345, 9.9e6]) {
      const step = niceStep(span);
      const mantissa = step / 10 ** Math.floor(Math.log10(step) + 1e-12);
      expect([1, 2, 5, 10]).toContainEqual(Number(mantissa.toPrecision(6)));
    }
  });

  it("covers the range with multiples of the step, inside it", () => {
    const ticks = niceTicks(11_850.3, 13_020.9);
    expect(ticks).toEqual([12_000, 12_200, 12_400, 12_600, 12_800, 13_000]);
    expect(niceTicks(-0.37, 0.41)).toEqual([-0.2, 0, 0.2, 0.4]);
    expect(niceTicks(5, 5)).toEqual([5]);
    expect(niceTicks(Number.NaN, 1)).toEqual([]);
  });

  it("labels carry the decimals the step needs", () => {
    expect(formatTicks([12_000, 12_200])).toEqual(["12000", "12200"]);
    expect(formatTicks([-0.2, -0, 0.2])).toEqual(["-0.2", "0.0", "0.2"]);
    expect(formatTicks([0.05, 0.1, 0.15])).toEqual(["0.05", "0.10", "0.15"]);
  });
});

describe("colormaps", () => {
  it("ship 256 entries each", () => {
    for (const hex of Object.values(SEQUENTIAL)) expect(hex).toHaveLength(256 * 6);
    expect(CATEGORICAL.length).toBeGreaterThanOrEqual(8);
  });

  it("map the range ends to the first and last entries and clamp outside", () => {
    expect(lutIndex(0, 0, 10)).toBe(0);
    expect(lutIndex(10, 0, 10)).toBe(255);
    expect(lutIndex(-3, 0, 10)).toBe(0);
    expect(lutIndex(99, 0, 10)).toBe(255);
    expect(lutIndex(5, 0, 10)).toBe(128);
    expect(lutIndex(5, 5, 5)).toBe(128);
  });

  it("look colors up in the table", () => {
    expect(lutColor("viridis", 0, 0, 1)).toEqual(hexToRgb(SEQUENTIAL.viridis.slice(0, 6)));
    expect(lutColor("magma", 1, 0, 1)).toEqual(hexToRgb(SEQUENTIAL.magma.slice(-6)));
    expect(Array.from(lut("turbo").slice(3 * 100, 3 * 101))).toEqual(hexToRgb(SEQUENTIAL.turbo.slice(600, 606)));
  });

  it("never color nulls", () => {
    const values = new Float32Array([0, Number.NaN, 1]);
    const column = { name: "v", type: "number" as const, buffer: "", on: "row" as const, min: 0, max: 1 };
    const paint = columnPaint(column, values, { cmap: "viridis", lo: 0, hi: 1, categories: [] }, [9, 9, 9]);
    expect(Array.from(paint.keep!)).toEqual([1, 0, 1]);
    const codes = new Int32Array([1, -1, 0]);
    const text = { name: "t", type: "text" as const, buffer: "", on: "row" as const, categories: ["b", "c"] };
    const painted = columnPaint(text, codes, { cmap: "viridis", lo: 0, hi: 1, categories: ["a", "b", "c"] }, [9, 9, 9]);
    expect(Array.from(painted.keep!)).toEqual([1, 0, 1]);
    expect(Array.from(painted.colors!.slice(0, 3))).toEqual(hexToRgb(CATEGORICAL[2]));
    expect(Array.from(painted.colors!.slice(6, 9))).toEqual(hexToRgb(CATEGORICAL[1]));
  });
});

describe("themes", () => {
  it("auto follows the host", () => {
    expect(resolveTheme("auto", true)).toEqual(THEMES.dark);
    expect(resolveTheme("auto", false)).toEqual(THEMES.light);
    expect(resolveTheme("light", true)).toEqual(THEMES.light);
  });

  it("a dict overrides keys of its base", () => {
    const theme = resolveTheme({ base: "dark", background: "#000000", bogus: "x" } as never, false);
    expect(theme.background).toBe("#000000");
    expect(theme.text).toBe(THEMES.dark.text);
    expect("bogus" in theme).toBe(false);
    expect(resolveTheme({ accent: "red" }, true).background).toBe(THEMES.dark.background);
  });

  it("switching keeps the overrides", () => {
    expect(withBase("auto", "dark")).toBe("dark");
    expect(withBase({ base: "light", grid: "#123456" }, "dark")).toEqual({ base: "dark", grid: "#123456" });
  });
});

describe("bounds", () => {
  const a: Box = { min: [0, 0, 0], max: [1, 1, 1] };
  const b: Box = { min: [5, -2, 0], max: [6, 0, 3] };

  it("box of the visible layers only", () => {
    expect(visibleBox([{ visible: true, box: a }, { visible: false, box: b }])).toEqual(a);
    expect(visibleBox([{ visible: true, box: a }, { visible: true, box: b }])).toEqual({
      min: [0, -2, 0],
      max: [6, 1, 3],
    });
    expect(visibleBox([{ visible: false, box: a }])).toBeNull();
  });

  it("box of points skips NaN", () => {
    expect(boxOf([1, 2, 3, Number.NaN, Number.NaN, Number.NaN, -1, 5, 0])).toEqual({ min: [-1, 2, 0], max: [1, 5, 3] });
  });

  it("back walls face away from the camera", () => {
    expect(backWalls(a, [10, 10, 10])).toEqual([0, 0, 0]);
    expect(backWalls(a, [-10, 10, -10])).toEqual([1, 0, 1]);
    expect(backWalls(a, [0.4, 0.6, 50])).toEqual([1, 0, 0]);
  });
});
