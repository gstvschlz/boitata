import { describe, expect, it } from "vitest";
import { decodePick, encodePick, formatValue, isClick, nearestHit, NO_ROW } from "../src/pick";

describe("id encoding", () => {
  it("round-trips layers and rows up to 24 bits", () => {
    for (const [layer, row] of [
      [0, 0],
      [3, 255],
      [7, 256],
      [254, 70_000],
      [1, NO_ROW - 1],
    ])
      expect(decodePick(...encodePick(layer, row))).toEqual({ layer, row });
  });

  it("reads the cleared background and rowless elements as nothing", () => {
    expect(decodePick(0, 0, 0, 0)).toBeNull();
    expect(decodePick(...encodePick(2, -1))).toBeNull();
    expect(decodePick(...encodePick(2, NO_ROW))).toBeNull();
  });

  it("takes the hit nearest the click in the window read back", () => {
    const pixels = new Uint8Array(4 * 5 * 3);
    pixels.set(encodePick(1, 10), 4 * (0 * 5 + 0));
    pixels.set(encodePick(2, 20), 4 * (1 * 5 + 3));
    expect(nearestHit(pixels, 5, 3, 2, 1)).toEqual({ layer: 2, row: 20 });
    expect(nearestHit(pixels, 5, 3, 0, 1)).toEqual({ layer: 1, row: 10 });
    expect(nearestHit(new Uint8Array(4 * 9), 3, 3, 1, 1)).toBeNull();
  });
});

describe("click or drag", () => {
  it("picks on a release near the press, never at the end of a drag", () => {
    expect(isClick([100, 100], [102, 103], 5)).toBe(true);
    expect(isClick([100, 100], [140, 100], 5)).toBe(false);
    expect(isClick([100, 100], [104, 103], 5)).toBe(false);
    expect(isClick(null, [100, 100], 5)).toBe(false);
  });
});

describe("card values", () => {
  it("shows nulls as an em dash, never NaN", () => {
    for (const value of [null, undefined, Number.NaN, Infinity]) expect(formatValue(value)).toBe("—");
  });

  it("drops float32 noise and keeps categories", () => {
    expect(formatValue(Math.fround(0.1))).toBe("0.1");
    expect(formatValue(Math.fround(1234.5678))).toBe("1234.568");
    expect(formatValue(-3)).toBe("-3");
    expect(formatValue("lens_2")).toBe("lens_2");
  });
});
