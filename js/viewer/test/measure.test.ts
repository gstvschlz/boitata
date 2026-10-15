import { describe, expect, it } from "vitest";
import { measure, metersPerPixel, scaleBar } from "../src/measure";

describe("measure", () => {
  it("reads azimuth clockwise from north", () => {
    expect(measure([0, 0, 0], [0, 10, 0]).azimuth).toBeCloseTo(0);
    expect(measure([0, 0, 0], [10, 0, 0]).azimuth).toBeCloseTo(90);
    expect(measure([0, 0, 0], [0, -10, 0]).azimuth).toBeCloseTo(180);
    expect(measure([0, 0, 0], [-10, 0, 0]).azimuth).toBeCloseTo(270);
    expect(measure([5, 5, 0], [6, 6, 0]).azimuth).toBeCloseTo(45);
  });
  it("reads dip positive down and distances in 3D and in plan", () => {
    const m = measure([0, 0, 100], [3, 0, 96]);
    expect(m.distance).toBeCloseTo(5);
    expect(m.horizontal).toBeCloseTo(3);
    expect(m.dip).toBeCloseTo((Math.atan2(4, 3) * 180) / Math.PI);
    expect(measure([0, 0, 0], [0, 1, 1]).dip).toBeCloseTo(-45);
    const down = measure([0, 0, 0], [0, 0, -2]);
    expect([down.dip, down.azimuth, down.horizontal]).toEqual([90, 0, 0]);
  });
});

describe("scale bar", () => {
  it("picks a round length near the target width", () => {
    const bar = scaleBar(0.37);
    expect(bar.length).toBe(50);
    expect(bar.pixels).toBeCloseTo(50 / 0.37);
  });
  it("spans the view's height at its distance", () => {
    expect(metersPerPixel(10, 90, 200) * 200).toBeCloseTo(20);
  });
});
