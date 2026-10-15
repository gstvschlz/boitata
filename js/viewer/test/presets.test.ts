import { describe, expect, it } from "vitest";
import {
  buildSection,
  cutAzimuth,
  faceOn,
  inSlab,
  orthogonalCut,
  presetCut,
  sectionBox,
  stepCut,
  traces,
} from "../src/section";

const close = (a: number[][], b: number[][]) => a.forEach((p, i) => p.forEach((v, k) => expect(v).toBeCloseTo(b[i][k])));

describe("section presets", () => {
  it("cuts along a bearing through a center", () => {
    close(presetCut([10, 20], 0, 100), [[10, -30], [10, 70]]);
    close(presetCut([0, 0], 90, 10), [[-5, 0], [5, 0]]);
    expect(cutAzimuth(presetCut([3, 4], 135, 8))).toBeCloseTo(135);
    expect(cutAzimuth([[0, 0], [0, -1]])).toBeCloseTo(180);
  });

  it("steps square to the cut, to the right of its direction", () => {
    close(stepCut([[0, 0], [0, 10]], 5), [[5, 0], [5, 10]]);
    close(stepCut([[0, 0], [10, 0], [10, 10]], -2), [[0, 2], [10, 2], [10, 12]]);
  });

  it("gives long and cross sections through the middle of the cut", () => {
    const cut: [number, number][] = [[0, 0], [10, 10]];
    const long = orthogonalCut(cut, "long", 20);
    const cross = orthogonalCut(cut, "cross", 20);
    expect(cutAzimuth(long)).toBeCloseTo(45);
    expect(cutAzimuth(cross)).toBeCloseTo(135);
    for (const [a, b] of [long, cross]) {
      expect((a[0] + b[0]) / 2).toBeCloseTo(5);
      expect((a[1] + b[1]) / 2).toBeCloseTo(5);
      expect(Math.hypot(b[0] - a[0], b[1] - a[1])).toBeCloseTo(20);
    }
  });

  it("faces a cut from its left, so it runs left to right on screen", () => {
    expect(faceOn(0)).toBe(270);
    expect(faceOn(90)).toBe(0);
    expect(faceOn(135)).toBe(45);
  });
});

describe("plan slab", () => {
  const s = buildSection([[-50, 0], [50, 0]], 10, 4, 0)!;
  it("keeps the points within half its width of its elevation, anywhere across", () => {
    expect(inSlab(s, [0, 300, 11])).toBe(true);
    expect(inSlab(s, [0, -300, 7.5])).toBe(false);
    expect(inSlab(s, [60, 0, 10])).toBe(false);
  });
  it("bounds the box to its elevations and traces its own polyline", () => {
    expect(sectionBox(s, { min: [0, 0, 0], max: [5, 5, 100] })).toEqual({ min: [0, 0, 8], max: [5, 5, 12] });
    expect(sectionBox(s, { min: [0, 0, 50], max: [5, 5, 100] })).toBeNull();
    const [trace, left, right] = traces(s, 100);
    close(trace, [[-50, 0], [50, 0]]);
    expect(left.length + right.length).toBe(0);
  });
});
