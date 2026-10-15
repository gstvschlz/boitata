import { describe, expect, it } from "vitest";
import { lodItems, meanOf, modeOf, octree, projectedPixels } from "../src/lod";
import { mergedColor, solidPaint } from "../src/paint";

function cloud(n: number, side = 100): Float32Array {
  const xyz = new Float32Array(3 * n);
  let s = 7;
  for (let i = 0; i < xyz.length; i++) {
    s = (s * 16807) % 2147483647;
    xyz[i] = (s / 2147483647) * side;
  }
  return xyz;
}

describe("aggregation", () => {
  const order = Uint32Array.from([0, 1, 2, 3, 4]);
  const all = new Uint8Array(5).fill(1);
  it("takes the mean of finite kept values", () => {
    const use = Uint8Array.from([1, 1, 0, 1, 1]);
    expect(meanOf([1, 2, 100, NaN, 3], order, 0, 5, use)).toBe(2);
    expect(meanOf([NaN, NaN], order, 0, 2, all)).toBeNaN();
  });
  it("takes the most frequent category, the lowest on a tie, and resets its counts", () => {
    const counts = new Int32Array(4);
    expect(modeOf([3, 1, 3, 1, 2], order, 0, 5, all, counts)).toBe(1);
    expect(modeOf([3, 1, 3, -1, 2], order, 0, 5, all, counts)).toBe(3);
    expect(modeOf([-1, -1], order, 0, 2, all, counts)).toBe(-1);
    expect([...counts]).toEqual([0, 0, 0, 0]);
  });
  it("colors a merged element by the mean through the scale", () => {
    const out = new Uint8Array(3);
    const paint = { ...solidPaint([9, 9, 9]), values: Float32Array.from([0, 10]), colorOf: (v: number, o: Uint8Array, at: number) => (o[at] = v) };
    mergedColor(paint, Uint32Array.from([0, 1]), 0, 2, all, out, 0, new Int32Array(0));
    expect(out[0]).toBe(5);
    mergedColor(solidPaint([9, 8, 7]), order, 0, 2, all, out, 0, new Int32Array(0));
    expect([...out]).toEqual([9, 8, 7]);
  });
});

describe("level selection", () => {
  it("shrinks a cell's screen size with distance", () => {
    expect(projectedPixels(1, 10, 90, 200)).toBeCloseTo(10);
    expect(projectedPixels(1, 20, 90, 200)).toBeCloseTo(5);
    expect(projectedPixels(5, 4, 30, 600)).toBe(Infinity);
  });

  const n = 50_000;
  const xyz = cloud(n);
  const t = octree(xyz, n);
  const use = new Uint8Array(n).fill(1);
  const kept = (items: ReturnType<typeof lodItems>) => {
    let total = 0;
    for (let k = 0; k < items.count; k++) total += items.to[k] - items.from[k];
    return total;
  };

  it("draws everything one by one up close and one cell from afar", () => {
    const near = lodItems(t, use, () => Infinity);
    expect(near.count).toBe(n);
    const far = lodItems(t, use, () => 0);
    expect(far.count).toBe(1);
    expect(kept(far)).toBe(n);
  });

  it("refines cells by their size on screen, covering every kept element once", () => {
    const eye = [-1500, 50, 50];
    const pixels = (c: number[], size: number) => projectedPixels(size, Math.hypot(c[0] - eye[0], c[1] - eye[1], c[2] - eye[2]), 30, 400);
    const items = lodItems(t, use, pixels);
    expect(items.count).toBeGreaterThan(1);
    expect(items.count).toBeLessThan(n);
    expect(kept(items)).toBe(n);
    const again = lodItems(octree(xyz, n), use, pixels);
    expect(again.count).toBe(items.count);
    expect([...again.from.subarray(0, again.count)]).toEqual([...items.from.subarray(0, items.count)]);
    expect([...items.ranks].sort((a, b) => a - b)).toEqual(Array.from({ length: items.count }, (_, k) => k));
  });

  it("leaves out elements it does not keep", () => {
    const half = Uint8Array.from({ length: n }, (_, i) => i % 2);
    const items = lodItems(t, half, () => Infinity);
    expect(items.count).toBe(n / 2);
  });
});
