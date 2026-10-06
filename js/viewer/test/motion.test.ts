import { afterEach, describe, expect, it, vi } from "vitest";
import { representations } from "../src/layers/index";
import { faceToVertex } from "../src/layers/points";
import {
  adapt,
  FLOOR,
  MIN_FRACTION,
  midpoints,
  motionFraction,
  motionPixelRatio,
  Settle,
  startFraction,
  stratifiedOrder,
  subsetCount,
  TARGET_MS,
} from "../src/motion";
import { formatRange } from "../src/ticks";

function grid(side: number): Float32Array {
  const xyz = new Float32Array(3 * side ** 3);
  let i = 0;
  for (let x = 0; x < side; x++)
    for (let y = 0; y < side; y++) for (let z = 0; z < side; z++) xyz.set([x, y, z], 3 * i++);
  return xyz;
}

describe("stratified order", () => {
  it("is a deterministic permutation", () => {
    const xyz = grid(7);
    const n = xyz.length / 3;
    const order = stratifiedOrder(xyz, n);
    expect([...order].sort((a, b) => a - b)).toEqual(Array.from({ length: n }, (_, i) => i));
    expect(stratifiedOrder(xyz, n)).toEqual(order);
  });

  it("spreads every prefix over space", () => {
    const side = 16;
    const xyz = grid(side);
    const order = stratifiedOrder(xyz, side ** 3);
    for (const k of [64, 512]) {
      const octants = new Array(8).fill(0);
      const halves = [0, 0];
      for (const i of order.subarray(0, k)) {
        const [x, y, z] = xyz.subarray(3 * i, 3 * i + 3);
        octants[(x >= side / 2 ? 4 : 0) + (y >= side / 2 ? 2 : 0) + (z >= side / 2 ? 1 : 0)]++;
        halves[x >= side / 4 && x < (3 * side) / 4 ? 1 : 0]++;
      }
      expect(octants).toEqual(new Array(8).fill(k / 8));
      expect(halves).toEqual([k / 2, k / 2]);
    }
  });

  it("handles NaN, flat and tiny inputs", () => {
    expect(Array.from(stratifiedOrder([0, 0, 0], 1))).toEqual([0]);
    const order = stratifiedOrder([0, 0, 0, Number.NaN, 1, 1, 2, 2, 2], 3);
    expect([...order].sort()).toEqual([0, 1, 2]);
  });

  it("orders segments by their midpoints", () => {
    expect(Array.from(midpoints(new Float32Array([0, 0, 0, 2, 4, 6])))).toEqual([1, 2, 3]);
  });
});

describe("motion quality", () => {
  it("maps the quality to a fraction", () => {
    expect(motionFraction("full", 0.25, 0.1)).toBe(1);
    expect(motionFraction("fast", 0.25, 0.1)).toBe(0.25);
    expect(motionFraction("auto", 0.25, 0.1)).toBe(0.1);
  });

  it("keeps small layers whole and a share of large ones", () => {
    expect(subsetCount(FLOOR, 0.01)).toBe(FLOOR);
    expect(subsetCount(100_000, 1)).toBe(100_000);
    expect(subsetCount(100_000, 0.25)).toBe(25_000);
    expect(subsetCount(100_000, 0.001)).toBe(FLOOR);
  });

  it("auto starts from the budget and adapts to the frame time", () => {
    expect(startFraction(1000)).toBe(1);
    expect(startFraction(1e7)).toBeCloseTo(0.025);
    expect(startFraction(1e12)).toBe(MIN_FRACTION);
    expect(adapt(0.5, TARGET_MS)).toBe(0.5);
    expect(adapt(0.5, 4 * TARGET_MS)).toBe(0.25);
    expect(adapt(0.5, 1.5 * TARGET_MS)).toBeCloseTo(0.5 / 1.5);
    expect(adapt(0.5, 10)).toBe(0.625);
    expect(adapt(0.9, 10)).toBe(1);
    expect(adapt(MIN_FRACTION, 1000)).toBe(MIN_FRACTION);
    expect(adapt(0.5, Number.NaN)).toBe(0.5);
  });

  it("lowers the pixel ratio with the fraction, to half at most", () => {
    expect(motionPixelRatio(2, 1)).toBe(2);
    expect(motionPixelRatio(2, 0.64)).toBeCloseTo(1.6);
    expect(motionPixelRatio(2, 0.01)).toBe(1);
  });
});

describe("settle", () => {
  afterEach(() => vi.useRealTimers());

  it("reports motion once and settles after the last poke", () => {
    vi.useFakeTimers();
    const changes: boolean[] = [];
    const settle = new Settle((moving) => changes.push(moving), 150);
    settle.poke();
    vi.advanceTimersByTime(100);
    settle.poke();
    vi.advanceTimersByTime(100);
    expect(changes).toEqual([true]);
    expect(settle.moving).toBe(true);
    vi.advanceTimersByTime(60);
    expect(changes).toEqual([true, false]);
    expect(settle.moving).toBe(false);
  });

  it("stops at once and only reports a change", () => {
    vi.useFakeTimers();
    const changes: boolean[] = [];
    const settle = new Settle((moving) => changes.push(moving));
    settle.stop();
    settle.poke();
    settle.stop();
    vi.runAllTimers();
    expect(changes).toEqual([true, false]);
  });
});

describe("representations", () => {
  it("list each kind's names, the default first", () => {
    expect(representations("drillholes")).toEqual(["lines", "tubes", "points"]);
    expect(representations("blocks")).toEqual(["cells", "wireframe", "points"]);
    expect(representations("mesh")).toEqual(["surface", "wireframe", "points"]);
    expect(representations("points")).toEqual(["points", "spheres"]);
  });

  it("mesh points under a face column show where a shown face touches them", () => {
    const paint = {
      on: "face" as const,
      colors: new Uint8Array([10, 20, 30, 40, 50, 60]),
      keep: new Uint8Array([1, 0]),
      solid: [0, 0, 0] as [number, number, number],
    };
    const vertex = faceToVertex(paint, new Uint32Array([0, 1, 2, 1, 3, 2]), 4);
    expect(Array.from(vertex.keep!)).toEqual([1, 1, 1, 0]);
    expect(Array.from(vertex.colors!.slice(0, 3))).toEqual([10, 20, 30]);
  });
});

describe("range text", () => {
  it("rounds outward to a precision that follows the span", () => {
    expect(formatRange(51.63, 601.2)).toEqual(["51", "602"]);
    expect(formatRange(0.2, 0.8)).toEqual(["0.20", "0.80"]);
    expect(formatRange(-0.0123, 0.0456)).toEqual(["-0.013", "0.046"]);
    expect(formatRange(12_000.4, 13_020.9)).toEqual(["12000", "13021"]);
    expect(formatRange(5, 5)).toEqual(["5.0", "5.0"]);
    expect(formatRange(0, 0)).toEqual(["0.00", "0.00"]);
  });
});
