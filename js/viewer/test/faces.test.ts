import { describe, expect, it } from "vitest";
import type { Vec3 } from "../src/bounds";
import { ALL_SIDES, exposedSides, type Grid, sideCount, SOLID, WHOLE } from "../src/faces";
import { boxInSlab, buildSection } from "../src/section";

function cube(n: number): Grid {
  return { count: [n, n, n], cells: Uint32Array.from({ length: n ** 3 }, (_, i) => i) };
}

describe("exposed sides", () => {
  it("a solid n³ grid shows 6n² sides", () => {
    for (const n of [1, 2, 5, 100]) {
      const grid = cube(n);
      const sides = exposedSides(grid, new Uint8Array(n ** 3).fill(SOLID));
      expect(sideCount(sides)).toBe(6 * n * n);
    }
  });

  it("removing an interior block exposes its six neighbours' sides", () => {
    const n = 5;
    const state = new Uint8Array(n ** 3).fill(SOLID);
    state[2 + n * (2 + n * 2)] = 0;
    expect(sideCount(exposedSides(cube(n), state))).toBe(6 * n * n + 6);
  });

  it("a whole block shows every side and does not hide its neighbours'", () => {
    const n = 3;
    const state = new Uint8Array(n ** 3).fill(SOLID);
    const middle = 1 + n * (1 + n);
    state[middle] = WHOLE;
    const sides = exposedSides(cube(n), state);
    expect(sides[middle]).toBe(ALL_SIDES);
    expect(sideCount(sides)).toBe(6 * n * n + 6 + 6);
  });

  it("follows the cells of a masked grid and leaves the scratch clean", () => {
    const grid: Grid = { count: [4, 1, 1], cells: Uint32Array.from([0, 1, 3]) };
    const scratch = new Uint8Array(4);
    const sides = exposedSides(grid, new Uint8Array(3).fill(SOLID), scratch);
    expect([...sides]).toEqual([ALL_SIDES & ~1, ALL_SIDES & ~2, ALL_SIDES]);
    expect([...scratch]).toEqual([0, 0, 0, 0]);
  });
});

describe("box against a slab", () => {
  const axes: Vec3[] = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
  const s = buildSection(
    [
      [0, 0],
      [10, 0],
    ],
    0,
    2,
    90,
  )!;
  it("is inside, across or outside", () => {
    expect(boxInSlab(s, [5, 0, 0], axes, [1, 1, 1])).toBe(2);
    expect(boxInSlab(s, [5, 1, 0], axes, [1, 1, 1])).toBe(1);
    expect(boxInSlab(s, [5, 0, 0], axes, [1, 4, 1])).toBe(1);
    expect(boxInSlab(s, [5, 3, 0], axes, [1, 1, 1])).toBe(0);
    expect(boxInSlab(s, [12, 0, 0], axes, [1, 1, 1])).toBe(0);
  });
});
