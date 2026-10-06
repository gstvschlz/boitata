import { describe, expect, it } from "vitest";
import { blocksBox, maskedBox, triangleMask, type Vec3 } from "../src/bounds";
import {
  categoryCounts,
  compile,
  type Condition,
  encoded,
  formatBound,
  fromState,
  histogram,
  LayerFilter,
  NULL,
  passMask,
  type Slot,
  sliderStep,
  slotPasses,
  snap,
  toState,
  writeFilter,
} from "../src/filter";
import type { ColumnSpec } from "../src/types";

const grade: ColumnSpec = { name: "grade", type: "number", buffer: "b0", on: "row", min: 0, max: 10 };
const lens: ColumnSpec = { name: "lens", type: "text", buffer: "b1", on: "row", categories: ["a", "b", "c"] };
const range = (lo: number | null, hi: number | null): Condition => ({ column: grade, lo, hi, categories: null });
const cats = (...kept: string[]): Condition => ({ column: lens, lo: null, hi: null, categories: new Set(kept) });

describe("histogram", () => {
  it("bins finite values over the range, the top bin closed", () => {
    expect(histogram([0, 0.9, 1, 5, 10, Number.NaN, 11, -1], 0, 10, 10)).toEqual([2, 1, 0, 0, 0, 1, 0, 0, 0, 1]);
  });

  it("puts every value of a constant column in one bin", () => {
    expect(histogram([3, 3, 3], 3, 3, 4)).toEqual([3, 0, 0, 0]);
  });

  it("counts categories without nulls", () => {
    expect(categoryCounts(new Int32Array([0, 2, -1, 2, 2]), 3)).toEqual([1, 0, 3]);
  });
});

describe("condition evaluation", () => {
  it("keeps a range inclusively and fails nulls", () => {
    const f = compile([range(2, 5)]);
    const slot: Slot = { on: "row", data: new Float32Array([1, 2, 5, 5.5, Number.NaN]), text: false };
    expect([0, 1, 2, 3, 4].map((i) => slotPasses(f, 0, encoded(slot, i)))).toEqual([false, true, true, false, false]);
  });

  it("leaves open ends unbounded but still fails nulls", () => {
    const f = compile([range(null, 3)]);
    expect(slotPasses(f, 0, -1e30)).toBe(true);
    expect(slotPasses(f, 0, 3)).toBe(true);
    expect(slotPasses(f, 0, NULL)).toBe(false);
    expect(slotPasses(compile([range(null, null)]), 0, NULL)).toBe(false);
  });

  it("keeps the checked categories and fails null codes", () => {
    const f = compile([cats("a", "c")]);
    expect([0, 1, 2, -1].map((code) => slotPasses(f, 0, code))).toEqual([true, false, true, false]);
  });

  it("combines conditions on the same association with and", () => {
    const f = compile([range(2, null), cats("b")]);
    const slots: Slot[] = [
      { on: "row", data: new Float32Array([1, 3, 3, 4]), text: false },
      { on: "row", data: new Int32Array([1, 1, 0, -1]), text: true },
    ];
    expect(Array.from(passMask(f, slots, "row", 4, null))).toEqual([0, 1, 0, 0]);
    expect(Array.from(passMask(f, slots, "row", 4, new Uint8Array([1, 0, 1, 1])))).toEqual([0, 0, 0, 0]);
  });

  it("only judges elements by the slots on their association", () => {
    const f = compile([{ column: { ...grade, on: "face" }, lo: 5, hi: null, categories: null }]);
    const slots: Slot[] = [{ on: "face", data: new Float32Array([1, 9]), text: false }];
    expect(Array.from(passMask(f, slots, "vertex", 3, null))).toEqual([1, 1, 1]);
    expect(Array.from(passMask(f, slots, "face", 2, null))).toEqual([0, 1]);
  });

  it("writes encoded values per slot, missing elements as nulls", () => {
    const slots: Slot[] = [
      { on: "face", data: new Float32Array([7, Number.NaN]), text: false },
      { on: "vertex", data: new Int32Array([2, -1]), text: true },
    ];
    const out = new Float32Array(4);
    writeFilter(slots, out, 0, (on) => (on === "face" ? 1 : 0));
    expect(Array.from(out.slice(0, 2))).toEqual([Math.fround(NULL), 2]);
    writeFilter(slots, out, 0, () => -1);
    expect(Array.from(out.slice(0, 2))).toEqual([Math.fround(NULL), -1]);
  });

  it("sets the uniforms the shader mirrors", () => {
    const filter = new LayerFilter();
    filter.conditions = [range(1, null), cats("b")];
    filter.slots = [
      { on: "face", data: new Float32Array(0), text: false },
      { on: "vertex", data: new Int32Array(0), text: true },
    ];
    filter.update();
    const u = filter.uniforms;
    expect(u.uFilterMode.value.toArray()).toEqual([1, 2, 0, 0]);
    expect(u.uFilterLo.value.x).toBe(1);
    expect(u.uFilterHi.value.x).toBeGreaterThan(1e37);
    expect(u.uFilterJoin.value.toArray()).toEqual([1, 0, 0, 0]);
    const texels = u.uFilterCats.value.image.data as Uint8Array;
    expect(Array.from(texels.slice(1024, 1027))).toEqual([0, 255, 0]);
  });
});

describe("filter state", () => {
  it("round-trips through the form Python reads, dropping unknown columns and categories", () => {
    const state = toState([range(50, null), cats("c", "a")]);
    expect(state).toEqual({ grade: { range: [50, null] }, lens: { categories: ["a", "c"] } });
    const back = fromState({ ...state, nope: { range: [0, 1] }, lens: { categories: ["a", "zz"] } }, [grade, lens]);
    expect(toState(back)).toEqual({ grade: { range: [50, null] }, lens: { categories: ["a"] } });
  });
});

describe("slider values", () => {
  it("snap to a hundredth of a tick step and print its decimals", () => {
    expect(sliderStep(100)).toBe(0.2);
    expect(snap(50.13, 0.2)).toBe(50.2);
    expect(formatBound(50.13, 100)).toBe("50.2");
    expect(formatBound(0.123456, 1)).toBe("0.124");
    expect(formatBound(12_345.6, 20_000)).toBe("12350");
    expect(formatBound(-0.0001, 1)).toBe("0.000");
  });
});

describe("filtered bounds", () => {
  it("fit only the kept points and segments", () => {
    const xyz = [0, 0, 0, 10, 10, 10, 5, -5, 2];
    expect(maskedBox(xyz, new Uint8Array([1, 0, 1]))).toEqual({ min: [0, -5, 0], max: [5, 0, 2] });
    expect(maskedBox(xyz, new Uint8Array([0, 0, 0]))).toBeNull();
    const segments = [0, 0, 0, 1, 1, 1, 9, 9, 9, 10, 10, 10];
    expect(maskedBox(segments, new Uint8Array([0, 1]), 2, [1, 0])).toEqual({ min: [0, 0, 0], max: [1, 1, 1] });
  });

  it("fit the kept blocks to their rotated extents", () => {
    const s = Math.SQRT1_2;
    const axes: Vec3[] = [[s, s, 0], [-s, s, 0], [0, 0, 1]];
    const box = blocksBox([0, 0, 0, 100, 0, 0], [2, 2, 2, 2, 2, 2], axes, new Uint8Array([1, 0]))!;
    expect(box.min.map((v) => Number(v.toFixed(6)))).toEqual([-1.414214, -1.414214, -1]);
    expect(box.max.map((v) => Number(v.toFixed(6)))).toEqual([1.414214, 1.414214, 1]);
  });

  it("keep a triangle only when its face and its three vertices are kept", () => {
    const triangles = [0, 1, 2, 1, 3, 2];
    expect(Array.from(triangleMask(triangles, new Uint8Array([1, 1]), new Uint8Array([1, 1, 1, 0])))).toEqual([1, 0]);
    expect(Array.from(triangleMask(triangles, new Uint8Array([0, 1]), null))).toEqual([0, 1]);
  });
});
