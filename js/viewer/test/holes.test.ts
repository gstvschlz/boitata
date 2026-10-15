import { describe, expect, it } from "vitest";
import { tubeRadii } from "../src/layers/holes";
import type { LayerSpec } from "../src/types";

const layer = (extra: Partial<LayerSpec>): LayerSpec =>
  ({ columns: [{ name: "cu", type: "number", buffer: "cu", on: "row", min: 1, max: 3 }], radius: 3, ...extra }) as LayerSpec;
const buffers = { cu: Float32Array.from([1, 2, 3, NaN]).buffer };

describe("tube radii", () => {
  it("is the layer's radius without a column", () => {
    expect([...tubeRadii(layer({}), buffers, null, 2)]).toEqual([3, 3]);
  });
  it("maps the column's range from a third of the radius to all of it, nulls the thinnest", () => {
    expect([...tubeRadii(layer({ radiusBy: "cu" }), buffers, null, 4)]).toEqual([1, 2, 3, 1]);
  });
  it("defaults to a share of the holes' extent", () => {
    const box = { min: [0, 0, 0] as [number, number, number], max: [400, 0, 0] as [number, number, number] };
    expect(tubeRadii(layer({ radius: null }), buffers, box, 1)[0]).toBeCloseTo(1);
  });
});
