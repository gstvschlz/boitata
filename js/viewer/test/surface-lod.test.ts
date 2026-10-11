import { describe, expect, it } from "vitest";
import * as THREE from "three";
import { LayerFilter } from "../src/filter";
import { surface } from "../src/layers/surface";
import { solidPaint } from "../src/paint";
import type { Buffers, LayerSpec } from "../src/types";

function grid(n: number): { positions: Float32Array; triangles: Uint32Array } {
  const positions: number[] = [];
  for (let y = 0; y <= n; y++) for (let x = 0; x <= n; x++) positions.push(x, y, 0);
  const triangles: number[] = [];
  const at = (x: number, y: number) => y * (n + 1) + x;
  for (let y = 0; y < n; y++)
    for (let x = 0; x < n; x++) {
      triangles.push(at(x, y), at(x + 1, y), at(x, y + 1));
      triangles.push(at(x + 1, y), at(x + 1, y + 1), at(x, y + 1));
    }
  return { positions: new Float32Array(positions), triangles: new Uint32Array(triangles) };
}

function layer(): LayerSpec {
  return {
    id: "m1",
    name: "mesh",
    kind: "mesh",
    representation: "surface",
    color: null,
    opacity: 1,
    visible: true,
    values: null,
    geometry: { positions: "positions", triangles: "triangles" },
    columns: [],
  };
}

describe("surface() motion detail", () => {
  it("shrinks and restores the draw range to a multiple of 3", () => {
    const { positions, triangles } = grid(60); // 7200 triangles, above the motion floor
    const buffers: Buffers = {
      positions: positions.buffer as ArrayBuffer,
      triangles: triangles.buffer as ArrayBuffer,
    };
    const rep = surface(layer(), buffers, new LayerFilter());
    rep.paint(solidPaint([200, 200, 200]));

    const geometry = (rep.object as THREE.Mesh).geometry;
    const full = geometry.drawRange.count;
    expect(full).toBe(3 * 7200);

    rep.detail!(0.1);
    expect(geometry.drawRange.count).toBeLessThan(full);
    expect(geometry.drawRange.count % 3).toBe(0);

    rep.detail!(1);
    expect(geometry.drawRange.count).toBe(full);
  });
});
