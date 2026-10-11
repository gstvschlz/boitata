import { describe, expect, it } from "vitest";
import * as THREE from "three";
import type { Vec3 } from "../src/bounds";
import { buildVolumeTexture, gridUVW } from "../src/contour";

describe("gridUVW", () => {
  const identity: [Vec3, Vec3, Vec3] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];

  it("maps an axis-aligned grid's own corner and far corner to 0 and 1", () => {
    const origin: Vec3 = [10, 20, 0];
    const size: Vec3 = [2, 2, 2];
    const count: [number, number, number] = [5, 5, 5]; // a 10 m cube
    expect(gridUVW([10, 20, 0], origin, identity, size, count)).toEqual([0, 0, 0]);
    const far = gridUVW([20, 30, 10], origin, identity, size, count);
    far.forEach((v) => expect(v).toBeCloseTo(1));
  });

  it("rotates a world point into the grid's own axes before normalizing", () => {
    const axes: [Vec3, Vec3, Vec3] = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]; // grid rotated 90 deg about z
    const size: Vec3 = [1, 1, 1];
    const count: [number, number, number] = [10, 10, 10];
    const [u, v] = gridUVW([5, 0, 0], [0, 0, 0], axes, size, count);
    expect(u).toBeCloseTo(0);
    expect(v).toBeCloseTo(-0.5);
  });
});

describe("buildVolumeTexture", () => {
  it("builds a trilinearly-filterable texture of the right shape", () => {
    const texture = buildVolumeTexture(new Float32Array([0, 1, 2, 3, 4, 5, 6, 7]), [2, 2, 2]);
    expect(texture.image.width).toBe(2);
    expect(texture.image.height).toBe(2);
    expect(texture.image.depth).toBe(2);
    expect(texture.minFilter).toBe(THREE.LinearFilter);
    expect(texture.magFilter).toBe(THREE.LinearFilter);
  });
});
