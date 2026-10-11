import * as THREE from "three";
import type { Vec3 } from "./bounds";

/**
 * World point `p` as a block grid's normalized (u, v, w), the same transform `shaders.ts` ports to GLSL for
 * the live contour sample: rotate into the grid's own axes, then divide by its full extent per axis.
 */
export function gridUVW(p: Vec3, origin: Vec3, axes: [Vec3, Vec3, Vec3], size: Vec3, count: [number, number, number]): Vec3 {
  const local: Vec3 = [p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]];
  const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  return [0, 1, 2].map((a) => dot(local, axes[a]) / (size[a] * count[a])) as Vec3;
}

/** A dense attribute grid as a trilinearly-filterable 3D texture; NaN cells stay NaN after sampling. */
export function buildVolumeTexture(values: Float32Array, count: [number, number, number]): THREE.Data3DTexture {
  const [nx, ny, nz] = count;
  const half = new Uint16Array(nx * ny * nz);
  for (let i = 0; i < half.length; i++) half[i] = THREE.DataUtils.toHalfFloat(values[i]);
  const texture = new THREE.Data3DTexture(half, nx, ny, nz);
  texture.format = THREE.RedFormat;
  texture.type = THREE.HalfFloatType;
  texture.minFilter = texture.magFilter = THREE.LinearFilter;
  texture.wrapS = texture.wrapT = texture.wrapR = THREE.ClampToEdgeWrapping;
  texture.needsUpdate = true;
  return texture;
}
