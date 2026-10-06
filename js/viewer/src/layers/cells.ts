import * as THREE from "three";
import type { Box, Vec3 } from "../bounds";
import { f32 } from "../buffers";
import { colorAt, kept, type Paint } from "../paint";
import { setOpacity, shadedMaterial } from "../shaders";
import type { Buffers, LayerSpec } from "../types";
import type { Representation } from "./index";

/** Blocks as instanced boxes along the model's axes, each scaled to its own (sub-)block size. */
export function cells(layer: LayerSpec, buffers: Buffers): Representation {
  const centers = f32(buffers, layer.geometry.centers);
  const sizes = f32(buffers, layer.geometry.sizes);
  const axes = (layer.axes ?? [[1, 0, 0], [0, 1, 0], [0, 0, 1]]) as [Vec3, Vec3, Vec3];
  const n = centers.length / 3;
  const material = shadedMaterial(true, layer.opacity);
  const mesh = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), material, Math.max(n, 1));
  mesh.frustumCulled = false;
  const colors = new THREE.InstancedBufferAttribute(new Uint8Array(3 * Math.max(n, 1)), 3, true);
  mesh.instanceColor = colors as unknown as THREE.InstancedBufferAttribute;
  mesh.count = 0;

  const min: Vec3 = [Infinity, Infinity, Infinity];
  const max: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < n; i++)
    for (let a = 0; a < 3; a++) {
      let half = 0;
      for (let k = 0; k < 3; k++) half += Math.abs(axes[k][a]) * sizes[3 * i + k];
      half /= 2;
      min[a] = Math.min(min[a], centers[3 * i + a] - half);
      max[a] = Math.max(max[a], centers[3 * i + a] + half);
    }
  const box: Box | null = n ? { min, max } : null;

  return {
    object: mesh,
    box,
    paint(paint: Paint) {
      const m = mesh.instanceMatrix.array as Float32Array;
      const c = colors.array as Uint8Array;
      let j = 0;
      for (let i = 0; i < n; i++) {
        if (!kept(paint, i)) continue;
        const o = 16 * j;
        for (let k = 0; k < 3; k++) {
          const s = sizes[3 * i + k];
          m[o + 4 * k] = axes[k][0] * s;
          m[o + 4 * k + 1] = axes[k][1] * s;
          m[o + 4 * k + 2] = axes[k][2] * s;
          m[o + 4 * k + 3] = 0;
        }
        m[o + 12] = centers[3 * i];
        m[o + 13] = centers[3 * i + 1];
        m[o + 14] = centers[3 * i + 2];
        m[o + 15] = 1;
        colorAt(paint, i, c, 3 * j);
        j++;
      }
      mesh.count = j;
      mesh.instanceMatrix.needsUpdate = true;
      colors.needsUpdate = true;
    },
    setOpacity: (opacity) => setOpacity(material, opacity),
    dispose() {
      mesh.geometry.dispose();
      material.dispose();
      mesh.dispose();
    },
  };
}
