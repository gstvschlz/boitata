import * as THREE from "three";
import { boxOf } from "../bounds";
import { f32 } from "../buffers";
import { colorAt, kept, type Paint } from "../paint";
import { pointMaterial, setOpacity } from "../shaders";
import type { Buffers, LayerSpec } from "../types";
import type { Representation } from "./index";

const SIZE = 6;

/** Points as round sprites of a fixed screen size. */
export function points(layer: LayerSpec, buffers: Buffers): Representation {
  const positions = f32(buffers, layer.geometry.positions);
  const n = positions.length / 3;
  const material = pointMaterial(SIZE, layer.opacity);
  const geometry = new THREE.BufferGeometry();
  const object = new THREE.Points(geometry, material);
  object.frustumCulled = false;

  return {
    object,
    box: boxOf(positions),
    paint(paint: Paint) {
      let count = 0;
      for (let i = 0; i < n; i++) if (kept(paint, i)) count++;
      const xyz = new Float32Array(3 * count);
      const colors = new Uint8Array(3 * count);
      let j = 0;
      for (let i = 0; i < n; i++) {
        if (!kept(paint, i)) continue;
        xyz.set(positions.subarray(3 * i, 3 * i + 3), 3 * j);
        colorAt(paint, i, colors, 3 * j);
        j++;
      }
      geometry.setAttribute("position", new THREE.BufferAttribute(xyz, 3));
      geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3, true));
    },
    setOpacity: (opacity) => setOpacity(material, opacity),
    dispose() {
      geometry.dispose();
      material.dispose();
    },
  };
}
