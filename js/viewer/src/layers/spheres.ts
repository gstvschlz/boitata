import * as THREE from "three";
import { boxOf, diagonal } from "../bounds";
import { f32 } from "../buffers";
import { filterArray, type LayerFilter, writeFilter } from "../filter";
import { stratifiedOrder, subsetCount } from "../motion";
import { colorAt, kept, type Paint } from "../paint";
import { setOpacity, shadedMaterial } from "../shaders";
import type { Buffers, LayerSpec } from "../types";
import type { Representation } from "./index";

const RADIUS_SHARE = 1 / 250;

/** Points as shaded spheres of `radius` meters. */
export function spheres(layer: LayerSpec, buffers: Buffers, filter: LayerFilter): Representation {
  const positions = f32(buffers, layer.geometry.positions);
  const n = positions.length / 3;
  const box = boxOf(positions);
  const radius = layer.radius ?? Math.max(diagonal(box) * RADIUS_SHARE, 1e-6);
  const order = stratifiedOrder(positions, n);
  const material = shadedMaterial(true, layer.opacity, filter.uniforms, { whole: true });
  const mesh = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(1, 2), material, Math.max(n, 1));
  mesh.frustumCulled = false;
  mesh.count = 0;
  const colors = new THREE.InstancedBufferAttribute(new Uint8Array(3 * Math.max(n, 1)), 3, true);
  mesh.instanceColor = colors as unknown as THREE.InstancedBufferAttribute;
  let drawn = 0;

  return {
    object: mesh,
    box,
    instances: n,
    paint(paint: Paint) {
      const m = mesh.instanceMatrix.array as Float32Array;
      const c = colors.array as Uint8Array;
      const f = filterArray(mesh.geometry, n, filter.slots, true);
      let j = 0;
      for (const i of order) {
        if (!kept(paint, i)) continue;
        const o = 16 * j;
        m.fill(0, o, o + 16);
        m[o] = m[o + 5] = m[o + 10] = radius;
        m.set(positions.subarray(3 * i, 3 * i + 3), o + 12);
        m[o + 15] = 1;
        colorAt(paint, i, c, 3 * j);
        if (f) writeFilter(filter.slots, f, 4 * j, () => i);
        j++;
      }
      mesh.count = drawn = j;
      mesh.instanceMatrix.needsUpdate = true;
      colors.needsUpdate = true;
    },
    setOpacity: (opacity) => setOpacity(material, opacity),
    detail(fraction) {
      mesh.count = subsetCount(drawn, fraction);
    },
    dispose() {
      mesh.geometry.dispose();
      material.dispose();
      mesh.dispose();
    },
  };
}
