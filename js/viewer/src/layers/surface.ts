import * as THREE from "three";
import { boxOf } from "../bounds";
import { f32, u32 } from "../buffers";
import { colorAt, kept, type Paint } from "../paint";
import { setOpacity, shadedMaterial } from "../shaders";
import type { Buffers, LayerSpec } from "../types";
import type { Representation } from "./index";

/** Triangles, smooth-shaded by vertex or flat by face; a triangle with a hidden vertex or face is left out. */
export function surface(layer: LayerSpec, buffers: Buffers): Representation {
  const positions = f32(buffers, layer.geometry.positions);
  const triangles = u32(buffers, layer.geometry.triangles);
  const nv = positions.length / 3;
  const nt = triangles.length / 3;
  const material = shadedMaterial(false, layer.opacity);
  const object = new THREE.Mesh(new THREE.BufferGeometry(), material);
  object.frustumCulled = false;
  object.renderOrder = 1;

  const smooth = new THREE.BufferGeometry();
  smooth.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  smooth.setIndex(new THREE.BufferAttribute(triangles, 1));
  smooth.computeVertexNormals();
  const normals = smooth.getAttribute("normal");

  const byFace = (paint: Paint) => {
    let count = 0;
    for (let t = 0; t < nt; t++) if (kept(paint, t)) count++;
    const xyz = new Float32Array(9 * count);
    const colors = new Uint8Array(9 * count);
    let j = 0;
    for (let t = 0; t < nt; t++) {
      if (!kept(paint, t)) continue;
      for (let c = 0; c < 3; c++) {
        const v = triangles[3 * t + c];
        xyz.set(positions.subarray(3 * v, 3 * v + 3), 9 * j + 3 * c);
        colorAt(paint, t, colors, 9 * j + 3 * c);
      }
      j++;
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(xyz, 3));
    g.setAttribute("color", new THREE.BufferAttribute(colors, 3, true));
    g.computeVertexNormals();
    return g;
  };

  const byVertex = (paint: Paint) => {
    const colors = new Uint8Array(3 * nv);
    for (let v = 0; v < nv; v++) colorAt(paint, paint.on === "vertex" ? v : 0, colors, 3 * v);
    const index: number[] = [];
    for (let t = 0; t < nt; t++) {
      const a = triangles[3 * t], b = triangles[3 * t + 1], c = triangles[3 * t + 2];
      if (paint.on !== "vertex" || (kept(paint, a) && kept(paint, b) && kept(paint, c))) index.push(a, b, c);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", smooth.getAttribute("position"));
    g.setAttribute("normal", normals);
    g.setAttribute("color", new THREE.BufferAttribute(colors, 3, true));
    g.setIndex(new THREE.BufferAttribute(new Uint32Array(index), 1));
    return g;
  };

  return {
    object,
    box: boxOf(positions),
    paint(paint: Paint) {
      const old = object.geometry;
      object.geometry = paint.on === "face" ? byFace(paint) : byVertex(paint);
      if (old !== smooth) old.dispose();
    },
    setOpacity: (opacity) => setOpacity(material, opacity),
    dispose() {
      object.geometry.dispose();
      smooth.dispose();
      material.dispose();
    },
  };
}
