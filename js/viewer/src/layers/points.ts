import * as THREE from "three";
import { boxOf } from "../bounds";
import { f32, u32 } from "../buffers";
import { filterArray, type LayerFilter, writeFilter } from "../filter";
import { stratifiedOrder, subsetCount } from "../motion";
import { colorAt, kept, type Paint } from "../paint";
import { HALO_ORDER, pointMaterial, setOpacity } from "../shaders";
import type { Buffers, ColumnSpec, LayerSpec } from "../types";
import type { Representation } from "./index";

const SIZE = 6;

/** Element i's index for a filter column on association `on`; all one association except on mesh vertices. */
type Elements = (paint: Paint) => { paint: Paint; index: (on: ColumnSpec["on"], i: number) => number };
const same: Elements = (paint) => ({ paint, index: (_, i) => i });

/** Round sprites of a fixed screen size at `positions`; element i takes paint entry i, after `elements`. */
function sprites(positions: Float32Array, layer: LayerSpec, filter: LayerFilter, elements: Elements = same): Representation {
  const n = positions.length / 3;
  const order = stratifiedOrder(positions, n);
  const material = pointMaterial(layer.pointSize ?? SIZE, layer.opacity, filter.uniforms);
  const halo = pointMaterial(layer.pointSize ?? SIZE, layer.opacity, filter.uniforms, true);
  const geometry = new THREE.BufferGeometry();
  const xyz = new THREE.BufferAttribute(new Float32Array(3 * n), 3);
  const colors = new THREE.BufferAttribute(new Uint8Array(3 * n), 3, true);
  geometry.setAttribute("position", xyz);
  geometry.setAttribute("color", colors);
  const outline = new THREE.Points(geometry, halo);
  const dots = new THREE.Points(geometry, material);
  outline.frustumCulled = dots.frustumCulled = false;
  outline.renderOrder = HALO_ORDER;
  dots.renderOrder = HALO_ORDER + 1;
  const object = new THREE.Group().add(outline, dots);
  let drawn = 0;

  return {
    object,
    box: boxOf(positions),
    instances: n,
    paint(painted: Paint) {
      const { paint, index } = elements(painted);
      const p = xyz.array as Float32Array;
      const c = colors.array as Uint8Array;
      const f = filterArray(geometry, n, filter.slots, false);
      let j = 0;
      for (const i of order) {
        if (!kept(paint, i)) continue;
        p.set(positions.subarray(3 * i, 3 * i + 3), 3 * j);
        colorAt(paint, i, c, 3 * j);
        if (f) writeFilter(filter.slots, f, 4 * j, (on) => index(on, i));
        j++;
      }
      drawn = j;
      xyz.needsUpdate = colors.needsUpdate = true;
      geometry.setDrawRange(0, drawn);
    },
    setOpacity(opacity) {
      setOpacity(material, opacity);
      setOpacity(halo, opacity, true);
    },
    detail(fraction) {
      geometry.setDrawRange(0, subsetCount(drawn, fraction));
    },
    dispose() {
      geometry.dispose();
      material.dispose();
      halo.dispose();
    },
  };
}

export const points = (layer: LayerSpec, buffers: Buffers, filter: LayerFilter) =>
  sprites(f32(buffers, layer.geometry.positions), layer, filter);

/** Block centers. */
export const blockPoints = (layer: LayerSpec, buffers: Buffers, filter: LayerFilter) =>
  sprites(f32(buffers, layer.geometry.centers), layer, filter);

/** Interval midpoints along the hole. */
export const holePoints = (layer: LayerSpec, buffers: Buffers, filter: LayerFilter) =>
  sprites(f32(buffers, layer.geometry.midpoints), layer, filter);

/**
 * Mesh vertices. Under a face column a vertex shows when a face around it does, in that face's color; a face
 * column's filter judges a vertex by that same face.
 */
export function meshPoints(layer: LayerSpec, buffers: Buffers, filter: LayerFilter): Representation {
  const positions = f32(buffers, layer.geometry.positions);
  const triangles = u32(buffers, layer.geometry.triangles);
  const nv = positions.length / 3;
  return sprites(positions, layer, filter, (paint) => {
    const faces = vertexFaces(triangles, nv, paint.on === "face" ? paint : null);
    return {
      paint: paint.on === "face" ? faceToVertex(paint, triangles, nv) : paint,
      index: (on, v) => (on === "face" ? faces[v] : v),
    };
  });
}

/** For each vertex, the last face around it (kept under `paint`, when given); -1 for none. */
export function vertexFaces(triangles: Uint32Array, nv: number, paint: Paint | null): Int32Array {
  const out = new Int32Array(nv).fill(-1);
  for (let t = 0; t < triangles.length / 3; t++) {
    if (paint && !kept(paint, t)) continue;
    for (let k = 0; k < 3; k++) out[triangles[3 * t + k]] = t;
  }
  return out;
}

export function faceToVertex(paint: Paint, triangles: Uint32Array, nv: number): Paint {
  const colors = new Uint8Array(3 * nv);
  const keep = new Uint8Array(nv);
  for (let t = 0; t < triangles.length / 3; t++) {
    if (!kept(paint, t)) continue;
    for (let k = 0; k < 3; k++) {
      const v = triangles[3 * t + k];
      keep[v] = 1;
      colorAt(paint, t, colors, 3 * v);
    }
  }
  return { on: "vertex", colors, keep, solid: paint.solid };
}
