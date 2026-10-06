import * as THREE from "three";
import { LineSegments2 } from "three/examples/jsm/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/examples/jsm/lines/LineSegmentsGeometry.js";
import { boxOf, diagonal } from "../bounds";
import { f32, u32 } from "../buffers";
import { midpoints, stratifiedOrder, subsetCount } from "../motion";
import { colorAt, kept, type Paint } from "../paint";
import { filterArray, type LayerFilter, writeFilter } from "../filter";
import { HALO, HALO_ORDER, lineMaterial, setOpacity, shadedMaterial, shared } from "../shaders";
import type { Buffers, LayerSpec } from "../types";
import type { Representation } from "./index";

const WIDTH = 2.5;
const RADIUS_SHARE = 1 / 400;
const SIDES = 10;

/** Segments split at survey stations, each with the interval row it belongs to, in a spatially even order. */
function segments(layer: LayerSpec, buffers: Buffers) {
  const positions = f32(buffers, layer.geometry.positions);
  const rows = u32(buffers, layer.geometry.rows);
  return { positions, rows, n: rows.length, box: boxOf(positions), order: stratifiedOrder(midpoints(positions), rows.length) };
}

/** Drill-hole intervals as screen-width segments in a thin halo; each segment takes its interval row's color. */
export function lines(layer: LayerSpec, buffers: Buffers, filter: LayerFilter): Representation {
  const s = segments(layer, buffers);
  const width = layer.lineWidth ?? WIDTH;
  const material = lineMaterial({ vertexColors: true, linewidth: width }, filter.uniforms);
  const halo = lineMaterial({ linewidth: width + 2 * HALO }, filter.uniforms);
  const group = new THREE.Group();
  let geometry: LineSegmentsGeometry | null = null;
  let drawn = 0;
  const rgb = new Uint8Array(3);

  const api: Representation = {
    object: group,
    box: s.box,
    instances: s.n,
    paint(paint: Paint) {
      group.clear();
      geometry?.dispose();
      geometry = null;
      let count = 0;
      for (let i = 0; i < s.n; i++) if (kept(paint, s.rows[i])) count++;
      drawn = count;
      if (!count) return;
      const xyz = new Float32Array(6 * count);
      const colors = new Float32Array(6 * count);
      geometry = new LineSegmentsGeometry();
      const f = filterArray(geometry, count, filter.slots, true);
      let j = 0;
      for (const i of s.order) {
        if (!kept(paint, s.rows[i])) continue;
        xyz.set(s.positions.subarray(6 * i, 6 * i + 6), 6 * j);
        colorAt(paint, s.rows[i], rgb, 0);
        for (let e = 0; e < 2; e++) for (let a = 0; a < 3; a++) colors[6 * j + 3 * e + a] = rgb[a] / 255;
        if (f) writeFilter(filter.slots, f, 4 * j, () => s.rows[i]);
        j++;
      }
      geometry.setPositions(xyz);
      geometry.setColors(colors);
      const outline = new LineSegments2(geometry, halo);
      const line = new LineSegments2(geometry, material);
      outline.frustumCulled = line.frustumCulled = false;
      outline.renderOrder = HALO_ORDER;
      line.renderOrder = HALO_ORDER + 1;
      group.add(outline, line);
    },
    setOpacity(opacity) {
      setOpacity(material, opacity);
      setOpacity(halo, opacity, true);
    },
    detail(fraction) {
      if (geometry) geometry.instanceCount = subsetCount(drawn, fraction);
    },
    frame(w, h) {
      material.resolution.set(w, h);
      halo.resolution.set(w, h);
      halo.color.copy(shared.uHalo.value);
    },
    dispose() {
      geometry?.dispose();
      material.dispose();
      halo.dispose();
    },
  };
  api.setOpacity(layer.opacity);
  return api;
}

/** Drill-hole intervals as shaded cylinders, `radius` meters, one per segment between survey stations. */
export function tubes(layer: LayerSpec, buffers: Buffers, filter: LayerFilter): Representation {
  const s = segments(layer, buffers);
  const radius = layer.radius ?? Math.max(diagonal(s.box) * RADIUS_SHARE, 1e-6);
  const material = shadedMaterial(true, layer.opacity, filter.uniforms);
  const mesh = new THREE.InstancedMesh(new THREE.CylinderGeometry(1, 1, 1, SIDES), material, Math.max(s.n, 1));
  mesh.frustumCulled = false;
  mesh.count = 0;
  const colors = new THREE.InstancedBufferAttribute(new Uint8Array(3 * Math.max(s.n, 1)), 3, true);
  mesh.instanceColor = colors as unknown as THREE.InstancedBufferAttribute;
  let drawn = 0;
  const d = new THREE.Vector3();
  const u = new THREE.Vector3();
  const w = new THREE.Vector3();

  return {
    object: mesh,
    box: s.box,
    instances: s.n,
    paint(paint: Paint) {
      const m = mesh.instanceMatrix.array as Float32Array;
      const c = colors.array as Uint8Array;
      const p = s.positions;
      const f = filterArray(mesh.geometry, s.n, filter.slots, true);
      let j = 0;
      for (const i of s.order) {
        if (!kept(paint, s.rows[i])) continue;
        d.set(p[6 * i + 3] - p[6 * i], p[6 * i + 4] - p[6 * i + 1], p[6 * i + 5] - p[6 * i + 2]);
        u.set(Math.abs(d.z) < 0.9 * d.length() ? 0 : 1, 0, Math.abs(d.z) < 0.9 * d.length() ? 1 : 0);
        u.cross(d).normalize();
        w.copy(d).normalize().cross(u);
        const o = 16 * j;
        m.set([u.x * radius, u.y * radius, u.z * radius, 0, d.x, d.y, d.z, 0, w.x * radius, w.y * radius, w.z * radius, 0], o);
        for (let a = 0; a < 3; a++) m[o + 12 + a] = (p[6 * i + a] + p[6 * i + 3 + a]) / 2;
        m[o + 15] = 1;
        colorAt(paint, s.rows[i], c, 3 * j);
        if (f) writeFilter(filter.slots, f, 4 * j, () => s.rows[i]);
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
