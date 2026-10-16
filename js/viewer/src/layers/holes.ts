import * as THREE from "three";
import { LineSegments2 } from "three/examples/jsm/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/examples/jsm/lines/LineSegmentsGeometry.js";
import { type Box, boxOf, diagonal } from "../bounds";
import { screenPixels, type ViewCamera, viewKey } from "../lod";
import { f32, u32 } from "../buffers";
import { midpoints, stratifiedOrder, subsetCount } from "../motion";
import { colorAt, kept, type Paint } from "../paint";
import { pickArray } from "../pick";
import { filterArray, type LayerFilter, writeFilter } from "../filter";
import { HALO, HALO_ORDER, lineMaterial, setOpacity, shadedMaterial, shared } from "../shaders";
import type { Buffers, LayerSpec } from "../types";
import type { Representation } from "./index";

const WIDTH = 2.5;
const RADIUS_SHARE = 1 / 400;
const SIDES = 10;
/** Tubes narrower than this on screen, in CSS pixels, draw as lines. */
export const TUBE_PIXELS = 3;
/** The thinnest tube `radiusBy` draws, as a share of the radius. */
const THINNEST = 1 / 3;

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
      const rows = pickArray(geometry, count, true);
      let j = 0;
      for (const i of s.order) {
        if (!kept(paint, s.rows[i])) continue;
        rows[j] = s.rows[i];
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
      outline.userData.halo = true;
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

/**
 * Each interval's tube radius: `radius` meters (default a share of the holes' extent), or with `radiusBy` a
 * numeric column's range mapped from a third of it to all of it, nulls the thinnest.
 */
export function tubeRadii(layer: LayerSpec, buffers: Buffers, box: Box | null, rows: number): Float32Array {
  const radius = layer.radius ?? Math.max(diagonal(box) * RADIUS_SHARE, 1e-6);
  const out = new Float32Array(rows).fill(radius);
  const column = layer.columns.find((c) => c.name === layer.radiusBy && c.type === "number");
  if (!column) return out;
  const data = f32(buffers, column.buffer);
  const [lo, hi] = [column.min ?? 0, column.max ?? 0];
  for (let r = 0; r < rows; r++) {
    const t = Number.isFinite(data[r]) ? (hi > lo ? (data[r] - lo) / (hi - lo) : 1) : 0;
    out[r] = radius * (THINNEST + (1 - THINNEST) * t);
  }
  return out;
}

/**
 * Drill-hole intervals as shaded tubes near the camera and screen-width lines further off: at rest, an interval
 * whose tube would be under `TUBE_PIXELS` wide on screen draws as a line.
 */
export function tubes(layer: LayerSpec, buffers: Buffers, filter: LayerFilter): Representation {
  const s = segments(layer, buffers);
  const n = s.rows.reduce((m, r) => Math.max(m, r + 1), 0);
  const radii = tubeRadii(layer, buffers, s.box, n);
  const near = cylinders(layer, buffers, filter, radii);
  const far = lines(layer, buffers, filter);
  const object = new THREE.Group().add(near.object, far.object);
  let painted: Paint | null = null;
  let seen = "";
  const mid = midpoints(s.positions);

  const split = (camera: ViewCamera | null, height: number) => {
    const paint = painted!;
    const close = new Uint8Array(n);
    const e = camera?.matrixWorld.elements;
    const pixels = camera && screenPixels(camera, height);
    for (let i = 0; i < s.n; i++) {
      const r = s.rows[i];
      if (close[r] || !e) continue;
      const distance = Math.hypot(mid[3 * i] - e[12], mid[3 * i + 1] - e[13], mid[3 * i + 2] - e[14]);
      if (pixels!(2 * radii[r], distance) >= TUBE_PIXELS) close[r] = 1;
    }
    const part = (want: number): Paint => {
      const keep = new Uint8Array(n);
      for (let r = 0; r < n; r++) keep[r] = close[r] === want && (!paint.keep || paint.keep[r]) ? 1 : 0;
      return { ...paint, keep };
    };
    near.paint(part(1));
    far.paint(part(0));
    if (camera) seen = viewKey(camera, height);
  };

  return {
    object,
    box: s.box,
    instances: s.n,
    paint(paint: Paint) {
      painted = paint;
      seen = "";
      split(null, 1);
    },
    setOpacity(opacity) {
      near.setOpacity(opacity);
      far.setOpacity(opacity);
    },
    detail(fraction) {
      near.detail!(fraction);
      far.detail!(fraction);
    },
    frame(w, h, camera, moving) {
      far.frame!(w, h, camera, moving);
      if (painted && (!seen || (!moving && seen !== viewKey(camera, h)))) split(camera, h);
    },
    dispose() {
      near.dispose();
      far.dispose();
    },
  };
}

/** Drill-hole intervals as shaded cylinders, each its row's radius, one per segment between survey stations. */
function cylinders(layer: LayerSpec, buffers: Buffers, filter: LayerFilter, radii: Float32Array): Representation {
  const s = segments(layer, buffers);
  const material = shadedMaterial(true, layer.opacity, filter.uniforms, { whole: true });
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
      const rows = pickArray(mesh.geometry, s.n, true);
      let j = 0;
      for (const i of s.order) {
        if (!kept(paint, s.rows[i])) continue;
        rows[j] = s.rows[i];
        d.set(p[6 * i + 3] - p[6 * i], p[6 * i + 4] - p[6 * i + 1], p[6 * i + 5] - p[6 * i + 2]);
        u.set(Math.abs(d.z) < 0.9 * d.length() ? 0 : 1, 0, Math.abs(d.z) < 0.9 * d.length() ? 1 : 0);
        u.cross(d).normalize();
        w.copy(d).normalize().cross(u);
        const o = 16 * j;
        const radius = radii[s.rows[i]];
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
