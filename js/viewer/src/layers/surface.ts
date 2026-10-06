import * as THREE from "three";
import { LineSegments2 } from "three/examples/jsm/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/examples/jsm/lines/LineSegmentsGeometry.js";
import { boxOf } from "../bounds";
import { f32, u32 } from "../buffers";
import { filterArray, type LayerFilter, writeFilter } from "../filter";
import { midpoints, stratifiedOrder, subsetCount } from "../motion";
import { colorAt, kept, type Paint } from "../paint";
import { lineMaterial, setOpacity, shadedMaterial } from "../shaders";
import type { Buffers, LayerSpec } from "../types";
import type { Representation } from "./index";

/**
 * Triangles, smooth-shaded by vertex or flat by face; a triangle with a hidden vertex or face is left out, and one
 * with a vertex or face failing the filter is discarded where it rasterizes.
 */
export function surface(layer: LayerSpec, buffers: Buffers, filter: LayerFilter): Representation {
  const positions = f32(buffers, layer.geometry.positions);
  const triangles = u32(buffers, layer.geometry.triangles);
  const nt = triangles.length / 3;
  const material = shadedMaterial(false, layer.opacity, filter.uniforms);
  const object = new THREE.Mesh(new THREE.BufferGeometry(), material);
  object.frustumCulled = false;
  object.renderOrder = 1;

  const smooth = new THREE.BufferGeometry();
  smooth.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  smooth.setIndex(new THREE.BufferAttribute(triangles, 1));
  smooth.computeVertexNormals();
  const normals = smooth.getAttribute("normal").array as Float32Array;

  /** One vertex per triangle corner: face colors and filters need it, flat shading by face too. */
  const corners = (paint: Paint) => {
    const byFace = paint.on === "face";
    const shown = (t: number) => {
      if (byFace) return kept(paint, t);
      if (paint.on !== "vertex") return true;
      return kept(paint, triangles[3 * t]) && kept(paint, triangles[3 * t + 1]) && kept(paint, triangles[3 * t + 2]);
    };
    let count = 0;
    for (let t = 0; t < nt; t++) if (shown(t)) count++;
    const xyz = new Float32Array(9 * count);
    const normal = new Float32Array(9 * count);
    const colors = new Uint8Array(9 * count);
    const g = new THREE.BufferGeometry();
    const f = filterArray(g, 3 * count, filter.slots, false);
    let j = 0;
    for (let t = 0; t < nt; t++) {
      if (!shown(t)) continue;
      for (let c = 0; c < 3; c++) {
        const v = triangles[3 * t + c];
        const at = 3 * (3 * j + c);
        xyz.set(positions.subarray(3 * v, 3 * v + 3), at);
        normal.set(normals.subarray(3 * v, 3 * v + 3), at);
        colorAt(paint, byFace ? t : paint.on === "vertex" ? v : 0, colors, at);
        if (f) writeFilter(filter.slots, f, 4 * (3 * j + c), (on) => (on === "face" ? t : v));
      }
      j++;
    }
    g.setAttribute("position", new THREE.BufferAttribute(xyz, 3));
    g.setAttribute("normal", new THREE.BufferAttribute(normal, 3));
    g.setAttribute("color", new THREE.BufferAttribute(colors, 3, true));
    if (byFace) g.computeVertexNormals();
    return g;
  };

  return {
    object,
    box: boxOf(positions),
    instances: nt,
    paint(paint: Paint) {
      const old = object.geometry;
      object.geometry = corners(paint);
      old.dispose();
    },
    setOpacity: (opacity) => setOpacity(material, opacity),
    dispose() {
      object.geometry.dispose();
      smooth.dispose();
      material.dispose();
    },
  };
}

const EDGE_WIDTH = 1;

/**
 * Triangle edges, each drawn once, `lineWidth` pixels wide. Under a vertex column an edge blends its two vertices'
 * colors and shows when both do; under a face column it takes the color of a shown face beside it. The filter keeps
 * an edge when both its vertices pass a vertex column's condition, and when either face beside it passes a face
 * column's.
 */
export function meshWireframe(layer: LayerSpec, buffers: Buffers, filter: LayerFilter): Representation {
  const positions = f32(buffers, layer.geometry.positions);
  const triangles = u32(buffers, layer.geometry.triangles);
  const nv = positions.length / 3;
  const seen = new Map<number, number>();
  const ends: number[] = [];
  const faces: number[] = [];
  for (let t = 0; t < triangles.length / 3; t++)
    for (let k = 0; k < 3; k++) {
      const a = triangles[3 * t + k];
      const b = triangles[3 * t + ((k + 1) % 3)];
      const key = Math.min(a, b) * nv + Math.max(a, b);
      const e = seen.get(key);
      if (e === undefined) {
        seen.set(key, ends.length / 2);
        ends.push(a, b);
        faces.push(t, -1);
      } else if (faces[2 * e + 1] < 0) faces[2 * e + 1] = t;
    }
  const ne = ends.length / 2;
  const xyz = new Float32Array(6 * ne);
  for (let e = 0; e < ne; e++)
    for (let k = 0; k < 2; k++) xyz.set(positions.subarray(3 * ends[2 * e + k], 3 * ends[2 * e + k] + 3), 6 * e + 3 * k);
  const order = stratifiedOrder(midpoints(xyz), ne);
  const material = lineMaterial({ vertexColors: true, linewidth: layer.lineWidth ?? EDGE_WIDTH }, filter.uniforms, true);
  const group = new THREE.Group();
  let geometry: LineSegmentsGeometry | null = null;
  let drawn = 0;
  const rgb = new Uint8Array(3);
  const push = (colors: number[]) => colors.push(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255);

  const api: Representation = {
    object: group,
    box: boxOf(positions),
    instances: ne,
    paint(paint: Paint) {
      group.clear();
      geometry?.dispose();
      geometry = null;
      const out: number[] = [];
      const colors: number[] = [];
      const shown: number[] = [];
      for (const e of order) {
        const a = ends[2 * e];
        const b = ends[2 * e + 1];
        if (paint.on === "face") {
          const t = [faces[2 * e], faces[2 * e + 1]].find((f) => f >= 0 && kept(paint, f));
          if (t === undefined) continue;
          colorAt(paint, t, rgb, 0);
          push(colors);
          push(colors);
        } else {
          const on = paint.on === "vertex";
          if (on && !(kept(paint, a) && kept(paint, b))) continue;
          colorAt(paint, on ? a : 0, rgb, 0);
          push(colors);
          colorAt(paint, on ? b : 0, rgb, 0);
          push(colors);
        }
        for (let k = 0; k < 6; k++) out.push(xyz[6 * e + k]);
        shown.push(e);
      }
      drawn = shown.length;
      if (!drawn) return;
      geometry = new LineSegmentsGeometry();
      geometry.setPositions(out);
      geometry.setColors(colors);
      const f = filterArray(geometry, drawn, filter.slots, true);
      const g = filterArray(geometry, drawn, filter.slots, true, "aFilter2");
      if (f && g)
        shown.forEach((e, j) => {
          writeFilter(filter.slots, f, 4 * j, (on) => (on === "face" ? faces[2 * e] : ends[2 * e]));
          writeFilter(filter.slots, g, 4 * j, (on) => (on === "face" ? faces[2 * e + 1] : ends[2 * e + 1]));
        });
      const line = new LineSegments2(geometry, material);
      line.frustumCulled = false;
      group.add(line);
    },
    setOpacity: (opacity) => setOpacity(material, opacity),
    detail(fraction) {
      if (geometry) geometry.instanceCount = subsetCount(drawn, fraction);
    },
    frame(w, h) {
      material.resolution.set(w, h);
    },
    dispose() {
      geometry?.dispose();
      material.dispose();
    },
  };
  api.setOpacity(layer.opacity);
  return api;
}
