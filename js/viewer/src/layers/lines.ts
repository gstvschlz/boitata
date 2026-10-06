import * as THREE from "three";
import { LineMaterial } from "three/examples/jsm/lines/LineMaterial.js";
import { LineSegments2 } from "three/examples/jsm/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/examples/jsm/lines/LineSegmentsGeometry.js";
import { boxOf } from "../bounds";
import { f32, u32 } from "../buffers";
import { colorAt, kept, type Paint } from "../paint";
import type { Buffers, LayerSpec } from "../types";
import type { Representation } from "./index";

const WIDTH = 2.5;

/** Drill-hole intervals as screen-width segments; each segment takes its interval row's color. */
export function lines(layer: LayerSpec, buffers: Buffers): Representation {
  const positions = f32(buffers, layer.geometry.positions);
  const rows = u32(buffers, layer.geometry.rows);
  const n = rows.length;
  const material = new LineMaterial({ vertexColors: true, linewidth: WIDTH, transparent: layer.opacity < 1 });
  material.opacity = layer.opacity;
  const group = new THREE.Group();
  let segments: LineSegments2 | null = null;
  const rgb = new Uint8Array(3);

  return {
    object: group,
    box: boxOf(positions),
    paint(paint: Paint) {
      if (segments) {
        group.remove(segments);
        segments.geometry.dispose();
        segments = null;
      }
      let count = 0;
      for (let i = 0; i < n; i++) if (kept(paint, rows[i])) count++;
      if (!count) return;
      const xyz = new Float32Array(6 * count);
      const colors = new Float32Array(6 * count);
      let j = 0;
      for (let i = 0; i < n; i++) {
        if (!kept(paint, rows[i])) continue;
        xyz.set(positions.subarray(6 * i, 6 * i + 6), 6 * j);
        colorAt(paint, rows[i], rgb, 0);
        for (let e = 0; e < 2; e++) for (let a = 0; a < 3; a++) colors[6 * j + 3 * e + a] = rgb[a] / 255;
        j++;
      }
      const geometry = new LineSegmentsGeometry();
      geometry.setPositions(xyz);
      geometry.setColors(colors);
      segments = new LineSegments2(geometry, material);
      segments.frustumCulled = false;
      group.add(segments);
    },
    setOpacity(opacity) {
      material.opacity = opacity;
      material.transparent = opacity < 1;
      material.depthWrite = opacity >= 1;
    },
    frame(width, height) {
      material.resolution.set(width, height);
    },
    dispose() {
      segments?.geometry.dispose();
      material.dispose();
    },
  };
}
