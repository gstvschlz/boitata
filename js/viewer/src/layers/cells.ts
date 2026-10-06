import * as THREE from "three";
import { blocksBox, type Box, type Vec3 } from "../bounds";
import { f32 } from "../buffers";
import { filterArray, type LayerFilter, writeFilter } from "../filter";
import { stratifiedOrder, subsetCount } from "../motion";
import { colorAt, kept, type Paint } from "../paint";
import { pickArray } from "../pick";
import { boxEdgeMaterial, depthMaterial, setOpacity, shadedMaterial } from "../shaders";
import type { Buffers, LayerSpec } from "../types";
import type { Representation } from "./index";

const EDGE_WIDTH = 1;

interface Blocks {
  centers: Float32Array;
  sizes: Float32Array;
  axes: [Vec3, Vec3, Vec3];
  n: number;
  box: Box | null;
  order: Uint32Array;
}

function blocks(layer: LayerSpec, buffers: Buffers): Blocks {
  const centers = f32(buffers, layer.geometry.centers);
  const sizes = f32(buffers, layer.geometry.sizes);
  const axes = (layer.axes ?? [[1, 0, 0], [0, 1, 0], [0, 0, 1]]) as [Vec3, Vec3, Vec3];
  const n = centers.length / 3;
  return { centers, sizes, axes, n, box: blocksBox(centers, sizes, axes, null), order: stratifiedOrder(centers, n) };
}

/** Writes block i's matrix (its axes scaled by its sizes, at its center) at instance j. */
function setMatrix(b: Blocks, i: number, m: Float32Array, j: number): void {
  const o = 16 * j;
  for (let k = 0; k < 3; k++) {
    const s = b.sizes[3 * i + k];
    m[o + 4 * k] = b.axes[k][0] * s;
    m[o + 4 * k + 1] = b.axes[k][1] * s;
    m[o + 4 * k + 2] = b.axes[k][2] * s;
    m[o + 4 * k + 3] = 0;
  }
  m[o + 12] = b.centers[3 * i];
  m[o + 13] = b.centers[3 * i + 1];
  m[o + 14] = b.centers[3 * i + 2];
  m[o + 15] = 1;
}

function boxes(b: Blocks, material: THREE.Material, colored: boolean) {
  const mesh = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), material, Math.max(b.n, 1));
  mesh.frustumCulled = false;
  mesh.count = 0;
  const colors = colored ? new THREE.InstancedBufferAttribute(new Uint8Array(3 * Math.max(b.n, 1)), 3, true) : null;
  if (colors) mesh.instanceColor = colors as unknown as THREE.InstancedBufferAttribute;
  return { mesh, colors };
}

/** Blocks as instanced boxes along the model's axes, each scaled to its own (sub-)block size. */
export function cells(layer: LayerSpec, buffers: Buffers, filter: LayerFilter): Representation {
  const b = blocks(layer, buffers);
  const material = shadedMaterial(true, layer.opacity, filter.uniforms, { caps: true });
  const { mesh, colors } = boxes(b, material, true);
  let drawn = 0;

  return {
    object: mesh,
    box: b.box,
    instances: b.n,
    paint(paint: Paint) {
      const m = mesh.instanceMatrix.array as Float32Array;
      const c = colors!.array as Uint8Array;
      const f = filterArray(mesh.geometry, b.n, filter.slots, true);
      const rows = pickArray(mesh.geometry, b.n, true);
      let j = 0;
      for (const i of b.order) {
        if (!kept(paint, i)) continue;
        setMatrix(b, i, m, j);
        colorAt(paint, i, c, 3 * j);
        rows[j] = i;
        if (f) writeFilter(filter.slots, f, 4 * j, () => i);
        j++;
      }
      mesh.count = drawn = j;
      mesh.instanceMatrix.needsUpdate = true;
      colors!.needsUpdate = true;
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

/** Quads of the 12 edges of a unit box centered on the origin. */
function edgeGeometry(n: number): THREE.InstancedBufferGeometry {
  const start: number[] = [];
  const end: number[] = [];
  const corner: number[] = [];
  const index: number[] = [];
  const at = (c: number) => [0, 1, 2].map((a) => ((c >> a) & 1) - 0.5);
  for (let a = 0; a < 3; a++)
    for (let c = 0; c < 8; c++) {
      if ((c >> a) & 1) continue;
      const v = start.length / 3;
      for (const [along, side] of [[0, -1], [0, 1], [1, -1], [1, 1]]) {
        start.push(...at(c));
        end.push(...at(c | (1 << a)));
        corner.push(along, side);
      }
      index.push(v, v + 1, v + 2, v + 2, v + 1, v + 3);
    }
  const g = new THREE.InstancedBufferGeometry();
  g.setAttribute("aStart", new THREE.Float32BufferAttribute(start, 3));
  g.setAttribute("aEnd", new THREE.Float32BufferAttribute(end, 3));
  g.setAttribute("aCorner", new THREE.Float32BufferAttribute(corner, 2));
  g.setIndex(index);
  const m = Math.max(n, 1);
  g.setAttribute("iCenter", new THREE.InstancedBufferAttribute(new Float32Array(3 * m), 3));
  g.setAttribute("iSize", new THREE.InstancedBufferAttribute(new Float32Array(3 * m), 3));
  g.setAttribute("iColor", new THREE.InstancedBufferAttribute(new Uint8Array(3 * m), 3, true));
  g.instanceCount = 0;
  return g;
}

/**
 * Block edges, `lineWidth` pixels wide in each block's color. Opaque, the blocks' faces still hide what is behind
 * them, so only the visible edges draw and a model of any size stays legible; below opacity 1 every edge shows.
 */
export function blockWireframe(layer: LayerSpec, buffers: Buffers, filter: LayerFilter): Representation {
  const b = blocks(layer, buffers);
  const material = boxEdgeMaterial(b.axes, layer.lineWidth ?? EDGE_WIDTH, layer.opacity, filter.uniforms);
  const geometry = edgeGeometry(b.n);
  const edges = new THREE.Mesh(geometry, material);
  edges.frustumCulled = false;
  const depth = depthMaterial(filter.uniforms);
  const { mesh: faces } = boxes(b, depth, false);
  faces.renderOrder = -2;
  const group = new THREE.Group();
  group.add(faces, edges);
  let drawn = 0;
  const attribute = (name: string) => geometry.getAttribute(name) as THREE.InstancedBufferAttribute;

  const api: Representation = {
    object: group,
    box: b.box,
    instances: b.n,
    paint(paint: Paint) {
      const center = attribute("iCenter").array as Float32Array;
      const size = attribute("iSize").array as Float32Array;
      const color = attribute("iColor").array as Uint8Array;
      const m = faces.instanceMatrix.array as Float32Array;
      const f = filterArray(geometry, b.n, filter.slots, true);
      if (f) faces.geometry.setAttribute("aFilter", geometry.getAttribute("aFilter"));
      const rows = pickArray(geometry, b.n, true);
      faces.geometry.setAttribute("aPick", geometry.getAttribute("aPick"));
      let j = 0;
      for (const i of b.order) {
        if (!kept(paint, i)) continue;
        center.set(b.centers.subarray(3 * i, 3 * i + 3), 3 * j);
        size.set(b.sizes.subarray(3 * i, 3 * i + 3), 3 * j);
        colorAt(paint, i, color, 3 * j);
        setMatrix(b, i, m, j);
        rows[j] = i;
        if (f) writeFilter(filter.slots, f, 4 * j, () => i);
        j++;
      }
      drawn = j;
      for (const name of ["iCenter", "iSize", "iColor"]) attribute(name).needsUpdate = true;
      faces.instanceMatrix.needsUpdate = true;
      api.detail!(1);
    },
    setOpacity(opacity) {
      setOpacity(material, opacity);
      faces.visible = opacity >= 1;
    },
    detail(fraction) {
      geometry.instanceCount = faces.count = subsetCount(drawn, fraction);
    },
    dispose() {
      geometry.dispose();
      material.dispose();
      faces.geometry.dispose();
      depth.dispose();
      faces.dispose();
    },
  };
  api.setOpacity(layer.opacity);
  return api;
}
