import * as THREE from "three";
import { blocksBox, type Box, type Vec3 } from "../bounds";
import { f32, u32 } from "../buffers";
import { buildVolumeTexture } from "../contour";
import { ALL_SIDES, cellCount, exposedSides, type Grid, HIDDEN, MAX_CELLS, SOLID, WHOLE } from "../faces";
import { filterArray, type LayerFilter, passMask, writeFilter } from "../filter";
import { stratifiedOrder, subsetCount } from "../motion";
import { colorAt, kept, mergeCounts, mergedColor, type Paint } from "../paint";
import { cameraPixels, LOD_MIN, lodItems, octree, type ViewCamera, viewKey } from "../lod";
import { pickArray } from "../pick";
import { boxInSlab, type Section } from "../section";
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
  /** The regular grid the blocks fill, when small enough to draw as a shell; null for whole boxes. */
  grid: Grid | null;
}

function blocks(layer: LayerSpec, buffers: Buffers): Blocks {
  const centers = f32(buffers, layer.geometry.centers);
  const sizes = f32(buffers, layer.geometry.sizes);
  const axes = (layer.axes ?? [[1, 0, 0], [0, 1, 0], [0, 0, 1]]) as [Vec3, Vec3, Vec3];
  const n = centers.length / 3;
  const grid: Grid | null =
    layer.grid && layer.geometry.cells ? { count: layer.grid, cells: u32(buffers, layer.geometry.cells) } : null;
  return {
    centers,
    sizes,
    axes,
    n,
    box: blocksBox(centers, sizes, axes, null),
    order: stratifiedOrder(centers, n),
    grid: grid && cellCount(grid) <= MAX_CELLS ? grid : null,
  };
}

/**
 * The sides each block of a regular grid shows: a block passing the paint, the filter (judged here; the GPU passes
 * every row this passes) and the section draws only the sides not shared with a solid neighbour. Blocks the section
 * cuts, and every block while unfolded or translucent, show all their sides, so caps and blending stay as with whole
 * boxes. Without a grid every kept block draws whole.
 */
class Shell {
  dirty = false;
  private cut: Section | null = null;
  private unfolded = false;
  private whole = false;
  private slab: Uint8Array | null = null;
  private scratch: Uint8Array | null = null;

  constructor(
    private b: Blocks,
    private filter: LayerFilter,
  ) {}

  section(s: Section | null, unfolded: boolean): void {
    if (!this.b.grid || (s === this.cut && unfolded === this.unfolded)) return;
    this.cut = s;
    this.unfolded = unfolded;
    this.slab = s ? slabStates(this.b, s) : null;
    this.dirty = true;
  }

  setWhole(whole: boolean): void {
    if (whole === this.whole) return;
    this.whole = whole;
    this.dirty = !!this.b.grid;
  }

  filtered(): void {
    this.dirty = !!this.b.grid;
  }

  /** Per row, the sides it draws; 0 leaves it out. */
  sides(paint: Paint): Uint8Array {
    this.dirty = false;
    const { grid, n } = this.b;
    if (!grid) {
      const out = new Uint8Array(n);
      for (let i = 0; i < n; i++) if (kept(paint, i)) out[i] = ALL_SIDES;
      return out;
    }
    const f = this.filter;
    const state = passMask(f.compiled, f.slots, "row", n, paint.on === "row" ? paint.keep : null);
    const whole = this.whole || this.unfolded;
    for (let i = 0; i < n; i++) {
      if (!state[i]) continue;
      const s = this.slab ? this.slab[i] : 2;
      state[i] = s === 0 ? HIDDEN : whole || s === 1 ? WHOLE : SOLID;
    }
    this.scratch ??= new Uint8Array(cellCount(grid));
    return exposedSides(grid, state, this.scratch);
  }
}

function slabStates(b: Blocks, s: Section): Uint8Array {
  const out = new Uint8Array(b.n);
  const c: Vec3 = [0, 0, 0];
  const z: Vec3 = [0, 0, 0];
  for (let i = 0; i < b.n; i++) {
    for (let a = 0; a < 3; a++) {
      c[a] = b.centers[3 * i + a];
      z[a] = b.sizes[3 * i + a];
    }
    out[i] = boxInSlab(s, c, b.axes, z);
  }
  return out;
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

/**
 * Writes at instance j the box along the model's axes around the kept blocks of `order[from:to]`; returns the
 * first of them, which a click on the box picks.
 */
function mergedMatrix(b: Blocks, order: Uint32Array, from: number, to: number, use: Uint8Array, m: Float32Array, j: number): number {
  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  let first = -1;
  for (let p = to - 1; p >= from; p--) {
    const i = order[p];
    if (!use[i]) continue;
    first = i;
    for (let k = 0; k < 3; k++) {
      const ax = b.axes[k];
      const along = ax[0] * b.centers[3 * i] + ax[1] * b.centers[3 * i + 1] + ax[2] * b.centers[3 * i + 2];
      const half = b.sizes[3 * i + k] / 2;
      lo[k] = Math.min(lo[k], along - half);
      hi[k] = Math.max(hi[k], along + half);
    }
  }
  const o = 16 * j;
  m.fill(0, o, o + 16);
  for (let k = 0; k < 3; k++) {
    const mid = (lo[k] + hi[k]) / 2;
    for (let a = 0; a < 3; a++) {
      m[o + 4 * k + a] = b.axes[k][a] * (hi[k] - lo[k]);
      m[o + 12 + a] += b.axes[k][a] * mid;
    }
  }
  m[o + 15] = 1;
  return first;
}

/** A unit box whose vertices name their side, in `ALL_SIDES` bit order, with room for each instance's sides. */
function boxGeometry(n: number): THREE.BoxGeometry {
  const g = new THREE.BoxGeometry(1, 1, 1);
  g.setAttribute("aSide", new THREE.Float32BufferAttribute(Array.from({ length: 24 }, (_, v) => Math.floor(v / 4)), 1));
  g.setAttribute("aSides", new THREE.InstancedBufferAttribute(new Float32Array(Math.max(n, 1)), 1));
  return g;
}

function sidesArray(geometry: THREE.BufferGeometry): Float32Array {
  const a = geometry.getAttribute("aSides") as THREE.InstancedBufferAttribute;
  a.needsUpdate = true;
  return a.array as Float32Array;
}

function boxes(b: Blocks, material: THREE.Material, colored: boolean) {
  const mesh = new THREE.InstancedMesh(boxGeometry(b.n), material, Math.max(b.n, 1));
  mesh.frustumCulled = false;
  mesh.count = 0;
  const colors = colored ? new THREE.InstancedBufferAttribute(new Uint8Array(3 * Math.max(b.n, 1)), 3, true) : null;
  if (colors) mesh.instanceColor = colors as unknown as THREE.InstancedBufferAttribute;
  return { mesh, colors };
}

/**
 * Blocks as instanced boxes along the model's axes, each scaled to its own (sub-)block size; a regular grid draws
 * only the sides that can show.
 */
export function cells(layer: LayerSpec, buffers: Buffers, filter: LayerFilter): Representation {
  const b = blocks(layer, buffers);
  const shell = new Shell(b, filter);
  const contour = layer.contour;
  const material = shadedMaterial(true, layer.opacity, filter.uniforms, { caps: true, contour: !!contour, sides: true });
  if (contour) {
    const [ax, ay, az] = (layer.axes ?? [[1, 0, 0], [0, 1, 0], [0, 0, 1]]) as [Vec3, Vec3, Vec3];
    const texture = buildVolumeTexture(f32(buffers, contour.buffer), contour.count);
    const u = material.uniforms;
    u.uContourOn.value = 1;
    u.uContourStep.value = contour.step;
    u.uContourTexture.value = texture;
    u.uContourOrigin.value.fromArray(contour.origin);
    u.uContourExtent.value.set(
      contour.size[0] * contour.count[0],
      contour.size[1] * contour.count[1],
      contour.size[2] * contour.count[2],
    );
    u.uContourAxes.value.set(ax[0], ay[0], az[0], ax[1], ay[1], az[1], ax[2], ay[2], az[2]);
  }
  const { mesh, colors } = boxes(b, material, true);
  let drawn = 0;
  let fraction = 1;
  let painted: Paint | null = null;
  let sides: Uint8Array = new Uint8Array(0);
  let counts: Int32Array = new Int32Array(0);
  const tree = b.n >= LOD_MIN ? octree(b.centers, b.n) : null;
  let view: { camera: ViewCamera; height: number } | null = null;
  let seen = "";
  shell.setWhole(layer.opacity < 1);

  /**
   * Writes the blocks that draw. Far from the camera, the drawn blocks of an octree cell too small on screen draw
   * as one box around them, with all its sides, in the color of their mean value or most frequent category.
   */
  const emit = () => {
    const paint = painted!;
    const m = mesh.instanceMatrix.array as Float32Array;
    const c = colors!.array as Uint8Array;
    const f = filterArray(mesh.geometry, b.n, filter.slots, true);
    const rows = pickArray(mesh.geometry, b.n, true);
    const shown = sidesArray(mesh.geometry);
    let j = 0;
    const put = (i: number, s: number) => {
      rows[j] = i;
      shown[j] = s;
      if (f) writeFilter(filter.slots, f, 4 * j, () => i);
      j++;
    };
    if (tree && view) {
      const items = lodItems(tree, sides, cameraPixels(view.camera, view.height));
      for (let k = 0; k < items.count; k++) {
        const [s, e] = [items.from[items.ranks[k]], items.to[items.ranks[k]]];
        if (e - s === 1) {
          const i = tree.order[s];
          setMatrix(b, i, m, j);
          colorAt(paint, i, c, 3 * j);
          put(i, sides[i]);
          continue;
        }
        mergedColor(paint, tree.order, s, e, sides, c, 3 * j, counts);
        put(mergedMatrix(b, tree.order, s, e, sides, m, j), ALL_SIDES);
      }
      seen = viewKey(view.camera, view.height);
    } else
      for (const i of b.order) {
        if (!sides[i]) continue;
        setMatrix(b, i, m, j);
        colorAt(paint, i, c, 3 * j);
        put(i, sides[i]);
      }
    drawn = j;
    mesh.count = subsetCount(drawn, fraction);
    mesh.instanceMatrix.needsUpdate = true;
    colors!.needsUpdate = true;
  };
  const fill = (paint: Paint) => {
    painted = paint;
    sides = shell.sides(paint);
    counts = mergeCounts(paint);
    emit();
  };

  return {
    object: mesh,
    box: b.box,
    instances: b.n,
    paint: fill,
    setOpacity(opacity) {
      setOpacity(material, opacity);
      shell.setWhole(opacity < 1);
    },
    detail(f) {
      fraction = f;
      mesh.count = subsetCount(drawn, f);
    },
    section: (s, unfolded) => shell.section(s, unfolded),
    filtered: () => shell.filtered(),
    frame(_w, height, camera, moving) {
      if (!painted) return;
      if (tree) view = { camera, height };
      if (shell.dirty) fill(painted);
      else if (tree && !moving && seen !== viewKey(camera, height)) emit();
    },
    dispose() {
      mesh.geometry.dispose();
      material.dispose();
      mesh.dispose();
      if (contour) (material.uniforms.uContourTexture.value as THREE.Data3DTexture).dispose();
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
  const shell = new Shell(b, filter);
  const material = boxEdgeMaterial(b.axes, layer.lineWidth ?? EDGE_WIDTH, layer.opacity, filter.uniforms);
  const geometry = edgeGeometry(b.n);
  const edges = new THREE.Mesh(geometry, material);
  edges.frustumCulled = false;
  const depth = depthMaterial(filter.uniforms, true);
  const { mesh: faces } = boxes(b, depth, false);
  faces.renderOrder = -2;
  const group = new THREE.Group();
  group.add(faces, edges);
  let drawn = 0;
  let fraction = 1;
  let painted: Paint | null = null;
  const attribute = (name: string) => geometry.getAttribute(name) as THREE.InstancedBufferAttribute;

  const api: Representation = {
    object: group,
    box: b.box,
    instances: b.n,
    paint(paint: Paint) {
      painted = paint;
      const sides = shell.sides(paint);
      const shown = sidesArray(faces.geometry);
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
        if (!sides[i]) continue;
        center.set(b.centers.subarray(3 * i, 3 * i + 3), 3 * j);
        size.set(b.sizes.subarray(3 * i, 3 * i + 3), 3 * j);
        colorAt(paint, i, color, 3 * j);
        setMatrix(b, i, m, j);
        rows[j] = i;
        shown[j] = sides[i];
        if (f) writeFilter(filter.slots, f, 4 * j, () => i);
        j++;
      }
      drawn = j;
      for (const name of ["iCenter", "iSize", "iColor"]) attribute(name).needsUpdate = true;
      faces.instanceMatrix.needsUpdate = true;
      api.detail!(fraction);
    },
    setOpacity(opacity) {
      setOpacity(material, opacity);
      faces.visible = opacity >= 1;
      shell.setWhole(opacity < 1);
    },
    detail(f) {
      fraction = f;
      geometry.instanceCount = faces.count = subsetCount(drawn, f);
    },
    section: (s, unfolded) => shell.section(s, unfolded),
    filtered: () => shell.filtered(),
    frame() {
      if (shell.dirty && painted) api.paint(painted);
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
