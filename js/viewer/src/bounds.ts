export type Vec3 = [number, number, number];
export interface Box {
  min: Vec3;
  max: Vec3;
}

export function union(boxes: Iterable<Box | null>): Box | null {
  let out: Box | null = null;
  for (const b of boxes) {
    if (!b) continue;
    if (!out) out = { min: [...b.min], max: [...b.max] };
    else
      for (let a = 0; a < 3; a++) {
        out.min[a] = Math.min(out.min[a], b.min[a]);
        out.max[a] = Math.max(out.max[a], b.max[a]);
      }
  }
  return out;
}

/** Box around the layers that are shown. */
export function visibleBox(layers: Iterable<{ visible: boolean; box: Box | null }>): Box | null {
  const boxes: (Box | null)[] = [];
  for (const layer of layers) if (layer.visible) boxes.push(layer.box);
  return union(boxes);
}

/** Box of flat xyz triples, skipping NaN. */
export function boxOf(xyz: ArrayLike<number>): Box | null {
  const min: Vec3 = [Infinity, Infinity, Infinity];
  const max: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i + 2 < xyz.length; i += 3)
    for (let a = 0; a < 3; a++) {
      const v = xyz[i + a];
      if (v < min[a]) min[a] = v;
      if (v > max[a]) max[a] = v;
    }
  return min[0] <= max[0] ? { min, max } : null;
}

/** Per axis, 0 when the wall farthest from `eye` is at the box's minimum, 1 at its maximum. */
export function backWalls(box: Box, eye: Vec3): [0 | 1, 0 | 1, 0 | 1] {
  const side = (a: number): 0 | 1 => (eye[a] >= (box.min[a] + box.max[a]) / 2 ? 0 : 1);
  return [side(0), side(1), side(2)];
}

export function lerpBox(a: Box, b: Box, t: number): Box {
  const mix = (u: Vec3, v: Vec3): Vec3 => [0, 1, 2].map((i) => u[i] + (v[i] - u[i]) * t) as Vec3;
  return { min: mix(a.min, b.min), max: mix(a.max, b.max) };
}

/** Length of a box's diagonal, 0 without one. */
export function diagonal(box: Box | null): number {
  return box ? Math.hypot(box.max[0] - box.min[0], box.max[1] - box.min[1], box.max[2] - box.min[2]) : 0;
}

/** Box of the xyz triples whose element `index(i)` is kept, `stride` triples per element (2 for segments). */
export function maskedBox(xyz: ArrayLike<number>, keep: Uint8Array | null, stride = 1, index?: ArrayLike<number>): Box | null {
  const min: Vec3 = [Infinity, Infinity, Infinity];
  const max: Vec3 = [-Infinity, -Infinity, -Infinity];
  const n = xyz.length / (3 * stride);
  for (let i = 0; i < n; i++) {
    if (keep && !keep[index ? index[i] : i]) continue;
    for (let s = 0; s < stride; s++)
      for (let a = 0; a < 3; a++) {
        const v = xyz[3 * (stride * i + s) + a];
        if (v < min[a]) min[a] = v;
        if (v > max[a]) max[a] = v;
      }
  }
  return min[0] <= max[0] ? { min, max } : null;
}

/** Box of the kept blocks, each `sizes` long along the model's unit `axes` around its center. */
export function blocksBox(centers: ArrayLike<number>, sizes: ArrayLike<number>, axes: Vec3[], keep: Uint8Array | null): Box | null {
  const min: Vec3 = [Infinity, Infinity, Infinity];
  const max: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < centers.length / 3; i++) {
    if (keep && !keep[i]) continue;
    for (let a = 0; a < 3; a++) {
      let half = 0;
      for (let k = 0; k < 3; k++) half += Math.abs(axes[k][a]) * sizes[3 * i + k];
      half /= 2;
      min[a] = Math.min(min[a], centers[3 * i + a] - half);
      max[a] = Math.max(max[a], centers[3 * i + a] + half);
    }
  }
  return min[0] <= max[0] ? { min, max } : null;
}

/** 1 for the triangles whose face and three vertices are kept. */
export function triangleMask(triangles: ArrayLike<number>, faces: Uint8Array | null, vertices: Uint8Array | null): Uint8Array {
  const out = new Uint8Array(triangles.length / 3);
  for (let t = 0; t < out.length; t++) {
    if (faces && !faces[t]) continue;
    if (vertices && !(vertices[triangles[3 * t]] && vertices[triangles[3 * t + 1]] && vertices[triangles[3 * t + 2]])) continue;
    out[t] = 1;
  }
  return out;
}
