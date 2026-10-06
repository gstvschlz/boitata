import * as THREE from "three";
import type { Box, Vec3 } from "./bounds";

/** Segments a section holds at most: each takes four vec4 uniforms in every material. */
export const MAX_SEGMENTS = 16;
/** Views looking down at least this steeply pick a quick cut's ends on a level plane, others on an upright one. */
const LEVEL_PICK_DIP = 20;

type XY = [number, number];
type Plane = [number, number, number, number];

/** A section as Python reads and writes it: real-world points, width in meters (null for the default), dip. */
export interface SectionState {
  points: number[][];
  width: number | null;
  dip: number;
  unfolded: boolean;
}

/** One straight piece of a section, in local coordinates. */
export interface Segment {
  a: XY;
  /** Unit direction in plan. */
  u: XY;
  length: number;
  /** Distance along the section where the segment starts. */
  chainage: number;
  /** Plane through the segment, tilted by the dip: signed distance `dot(n, p) + d`. */
  slab: Plane;
  /** Upright planes bounding the segment, positive inside: across its start, and the bisectors at bends. */
  start: Plane;
  end: Plane;
  /** Unfolding motion in plan, `x' = c x + s y + tx, y' = -s x + c y + ty`, as (c, s, tx, ty). */
  move: Plane;
}

export interface Section {
  segments: Segment[];
  /** Elevation the trace hinges on when the section dips. */
  z: number;
  half: number;
  dip: number;
  length: number;
}

const dot2 = (a: XY, b: XY) => a[0] * b[0] + a[1] * b[1];
const plane = (n: XY, p: XY): Plane => [n[0], n[1], 0, -dot2(n, p)];
const side = (q: Plane, p: Vec3) => q[0] * p[0] + q[1] * p[1] + q[2] * p[2] + q[3];

/** Vertices with consecutive repeats in plan dropped. */
export function distinct(points: XY[]): XY[] {
  return points.filter((p, i) => i === 0 || p[0] !== points[i - 1][0] || p[1] !== points[i - 1][1]);
}

/**
 * A section along a polyline in plan, `width` thick around the surface that runs through it at elevation `z` and
 * dips `dip` degrees to the right of the direction it is drawn in (90 is upright). Null for fewer than two
 * distinct vertices.
 */
export function buildSection(points: XY[], z: number, width: number, dip: number): Section | null {
  const p = distinct(points).slice(0, MAX_SEGMENTS + 1);
  if (p.length < 2) return null;
  const d = THREE.MathUtils.degToRad(dip);
  const units = p.slice(1).map((q, i) => {
    const step: XY = [q[0] - p[i][0], q[1] - p[i][1]];
    const length = Math.hypot(...step);
    return { u: [step[0] / length, step[1] / length] as XY, length };
  });
  const bisector = (i: number): XY => {
    const [a, b] = [units[i - 1].u, units[i].u];
    const s: XY = [a[0] + b[0], a[1] + b[1]];
    const n = Math.hypot(...s);
    return n > 1e-9 ? [s[0] / n, s[1] / n] : b;
  };
  let chainage = 0;
  const segments = units.map(({ u, length }, i) => {
    const a = p[i];
    const n: XY = [u[1], -u[0]];
    const normal: Vec3 = [Math.sin(d) * n[0], Math.sin(d) * n[1], Math.cos(d)];
    const slab: Plane = [...normal, -(normal[0] * a[0] + normal[1] * a[1] + normal[2] * z)];
    const start = plane(i === 0 ? u : bisector(i), a);
    const toward = i === units.length - 1 ? u : bisector(i + 1);
    const end = plane([-toward[0], -toward[1]], p[i + 1]);
    const move: Plane = [u[0], u[1], chainage - (u[0] * a[0] + u[1] * a[1]), u[1] * a[0] - u[0] * a[1]];
    const segment = { a, u, length, chainage, slab, start, end, move };
    chainage += length;
    return segment;
  });
  return { segments, z, half: width / 2, dip, length: chainage };
}

/** Whether a point lies in the slab: within half the width of the surface, between the section's ends. */
export function inSlab(s: Section, p: Vec3): boolean {
  return s.segments.some((g) => side(g.start, p) >= 0 && side(g.end, p) >= 0 && Math.abs(side(g.slab, p)) <= s.half);
}

/** Index of the segment whose stretch holds a point: the first one whose end it has not passed. */
export function segmentOf(s: Section, p: Vec3): number {
  const last = s.segments.length - 1;
  for (let j = 0; j < last; j++) if (side(s.segments[j].end, p) >= 0) return j;
  return last;
}

/** A point of the unfolded section: distance along it, offset to the left of its direction, elevation. */
export function unfold(s: Section, p: Vec3, ref: Vec3 = p): Vec3 {
  const [c, sn, tx, ty] = s.segments[segmentOf(s, ref)].move;
  return [c * p[0] + sn * p[1] + tx, -sn * p[0] + c * p[1] + ty, p[2]];
}

/** `point` moved onto the nearest bearing from `from` that is a multiple of 45 degrees, keeping its projection. */
export function snap45(from: XY, point: XY): XY {
  const step: XY = [point[0] - from[0], point[1] - from[1]];
  const bearing = THREE.MathUtils.degToRad(Math.round(THREE.MathUtils.radToDeg(Math.atan2(step[0], step[1])) / 45) * 45);
  const dir: XY = [Math.sin(bearing), Math.cos(bearing)];
  const along = Math.max(dot2(step, dir), 0);
  return [from[0] + along * dir[0], from[1] + along * dir[1]];
}

export interface Ray {
  origin: Vec3;
  direction: Vec3;
}

/**
 * Ends of a quick cut from the rays under the two ends of the swept screen line: on the level plane through
 * `center` when the view looks down at least 20 degrees, else on the upright plane through `center` facing the
 * camera. Null when a ray misses its plane.
 */
export function pickCut(rays: [Ray, Ray], center: Vec3, forward: Vec3): [Vec3, Vec3] | null {
  const level = -forward[2] >= Math.sin(THREE.MathUtils.degToRad(LEVEL_PICK_DIP));
  const flat = Math.hypot(forward[0], forward[1]) || 1;
  const normal: Vec3 = level ? [0, 0, 1] : [forward[0] / flat, forward[1] / flat, 0];
  const hits = rays.map(({ origin, direction }) => {
    const along = normal[0] * direction[0] + normal[1] * direction[1] + normal[2] * direction[2];
    if (Math.abs(along) < 1e-9) return null;
    const t = ((center[0] - origin[0]) * normal[0] + (center[1] - origin[1]) * normal[1] + (center[2] - origin[2]) * normal[2]) / along;
    if (t <= 0) return null;
    return [0, 1, 2].map((k) => origin[k] + t * direction[k]) as Vec3;
  });
  return hits[0] && hits[1] ? [hits[0], hits[1]] : null;
}

/** Plan trace of the section's surface at elevation `z`, with its two slab edges, as polylines. */
export function traces(s: Section, z: number): [XY[], XY[], XY[]] {
  const d = THREE.MathUtils.degToRad(s.dip);
  const shift = (z - s.z) / Math.tan(d);
  const edge = s.half / Math.sin(d);
  const along = (offset: number): XY[] => {
    const out: XY[] = [];
    s.segments.forEach((g, j) => {
      const n: XY = [g.u[1], -g.u[0]];
      const at = (p: XY): XY => [p[0] + n[0] * offset, p[1] + n[1] * offset];
      const b: XY = [g.a[0] + g.u[0] * g.length, g.a[1] + g.u[1] * g.length];
      if (j === 0) out.push(at(g.a));
      if (j === s.segments.length - 1) out.push(at(b));
      else {
        // miter: the offset lines of this segment and the next meet on the bisector through the vertex
        const next = s.segments[j + 1].u;
        const m: XY = [n[0] + next[1], n[1] - next[0]];
        const k = 1 + dot2(n, [next[1], -next[0]]);
        out.push(k > 1e-6 ? [b[0] + (m[0] * offset) / k, b[1] + (m[1] * offset) / k] : at(b));
      }
    });
    return out;
  };
  return [along(-shift), along(-shift - edge), along(-shift + edge)];
}

/** The section's plan bounds at the elevations of `box`, widened by the slab; null when it misses the box. */
export function sectionBox(s: Section, box: Box): Box | null {
  const xy = [box.min[2], box.max[2]].flatMap((z) => traces(s, z).flat());
  const min: Vec3 = [Math.max(box.min[0], Math.min(...xy.map((p) => p[0]))), Math.max(box.min[1], Math.min(...xy.map((p) => p[1]))), box.min[2]];
  const max: Vec3 = [Math.min(box.max[0], Math.max(...xy.map((p) => p[0]))), Math.min(box.max[1], Math.max(...xy.map((p) => p[1]))), box.max[2]];
  return min[0] <= max[0] && min[1] <= max[1] ? { min, max } : null;
}

/** Uniforms every material shares; the viewer writes its section into them before each frame. */
export const sectionUniforms = {
  uSecCount: { value: 0 },
  uSecHalf: { value: 0 },
  uSecUnfolded: { value: 0 },
  uSecSlab: { value: Array.from({ length: MAX_SEGMENTS }, () => new THREE.Vector4()) },
  uSecStart: { value: Array.from({ length: MAX_SEGMENTS }, () => new THREE.Vector4()) },
  uSecEnd: { value: Array.from({ length: MAX_SEGMENTS }, () => new THREE.Vector4()) },
  uSecMove: { value: Array.from({ length: MAX_SEGMENTS }, () => new THREE.Vector4()) },
};

export function writeSection(s: Section | null, unfolded: boolean): void {
  const u = sectionUniforms;
  u.uSecCount.value = s ? s.segments.length : 0;
  u.uSecHalf.value = s ? s.half : 0;
  u.uSecUnfolded.value = s && unfolded ? 1 : 0;
  s?.segments.forEach((g, j) => {
    u.uSecSlab.value[j].fromArray(g.slab);
    u.uSecStart.value[j].fromArray(g.start);
    u.uSecEnd.value[j].fromArray(g.end);
    u.uSecMove.value[j].fromArray(g.move);
  });
}

/** The GLSL mirror of `inSlab`, `segmentOf` and `unfold`, plus the inverse motion and the distance to a cut face. */
export const SECTION_GLSL = /* glsl */ `
uniform int uSecCount;
uniform float uSecHalf;
uniform float uSecUnfolded;
uniform vec4 uSecSlab[${MAX_SEGMENTS}];
uniform vec4 uSecStart[${MAX_SEGMENTS}];
uniform vec4 uSecEnd[${MAX_SEGMENTS}];
uniform vec4 uSecMove[${MAX_SEGMENTS}];
float secSide(vec4 q, vec3 p) { return dot(q.xyz, p) + q.w; }
bool secIn(int j, vec3 p) {
  return secSide(uSecStart[j], p) >= 0.0 && secSide(uSecEnd[j], p) >= 0.0 && abs(secSide(uSecSlab[j], p)) <= uSecHalf;
}
int secHolding(vec3 p) {
  for (int j = 0; j < ${MAX_SEGMENTS}; j++) {
    if (j >= uSecCount) break;
    if (secIn(j, p)) return j;
  }
  return -1;
}
bool sectionKeep(vec3 p) { return uSecCount == 0 || secHolding(p) >= 0; }
int secSegment(vec3 p) {
  for (int j = 0; j < ${MAX_SEGMENTS}; j++) {
    if (j >= uSecCount - 1 || secSide(uSecEnd[j], p) >= 0.0) return j;
  }
  return 0;
}
vec3 sectionPlace(vec3 p, vec3 ref) {
  if (uSecUnfolded < 0.5 || uSecCount == 0) return p;
  vec4 m = uSecMove[secSegment(ref)];
  return vec3(m.x * p.x + m.y * p.y + m.z, -m.y * p.x + m.x * p.y + m.w, p.z);
}
vec3 sectionTurn(vec3 v, vec3 ref) {
  if (uSecUnfolded < 0.5 || uSecCount == 0) return v;
  vec4 m = uSecMove[secSegment(ref)];
  return vec3(m.x * v.x + m.y * v.y, -m.y * v.x + m.x * v.y, v.z);
}
vec3 sectionUnplace(vec3 q, vec3 ref) {
  if (uSecUnfolded < 0.5 || uSecCount == 0) return q;
  vec4 m = uSecMove[secSegment(ref)];
  vec2 r = q.xy - m.zw;
  return vec3(m.x * r.x - m.y * r.y, m.y * r.x + m.x * r.y, q.z);
}
float sectionEdge(vec3 p) {
  int j = secHolding(p);
  if (j < 0) return 1e30;
  float e = uSecHalf - abs(secSide(uSecSlab[j], p));
  if (j == 0) e = min(e, secSide(uSecStart[j], p));
  if (j == uSecCount - 1) e = min(e, secSide(uSecEnd[j], p));
  return e;
}
`;
