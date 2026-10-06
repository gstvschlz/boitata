import * as THREE from "three";
import { backWalls, type Box, type Vec3 } from "./bounds";
import type { Theme } from "./theme";
import { formatTicks, niceTicks } from "./ticks";

export interface Label {
  x: number;
  y: number;
  text: string;
  title: boolean;
}

const FONT = "11px system-ui, -apple-system, 'Segoe UI', sans-serif";
const TITLE_FONT = "600 11px system-ui, -apple-system, 'Segoe UI', sans-serif";

/** Bounding box, back-wall grid and real-world tick labels (HTML overlay). */
export class Axes {
  readonly group = new THREE.Group();
  readonly overlay = document.createElement("div");
  show = { box: true, grid: true, ticks: true };
  labels: Label[] = [];
  private box: Box | null = null;
  private walls = "";
  private outline = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial());
  private grid = new THREE.LineSegments(
    new THREE.BufferGeometry(),
    new THREE.LineBasicMaterial({ depthWrite: false }),
  );
  private ticks: number[][] = [[], [], []];
  private pool: HTMLDivElement[] = [];

  constructor(
    private origin: Vec3,
    private titles: [string, string, string],
  ) {
    this.overlay.className = "btv-labels";
    this.grid.renderOrder = -1;
    this.outline.frustumCulled = this.grid.frustumCulled = false;
    this.group.add(this.outline, this.grid);
  }

  setTheme(theme: Theme): void {
    (this.outline.material as THREE.LineBasicMaterial).color.set(theme.box);
    (this.grid.material as THREE.LineBasicMaterial).color.set(theme.grid);
  }

  setBox(box: Box | null): void {
    this.box = box;
    this.walls = "";
    if (!box) {
      this.ticks = [[], [], []];
      this.outline.geometry.setAttribute("position", new THREE.Float32BufferAttribute([], 3));
      return;
    }
    this.ticks = [0, 1, 2].map((a) =>
      niceTicks(box.min[a] + this.origin[a], box.max[a] + this.origin[a], 5).map((t) => t - this.origin[a]),
    );
    const p: number[] = [];
    const corner = (i: number) => [0, 1, 2].map((a) => ((i >> a) & 1 ? box.max[a] : box.min[a]));
    for (let i = 0; i < 8; i++)
      for (let a = 0; a < 3; a++) if (!((i >> a) & 1)) p.push(...corner(i), ...corner(i | (1 << a)));
    this.outline.geometry.setAttribute("position", new THREE.Float32BufferAttribute(p, 3));
  }

  /** Re-picks the back walls and places the labels for the current camera. */
  update(camera: THREE.Camera, width: number, height: number): void {
    const box = this.box;
    this.outline.visible = this.show.box && !!box;
    this.grid.visible = this.show.grid && !!box;
    this.labels = [];
    if (!box) return this.place();
    const eye = camera.position.toArray() as Vec3;
    const walls = backWalls(box, eye);
    const key = walls.join("");
    if (key !== this.walls) {
      this.walls = key;
      this.buildGrid(box, walls);
    }
    if (this.show.ticks) this.buildLabels(box, camera, width, height);
    this.place();
  }

  private buildGrid(box: Box, walls: [0 | 1, 0 | 1, 0 | 1]): void {
    const p: number[] = [];
    for (let a = 0; a < 3; a++) {
      const at = walls[a] ? box.max[a] : box.min[a];
      for (const b of [0, 1, 2]) {
        if (b === a) continue;
        const c = 3 - a - b;
        for (const t of this.ticks[b]) {
          const u: Vec3 = [0, 0, 0];
          const v: Vec3 = [0, 0, 0];
          u[a] = v[a] = at;
          u[b] = v[b] = t;
          u[c] = box.min[c];
          v[c] = box.max[c];
          p.push(...u, ...v);
        }
      }
    }
    this.grid.geometry.setAttribute("position", new THREE.Float32BufferAttribute(p, 3));
  }

  /**
   * Labels each axis along one of its four parallel box edges: of the two outermost on screen (silhouette edges),
   * the lower one for x and y, the left one for z, offset away from the box.
   */
  private buildLabels(box: Box, camera: THREE.Camera, w: number, h: number): void {
    const project = (p: Vec3) => {
      const v = new THREE.Vector3(...p).project(camera);
      return { x: ((v.x + 1) / 2) * w, y: ((1 - v.y) / 2) * h, z: v.z };
    };
    const mid = [0, 1, 2].map((a) => (box.min[a] + box.max[a]) / 2) as Vec3;
    const center = project(mid);
    const side = (a: number, s: number) => (s ? box.max[a] : box.min[a]);
    for (let a = 0; a < 3; a++) {
      const [b, c] = [0, 1, 2].filter((k) => k !== a);
      const tip = [...mid] as Vec3;
      tip[a] = box.max[a];
      const base = [...mid] as Vec3;
      base[a] = box.min[a];
      const t = project(tip);
      const s = project(base);
      const length = Math.hypot(t.x - s.x, t.y - s.y);
      if (length < 40) continue;
      const n = { x: -(t.y - s.y) / length, y: (t.x - s.x) / length };
      const edges = [0, 1].flatMap((sb) =>
        [0, 1].map((sc) => {
          const at = (v: number): Vec3 => {
            const p: Vec3 = [0, 0, 0];
            p[a] = v;
            p[b] = side(b, sb);
            p[c] = side(c, sc);
            return p;
          };
          const m = project(at(mid[a]));
          return { at, m, offset: (m.x - center.x) * n.x + (m.y - center.y) * n.y };
        }),
      );
      edges.sort((p, q) => p.offset - q.offset);
      const [lo, hi] = [edges[0], edges[3]];
      const edge = a < 2 ? (lo.m.y >= hi.m.y ? lo : hi) : lo.m.x <= hi.m.x ? lo : hi;
      if ([box.min[a], box.max[a]].some((v) => project(edge.at(v)).z > 1)) continue;
      const sign = edge.offset >= 0 ? 1 : -1;
      const shift = (p: { x: number; y: number }, d: number) => ({ x: p.x + n.x * sign * d, y: p.y + n.y * sign * d });
      const ticks = this.ticks[a];
      const texts = formatTicks(ticks.map((v) => v + this.origin[a]));
      const longest = Math.max(...texts.map((x) => x.length), 1);
      const spacing = ticks.length > 1 ? length * ((ticks[1] - ticks[0]) / (box.max[a] - box.min[a])) : length;
      const footprint = Math.abs(n.y) * (longest * 6.5 + 8) + Math.abs(n.x) * 16;
      const every = Math.max(1, Math.ceil(footprint / Math.max(spacing, 1)));
      const away = Math.abs(n.x) * (longest * 3.4 + 6) + Math.abs(n.y) * 10 + 4;
      ticks.forEach((v, i) => {
        if (i % every) return;
        const q = shift(project(edge.at(v)), away);
        this.labels.push({ x: q.x, y: q.y, text: texts[i], title: false });
      });
      const title = shift(edge.m, 2 * away + 12);
      this.labels.push({ x: title.x, y: title.y, text: this.titles[a], title: true });
    }
  }

  private place(): void {
    while (this.pool.length < this.labels.length) {
      const div = document.createElement("div");
      div.className = "btv-label";
      this.overlay.appendChild(div);
      this.pool.push(div);
    }
    this.pool.forEach((div, i) => {
      const label = this.labels[i];
      div.style.display = label ? "" : "none";
      if (!label) return;
      div.textContent = label.text;
      div.classList.toggle("btv-title", label.title);
      div.style.transform = `translate(${label.x}px, ${label.y}px) translate(-50%, -50%)`;
    });
  }

  /** Draws the labels on a screenshot. */
  draw(ctx: CanvasRenderingContext2D, scale: number, theme: Theme): void {
    ctx.save();
    ctx.scale(scale, scale);
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    for (const label of this.labels) {
      ctx.font = label.title ? TITLE_FONT : FONT;
      ctx.fillStyle = label.title ? theme.text : theme.muted;
      ctx.fillText(label.text, label.x, label.y);
    }
    ctx.restore();
  }
}
