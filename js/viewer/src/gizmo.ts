import * as THREE from "three";
import type { Theme } from "./theme";

const NS = "http://www.w3.org/2000/svg";
const SIZE = 128;
const C = SIZE / 2;
const R = 22;

const FACES = [
  { key: "up", label: "Up", normal: new THREE.Vector3(0, 0, 1), a: new THREE.Vector3(1, 0, 0), b: new THREE.Vector3(0, 1, 0), hint: "Plan view" },
  { key: "down", label: "Down", normal: new THREE.Vector3(0, 0, -1), a: new THREE.Vector3(1, 0, 0), b: new THREE.Vector3(0, 1, 0), hint: "View from below" },
  { key: "north", label: "North", normal: new THREE.Vector3(0, 1, 0), a: new THREE.Vector3(1, 0, 0), b: new THREE.Vector3(0, 0, 1), hint: "View from the north" },
  { key: "south", label: "South", normal: new THREE.Vector3(0, -1, 0), a: new THREE.Vector3(1, 0, 0), b: new THREE.Vector3(0, 0, 1), hint: "View from the south" },
  { key: "west", label: "West", normal: new THREE.Vector3(-1, 0, 0), a: new THREE.Vector3(0, 1, 0), b: new THREE.Vector3(0, 0, 1), hint: "View from the west" },
  { key: "east", label: "East", normal: new THREE.Vector3(1, 0, 0), a: new THREE.Vector3(0, 1, 0), b: new THREE.Vector3(0, 0, 1), hint: "View from the east" },
] as const;

export type Snap = (typeof FACES)[number]["key"];

const CORNERS = [
  [1, 1],
  [1, -1],
  [-1, -1],
  [-1, 1],
] as const;

/** Orientation gizmo: a cube face-labeled Top/Bottom/Front/Back/Left/Right; clicking a face snaps the camera to it. */
export class Gizmo {
  readonly el = document.createElementNS(NS, "svg");
  private theme: Theme | null = null;

  constructor(private onSnap: (snap: Snap) => void) {
    this.el.setAttribute("class", "btv-gizmo");
    this.el.setAttribute("width", String(SIZE));
    this.el.setAttribute("height", String(SIZE));
    this.el.setAttribute("viewBox", `0 0 ${SIZE} ${SIZE}`);
  }

  setTheme(theme: Theme): void {
    this.theme = theme;
  }

  update(camera: THREE.Camera): void {
    const right = new THREE.Vector3(1, 0, 0).applyQuaternion(camera.quaternion);
    const up = new THREE.Vector3(0, 1, 0).applyQuaternion(camera.quaternion);
    const back = new THREE.Vector3(0, 0, 1).applyQuaternion(camera.quaternion);
    const project = (v: THREE.Vector3) => ({ x: C + v.dot(right), y: C - v.dot(up) });
    const visible = FACES.map((face) => ({ face, depth: face.normal.dot(back) }))
      .filter((f) => f.depth > 0.02)
      .sort((a, b) => a.depth - b.depth);
    const fill = this.theme?.raised ?? "#F4F6F8";
    const stroke = this.theme?.border ?? "#D5DDE3";
    const text = this.theme?.text ?? "#17303F";
    this.el.replaceChildren();
    for (const { face } of visible) {
      const corners = CORNERS.map(([s, t]) =>
        project(face.normal.clone().multiplyScalar(R).addScaledVector(face.a, R * s).addScaledVector(face.b, R * t)),
      );
      const center = project(face.normal.clone().multiplyScalar(R));
      const g = document.createElementNS(NS, "g");
      g.setAttribute("class", "btv-gizmo-face");
      g.setAttribute("data-snap", face.key);
      const quad = document.createElementNS(NS, "path");
      quad.setAttribute("class", "btv-face");
      quad.setAttribute("d", `M${corners.map((p) => `${p.x},${p.y}`).join("L")}Z`);
      quad.setAttribute("fill", fill);
      quad.setAttribute("stroke", stroke);
      const label = document.createElementNS(NS, "text");
      label.setAttribute("x", String(center.x));
      label.setAttribute("y", String(center.y));
      label.setAttribute("fill", text);
      label.setAttribute("font-size", "9.5");
      label.setAttribute("font-weight", "600");
      label.setAttribute("font-family", "system-ui, -apple-system, 'Segoe UI', sans-serif");
      label.setAttribute("text-anchor", "middle");
      label.setAttribute("dominant-baseline", "central");
      label.textContent = face.label;
      const title = document.createElementNS(NS, "title");
      title.textContent = face.hint;
      g.append(title, quad, label);
      g.addEventListener("click", (e) => {
        e.stopPropagation();
        this.onSnap(face.key);
      });
      this.el.appendChild(g);
    }
  }
}
