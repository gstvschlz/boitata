import * as THREE from "three";
import type { Theme } from "./theme";

const NS = "http://www.w3.org/2000/svg";
const SIZE = 104;
const R = 36;
const AXES = [
  { key: "east", label: "E", dir: new THREE.Vector3(1, 0, 0), color: "#D1495B" },
  { key: "north", label: "N", dir: new THREE.Vector3(0, 1, 0), color: "#2E9E6A" },
  { key: "plan", label: "Up", dir: new THREE.Vector3(0, 0, 1), color: "#3B7DD8" },
] as const;

export type Snap = (typeof AXES)[number]["key"];

/** Orientation gizmo: east, north and up as seen from the camera; clicking an axis looks back along it. */
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
    const c = SIZE / 2;
    const items = AXES.map((axis) => ({
      ...axis,
      x: c + R * axis.dir.dot(right),
      y: c - R * axis.dir.dot(up),
      depth: axis.dir.dot(back),
    })).sort((a, b) => a.depth - b.depth);
    const text = this.theme?.dark ? "#0B151C" : "#FFFFFF";
    this.el.replaceChildren();
    for (const item of items) {
      const g = document.createElementNS(NS, "g");
      g.setAttribute("class", "btv-gizmo-axis");
      g.setAttribute("data-snap", item.key);
      const line = document.createElementNS(NS, "line");
      for (const [k, v] of Object.entries({ x1: c, y1: c, x2: item.x, y2: item.y })) line.setAttribute(k, String(v));
      line.setAttribute("stroke", item.color);
      line.setAttribute("stroke-width", "2");
      const dot = document.createElementNS(NS, "circle");
      dot.setAttribute("cx", String(item.x));
      dot.setAttribute("cy", String(item.y));
      dot.setAttribute("r", item.label === "Up" ? "11" : "9");
      dot.setAttribute("fill", item.color);
      dot.setAttribute("opacity", item.depth < -0.5 ? "0.6" : "1");
      const label = document.createElementNS(NS, "text");
      label.setAttribute("x", String(item.x));
      label.setAttribute("y", String(item.y));
      label.setAttribute("fill", text);
      label.setAttribute("font-size", "9");
      label.setAttribute("font-weight", "700");
      label.setAttribute("font-family", "system-ui, -apple-system, 'Segoe UI', sans-serif");
      label.setAttribute("text-anchor", "middle");
      label.setAttribute("dominant-baseline", "central");
      label.textContent = item.label;
      const title = document.createElementNS(NS, "title");
      title.textContent = { east: "View from the east", north: "View from the north", plan: "Plan view" }[item.key];
      g.append(title, line, dot, label);
      g.addEventListener("click", (e) => {
        e.stopPropagation();
        this.onSnap(item.key);
      });
      this.el.appendChild(g);
    }
  }
}
