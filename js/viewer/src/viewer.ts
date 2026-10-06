import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { Axes } from "./axes";
import { type Box, lerpBox, union, visibleBox } from "./bounds";
import { f32, i32 } from "./buffers";
import { categoryColor, COLORMAPS, cssToRgb, DEFAULT_COLORMAP, lut, type RGB, rgbToHex } from "./color";
import { colorBarSvg } from "./colorbar";
import { Gizmo, type Snap } from "./gizmo";
import { hostIsDark, watchHost } from "./host";
import { icon, iconButton, type IconName, setIcon } from "./icons";
import { represent, representations, type Representation } from "./layers/index";
import {
  adapt,
  FAST_FRACTION,
  motionFraction,
  motionPixelRatio,
  type Quality,
  Settle,
  startFraction,
} from "./motion";
import { columnPaint, solidPaint, type Scale } from "./paint";
import { type Renderer, webgl } from "./renderer";
import { shared } from "./shaders";
import { CSS } from "./styles";
import { resolveTheme, type Theme, type ThemeSpec, themeBase, withBase } from "./theme";
import { formatRange } from "./ticks";
import type { Buffers, ColumnSpec, Kind, LayerSpec, SceneSpec } from "./types";

const KINDS: [Kind, string][] = [
  ["drillholes", "Drill holes"],
  ["points", "Points"],
  ["mesh", "Meshes"],
  ["blocks", "Block models"],
];
const REFIT_MS = 320;
/** Frames further apart than this belong to different movements and are not timed. */
const GAP_MS = 250;
const QUALITIES: [Quality, string, string][] = [
  ["auto", "Auto", "Draw fewer instances while moving, as many as the frame rate allows"],
  ["fast", "Fast", "Draw a fixed share of the instances while moving"],
  ["full", "Full", "Draw everything while moving"],
];
const capitalize = (s: string) => s[0].toUpperCase() + s.slice(1);

interface Layer {
  spec: LayerSpec;
  rep: Representation;
  visible: boolean;
  column: ColumnSpec | null;
  color: RGB | null;
  opacity: number;
  open: boolean;
}

interface Variable {
  cmap: string;
  clim: [number, number] | null;
  label: string | null;
}

export interface MountOptions {
  /** Fill the host's height instead of the scene's `height`. */
  fill?: boolean;
}

export class Viewer {
  readonly root = document.createElement("div");
  private renderer: Renderer;
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(30, 1, 0.1, 1e7);
  private controls: OrbitControls;
  private axes: Axes;
  private gizmo: Gizmo;
  private layers: Layer[];
  private variables = new Map<string, Variable>();
  private data = new Map<string, Float32Array | Int32Array>();
  private themeSpec: ThemeSpec;
  private theme!: Theme;
  private width = 1;
  private height = 1;
  private pending = false;
  private box: Box | null = null;
  private refit: { from: Box; to: Box; start: number } | null = null;
  private show = { box: true, grid: true, ticks: true, gizmo: true };
  private collapsed = new Set<string>();
  private shot = { scale: 2, panel: false };
  private panel = document.createElement("div");
  private body = document.createElement("div");
  private reopen: HTMLButtonElement;
  private themeButton: HTMLButtonElement;
  private shotMenu = document.createElement("div");
  private bars = document.createElement("div");
  private barEls = new Map<string, HTMLDivElement>();
  private uid = Math.random().toString(36).slice(2, 8);
  private cleanup: (() => void)[] = [];
  private quality: Quality;
  private fastFraction: number;
  private autoFraction: number | null = null;
  private settle = new Settle(() => this.applyDetail());
  private lastMove = 0;
  private pixelRatio = window.devicePixelRatio || 1;

  constructor(host: HTMLElement, private spec: SceneSpec, private buffers: Buffers, options: MountOptions = {}) {
    const shadow = host.shadowRoot ?? host.attachShadow({ mode: "open" });
    shadow.replaceChildren();
    const style = document.createElement("style");
    style.textContent = CSS;
    shadow.append(style, this.root);
    this.root.className = "btv-root";
    this.root.tabIndex = 0;
    this.root.style.height = options.fill ? "100%" : `${spec.height}px`;
    if (options.fill) host.style.height = "100%";

    this.renderer = webgl();
    this.root.appendChild(this.renderer.canvas);
    this.camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, this.renderer.canvas);
    this.controls.zoomToCursor = true;
    this.controls.addEventListener("change", () => {
      if (this.quality !== "full") this.settle.poke();
      this.requestRender();
    });
    const motion = spec.motion ?? "auto";
    this.quality = typeof motion === "number" ? "fast" : motion;
    this.fastFraction = typeof motion === "number" ? motion : FAST_FRACTION;
    this.cleanup.push(() => this.settle.stop());

    this.axes = new Axes(spec.origin, spec.axes);
    this.scene.add(this.axes.group);
    this.gizmo = new Gizmo((snap) => this.snap(snap));
    this.themeSpec = spec.theme;

    this.layers = spec.layers.map((layer) => {
      const rep = represent(layer, buffers);
      rep.object.visible = layer.visible;
      this.scene.add(rep.object);
      return {
        spec: { ...layer },
        rep,
        visible: layer.visible,
        column: layer.columns.find((c) => c.name === layer.values) ?? null,
        color: layer.color ? cssToRgb(layer.color) : null,
        opacity: layer.opacity,
        open: false,
      };
    });
    for (const [name, v] of Object.entries(spec.variables)) {
      this.variables.set(name, { cmap: v.cmap ?? DEFAULT_COLORMAP, clim: v.clim ?? null, label: v.label ?? null });
    }

    this.bars.className = "btv-bars";
    this.root.append(this.axes.overlay, this.bars, this.gizmo.el);
    this.reopen = iconButton("layers", "Show panel (h)", () => this.togglePanel());
    this.reopen.classList.add("btv-open");
    this.reopen.hidden = true;
    this.themeButton = iconButton("moon", "Switch theme (t)", () => this.toggleTheme());
    this.buildPanel();
    this.root.append(this.panel, this.reopen);

    this.applyTheme(true);
    this.box = visibleBox(this.layers.map((l) => ({ visible: l.visible, box: l.rep.box })));
    this.axes.setBox(this.box);
    this.setView(spec.view.azimuth, spec.view.dip);
    this.fit();

    const resize = new ResizeObserver(() => this.resize());
    resize.observe(this.root);
    this.cleanup.push(() => resize.disconnect());
    this.cleanup.push(watchHost(() => themeBase(this.themeSpec) === "auto" && this.applyTheme()));
    this.root.addEventListener("keydown", (e) => this.key(e));
    this.renderer.canvas.addEventListener("pointerdown", () => this.root.focus({ preventScroll: true }));
    this.resize();
  }

  dispose(): void {
    this.cleanup.forEach((f) => f());
    this.controls.dispose();
    this.layers.forEach((l) => l.rep.dispose());
    this.renderer.dispose();
  }

  requestRender(): void {
    if (this.pending) return;
    this.pending = true;
    requestAnimationFrame(() => this.frame());
  }

  private frame(): void {
    this.pending = false;
    if (this.refit) {
      const t = Math.min(1, (performance.now() - this.refit.start) / REFIT_MS);
      const eased = 1 - (1 - t) ** 3;
      this.axes.setBox(lerpBox(this.refit.from, this.refit.to, eased));
      if (t < 1) this.requestRender();
      else this.refit = null;
    }
    if (this.settle.moving && this.quality === "auto" && this.autoFraction !== null) {
      const now = performance.now();
      if (this.lastMove && now - this.lastMove < GAP_MS) {
        const next = adapt(this.autoFraction, now - this.lastMove);
        if (Math.abs(next - this.autoFraction) > 0.01 * this.autoFraction) {
          this.autoFraction = next;
          this.applyDetail();
        }
      }
      this.lastMove = now;
    }
    const forward = this.camera.getWorldDirection(new THREE.Vector3());
    shared.uLight.value.copy(forward.negate().add(new THREE.Vector3(0, 0, 0.3))).normalize();
    shared.uPixelRatio.value = this.pixelRatio;
    shared.uResolution.value.set(this.width, this.height);
    for (const layer of this.layers) layer.rep.frame?.(this.width, this.height);
    this.axes.update(this.camera, this.width, this.height);
    this.gizmo.update(this.camera);
    this.renderer.render(this.scene, this.camera);
  }

  private resize(): void {
    this.width = Math.max(1, this.root.clientWidth);
    this.height = Math.max(1, this.root.clientHeight);
    this.renderer.setSize(this.width, this.height, this.pixelRatio);
    this.camera.aspect = this.width / this.height;
    this.camera.updateProjectionMatrix();
    this.requestRender();
  }

  // ---- motion quality ----

  /** Share of each layer's instances drawn now: 1 at rest, the quality's fraction while the camera moves. */
  private fraction(): number {
    if (!this.settle.moving) return 1;
    if (this.quality === "auto" && this.autoFraction === null) {
      let instances = 0;
      for (const l of this.layers) if (l.visible) instances += l.rep.instances;
      this.autoFraction = startFraction(instances);
    }
    return motionFraction(this.quality, this.fastFraction, this.autoFraction ?? 1);
  }

  /** Draws each layer at the current fraction and sets the pixel ratio to match; called when motion starts or stops. */
  private applyDetail(): void {
    if (!this.settle.moving) this.lastMove = 0;
    const f = this.fraction();
    for (const layer of this.layers) layer.rep.detail?.(f);
    const dpr = window.devicePixelRatio || 1;
    const ratio = this.settle.moving ? motionPixelRatio(dpr, f) : dpr;
    if (ratio !== this.pixelRatio) {
      this.pixelRatio = ratio;
      this.renderer.setSize(this.width, this.height, ratio);
    }
    this.requestRender();
  }

  setQuality(quality: Quality): void {
    this.quality = quality;
    this.settle.stop();
    this.renderBody();
  }

  // ---- camera ----

  private extent(): Box | null {
    return this.box ?? union(this.layers.map((l) => l.rep.box));
  }

  /** Frames the shown layers, keeping the viewing direction. */
  fit(): void {
    const box = this.extent();
    if (!box) return;
    const center = new THREE.Vector3(...box.min).add(new THREE.Vector3(...box.max)).multiplyScalar(0.5);
    const radius = Math.max(new THREE.Vector3(...box.max).distanceTo(new THREE.Vector3(...box.min)) / 2, 1e-3);
    const direction = this.camera.getWorldDirection(new THREE.Vector3());
    const half = THREE.MathUtils.degToRad(this.camera.fov / 2);
    const fitHeight = radius / Math.sin(half);
    const distance = 1.08 * (this.camera.aspect < 1 ? fitHeight / this.camera.aspect : fitHeight);
    this.camera.near = radius / 500;
    this.camera.far = radius * 400;
    this.camera.updateProjectionMatrix();
    this.camera.position.copy(center).addScaledVector(direction, -distance);
    this.controls.target.copy(center);
    this.controls.update();
    this.requestRender();
  }

  /** Looks along `azimuth` (degrees clockwise from north) and `dip` (degrees below horizontal). */
  setView(azimuth: number, dip: number): void {
    const a = THREE.MathUtils.degToRad(azimuth);
    const d = THREE.MathUtils.degToRad(Math.max(-89.9, Math.min(89.9, dip)));
    const direction = new THREE.Vector3(Math.sin(a) * Math.cos(d), Math.cos(a) * Math.cos(d), -Math.sin(d));
    const box = this.extent();
    const target = box
      ? new THREE.Vector3(...box.min).add(new THREE.Vector3(...box.max)).multiplyScalar(0.5)
      : this.controls.target.clone();
    const distance = Math.max(this.camera.position.distanceTo(this.controls.target), 1);
    this.controls.target.copy(target);
    this.camera.position.copy(target).addScaledVector(direction, -distance);
    this.camera.lookAt(target);
    this.controls.update();
    this.requestRender();
  }

  private snap(snap: Snap): void {
    const [azimuth, dip] = { plan: [0, 90], east: [270, 0], north: [180, 0] }[snap];
    this.setView(azimuth, dip);
  }

  // ---- layers and color ----

  private columnData(layer: Layer, column: ColumnSpec): Float32Array | Int32Array {
    const key = `${layer.spec.id}/${column.name}`;
    let data = this.data.get(key);
    if (!data) {
      data = column.type === "text" ? i32(this.buffers, column.buffer) : f32(this.buffers, column.buffer);
      this.data.set(key, data);
    }
    return data;
  }

  private variable(name: string): Variable {
    let v = this.variables.get(name);
    if (!v) {
      v = { cmap: DEFAULT_COLORMAP, clim: null, label: null };
      this.variables.set(name, v);
    }
    return v;
  }

  /** Map, range and categories of a variable, shared by every layer it colors. */
  private scale(name: string): Scale {
    const v = this.variable(name);
    const colored = this.layers.map((l) => l.column).filter((c): c is ColumnSpec => c?.name === name);
    const categories = new Set<string>();
    for (const layer of this.layers)
      for (const c of layer.spec.columns) if (c.name === name && c.type === "text") c.categories?.forEach((x) => categories.add(x));
    let lo = Infinity;
    let hi = -Infinity;
    for (const c of colored)
      if (c.type === "number" && c.min !== undefined && c.max !== undefined) {
        lo = Math.min(lo, c.min);
        hi = Math.max(hi, c.max);
      }
    if (v.clim) [lo, hi] = v.clim;
    return { cmap: v.cmap, lo, hi, categories: [...categories].sort() };
  }

  private solid(layer: Layer): RGB {
    return layer.color ?? cssToRgb(this.theme.layer);
  }

  private paint(layer: Layer): void {
    const c = layer.column;
    layer.rep.paint(
      c
        ? columnPaint(c, this.columnData(layer, c), this.scale(c.name), this.solid(layer))
        : solidPaint(this.solid(layer)),
    );
    layer.rep.detail?.(this.fraction());
  }

  /** Redraws a layer in another representation, keeping its color, range, opacity and visibility. */
  setRepresentation(layer: Layer, name: string): void {
    if (name === layer.spec.representation || !representations(layer.spec.kind).includes(name)) return;
    this.scene.remove(layer.rep.object);
    layer.rep.dispose();
    layer.spec.representation = name;
    layer.rep = represent(layer.spec, this.buffers);
    layer.rep.object.visible = layer.visible;
    layer.rep.setOpacity(layer.opacity);
    this.scene.add(layer.rep.object);
    this.paint(layer);
    this.refitBox();
    this.renderBody();
    this.requestRender();
  }

  private refitBox(): void {
    const to = visibleBox(this.layers.map((l) => ({ visible: l.visible, box: l.rep.box })));
    if (this.box && to) this.refit = { from: this.box, to, start: performance.now() };
    else this.axes.setBox(to);
    this.box = to;
  }

  /** Repaints the layers colored by `names` (all when omitted) and redraws what depends on them. */
  private refresh(names?: Set<string>): void {
    for (const layer of this.layers) {
      if (!names || (layer.column && names.has(layer.column.name)) || (!layer.column && names.has(""))) this.paint(layer);
    }
    this.renderBody();
    this.updateBars();
    this.requestRender();
  }

  private setVisible(layer: Layer, visible: boolean): void {
    layer.visible = visible;
    layer.rep.object.visible = visible;
  }

  private toggleLayer(layer: Layer, solo: boolean): void {
    if (solo) {
      const alone = this.layers.every((l) => l.visible === (l === layer));
      for (const l of this.layers) this.setVisible(l, alone || l === layer);
    } else this.setVisible(layer, !layer.visible);
    this.refitBox();
    this.renderBody();
    this.updateBars();
    this.requestRender();
  }

  // ---- theme ----

  private applyTheme(all = false): void {
    this.theme = resolveTheme(this.themeSpec, hostIsDark());
    const t = this.theme;
    const vars: Record<string, string> = {
      bg: t.background, panel: t.panel, raised: t.raised, border: t.border, text: t.text, muted: t.muted, accent: t.accent,
    };
    for (const [k, v] of Object.entries(vars)) this.root.style.setProperty(`--btv-${k}`, v);
    this.root.style.colorScheme = t.dark ? "dark" : "light";
    this.renderer.setBackground(t.background);
    shared.uHalo.value.set(t.halo);
    this.axes.setTheme(t);
    this.gizmo.setTheme(t);
    setIcon(this.themeButton, t.dark ? "sun" : "moon");
    for (const layer of this.layers) if (all || (!layer.column && !layer.color)) this.paint(layer);
    this.renderBody();
    this.updateBars();
    this.requestRender();
  }

  toggleTheme(): void {
    this.themeSpec = withBase(this.themeSpec, this.theme.dark ? "light" : "dark");
    this.applyTheme();
  }

  // ---- panel ----

  private buildPanel(): void {
    this.panel.className = "btv-panel";
    const head = document.createElement("div");
    head.className = "btv-head";
    const brand = document.createElement("span");
    brand.className = "btv-brand";
    brand.textContent = "Scene";
    head.append(
      brand,
      this.themeButton,
      iconButton("fit", "Fit view (f)", () => this.fit()),
      iconButton("camera", "Screenshot", () => (this.shotMenu.hidden = !this.shotMenu.hidden)),
      iconButton("hide", "Hide panel (h)", () => this.togglePanel()),
    );
    this.buildShotMenu();
    this.body.className = "btv-body";
    this.panel.append(head, this.shotMenu, this.body);
  }

  private buildShotMenu(): void {
    this.shotMenu.className = "btv-shot";
    this.shotMenu.hidden = true;
    const seg = document.createElement("div");
    seg.className = "btv-seg";
    for (const scale of [1, 2, 4]) {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = `${scale}x`;
      b.classList.toggle("btv-on", scale === this.shot.scale);
      b.addEventListener("click", () => {
        this.shot.scale = scale;
        seg.querySelectorAll("button").forEach((x) => x.classList.toggle("btv-on", x === b));
      });
      seg.appendChild(b);
    }
    const label = document.createElement("label");
    const check = document.createElement("input");
    check.type = "checkbox";
    check.checked = this.shot.panel;
    check.addEventListener("change", () => (this.shot.panel = check.checked));
    label.append(check, "panel");
    this.shotMenu.append(seg, label, iconButton("download", "Download PNG", () => void this.download()));
  }

  togglePanel(): void {
    this.panel.hidden = !this.panel.hidden;
    this.reopen.hidden = !this.panel.hidden;
  }

  private section(title: string, key: string, count?: number): HTMLButtonElement {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "btv-section";
    b.append(icon(this.collapsed.has(key) ? "chevronRight" : "chevronDown"), title);
    if (count !== undefined) {
      const n = document.createElement("span");
      n.className = "btv-count";
      n.textContent = String(count);
      b.appendChild(n);
    }
    b.addEventListener("click", () => {
      if (!this.collapsed.delete(key)) this.collapsed.add(key);
      this.renderBody();
    });
    return b;
  }

  private renderBody(): void {
    const scroll = this.body.scrollTop;
    this.body.replaceChildren();
    for (const [kind, title] of KINDS) {
      const layers = this.layers.filter((l) => l.spec.kind === kind);
      if (!layers.length) continue;
      this.body.appendChild(this.section(title, kind, layers.length));
      if (this.collapsed.has(kind)) continue;
      const group = document.createElement("div");
      group.className = "btv-group";
      for (const layer of layers) group.append(...this.layerRow(layer));
      this.body.appendChild(group);
    }
    this.body.appendChild(this.section("View", "view"));
    if (!this.collapsed.has("view")) {
      const box = document.createElement("div");
      box.className = "btv-settings btv-view";
      const tools = document.createElement("div");
      tools.className = "btv-tools";
      const toggles: [keyof Viewer["show"], IconName, string][] = [
        ["box", "box", "Bounding box"],
        ["grid", "grid", "Grid"],
        ["ticks", "ruler", "Tick labels"],
        ["gizmo", "axes", "Orientation"],
      ];
      for (const [key, name, text] of toggles) {
        const button = iconButton(name, text, () => {
          this.show[key] = !this.show[key];
          button.setAttribute("aria-pressed", String(this.show[key]));
          this.axes.show = { box: this.show.box, grid: this.show.grid, ticks: this.show.ticks };
          this.gizmo.el.style.display = this.show.gizmo ? "" : "none";
          this.requestRender();
        });
        button.classList.add("btv-tool");
        button.setAttribute("aria-pressed", String(this.show[key]));
        tools.appendChild(button);
      }
      const quality = segmented(
        QUALITIES.map(([value, text, title]) => ({ value, text, title })),
        this.quality,
        (value) => this.setQuality(value as Quality),
      );
      quality.setAttribute("aria-label", "Quality while moving");
      box.append(label("Show"), tools, label("Quality"), quality);
      this.body.appendChild(box);
    }
    this.body.scrollTop = scroll;
  }

  private key_(layer: Layer): HTMLElement {
    const key = document.createElement("span");
    key.className = "btv-key";
    const c = layer.column;
    if (!c) {
      key.classList.add("btv-swatch");
      key.style.background = rgbToHex(this.solid(layer));
    } else if (c.type === "text") {
      const n = Math.max(1, Math.min(5, this.scale(c.name).categories.length));
      key.style.background = `linear-gradient(to right, ${Array.from({ length: n }, (_, i) => {
        const hex = rgbToHex(categoryColor(i));
        return `${hex} ${(i / n) * 100}% ${((i + 1) / n) * 100}%`;
      }).join(", ")})`;
    } else {
      const table = lut(this.scale(c.name).cmap);
      const stops = [0, 64, 128, 192, 255].map((i) => rgbToHex([table[3 * i], table[3 * i + 1], table[3 * i + 2]]));
      key.style.background = `linear-gradient(to right, ${stops.join(", ")})`;
    }
    key.title = c ? c.name : "Solid color";
    return key;
  }

  private layerRow(layer: Layer): HTMLElement[] {
    const row = document.createElement("div");
    row.className = "btv-row";
    row.classList.toggle("btv-off", !layer.visible);
    const eye = iconButton(layer.visible ? "eye" : "eyeOff", "Show or hide (Alt-click: only this one)", (e) =>
      this.toggleLayer(layer, e.altKey),
    );
    const name = document.createElement("span");
    name.className = "btv-name";
    name.textContent = layer.spec.name;
    name.title = layer.spec.name;
    const more = iconButton("ellipsis", "Settings", () => {
      layer.open = !layer.open;
      this.renderBody();
    });
    more.classList.toggle("btv-on", layer.open);
    row.append(eye, name, this.key_(layer), more);
    return layer.open ? [row, this.settings(layer)] : [row];
  }

  private settings(layer: Layer): HTMLElement {
    const box = document.createElement("div");
    box.className = "btv-settings";
    const field = (text: string, control: HTMLElement) => box.append(label(text), control);

    const names = representations(layer.spec.kind);
    if (names.length > 1) {
      const show = segmented(
        names.map((name) => ({ value: name, text: capitalize(name), title: `Draw as ${name}` })),
        layer.spec.representation,
        (name) => this.setRepresentation(layer, name),
      );
      show.setAttribute("aria-label", "Representation");
      field("Show as", show);
    }

    const by = document.createElement("select");
    by.add(new Option("Solid color", ""));
    for (const c of layer.spec.columns) {
      const option = new Option(c.name, c.name, false, c === layer.column);
      option.title = c.name;
      by.add(option);
    }
    by.value = layer.column?.name ?? "";
    by.title = layer.column?.name ?? "Solid color";
    by.addEventListener("change", () => {
      const before = layer.column?.name;
      layer.column = layer.spec.columns.find((c) => c.name === by.value) ?? null;
      this.refresh(new Set([before ?? "", layer.column?.name ?? ""]));
    });
    field("Color by", by);

    const c = layer.column;
    if (!c) {
      const pick = document.createElement("input");
      pick.type = "color";
      pick.value = rgbToHex(this.solid(layer));
      pick.addEventListener("input", () => {
        layer.color = cssToRgb(pick.value);
        this.paint(layer);
        this.requestRender();
      });
      pick.addEventListener("change", () => this.renderBody());
      field("Color", pick);
    } else if (c.type === "number") {
      const v = this.variable(c.name);
      const cmap = document.createElement("select");
      for (const name of COLORMAPS) cmap.add(new Option(name, name, false, name === v.cmap));
      cmap.addEventListener("change", () => {
        v.cmap = cmap.value;
        this.refresh(new Set([c.name]));
      });
      field("Colormap", cmap);
      const s = this.scale(c.name);
      const range = document.createElement("div");
      range.className = "btv-range";
      const input = (text: string) => {
        const el = document.createElement("input");
        el.type = "number";
        el.step = "any";
        el.value = text;
        el.title = text;
        return el;
      };
      const [loText, hiText] = formatRange(s.lo, s.hi);
      const lo = input(loText);
      const hi = input(hiText);
      const apply = () => {
        const a = Number(lo.value);
        const b = Number(hi.value);
        if (Number.isFinite(a) && Number.isFinite(b) && b > a) v.clim = [a, b];
        this.refresh(new Set([c.name]));
      };
      lo.addEventListener("change", apply);
      hi.addEventListener("change", apply);
      const reset = iconButton("reset", "Reset range", () => {
        v.clim = this.spec.variables[c.name]?.clim ?? null;
        this.refresh(new Set([c.name]));
      });
      range.append(lo, hi, reset);
      field("Range", range);
    }

    const opacity = document.createElement("input");
    opacity.type = "range";
    opacity.min = "0";
    opacity.max = "1";
    opacity.step = "0.05";
    opacity.value = String(layer.opacity);
    opacity.addEventListener("input", () => {
      layer.opacity = Number(opacity.value);
      layer.rep.setOpacity(layer.opacity);
      this.requestRender();
    });
    field("Opacity", opacity);
    return box;
  }

  // ---- color bars ----

  private updateBars(): void {
    const names = [...new Set(this.layers.filter((l) => l.visible && l.column).map((l) => l.column!.name))];
    for (const [name, el] of this.barEls)
      if (!names.includes(name)) {
        el.remove();
        this.barEls.delete(name);
      }
    names.forEach((name, i) => {
      let el = this.barEls.get(name);
      if (!el) {
        el = document.createElement("div");
        el.className = "btv-bar";
        this.draggable(el);
        this.bars.appendChild(el);
        this.barEls.set(name, el);
      }
      const s = this.scale(name);
      const text = this.layers.some((l) => l.column?.name === name && l.column.type === "text");
      el.innerHTML = colorBarSvg(
        { title: this.variable(name).label ?? name, cmap: s.cmap, lo: s.lo, hi: s.hi, categories: text ? s.categories : null },
        this.theme,
        `btv-${this.uid}-${i}`,
      );
    });
  }

  private draggable(el: HTMLDivElement): void {
    el.addEventListener("pointerdown", (e) => {
      e.preventDefault();
      el.setPointerCapture(e.pointerId);
      const root = this.root.getBoundingClientRect();
      const r = el.getBoundingClientRect();
      const dx = e.clientX - r.left;
      const dy = e.clientY - r.top;
      if (!el.classList.contains("btv-moved")) {
        el.classList.add("btv-moved");
        this.root.appendChild(el);
        el.style.left = `${r.left - root.left}px`;
        el.style.top = `${r.top - root.top}px`;
      }
      const move = (m: PointerEvent) => {
        el.style.left = `${Math.max(0, Math.min(root.width - r.width, m.clientX - root.left - dx))}px`;
        el.style.top = `${Math.max(0, Math.min(root.height - r.height, m.clientY - root.top - dy))}px`;
      };
      const up = () => {
        el.removeEventListener("pointermove", move);
        el.removeEventListener("pointerup", up);
      };
      el.addEventListener("pointermove", move);
      el.addEventListener("pointerup", up);
    });
  }

  // ---- keys and screenshots ----

  private key(e: KeyboardEvent): void {
    const target = e.composedPath()[0] as HTMLElement;
    if (e.ctrlKey || e.metaKey || e.altKey || /^(INPUT|SELECT|TEXTAREA)$/.test(target?.tagName ?? "")) return;
    const action = { f: () => this.fit(), h: () => this.togglePanel(), t: () => this.toggleTheme() }[e.key.toLowerCase()];
    if (!action) return;
    e.preventDefault();
    e.stopPropagation();
    action();
  }

  /** The current view as a PNG blob at `scale`, with color bars, labels and gizmo, and the panel if asked. */
  async screenshot(scale: number, panel: boolean): Promise<Blob> {
    this.settle.stop();
    this.frame();
    shared.uPixelRatio.value = scale;
    const canvas = this.renderer.snapshot(this.scene, this.camera, scale);
    shared.uPixelRatio.value = this.pixelRatio;
    this.requestRender();
    const ratio = canvas.width / this.width;
    const ctx = canvas.getContext("2d")!;
    if (this.show.ticks) this.axes.draw(ctx, ratio, this.theme);
    const root = this.root.getBoundingClientRect();
    const place = (el: Element) => {
      const r = el.getBoundingClientRect();
      return [(r.left - root.left) * ratio, (r.top - root.top) * ratio, r.width * ratio, r.height * ratio] as const;
    };
    const overlays: Element[] = [...this.barEls.values()].map((el) => el.firstElementChild!).filter(Boolean);
    if (this.show.gizmo) overlays.push(this.gizmo.el);
    for (const svg of overlays) {
      const image = await svgImage(new XMLSerializer().serializeToString(svg));
      ctx.drawImage(image, ...place(svg));
    }
    if (panel && !this.panel.hidden) {
      try {
        const copy = document.createElement("canvas");
        copy.width = canvas.width;
        copy.height = canvas.height;
        const c2 = copy.getContext("2d")!;
        c2.drawImage(canvas, 0, 0);
        c2.drawImage(await svgImage(this.panelSvg()), ...place(this.panel));
        copy.toDataURL();
        return await toBlob(copy);
      } catch {
        // the browser taints canvases that draw HTML through SVG; keep the shot without the panel
      }
    }
    return toBlob(canvas);
  }

  private panelSvg(): string {
    const r = this.panel.getBoundingClientRect();
    const vars = this.root.getAttribute("style") ?? "";
    const copy = this.panel.cloneNode(true) as HTMLElement;
    const live = this.panel.querySelectorAll<HTMLInputElement | HTMLSelectElement>("input, select");
    copy.querySelectorAll<HTMLInputElement | HTMLSelectElement>("input, select").forEach((el, i) => {
      const from = live[i];
      if (from instanceof HTMLSelectElement) el.querySelectorAll("option")[from.selectedIndex]?.setAttribute("selected", "");
      else if (from.type === "checkbox") from.checked ? el.setAttribute("checked", "") : el.removeAttribute("checked");
      else el.setAttribute("value", from.value);
    });
    const html = new XMLSerializer().serializeToString(copy);
    return (
      `<svg xmlns="http://www.w3.org/2000/svg" width="${r.width}" height="${r.height}">` +
      `<foreignObject width="100%" height="100%"><div xmlns="http://www.w3.org/1999/xhtml" class="btv-root" style="${vars.replace(/"/g, "'")};height:100%;background:transparent">` +
      `<style>${CSS.replace(/</g, "&lt;")} .btv-panel{position:static;max-height:none;height:100%}</style>${html}</div></foreignObject></svg>`
    );
  }

  private async download(): Promise<void> {
    const blob = await this.screenshot(this.shot.scale, this.shot.panel);
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "scene.png";
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }
}

function label(text: string): HTMLLabelElement {
  const el = document.createElement("label");
  el.textContent = text;
  return el;
}

/** One-of-several buttons; the pressed one carries `aria-pressed`. */
function segmented(
  options: { value: string; text: string; title: string }[],
  value: string,
  onChange: (value: string) => void,
): HTMLDivElement {
  const seg = document.createElement("div");
  seg.className = "btv-seg";
  seg.setAttribute("role", "group");
  for (const option of options) {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = option.text;
    b.title = option.title;
    b.classList.toggle("btv-on", option.value === value);
    b.setAttribute("aria-pressed", String(option.value === value));
    b.addEventListener("click", () => onChange(option.value));
    seg.appendChild(b);
  }
  return seg;
}

function svgImage(svg: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = reject;
    image.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(svg);
  });
}

function toBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("empty canvas"))), "image/png"));
}
