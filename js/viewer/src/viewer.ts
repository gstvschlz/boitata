import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { LineMaterial } from "three/examples/jsm/lines/LineMaterial.js";
import { LineSegments2 } from "three/examples/jsm/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/examples/jsm/lines/LineSegmentsGeometry.js";
import { Axes } from "./axes";
import { blocksBox, type Box, diagonal, lerpBox, maskedBox, triangleMask, union, visibleBox, type Vec3 } from "./bounds";
import { f32, i32, u32 } from "./buffers";
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
import {
  barHeights,
  categoryCounts,
  type Condition,
  count,
  type FilterState,
  formatBound,
  fromState,
  histogram,
  LayerFilter,
  openCondition,
  passMask,
  SLOTS,
  sliderStep,
  snap,
  toState,
} from "./filter";
import { columnPaint, type Paint, solidPaint, type Scale } from "./paint";
import { type Action, keyAction, SHORTCUTS } from "./keys";
import { formatValue, type Hit, isClick, nearestHit, PICK_RADIUS, round7 } from "./pick";
import { type Renderer, webgl } from "./renderer";
import {
  buildSection,
  distinct,
  MAX_SEGMENTS,
  pickCut,
  type Ray,
  type Section,
  sectionBox,
  type SectionState,
  snap45,
  traces,
  unfold,
  writeSection,
} from "./section";
import { HALO_ORDER, shared } from "./shaders";
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
/** Default section width, as a share of the diagonal of the shown layers' bounds. */
const WIDTH_SHARE = 1 / 20;
/** A pointer moving less than this many pixels between press and release clicks. */
const CLICK_PX = 5;
const WHEEL_STEP = 1.15;

type XY = [number, number];
interface CameraState {
  position: THREE.Vector3;
  target: THREE.Vector3;
}

/** A section in local coordinates; its width and dip are the viewer's. */
interface Cut {
  points: XY[];
  z: number;
  unfolded: boolean;
}

interface Layer {
  spec: LayerSpec;
  rep: Representation;
  visible: boolean;
  column: ColumnSpec | null;
  color: RGB | null;
  opacity: number;
  open: boolean;
  filter: LayerFilter;
  painted: Paint | null;
  /** Bounds of what it shows: rows that are neither null in its color column nor filtered out. */
  box: Box | null;
  counts: { shown: number; total: number; nulls: number };
  readout: HTMLElement | null;
  picking: boolean;
}

interface Variable {
  cmap: string;
  clim: [number, number] | null;
  label: string | null;
}

/** A picked element as Python reads it: real-world position, values by column, null for nulls. */
export interface PickState {
  layer: string;
  row: number;
  values: Record<string, number | string | null>;
  position: [number, number, number];
}

export interface MountOptions {
  /** Fill the host's height instead of the scene's `height`. */
  fill?: boolean;
  /** Filters by layer id, overriding the layers' own; from the widget's synced state. */
  filters?: Record<string, FilterState>;
  /** Called after the user edits a filter, with every layer's. */
  onFilters?: (filters: Record<string, FilterState>) => void;
  /** Section overriding the scene's own, empty for none; from the widget's synced state. */
  section?: SectionState | Record<string, never>;
  /** Called after the user cuts, edits, unfolds or clears the section; empty when cleared. */
  onSection?: (section: SectionState | Record<string, never>) => void;
  /** Called after a click picks an element, or dismisses the pick with an empty object. */
  onPick?: (picked: PickState | Record<string, never>) => void;
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

  private histograms = new Map<string, number[]>();
  private measureTimer: ReturnType<typeof setTimeout> | undefined;
  private dirty = new Set<Layer>();

  private cut: Cut | null = null;
  private cutSection: Section | null = null;
  private cutWidth: number | null = null;
  private cutDip = 90;
  private drawing: { points: XY[]; cursor: XY | null; shift: boolean; view: CameraState } | null = null;
  private quick: { from: XY; to: XY } | null = null;
  private press: XY | null = null;
  private folded: CameraState | null = null;
  private tween: { from: CameraState; to: CameraState; start: number } | null = null;
  private outline = new THREE.Group();
  private outlineLines = new LineMaterial({ linewidth: 2, depthTest: false, transparent: true });
  private outlineEdges = new LineMaterial({ linewidth: 1, depthTest: false, transparent: true, opacity: 0.75 });
  private outlineDots = new THREE.PointsMaterial({ size: 7, sizeAttenuation: false, depthTest: false, transparent: true });
  private sweep = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  private sectionControls: { width: HTMLInputElement[]; dip: HTMLInputElement[]; hint: HTMLElement | null } = {
    width: [],
    dip: [],
    hint: null,
  };
  private emitTimer: ReturnType<typeof setTimeout> | undefined;

  private surfaceBias = 0;
  private clickAt: XY | null = null;
  private picked: PickState | null = null;
  private card = document.createElement("div");
  private help = document.createElement("div");
  private fullButton: HTMLButtonElement;
  private helpButton = iconButton("keyboard", "Keyboard shortcuts (?)", () => this.toggleHelp());
  private highlight = new THREE.Group();
  private highlightLines = new LineMaterial({
    linewidth: 2.5,
    polygonOffset: true,
    polygonOffsetFactor: -2,
    polygonOffsetUnits: -8,
  });
  private highlightRim = new LineMaterial({ linewidth: 8, transparent: true, depthWrite: false });
  private highlightOver = new LineMaterial({ linewidth: 2.5, depthTest: false, transparent: true });
  private highlightRing = ringMaterial();

  constructor(
    host: HTMLElement,
    private spec: SceneSpec,
    private buffers: Buffers,
    private options: MountOptions = {},
  ) {
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

    this.layers = spec.layers.map((layer, index) => {
      const filter = new LayerFilter();
      filter.uniforms.uLayer.value = index;
      const l: Layer = {
        spec: { ...layer },
        rep: null as unknown as Representation,
        visible: layer.visible,
        column: layer.columns.find((c) => c.name === layer.values) ?? null,
        color: layer.color ? cssToRgb(layer.color) : null,
        opacity: layer.opacity,
        open: false,
        filter,
        painted: null,
        box: null,
        counts: { shown: 0, total: 0, nulls: 0 },
        readout: null,
        picking: false,
      };
      this.setSlots(l, fromState(options.filters ? options.filters[layer.id] : layer.filter, layer.columns));
      l.rep = represent(layer, buffers, filter);
      l.rep.object.visible = layer.visible;
      this.scene.add(l.rep.object);
      return l;
    });
    this.surfaceBias = surfaceBias(spec.layers, buffers);
    for (const [name, v] of Object.entries(spec.variables)) {
      this.variables.set(name, { cmap: v.cmap ?? DEFAULT_COLORMAP, clim: v.clim ?? null, label: v.label ?? null });
    }

    this.outline.renderOrder = 10;
    this.scene.add(this.outline, this.highlight);
    this.sweep.setAttribute("class", "btv-sweep");
    this.bars.className = "btv-bars";
    this.card.className = "btv-card";
    this.card.hidden = true;
    this.help.className = "btv-card btv-help";
    this.help.hidden = true;
    this.root.append(this.axes.overlay, this.sweep, this.bars, this.gizmo.el, this.card);
    this.reopen = iconButton("layers", "Show panel (H)", () => this.togglePanel());
    this.reopen.classList.add("btv-open");
    this.reopen.hidden = true;
    this.themeButton = iconButton("moon", "Switch theme (T)", () => this.toggleTheme());
    this.fullButton = iconButton("maximize", "Fullscreen", () => void this.toggleFullscreen());
    if (!document.fullscreenEnabled) this.fullscreenOff();
    const fullscreen = () => this.fullscreenChanged();
    document.addEventListener("fullscreenchange", fullscreen);
    this.cleanup.push(() => document.removeEventListener("fullscreenchange", fullscreen));
    this.buildPanel();
    this.root.append(this.panel, this.reopen, this.help);
    this.root.addEventListener(
      "pointerdown",
      (e) => {
        const path = e.composedPath();
        if (!this.help.hidden && !path.includes(this.help) && !path.includes(this.helpButton)) this.toggleHelp(false);
      },
      { capture: true },
    );

    this.applyTheme(true);
    this.box = visibleBox(this.layers);
    this.axes.setBox(this.box);
    this.setView(spec.view.azimuth, spec.view.dip);
    this.fit();
    const section = options.section !== undefined ? options.section : spec.section;
    if (section && "points" in section) {
      this.setSection(section);
      this.refit = null;
      this.axes.setBox(this.box);
      if (!this.cut?.unfolded) this.fit();
    }

    const resize = new ResizeObserver(() => this.resize());
    resize.observe(this.root);
    this.cleanup.push(() => resize.disconnect());
    this.cleanup.push(watchHost(() => themeBase(this.themeSpec) === "auto" && this.applyTheme()));
    this.root.addEventListener("keydown", (e) => this.key(e));
    this.root.addEventListener("keyup", (e) => e.key === "Shift" && this.drawing && this.setShift(false));
    const canvas = this.renderer.canvas;
    canvas.addEventListener("pointerdown", (e) => this.pointerDown(e), { capture: true });
    canvas.addEventListener("pointermove", (e) => this.pointerMove(e));
    canvas.addEventListener("pointerup", (e) => this.pointerUp(e));
    canvas.addEventListener("wheel", (e) => this.wheel(e), { capture: true, passive: false });
    this.resize();
  }

  dispose(): void {
    this.cleanup.forEach((f) => f());
    clearTimeout(this.measureTimer);
    clearTimeout(this.emitTimer);
    this.clearOutline();
    this.clearHighlight();
    for (const m of [this.outlineLines, this.outlineEdges, this.outlineDots, this.highlightLines, this.highlightRim, this.highlightOver, this.highlightRing])
      m.dispose();
    this.controls.dispose();
    this.layers.forEach((l) => {
      l.rep.dispose();
      l.filter.dispose();
    });
    this.renderer.dispose();
  }

  requestRender(): void {
    if (this.pending) return;
    this.pending = true;
    requestAnimationFrame(() => this.frame());
  }

  private frame(): void {
    this.pending = false;
    if (this.tween) {
      const t = Math.min(1, (performance.now() - this.tween.start) / REFIT_MS);
      const eased = 1 - (1 - t) ** 3;
      const { from, to } = this.tween;
      this.camera.position.lerpVectors(from.position, to.position, eased);
      this.controls.target.lerpVectors(from.target, to.target, eased);
      this.camera.lookAt(this.controls.target);
      if (t < 1) this.requestRender();
      else {
        this.tween = null;
        this.controls.update();
      }
    }
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
    this.shareUniforms();
    this.axes.update(this.camera, this.width, this.height);
    this.gizmo.update(this.camera);
    this.renderer.render(this.scene, this.camera);
  }

  /** Writes this viewer's state into the uniforms its materials share with any other viewer on the page. */
  private shareUniforms(): void {
    writeSection(this.drawing ? null : this.cutSection, !!this.cut?.unfolded);
    shared.uAccent.value.set(this.theme.accent);
    for (const m of [this.outlineLines, this.outlineEdges, this.highlightLines, this.highlightRim, this.highlightOver])
      m.resolution.set(this.width, this.height);
    const forward = this.camera.getWorldDirection(new THREE.Vector3());
    shared.uLight.value.copy(forward.negate().add(new THREE.Vector3(0, 0, 0.3))).normalize();
    shared.uPixelRatio.value = this.pixelRatio;
    shared.uResolution.value.set(this.width, this.height);
    shared.uSurfaceBias.value = this.surfaceBias;
    for (const layer of this.layers) layer.rep.frame?.(this.width, this.height);
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
    layer.painted = c
      ? columnPaint(c, this.columnData(layer, c), this.scale(c.name), this.solid(layer))
      : solidPaint(this.solid(layer));
    layer.rep.paint(layer.painted);
    layer.rep.detail?.(this.fraction());
    this.measure(layer);
  }

  /** Redraws a layer in another representation, keeping its color, range, opacity, filter and visibility. */
  setRepresentation(layer: Layer, name: string): void {
    if (name === layer.spec.representation || !representations(layer.spec.kind).includes(name)) return;
    this.dismiss();
    this.scene.remove(layer.rep.object);
    layer.rep.dispose();
    layer.spec.representation = name;
    layer.rep = represent(layer.spec, this.buffers, layer.filter);
    layer.rep.object.visible = layer.visible;
    layer.rep.setOpacity(layer.opacity);
    this.scene.add(layer.rep.object);
    this.paint(layer);
    this.refitBox();
    this.renderBody();
    this.requestRender();
  }

  // ---- filter ----

  private setSlots(layer: Layer, conditions: Condition[]): void {
    const f = layer.filter;
    f.conditions = conditions.slice(0, SLOTS);
    f.slots = f.conditions.map((c) => ({ on: c.column.on, data: this.columnData(layer, c.column), text: c.column.type === "text" }));
    f.update();
  }

  /** Sets a layer's conditions: their columns go into its buffers, so it repaints. */
  setConditions(layer: Layer, conditions: Condition[], emit = true): void {
    this.dismiss();
    this.setSlots(layer, conditions);
    this.paint(layer);
    this.refitBox();
    this.renderBody();
    this.requestRender();
    if (emit) this.emitFilters();
  }

  /** A bound or category changed: new uniforms now, counts and bounds once the edits pause. */
  private filterEdited(layer: Layer): void {
    this.dismiss();
    layer.filter.update();
    this.requestRender();
    this.dirty.add(layer);
    clearTimeout(this.measureTimer);
    this.measureTimer = setTimeout(() => {
      for (const l of this.dirty) {
        this.measure(l);
        this.updateReadout(l);
      }
      this.dirty.clear();
      this.refitBox();
      this.requestRender();
      this.emitFilters();
    }, 120);
  }

  /** Every layer's filter, by layer id, as Python reads it. */
  filters(): Record<string, FilterState> {
    const out: Record<string, FilterState> = {};
    for (const l of this.layers) if (l.filter.active) out[l.spec.id] = toState(l.filter.conditions);
    return out;
  }

  /** Applies filters from Python; layers left out lose theirs. */
  setFilters(filters: Record<string, FilterState>): void {
    for (const layer of this.layers) {
      const conditions = fromState(filters?.[layer.spec.id], layer.spec.columns);
      if (JSON.stringify(toState(conditions)) !== JSON.stringify(toState(layer.filter.conditions)))
        this.setConditions(layer, conditions, false);
    }
  }

  private emitFilters(): void {
    this.options.onFilters?.(this.filters());
  }

  /** Counts what a layer shows and the bounds of it: one pass over its rows, after a paint or a filter edit. */
  private measure(layer: Layer): void {
    const { kind, geometry } = layer.spec;
    const f = layer.filter;
    const paint = layer.painted;
    const keep = (on: ColumnSpec["on"]) => (paint?.on === on ? paint.keep : null);
    const sum = (mask: Uint8Array) => mask.reduce((s, v) => s + v, 0);
    if (kind === "mesh") {
      const positions = f32(this.buffers, geometry.positions);
      const triangles = u32(this.buffers, geometry.triangles);
      const nv = positions.length / 3;
      const nt = triangles.length / 3;
      const shown = triangleMask(
        triangles,
        passMask(f.compiled, f.slots, "face", nt, keep("face")),
        passMask(f.compiled, f.slots, "vertex", nv, keep("vertex")),
      );
      const colored = triangleMask(triangles, keep("face"), keep("vertex"));
      const vertices = new Uint8Array(nv);
      for (let t = 0; t < nt; t++)
        if (shown[t]) for (let k = 0; k < 3; k++) vertices[triangles[3 * t + k]] = 1;
      layer.counts = { shown: sum(shown), total: nt, nulls: nt - sum(colored) };
      layer.box = maskedBox(positions, vertices);
      return;
    }
    let n: number;
    let box: (mask: Uint8Array) => Box | null;
    if (kind === "blocks") {
      const centers = f32(this.buffers, geometry.centers);
      const sizes = f32(this.buffers, geometry.sizes);
      const axes = (layer.spec.axes ?? [[1, 0, 0], [0, 1, 0], [0, 0, 1]]) as Vec3[];
      n = centers.length / 3;
      box = (mask) => blocksBox(centers, sizes, axes, mask);
    } else if (kind === "drillholes") {
      const positions = f32(this.buffers, geometry.positions);
      const rows = u32(this.buffers, geometry.rows);
      n = rows.reduce((m, r) => Math.max(m, r + 1), 0);
      box = (mask) => maskedBox(positions, mask, 2, rows);
    } else {
      const positions = f32(this.buffers, geometry.positions);
      n = positions.length / 3;
      box = (mask) => maskedBox(positions, mask);
    }
    const shown = passMask(f.compiled, f.slots, "row", n, keep("row"));
    const kept = keep("row");
    layer.counts = { shown: sum(shown), total: n, nulls: kept ? n - sum(kept) : 0 };
    layer.box = layer.counts.shown === n ? layer.rep.box : box(shown);
  }

  private readoutText(layer: Layer): string {
    const { shown, total, nulls } = layer.counts;
    if (!layer.filter.active && !nulls) return "";
    return `${count(shown)} of ${count(total)} shown${nulls ? ` (${count(nulls)} null)` : ""}`;
  }

  private updateReadout(layer: Layer): void {
    if (!layer.readout) return;
    layer.readout.textContent = this.readoutText(layer);
    layer.readout.hidden = !layer.readout.textContent;
  }

  private refitBox(): void {
    const to = this.shownBox();
    if (this.box && to) this.refit = { from: this.box, to, start: performance.now() };
    else this.axes.setBox(to);
    this.box = to;
  }

  /** Repaints the layers colored by `names` (all when omitted) and redraws what depends on them. */
  private refresh(names?: Set<string>): void {
    for (const layer of this.layers) {
      if (!names || (layer.column && names.has(layer.column.name)) || (!layer.column && names.has(""))) this.paint(layer);
    }
    this.refitBox();
    this.renderBody();
    this.updateBars();
    this.requestRender();
  }

  private setVisible(layer: Layer, visible: boolean): void {
    layer.visible = visible;
    layer.rep.object.visible = visible;
  }

  private toggleLayer(layer: Layer, solo: boolean): void {
    this.dismiss();
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
    shared.uAccent.value.set(t.accent);
    for (const m of [this.outlineLines, this.outlineEdges, this.outlineDots, this.highlightLines, this.highlightRim, this.highlightOver])
      m.color.set(t.accent);
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
      iconButton("fit", "Fit view (R)", () => this.fit()),
      this.fullButton,
      iconButton("camera", "Screenshot", () => (this.shotMenu.hidden = !this.shotMenu.hidden)),
      this.helpButton,
      iconButton("hide", "Hide panel (H)", () => this.togglePanel()),
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
    this.sectionControls = { width: [], dip: [], hint: null };
    this.body.appendChild(this.section("Section", "section"));
    if (!this.collapsed.has("section")) this.body.appendChild(this.sectionPanel());
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
          this.showGizmo();
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
    if (layer.filter.active) {
      const mark = icon("filter");
      mark.classList.add("btv-mark");
      mark.setAttribute("aria-label", "Filtered");
      name.after(mark);
    }
    const readout = document.createElement("div");
    readout.className = "btv-readout";
    layer.readout = readout;
    this.updateReadout(layer);
    return layer.open ? [row, readout, this.settings(layer)] : [row, readout];
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
    box.appendChild(this.filterSection(layer));
    return box;
  }

  private filterSection(layer: Layer): HTMLElement {
    const section = document.createElement("div");
    section.className = "btv-filter";
    const head = document.createElement("div");
    head.className = "btv-filter-head";
    const conditions = layer.filter.conditions;
    const full = conditions.length >= SLOTS;
    const add = iconButton("plus", full ? `At most ${SLOTS} conditions per layer` : "Add condition", () => {
      layer.picking = !layer.picking;
      this.renderBody();
    });
    add.classList.add("btv-add");
    add.append("Add");
    add.classList.toggle("btv-on", layer.picking);
    head.append(label("Filter"), add);
    section.appendChild(head);
    if (layer.picking && full) {
      const note = document.createElement("div");
      note.className = "btv-note";
      note.textContent = `A layer filters on ${SLOTS} columns at most; remove one to add another.`;
      section.appendChild(note);
    } else if (layer.picking) {
      const pick = document.createElement("select");
      pick.setAttribute("aria-label", "Column to filter");
      pick.add(new Option("Choose a column", "", true, true));
      pick.options[0].disabled = true;
      for (const c of layer.spec.columns)
        if (!conditions.some((x) => x.column === c)) pick.add(new Option(c.name, c.name));
      pick.addEventListener("change", () => {
        const column = layer.spec.columns.find((c) => c.name === pick.value);
        if (!column) return;
        layer.picking = false;
        this.setConditions(layer, [...conditions, openCondition(column)]);
      });
      section.appendChild(pick);
    }
    for (const condition of conditions) section.appendChild(this.conditionCard(layer, condition));
    return section;
  }

  private conditionCard(layer: Layer, condition: Condition): HTMLElement {
    const card = document.createElement("div");
    card.className = "btv-cond";
    const head = document.createElement("div");
    head.className = "btv-cond-head";
    const name = document.createElement("span");
    name.className = "btv-name";
    name.textContent = name.title = condition.column.name;
    const remove = iconButton("x", `Remove the condition on ${condition.column.name}`, () =>
      this.setConditions(
        layer,
        layer.filter.conditions.filter((c) => c !== condition),
      ),
    );
    head.append(icon("filter"), name, remove);
    card.append(head, condition.column.type === "text" ? this.categoryControl(layer, condition) : this.rangeControl(layer, condition));
    return card;
  }

  private columnHistogram(layer: Layer, column: ColumnSpec): number[] {
    const key = `${layer.spec.id}/${column.name}`;
    let counts = this.histograms.get(key);
    if (!counts) {
      const data = this.columnData(layer, column);
      counts =
        column.type === "text"
          ? categoryCounts(data as Int32Array, column.categories?.length ?? 0)
          : histogram(data, column.min ?? 0, column.max ?? 0);
      this.histograms.set(key, counts);
    }
    return counts;
  }

  /** Two-handle slider over the column's histogram, with editable bounds; a handle at an end leaves it open. */
  private rangeControl(layer: Layer, condition: Condition): HTMLElement {
    const min = condition.column.min ?? 0;
    const max = condition.column.max ?? 0;
    const span = max - min;
    const step = sliderStep(span);
    const wrap = document.createElement("div");
    wrap.className = "btv-slider";
    const counts = this.columnHistogram(layer, condition.column);
    const heights = barHeights(counts);
    const bars = heights
      .map((h, i) => (h ? `<rect x="${i + 0.08}" y="${1 - h}" width="0.84" height="${h}"/>` : ""))
      .join("");
    const clip = `btv-${this.uid}-${layer.spec.id}-${condition.column.name.replace(/[^\w-]/g, "_")}`;
    const hist = document.createElement("div");
    hist.className = "btv-hist";
    hist.innerHTML =
      `<svg viewBox="0 0 ${counts.length} 1" preserveAspectRatio="none" aria-hidden="true">` +
      `<clipPath id="${clip}"><rect y="0" height="1" x="0" width="0"/></clipPath>` +
      `<g class="btv-bars-all">${bars}</g><g class="btv-bars-on" clip-path="url(#${clip})">${bars}</g></svg>`;
    const window_ = hist.querySelector("clipPath rect")!;
    const track = document.createElement("div");
    track.className = "btv-track";
    const fill = document.createElement("div");
    fill.className = "btv-fill";
    const handles = ["lo", "hi"].map((end) => {
      const h = document.createElement("div");
      h.className = "btv-handle";
      h.tabIndex = 0;
      h.setAttribute("role", "slider");
      h.setAttribute("aria-label", end === "lo" ? "Minimum" : "Maximum");
      return h;
    });
    track.append(fill, ...handles);
    const inputs = ["Minimum", "Maximum"].map((title) => {
      const el = document.createElement("input");
      el.type = "number";
      el.step = "any";
      el.title = title;
      el.setAttribute("aria-label", title);
      return el;
    });
    const at = (v: number) => (span > 0 ? Math.min(1, Math.max(0, (v - min) / span)) : 0);
    const show = () => {
      const lo = condition.lo ?? min;
      const hi = condition.hi ?? max;
      const [a, b] = [at(lo), at(hi)];
      window_.setAttribute("x", String(a * counts.length));
      window_.setAttribute("width", String(Math.max(0, b - a) * counts.length));
      fill.style.left = `${a * 100}%`;
      fill.style.width = `${Math.max(0, b - a) * 100}%`;
      handles[0].style.left = `${a * 100}%`;
      handles[1].style.left = `${b * 100}%`;
      handles[0].setAttribute("aria-valuenow", String(lo));
      handles[1].setAttribute("aria-valuenow", String(hi));
      [lo, hi].forEach((v, k) => {
        if ((this.root.getRootNode() as Document | ShadowRoot).activeElement !== inputs[k])
          inputs[k].value = formatBound(v, span);
      });
    };
    /** Sets one end from a value: snapped, kept on its side of the other, open at the column's end. */
    const set = (end: 0 | 1, value: number, snapped = true) => {
      let v = snapped ? snap(value, step) : value;
      if (end === 0) {
        v = Math.min(v, condition.hi ?? max);
        condition.lo = v <= min ? null : v;
      } else {
        v = Math.max(v, condition.lo ?? min);
        condition.hi = v >= max ? null : v;
      }
      show();
      this.filterEdited(layer);
    };
    track.addEventListener("pointerdown", (e) => {
      e.preventDefault();
      const r = track.getBoundingClientRect();
      const value = (x: number) => min + Math.min(1, Math.max(0, (x - r.left) / Math.max(r.width, 1))) * span;
      const v = value(e.clientX);
      const lo = condition.lo ?? min;
      const hi = condition.hi ?? max;
      const end: 0 | 1 = Math.abs(v - lo) <= Math.abs(v - hi) && !(lo === hi && v > hi) ? 0 : 1;
      track.setPointerCapture(e.pointerId);
      handles[end].focus({ preventScroll: true });
      set(end, v);
      const move = (m: PointerEvent) => set(end, value(m.clientX));
      const up = () => {
        track.removeEventListener("pointermove", move);
        track.removeEventListener("pointerup", up);
      };
      track.addEventListener("pointermove", move);
      track.addEventListener("pointerup", up);
    });
    handles.forEach((h, end) =>
      h.addEventListener("keydown", (e) => {
        const d = { ArrowLeft: -1, ArrowDown: -1, ArrowRight: 1, ArrowUp: 1 }[e.key];
        if (!d) return;
        e.preventDefault();
        e.stopPropagation();
        const current = end === 0 ? (condition.lo ?? min) : (condition.hi ?? max);
        set(end as 0 | 1, current + d * step * (e.shiftKey ? 10 : 1));
      }),
    );
    inputs.forEach((el, end) =>
      el.addEventListener("change", () => {
        const text = el.value.trim();
        const v = Number(text);
        if (!text) set(end as 0 | 1, end === 0 ? min : max);
        else if (Number.isFinite(v)) set(end as 0 | 1, v, false);
        show();
        inputs[end].value = formatBound(end === 0 ? (condition.lo ?? min) : (condition.hi ?? max), span);
      }),
    );
    const range = document.createElement("div");
    range.className = "btv-range";
    range.append(...inputs);
    wrap.append(hist, track, range);
    show();
    return wrap;
  }

  /** The column's categories as checkboxes with their row counts, and all/none shortcuts. */
  private categoryControl(layer: Layer, condition: Condition): HTMLElement {
    const wrap = document.createElement("div");
    wrap.className = "btv-cats";
    const names = condition.column.categories ?? [];
    const counts = this.columnHistogram(layer, condition.column);
    const kept = condition.categories ?? new Set<string>();
    const boxes: HTMLInputElement[] = [];
    const quick = document.createElement("div");
    quick.className = "btv-quick";
    for (const [text, all] of [["All", true], ["None", false]] as const) {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = text;
      b.addEventListener("click", () => {
        kept.clear();
        if (all) names.forEach((n) => kept.add(n));
        boxes.forEach((x) => (x.checked = all));
        this.filterEdited(layer);
      });
      quick.appendChild(b);
    }
    const list = document.createElement("div");
    list.className = "btv-list";
    names.forEach((name, i) => {
      const row = document.createElement("label");
      const check = document.createElement("input");
      check.type = "checkbox";
      check.checked = kept.has(name);
      check.addEventListener("change", () => {
        if (check.checked) kept.add(name);
        else kept.delete(name);
        this.filterEdited(layer);
      });
      boxes.push(check);
      const text = document.createElement("span");
      text.className = "btv-name";
      text.textContent = text.title = name;
      const n = document.createElement("span");
      n.className = "btv-count";
      n.textContent = count(counts[i] ?? 0);
      row.append(check, text, n);
      list.appendChild(row);
    });
    condition.categories = kept;
    wrap.append(quick, list);
    return wrap;
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

  // ---- section ----

  private cameraState(): CameraState {
    return { position: this.camera.position.clone(), target: this.controls.target.clone() };
  }

  private restoreCamera(state: CameraState): void {
    this.camera.position.copy(state.position);
    this.controls.target.copy(state.target);
    this.camera.lookAt(state.target);
    this.controls.update();
  }

  /** The camera `fit` would give from `from`, without moving there. */
  private fitted(from: CameraState): CameraState {
    const now = this.cameraState();
    this.restoreCamera(from);
    this.fit();
    const out = this.cameraState();
    this.restoreCamera(now);
    return out;
  }

  /** The camera `setView` and `fit` would give, without moving there. */
  private viewState(azimuth: number, dip: number): CameraState {
    const now = this.cameraState();
    this.setView(azimuth, dip);
    this.fit();
    const out = this.cameraState();
    this.restoreCamera(now);
    return out;
  }

  private glide(to: CameraState): void {
    this.tween = { from: this.cameraState(), to, start: performance.now() };
    this.requestRender();
  }

  /** Center of the shown layers' bounds, whatever the section: quick cuts pick on it, drawn ones hinge on it. */
  private middle(): Vec3 {
    const box = visibleBox(this.layers);
    return box ? ([0, 1, 2].map((a) => (box.min[a] + box.max[a]) / 2) as Vec3) : [0, 0, 0];
  }

  private widthRange(): [number, number] {
    const d = Math.max(diagonal(visibleBox(this.layers)), 1e-3);
    return [d / 500, d / 2];
  }

  private sectionWidth(): number {
    return (this.cutWidth ??= Math.max(diagonal(visibleBox(this.layers)) * WIDTH_SHARE, 1e-3));
  }

  /** Box the axes frame: the shown layers', narrowed to the section in plan, or the unfolded section's own. */
  private shownBox(): Box | null {
    const box = visibleBox(this.layers);
    const s = this.cutSection;
    if (!box || !s || this.drawing) return box;
    if (!this.cut?.unfolded) return sectionBox(s, box) ?? box;
    const d = THREE.MathUtils.degToRad(s.dip);
    const across = s.half / Math.sin(d) + Math.max(Math.abs(box.min[2] - s.z), Math.abs(box.max[2] - s.z)) / Math.tan(d);
    return { min: [0, -across, box.min[2]], max: [s.length, across, box.max[2]] };
  }

  /** The section as Python reads it, in real-world coordinates; empty without one. */
  sectionState(): SectionState | Record<string, never> {
    const c = this.cut;
    if (!c || !this.cutSection) return {};
    const o = this.spec.origin;
    return {
      points: distinct(c.points).map((p) => [p[0] + o[0], p[1] + o[1], c.z + o[2]]),
      width: this.sectionWidth(),
      dip: this.cutDip,
      unfolded: c.unfolded,
    };
  }

  /** Applies a section from Python; empty clears it. */
  setSection(state: SectionState | Record<string, never>): void {
    if (JSON.stringify(state ?? {}) === JSON.stringify(this.sectionState())) return;
    if (this.drawing) this.endDrawing();
    if (!state || !("points" in state)) return this.setCut(null, false);
    const o = this.spec.origin;
    const zs = state.points.filter((p) => p.length > 2).map((p) => p[2] - o[2]);
    if (state.width !== null && state.width > 0) this.cutWidth = state.width;
    if (state.dip > 0 && state.dip <= 90) this.cutDip = state.dip;
    this.setCut(
      {
        points: state.points.map((p) => [p[0] - o[0], p[1] - o[1]] as XY),
        z: zs.length ? zs.reduce((a, b) => a + b, 0) / zs.length : this.middle()[2],
        unfolded: false,
      },
      false,
    );
    if (state.unfolded) this.setUnfolded(true, false);
  }

  private emitSection(): void {
    clearTimeout(this.emitTimer);
    this.options.onSection?.(this.sectionState());
  }

  /** Cuts along `cut`, or removes the section with null; folds back first. */
  private setCut(cut: Cut | null, emit = true): void {
    if (this.cut?.unfolded && this.folded) {
      this.tween = null;
      this.restoreCamera(this.folded);
    }
    this.folded = null;
    this.cut = cut ? { ...cut, unfolded: false } : null;
    this.applySection();
    this.showGizmo();
    this.renderBody();
    if (!emit) return;
    this.glide(this.fitted(this.tween?.to ?? this.cameraState()));
    this.emitSection();
  }

  /** Rebuilds the section from the cut, width and dip, with its axes, box and outline. */
  private applySection(): void {
    this.dismiss();
    const c = this.cut;
    this.cutSection = c ? buildSection(c.points, c.z, this.sectionWidth(), this.cutDip) : null;
    if (!this.cutSection) this.cut = null;
    if (this.cut?.unfolded) {
      const unit = /\(([^)]*)\)$/.exec(this.spec.axes[0])?.[1];
      const suffix = unit ? ` (${unit})` : "";
      this.axes.setFrame([0, 0, this.spec.origin[2]], [`Distance${suffix}`, `Offset${suffix}`, `Elevation${suffix}`]);
    } else this.axes.setFrame(this.spec.origin, this.spec.axes);
    this.refitBox();
    this.drawOutline();
    this.requestRender();
  }

  toggleUnfolded(): void {
    if (this.cut && !this.drawing) this.setUnfolded(!this.cut.unfolded);
  }

  /** Lays the section out flat, distance along it to the right and elevation up, or folds it back. */
  private setUnfolded(on: boolean, emit = true): void {
    const c = this.cut;
    if (!c || c.unfolded === on) return;
    if (on) this.folded = this.cameraState();
    c.unfolded = on;
    this.applySection();
    this.refit = null;
    this.axes.setBox(this.box);
    if (on) {
      this.setView(0, 0);
      this.fit();
    } else if (this.folded) this.restoreCamera(this.folded);
    this.tween = null;
    if (!on) this.folded = null;
    this.showGizmo();
    this.renderBody();
    if (emit) this.emitSection();
  }

  /** The gizmo shows unless turned off or the section is unfolded, where north and east lose their meaning. */
  private showGizmo(): void {
    this.gizmo.el.style.display = this.show.gizmo && !this.cut?.unfolded ? "" : "none";
  }

  clearSection(): void {
    if (this.drawing) this.endDrawing();
    else if (this.cut) this.setCut(null);
  }

  setWidth(width: number): void {
    if (!(width > 0) || !Number.isFinite(width)) return this.syncControls();
    this.cutWidth = width;
    this.sectionEdited();
  }

  setDip(dip: number): void {
    if (!(dip > 0 && dip <= 90)) return this.syncControls();
    this.cutDip = dip;
    this.sectionEdited();
  }

  /** The width or dip changed: the section and outline now, the panel's other control, Python once edits pause. */
  private sectionEdited(): void {
    this.applySection();
    this.syncControls();
    this.drawSweep();
    if (!this.cut) return;
    clearTimeout(this.emitTimer);
    this.emitTimer = setTimeout(() => this.emitSection(), 150);
  }

  // drawing

  toggleDrawing(): void {
    if (this.drawing) this.finishDrawing();
    else this.startDrawing();
  }

  /** Switches to a plan view where clicks add the vertices of a polyline. */
  private startDrawing(): void {
    this.endQuick();
    if (this.cut?.unfolded) this.setUnfolded(false);
    this.drawing = { points: [], cursor: null, shift: false, view: this.cameraState() };
    this.controls.enableRotate = false;
    this.controls.mouseButtons.LEFT = THREE.MOUSE.PAN;
    this.refitBox();
    this.glide(this.viewState(0, 90));
    this.drawOutline();
    this.renderBody();
  }

  /** Leaves the drawer for the view it started from; returns the vertices drawn. */
  private endDrawing(): XY[] {
    const d = this.drawing;
    if (!d) return [];
    this.drawing = null;
    this.press = null;
    this.controls.enableRotate = true;
    this.controls.mouseButtons.LEFT = THREE.MOUSE.ROTATE;
    this.glide(d.view);
    this.refitBox();
    this.drawOutline();
    this.renderBody();
    return d.points;
  }

  finishDrawing(): void {
    const points = this.endDrawing();
    if (distinct(points).length >= 2) this.setCut({ points, z: this.middle()[2], unfolded: false });
  }

  private drawCursor(): XY | null {
    const d = this.drawing;
    if (!d?.cursor) return null;
    const last = d.points[d.points.length - 1];
    return d.shift && last ? snap45(last, d.cursor) : d.cursor;
  }

  private addVertex(at: XY): void {
    const d = this.drawing!;
    if (d.points.length > MAX_SEGMENTS) return;
    const last = d.points[d.points.length - 1];
    const p = d.shift && last ? snap45(last, at) : at;
    if (last && last[0] === p[0] && last[1] === p[1]) return;
    d.points.push(p);
    this.drawOutline();
    this.updateHint();
  }

  private undoVertex(): void {
    this.drawing?.points.pop();
    this.drawOutline();
    this.updateHint();
  }

  private setShift(on: boolean): void {
    if (!this.drawing || this.drawing.shift === on) return;
    this.drawing.shift = on;
    this.drawOutline();
  }

  // outline

  private clearOutline(): void {
    for (const child of this.outline.children) (child as THREE.Mesh).geometry.dispose();
    this.outline.clear();
  }

  private addLines(positions: number[], material: LineMaterial): void {
    if (!positions.length) return;
    const line = new LineSegments2(new LineSegmentsGeometry().setPositions(positions), material);
    line.frustumCulled = false;
    line.renderOrder = 10;
    this.outline.add(line);
  }

  /**
   * The section in the accent: its trace and slab edges on the top of the box, or its ends and bends once
   * unfolded; while drawing, the polyline so far, its vertices and the slab it would cut.
   */
  private drawOutline(): void {
    this.clearOutline();
    const box = this.box ?? visibleBox(this.layers);
    if (!box) return;
    const top = box.max[2];
    const pairs = (line: XY[]) => line.slice(1).flatMap((p, i) => [...line[i], top, ...p, top]);
    const d = this.drawing;
    let s = this.cutSection;
    if (d) {
      const cursor = this.drawCursor();
      const path = cursor ? [...d.points, cursor] : d.points;
      s = buildSection(path, this.middle()[2], this.sectionWidth(), this.cutDip);
      const dots = new THREE.BufferGeometry();
      dots.setAttribute("position", new THREE.Float32BufferAttribute(path.flatMap((p) => [...p, top]), 3));
      const points = new THREE.Points(dots, this.outlineDots);
      points.frustumCulled = false;
      points.renderOrder = 11;
      this.outline.add(points);
    }
    if (s && this.cut?.unfolded && !d) {
      const at = [0, ...s.segments.map((g) => g.chainage + g.length)];
      this.addLines(at.flatMap((x) => [x, 0, box.min[2], x, 0, box.max[2]]), this.outlineEdges);
    } else if (s) {
      const [trace, left, right] = traces(s, top);
      this.addLines(pairs(trace), this.outlineLines);
      this.addLines([...pairs(left), ...pairs(right)], this.outlineEdges);
    }
    this.requestRender();
  }

  // pointer

  private screen(e: MouseEvent): XY {
    const r = this.renderer.canvas.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  }

  private ray(at: XY): Ray {
    const caster = new THREE.Raycaster();
    caster.setFromCamera(new THREE.Vector2((at[0] / this.width) * 2 - 1, 1 - (at[1] / this.height) * 2), this.camera);
    return { origin: caster.ray.origin.toArray() as Vec3, direction: caster.ray.direction.toArray() as Vec3 };
  }

  /** Where the pointer meets the level plane through the top of the box, where the drawer draws. */
  private planAt(at: XY): XY | null {
    const { origin, direction } = this.ray(at);
    const z = (this.box ?? visibleBox(this.layers))?.max[2] ?? 0;
    if (Math.abs(direction[2]) < 1e-9) return null;
    const t = (z - origin[2]) / direction[2];
    return t > 0 ? [origin[0] + t * direction[0], origin[1] + t * direction[1]] : null;
  }

  private pointerDown(e: PointerEvent): void {
    this.root.focus({ preventScroll: true });
    if (e.button !== 0) return;
    const at = this.screen(e);
    if (this.drawing) {
      this.press = at;
      return;
    }
    if (!e.shiftKey || this.cut?.unfolded) {
      this.clickAt = at;
      return;
    }
    e.preventDefault();
    e.stopImmediatePropagation();
    this.renderer.canvas.setPointerCapture(e.pointerId);
    this.quick = { from: at, to: at };
    this.drawSweep();
  }

  private pointerMove(e: PointerEvent): void {
    if (this.quick) {
      this.quick.to = this.screen(e);
      this.drawSweep();
    } else if (this.drawing) {
      this.drawing.cursor = this.planAt(this.screen(e));
      this.drawing.shift = e.shiftKey;
      this.drawOutline();
    }
  }

  private pointerUp(e: PointerEvent): void {
    const at = this.screen(e);
    const click = isClick(this.clickAt, at, CLICK_PX);
    this.clickAt = null;
    if (click && !this.drawing && !this.quick) return this.pick(at);
    if (this.quick) {
      const { from } = this.quick;
      this.endQuick();
      if (Math.hypot(at[0] - from[0], at[1] - from[1]) >= 2 * CLICK_PX) this.quickCut(from, at);
      return;
    }
    if (this.drawing && this.press && Math.hypot(at[0] - this.press[0], at[1] - this.press[1]) < CLICK_PX) {
      const p = this.planAt(at);
      this.drawing.shift = e.shiftKey;
      if (p) this.addVertex(p);
    }
    this.press = null;
  }

  /** The wheel zooms; Shift+wheel sets the width while cutting or drawing, or of the section there is. */
  private wheel(e: WheelEvent): void {
    if (e.ctrlKey || !e.shiftKey || !(this.quick || this.drawing || this.cut)) return;
    e.preventDefault();
    e.stopImmediatePropagation();
    const delta = e.deltaY || e.deltaX;
    if (delta) this.setWidth(this.sectionWidth() * (delta < 0 ? WHEEL_STEP : 1 / WHEEL_STEP));
  }

  /** Cuts the upright section under a line swept across the screen, its ends picked as `pickCut` says. */
  private quickCut(from: XY, to: XY): void {
    const center = this.middle();
    const forward = this.camera.getWorldDirection(new THREE.Vector3()).toArray() as Vec3;
    const ends = pickCut([this.ray(from), this.ray(to)], center, forward);
    const points = ends?.map((p) => [p[0], p[1]] as XY) ?? [];
    if (distinct(points).length === 2) this.setCut({ points, z: center[2], unfolded: false });
  }

  private endQuick(): void {
    this.quick = null;
    this.drawSweep();
  }

  private drawSweep(): void {
    const q = this.quick;
    this.sweep.replaceChildren();
    if (!q) return;
    const ns = "http://www.w3.org/2000/svg";
    const line = document.createElementNS(ns, "line");
    for (const [k, v] of Object.entries({ x1: q.from[0], y1: q.from[1], x2: q.to[0], y2: q.to[1] }))
      line.setAttribute(k, String(v));
    const text = document.createElementNS(ns, "text");
    text.setAttribute("x", String(q.to[0] + 10));
    text.setAttribute("y", String(q.to[1] - 10));
    text.textContent = `width ${formatLength(this.sectionWidth())}`;
    this.sweep.append(line, text);
  }

  // panel

  private sectionPanel(): HTMLElement {
    const box = document.createElement("div");
    box.className = "btv-settings btv-view";
    const tools = document.createElement("div");
    tools.className = "btv-tools";
    const tool = (name: IconName, title: string, pressed: boolean, enabled: boolean, action: () => void) => {
      const b = iconButton(name, title, action);
      b.classList.add("btv-tool");
      b.setAttribute("aria-pressed", String(pressed));
      b.disabled = !enabled;
      tools.appendChild(b);
    };
    const c = this.cut;
    tool("draw", this.drawing ? "Cut along the polyline (Enter)" : "Draw a section (S)", !!this.drawing, true, () =>
      this.toggleDrawing(),
    );
    tool(c?.unfolded ? "fold" : "unfold", c?.unfolded ? "Fold back (U)" : "Unfold (U)", !!c?.unfolded, !!c && !this.drawing, () =>
      this.toggleUnfolded(),
    );
    tool("clear", this.drawing ? "Cancel drawing (Esc)" : "Clear the section (X)", false, !!c || !!this.drawing, () =>
      this.clearSection(),
    );
    const pair = (title: string, slider: [number, number, number], set: (slider: number | null, typed: number | null) => void) => {
      const row = document.createElement("div");
      row.className = "btv-range btv-pair";
      const range = document.createElement("input");
      range.type = "range";
      [range.min, range.max, range.step] = slider.map(String);
      range.setAttribute("aria-label", title);
      const typed = document.createElement("input");
      typed.type = "number";
      typed.step = "any";
      typed.title = title;
      typed.setAttribute("aria-label", title);
      range.addEventListener("input", () => set(Number(range.value), null));
      typed.addEventListener("change", () => set(null, Number(typed.value)));
      row.append(range, typed);
      return { row, inputs: [range, typed] };
    };
    const width = pair("Width", [0, 1000, 1], (slider, typed) => {
      const [lo, hi] = this.widthRange();
      this.setWidth(slider === null ? typed! : lo * (hi / lo) ** (slider / 1000));
    });
    const dip = pair("Dip", [5, 90, 1], (slider, typed) => this.setDip((slider ?? typed)!));
    const hint = document.createElement("div");
    hint.className = "btv-note btv-hint";
    this.sectionControls = { width: width.inputs, dip: dip.inputs, hint };
    box.append(label("Cut"), tools, label("Width"), width.row, label("Dip"), dip.row, hint);
    this.syncControls();
    return box;
  }

  private syncControls(): void {
    const active = (this.root.getRootNode() as Document | ShadowRoot).activeElement;
    const set = (el: HTMLInputElement | undefined, value: string) => {
      if (el && el !== active) el.value = value;
    };
    const [lo, hi] = this.widthRange();
    const w = this.sectionWidth();
    const [wRange, wTyped] = this.sectionControls.width;
    set(wRange, String(Math.round((1000 * Math.log(Math.min(Math.max(w, lo), hi) / lo)) / Math.log(hi / lo))));
    set(wTyped, formatLength(w));
    const [dRange, dTyped] = this.sectionControls.dip;
    set(dRange, String(this.cutDip));
    set(dTyped, formatLength(this.cutDip));
    this.updateHint();
  }

  private updateHint(): void {
    const hint = this.sectionControls.hint;
    if (!hint) return;
    const d = this.drawing;
    const s = this.cutSection;
    if (d) {
      const n = d.points.length;
      hint.textContent = `${n} point${n === 1 ? "" : "s"}. Click to add, Shift locks 45°, Shift+wheel sets the width, Backspace undoes, Enter cuts, Esc cancels.`;
    } else if (s && this.cut) {
      const n = s.segments.length + 1;
      hint.textContent = `${n} points, ${formatLength(s.length)} long. Shift+wheel sets the width, U ${this.cut.unfolded ? "folds back" : "unfolds"}, X clears.`;
    } else hint.textContent = "Shift-drag across the view for a straight cut (Shift+wheel sets its width), or draw a polyline (S).";
  }

  // ---- picking ----

  /** The element under CSS pixel `at`, from an id pass drawn for this click in full detail; null for none. */
  private hitAt(at: XY): Hit | null {
    this.settle.stop();
    this.shareUniforms();
    const hidden: THREE.Object3D[] = [];
    const hide = (o: THREE.Object3D) => {
      if (!o.visible) return;
      o.visible = false;
      hidden.push(o);
    };
    [this.axes.group, this.outline, this.highlight].forEach(hide);
    const saved: [THREE.Material, THREE.Blending, boolean][] = [];
    this.scene.traverse((o) => {
      if (o.userData.halo) hide(o);
      const m = (o as THREE.Mesh).material;
      if (!m || Array.isArray(m)) return;
      saved.push([m, m.blending, m.colorWrite]);
      m.blending = THREE.NoBlending;
      if (m.userData.pickColor) m.colorWrite = true;
    });
    shared.uPicking.value = 1;
    shared.uPixelRatio.value = 1;
    const w = this.renderer.pick(this.scene, this.camera, at[0], at[1], PICK_RADIUS);
    shared.uPicking.value = 0;
    shared.uPixelRatio.value = this.pixelRatio;
    for (const [m, blending, colorWrite] of saved.reverse()) {
      m.blending = blending;
      m.colorWrite = colorWrite;
    }
    hidden.forEach((o) => (o.visible = true));
    this.requestRender();
    return nearestHit(w.pixels, w.width, w.height, w.x, w.y);
  }

  /** Inspects the element under CSS pixel `at`: its card, its highlight, and Python's `picked`; empty space dismisses. */
  pick(at: XY): void {
    const hit = this.hitAt(at);
    const layer = hit ? this.layers[hit.layer] : undefined;
    if (!hit || !layer) return this.dismiss();
    const element = this.inspect(layer, hit.row, at);
    this.picked = element.state;
    this.drawHighlight(element);
    this.showCard(layer.spec.name, element.card, at);
    this.options.onPick?.(element.state);
  }

  /** Closes the card and drops the highlight and the pick. */
  dismiss(): void {
    if (!this.picked) return;
    this.picked = null;
    this.card.hidden = true;
    this.clearHighlight();
    this.requestRender();
    this.options.onPick?.({});
  }

  /** The last pick as Python reads it; null without one. */
  pickState(): PickState | null {
    return this.picked;
  }

  /** Where a local point draws: moved with its segment's stretch (that of `ref`) when the section is unfolded. */
  private place(p: Vec3, ref: Vec3 = p): Vec3 {
    return this.cut?.unfolded && this.cutSection ? unfold(this.cutSection, p, ref) : p;
  }

  /**
   * A picked row's values, real-world position and card lines, and its highlight in the accent: a ring around its
   * position, which keeps a small element easy to find, and the edges of a block or a mesh triangle, a rim under a
   * drill-hole interval drawn as lines, or its axis over tubes. A mesh row is a triangle; its vertex columns are
   * read at the corner nearest the click.
   */
  private inspect(layer: Layer, row: number, at: XY): Inspected {
    const { kind, geometry, representation: rep } = layer.spec;
    const xyz = (a: Float32Array, i: number): Vec3 => [a[3 * i], a[3 * i + 1], a[3 * i + 2]];
    const out: Inspected = { state: null as unknown as PickState, card: [], lines: [], rings: [], rim: 0, over: false };
    const extra: [string, string][] = [];
    let position: Vec3;
    let vertex = row;
    if (kind === "blocks") {
      const sizes = f32(this.buffers, geometry.sizes);
      position = xyz(f32(this.buffers, geometry.centers), row);
      const size = xyz(sizes, row);
      if (rep !== "points")
        for (const p of boxEdges(position, size, (layer.spec.axes ?? IDENTITY) as Vec3[])) out.lines.push(...this.place(p, position));
      extra.push(["Block size", size.map((v) => formatValue(v)).join(" × ")]);
    } else if (kind === "drillholes") {
      position = xyz(f32(this.buffers, geometry.midpoints), row);
      if (rep !== "points") {
        const ends = f32(this.buffers, geometry.positions);
        const rows = u32(this.buffers, geometry.rows);
        for (let i = 0; i < rows.length; i++) {
          if (rows[i] !== row) continue;
          const [a, b] = [xyz(ends, 2 * i), xyz(ends, 2 * i + 1)];
          const mid: Vec3 = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2];
          out.lines.push(...this.place(a, mid), ...this.place(b, mid));
        }
        if (rep === "lines") out.rim = (layer.spec.lineWidth ?? 2.5) + 6;
        else out.over = true;
      }
    } else if (kind === "points") {
      position = xyz(f32(this.buffers, geometry.positions), row);
    } else {
      const points = f32(this.buffers, geometry.positions);
      const triangles = u32(this.buffers, geometry.triangles);
      const corners = [0, 1, 2].map((k) => triangles[3 * row + k]);
      const screen = corners.map((v) => this.toScreen(this.place(xyz(points, v))));
      const nearest = screen.reduce((best, s, k) => (dist2(s, at) < dist2(screen[best], at) ? k : best), 0);
      vertex = corners[nearest];
      position = xyz(points, vertex);
      if (rep !== "points") {
        const v = corners.map((c) => this.place(xyz(points, c)));
        out.lines.push(...v[0], ...v[1], ...v[1], ...v[2], ...v[2], ...v[0]);
        if (!this.cut?.unfolded) position = this.onTriangle(at, v) ?? position;
      }
    }
    out.rings.push(...this.place(position));

    const values: PickState["values"] = {};
    for (const c of layer.spec.columns) {
      const data = this.columnData(layer, c);
      const v = data[c.on === "vertex" ? vertex : row];
      values[c.name] = c.type === "text" ? (c.categories?.[v] ?? null) : Number.isFinite(v) ? round7(v) : null;
    }
    const o = this.spec.origin;
    const world: [number, number, number] = [position[0] + o[0], position[1] + o[1], position[2] + o[2]];
    out.state = { layer: layer.spec.name, row, values, position: world };

    const holes = layer.spec.holes;
    const own = new Set(holes ? [holes.hole, holes.from, holes.to] : []);
    if (holes) {
      const depths = geometry.depths ? f32(this.buffers, geometry.depths) : null;
      out.card.push([
        ["Hole", formatValue(values[holes.hole])],
        ["From", formatValue(depths ? round7(depths[2 * row]) : null)],
        ["To", formatValue(depths ? round7(depths[2 * row + 1]) : null)],
      ]);
    }
    const columns = layer.spec.columns.filter((c) => !own.has(c.name));
    if (columns.length) out.card.push(columns.map((c) => [c.name, formatValue(values[c.name])]));
    out.card.push([...this.spec.axes.map((title, k): [string, string] => [title, world[k].toFixed(2)]), ...extra]);
    return out;
  }

  private toScreen(p: Vec3): XY {
    const v = new THREE.Vector3(...p).project(this.camera);
    return [((v.x + 1) / 2) * this.width, ((1 - v.y) / 2) * this.height];
  }

  /** Where the ray under CSS pixel `at` meets the plane of triangle `v`; null when it runs parallel. */
  private onTriangle(at: XY, v: Vec3[]): Vec3 | null {
    const { origin, direction } = this.ray(at);
    const plane = new THREE.Plane().setFromCoplanarPoints(...(v.map((p) => new THREE.Vector3(...p)) as [THREE.Vector3, THREE.Vector3, THREE.Vector3]));
    const hit = new THREE.Ray(new THREE.Vector3(...origin), new THREE.Vector3(...direction)).intersectPlane(plane, new THREE.Vector3());
    return hit ? (hit.toArray() as Vec3) : null;
  }

  private clearHighlight(): void {
    for (const child of this.highlight.children) (child as THREE.Mesh).geometry.dispose();
    this.highlight.clear();
  }

  private drawHighlight(element: Inspected): void {
    this.clearHighlight();
    if (element.lines.length) {
      const material = element.rim ? this.highlightRim : element.over ? this.highlightOver : this.highlightLines;
      if (element.rim) this.highlightRim.linewidth = element.rim;
      const line = new LineSegments2(new LineSegmentsGeometry().setPositions(element.lines), material);
      line.frustumCulled = false;
      line.renderOrder = element.rim ? HALO_ORDER + 0.5 : 12;
      this.highlight.add(line);
    }
    if (element.rings.length) {
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.Float32BufferAttribute(element.rings, 3));
      const ring = new THREE.Points(g, this.highlightRing);
      ring.frustumCulled = false;
      ring.renderOrder = 12;
      this.highlight.add(ring);
    }
    this.requestRender();
  }

  /** The inspect card beside the click, kept inside the view: the layer's name, then groups of lines. */
  private showCard(name: string, groups: [string, string][][], at: XY): void {
    const head = document.createElement("div");
    head.className = "btv-card-head";
    const title = document.createElement("span");
    title.className = "btv-name";
    title.textContent = title.title = name;
    head.append(title, iconButton("x", "Close (Esc)", () => this.dismiss()));
    const list = document.createElement("dl");
    groups.forEach((lines, g) => {
      if (g) list.appendChild(document.createElement("hr"));
      for (const [key, value] of lines) {
        const dt = document.createElement("dt");
        const dd = document.createElement("dd");
        dt.textContent = dt.title = key;
        dd.textContent = dd.title = value;
        list.append(dt, dd);
      }
    });
    this.card.replaceChildren(head, list);
    this.card.hidden = false;
    const w = this.card.offsetWidth;
    const h = this.card.offsetHeight;
    const gap = 14;
    let x = at[0] + gap;
    if (x + w > this.width - 8) x = at[0] - gap - w;
    let y = at[1] + gap;
    if (y + h > this.height - 8) y = this.height - 8 - h;
    this.card.style.left = `${Math.max(8, x)}px`;
    this.card.style.top = `${Math.max(8, y)}px`;
  }

  // ---- keys and screenshots ----

  private key(e: KeyboardEvent): void {
    const target = e.composedPath()[0] as HTMLElement;
    const inField = /^(INPUT|SELECT|TEXTAREA)$/.test(target?.tagName ?? "");
    const cancellable = !!(this.drawing || this.quick || this.picked || !this.help.hidden);
    const action = keyAction(e, inField, { drawing: !!this.drawing, cancellable });
    if (!action) return;
    e.preventDefault();
    e.stopPropagation();
    this.act(action);
  }

  act(action: Action): void {
    ({
      fit: () => this.fit(),
      plan: () => this.setView(0, 90),
      draw: () => this.toggleDrawing(),
      finish: () => this.finishDrawing(),
      undo: () => this.undoVertex(),
      lock: () => this.setShift(true),
      unfold: () => this.toggleUnfolded(),
      clear: () => this.clearSection(),
      panel: () => this.togglePanel(),
      theme: () => this.toggleTheme(),
      help: () => this.toggleHelp(),
      cancel: () => {
        if (this.drawing) this.endDrawing();
        else if (this.quick) this.endQuick();
        else if (!this.help.hidden) this.toggleHelp(false);
        else this.dismiss();
      },
    })[action]();
  }

  /** Shows or hides the card listing the keyboard shortcuts. */
  toggleHelp(show = this.help.hidden): void {
    if (!this.help.childElementCount) {
      const head = document.createElement("div");
      head.className = "btv-card-head";
      const title = document.createElement("span");
      title.className = "btv-name";
      title.textContent = "Keyboard shortcuts";
      head.append(title, iconButton("x", "Close (Esc)", () => this.toggleHelp(false)));
      const list = document.createElement("dl");
      for (const [key, text] of SHORTCUTS) {
        const dt = document.createElement("dt");
        const kbd = document.createElement("kbd");
        kbd.textContent = key;
        dt.appendChild(kbd);
        const dd = document.createElement("dd");
        dd.textContent = text;
        list.append(dt, dd);
      }
      this.help.append(head, list);
    }
    this.help.hidden = !show;
  }

  /** Enters or leaves fullscreen; where the page forbids it, the button says so and stays off. */
  private async toggleFullscreen(): Promise<void> {
    if (this.fullButton.getAttribute("aria-disabled") === "true") return;
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else await this.root.requestFullscreen();
    } catch {
      this.fullscreenOff();
    }
  }

  private fullscreenOff(): void {
    this.fullButton.setAttribute("aria-disabled", "true");
    this.fullButton.setAttribute("aria-label", "Fullscreen is unavailable");
    this.fullButton.title = "Fullscreen is unavailable: the page embedding the viewer does not allow it";
  }

  private fullscreenChanged(): void {
    const on = this.root.matches(":fullscreen");
    setIcon(this.fullButton, on ? "minimize" : "maximize");
    this.fullButton.title = on ? "Leave fullscreen (Esc)" : "Fullscreen";
    this.fullButton.setAttribute("aria-label", this.fullButton.title);
  }

  /** Resolves once camera glides and box refits have ended and the last requested frame is drawn. */
  async idle(): Promise<void> {
    while (this.tween || this.refit || this.pending) await new Promise((r) => requestAnimationFrame(r));
  }

  /**
   * The current view as a PNG blob at `scale`, in full detail, with color bars, labels and gizmo, the panel if
   * asked, on a transparent background if asked. The screenshot button and Python both take it.
   */
  async screenshot(scale: number, panel: boolean, transparent = false): Promise<Blob> {
    this.settle.stop();
    await this.idle();
    this.frame();
    shared.uPixelRatio.value = scale;
    const canvas = this.renderer.snapshot(this.scene, this.camera, scale, transparent);
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
    if (this.show.gizmo && !this.cut?.unfolded) overlays.push(this.gizmo.el);
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

/** What a pick shows: Python's payload, the card's groups of lines, and the highlight in local coordinates. */
interface Inspected {
  state: PickState;
  card: [string, string][][];
  lines: number[];
  rings: number[];
  /** Width of the rim drawn under picked lines, in CSS pixels; 0 for none. */
  rim: number;
  /** Whether the lines draw over everything, as the axis of a tube does. */
  over: boolean;
}

const IDENTITY: Vec3[] = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
];

/**
 * How far mesh surfaces draw toward the camera: half the largest block side of the scene, the most a block built
 * from a surface strays from it, so the surface shows over such blocks instead of in stripes between them.
 */
function surfaceBias(layers: LayerSpec[], buffers: Buffers): number {
  let side = 0;
  for (const layer of layers) {
    if (layer.kind !== "blocks") continue;
    const sizes = f32(buffers, layer.geometry.sizes);
    for (let i = 0; i < sizes.length; i++) if (sizes[i] > side) side = sizes[i];
  }
  return side / 2;
}

const dist2 = (a: XY, b: XY) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2;

/** End points of the 12 edges of a box at `center`, `size` along each of `axes`. */
function boxEdges(center: Vec3, size: Vec3, axes: Vec3[]): Vec3[] {
  const corner = (c: number): Vec3 =>
    [0, 1, 2].map((d) => center[d] + [0, 1, 2].reduce((s, k) => s + (((c >> k) & 1) - 0.5) * size[k] * axes[k][d], 0)) as Vec3;
  const out: Vec3[] = [];
  for (let a = 0; a < 3; a++) for (let c = 0; c < 8; c++) if (!((c >> a) & 1)) out.push(corner(c), corner(c | (1 << a)));
  return out;
}

/** An accent ring of a fixed screen size around picked points, drawn over everything. */
function ringMaterial(): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    uniforms: { uAccent: shared.uAccent, uPixelRatio: shared.uPixelRatio },
    depthTest: false,
    transparent: true,
    vertexShader: /* glsl */ `
      uniform float uPixelRatio;
      void main() {
        gl_PointSize = 22.0 * uPixelRatio;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }`,
    fragmentShader: /* glsl */ `
      uniform vec3 uAccent;
      void main() {
        float r = 2.0 * length(gl_PointCoord - 0.5);
        if (r > 1.0 || r < 0.75) discard;
        gl_FragColor = vec4(uAccent, 1.0);
      }`,
  });
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

/** A length or an angle for the panel, to three significant digits. */
function formatLength(value: number): string {
  return String(Number(value.toPrecision(3)));
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
