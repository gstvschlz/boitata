import { own } from "./buffers";
import type { FilterState } from "./filter";
import type { SectionState } from "./section";
import type { Buffers, SceneSpec } from "./types";
import { type MountOptions, type PickState, Viewer, type ViewState } from "./viewer";

interface Model {
  get(name: "spec"): SceneSpec;
  get(name: "buffers"): Record<string, ArrayBuffer | ArrayBufferView>;
  get(name: "filters"): Record<string, FilterState> | undefined;
  get(name: "section"): SectionState | Record<string, never> | undefined;
  get(name: "view"): ViewState | Record<string, never> | undefined;
  get(name: "key"): string | undefined;
  set(name: "filters", value: Record<string, FilterState>): void;
  set(name: "section", value: SectionState | Record<string, never>): void;
  set(name: "picked", value: PickState | Record<string, never>): void;
  set(name: "view", value: ViewState): void;
  save_changes(): void;
  on(event: string, callback: () => void): void;
  off(event: string, callback: () => void): void;
}

/**
 * anywidget entry point; the filters and the section sync both ways with Python, the pick and the view one way.
 * The host element carries the widget's key, which tells the notebook's iframe fallback that the widget rendered.
 */
function render({ model, el }: { model: Model; el: HTMLElement }): () => void {
  el.setAttribute("data-btv-key", model.get("key") ?? "");
  const raw = model.get("buffers");
  const buffers: Buffers = {};
  for (const [key, view] of Object.entries(raw)) buffers[key] = own(view);
  const view = model.get("view");
  const viewer = new Viewer(el, model.get("spec"), buffers, {
    filters: model.get("filters"),
    onFilters(filters) {
      model.set("filters", filters);
      model.save_changes();
    },
    section: model.get("section") ?? {},
    onSection(section) {
      model.set("section", section);
      model.save_changes();
    },
    onPick(picked) {
      model.set("picked", picked);
      model.save_changes();
    },
    state: view && "camera" in view ? (view as ViewState) : undefined,
    onState(state) {
      model.set("view", state);
      model.save_changes();
    },
  });
  const filters = () => viewer.setFilters(model.get("filters") ?? {});
  const section = () => viewer.setSection(model.get("section") ?? {});
  model.on("change:filters", filters);
  model.on("change:section", section);
  return () => {
    model.off("change:filters", filters);
    model.off("change:section", section);
    viewer.dispose();
  };
}

/** Standalone page entry point: buffers arrive base64-encoded, with the view a widget last had if any. */
async function mount(
  el: HTMLElement,
  data: { spec: SceneSpec; buffers: Record<string, string>; state?: ViewState },
  options: MountOptions = {},
): Promise<Viewer> {
  const buffers: Buffers = {};
  await Promise.all(
    Object.entries(data.buffers).map(async ([key, b64]) => {
      buffers[key] = await (await fetch(`data:application/octet-stream;base64,${b64}`)).arrayBuffer();
    }),
  );
  return new Viewer(el, data.spec, buffers, { state: data.state, ...options });
}

export default { render, mount };
