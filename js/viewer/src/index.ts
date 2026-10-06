import { own } from "./buffers";
import type { FilterState } from "./filter";
import type { Buffers, SceneSpec } from "./types";
import { type MountOptions, Viewer } from "./viewer";

interface Model {
  get(name: "spec"): SceneSpec;
  get(name: "buffers"): Record<string, ArrayBuffer | ArrayBufferView>;
  get(name: "filters"): Record<string, FilterState> | undefined;
  set(name: "filters", value: Record<string, FilterState>): void;
  save_changes(): void;
  on(event: string, callback: () => void): void;
  off(event: string, callback: () => void): void;
}

/** anywidget entry point; the filters sync both ways with Python. */
function render({ model, el }: { model: Model; el: HTMLElement }): () => void {
  const raw = model.get("buffers");
  const buffers: Buffers = {};
  for (const [key, view] of Object.entries(raw)) buffers[key] = own(view);
  const viewer = new Viewer(el, model.get("spec"), buffers, {
    filters: model.get("filters"),
    onFilters(filters) {
      model.set("filters", filters);
      model.save_changes();
    },
  });
  const changed = () => viewer.setFilters(model.get("filters") ?? {});
  model.on("change:filters", changed);
  return () => {
    model.off("change:filters", changed);
    viewer.dispose();
  };
}

/** Standalone page entry point: buffers arrive base64-encoded. */
async function mount(
  el: HTMLElement,
  data: { spec: SceneSpec; buffers: Record<string, string> },
  options: MountOptions = {},
): Promise<Viewer> {
  const buffers: Buffers = {};
  await Promise.all(
    Object.entries(data.buffers).map(async ([key, b64]) => {
      buffers[key] = await (await fetch(`data:application/octet-stream;base64,${b64}`)).arrayBuffer();
    }),
  );
  return new Viewer(el, data.spec, buffers, options);
}

export default { render, mount };
