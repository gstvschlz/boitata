import { own } from "./buffers";
import type { FilterState } from "./filter";
import type { SectionState } from "./section";
import type { Buffers, SceneSpec } from "./types";
import { type MountOptions, type PickState, Viewer } from "./viewer";

/** A screenshot Python asks the widget for; the answer carries the PNG as a binary buffer. */
interface ShotRequest {
  type: "screenshot";
  id: number;
  scale: number;
  panel: boolean;
  transparent: boolean;
}

interface Model {
  get(name: "spec"): SceneSpec;
  get(name: "buffers"): Record<string, ArrayBuffer | ArrayBufferView>;
  get(name: "filters"): Record<string, FilterState> | undefined;
  get(name: "section"): SectionState | Record<string, never> | undefined;
  set(name: "filters", value: Record<string, FilterState>): void;
  set(name: "section", value: SectionState | Record<string, never>): void;
  set(name: "picked", value: PickState | Record<string, never>): void;
  save_changes(): void;
  send(content: object, callbacks?: unknown, buffers?: ArrayBuffer[]): void;
  on(event: "msg:custom", callback: (msg: ShotRequest) => void): void;
  on(event: string, callback: () => void): void;
  off(event: "msg:custom", callback: (msg: ShotRequest) => void): void;
  off(event: string, callback: () => void): void;
}

/**
 * anywidget entry point; the filters and the section sync both ways with Python, the pick one way. The view tells
 * Python it is shown, so screenshots ask it, and answers their requests through `Viewer.screenshot`.
 */
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
    section: model.get("section") ?? {},
    onSection(section) {
      model.set("section", section);
      model.save_changes();
    },
    onPick(picked) {
      model.set("picked", picked);
      model.save_changes();
    },
  });
  const filters = () => viewer.setFilters(model.get("filters") ?? {});
  const section = () => viewer.setSection(model.get("section") ?? {});
  const message = (msg: ShotRequest) => {
    if (msg?.type !== "screenshot") return;
    viewer
      .screenshot(msg.scale, msg.panel, msg.transparent)
      .then(async (blob) => model.send({ type: "screenshot", id: msg.id }, undefined, [await blob.arrayBuffer()]))
      .catch((e: unknown) => model.send({ type: "screenshot", id: msg.id, error: String(e) }));
  };
  model.on("change:filters", filters);
  model.on("change:section", section);
  model.on("msg:custom", message);
  model.send({ type: "view", shown: true });
  return () => {
    model.off("change:filters", filters);
    model.off("change:section", section);
    model.off("msg:custom", message);
    model.send({ type: "view", shown: false });
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
