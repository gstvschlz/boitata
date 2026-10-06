import type { Buffers } from "./types";

function get(buffers: Buffers, key: string): ArrayBuffer {
  const b = buffers[key];
  if (!b) throw new Error(`missing buffer ${key}`);
  return b;
}

export const f32 = (buffers: Buffers, key: string) => new Float32Array(get(buffers, key));
export const u32 = (buffers: Buffers, key: string) => new Uint32Array(get(buffers, key));
export const i32 = (buffers: Buffers, key: string) => new Int32Array(get(buffers, key));

/** Copies a view (an anywidget DataView, a typed array) into an aligned ArrayBuffer of its own. */
export function own(view: ArrayBuffer | ArrayBufferView): ArrayBuffer {
  if (view instanceof ArrayBuffer) return view;
  const out = new ArrayBuffer(view.byteLength);
  new Uint8Array(out).set(new Uint8Array(view.buffer as ArrayBuffer, view.byteOffset, view.byteLength));
  return out;
}
