import type { Snap } from "./gizmo";

export type Arrow = "left" | "right" | "up" | "down";

/** What a key does in the viewer. */
export type Action =
  | "fit"
  | "plan"
  | "draw"
  | "finish"
  | "undo"
  | "lock"
  | "unfold"
  | "clear"
  | "panel"
  | "theme"
  | "cancel"
  | "help"
  | "edl"
  | "ortho"
  | "measure"
  | "forward"
  | "back"
  | "zoomIn"
  | "zoomOut"
  | `orbit-${Arrow}`
  | `pan-${Arrow}`
  | `view-${Snap}`;

/** The state a key's meaning depends on. */
export interface KeyState {
  /** A polyline is being drawn. */
  drawing: boolean;
  /** Something Esc closes or cancels: a drawing, a quick cut, the pick card, the help card. */
  cancellable: boolean;
  /** Focus is in the panel, where arrows and Page Up/Down are left to it. */
  inPanel?: boolean;
}

const ALWAYS: Record<string, Action> = {
  r: "fit",
  p: "plan",
  s: "draw",
  u: "unfold",
  x: "clear",
  h: "panel",
  t: "theme",
  e: "edl",
  o: "ortho",
  m: "measure",
  "]": "forward",
  "[": "back",
  "?": "help",
};

const DRAWING: Record<string, Action> = { enter: "finish", backspace: "undo", shift: "lock" };

const ARROWS: Record<string, Arrow> = { arrowleft: "left", arrowright: "right", arrowup: "up", arrowdown: "down" };

/** Number keys laid out as a compass on the keypad: 8 north, 2 south, 4 west, 6 east, 5 down from above, 0 from below. */
const VIEWS: Record<string, Snap> = { "8": "north", "2": "south", "4": "west", "6": "east", "5": "up", "0": "down" };

/**
 * The action of a key pressed while the viewer has focus; null leaves the key to the page. Keys with Ctrl, Alt or
 * Meta, and keys typed into a field, are never the viewer's. While drawing, the camera keeps to plan: it pans and
 * zooms but does not orbit or turn to another view.
 */
export function keyAction(
  e: { key: string; shiftKey?: boolean; ctrlKey: boolean; altKey: boolean; metaKey: boolean },
  inField: boolean,
  state: KeyState,
): Action | null {
  if (e.ctrlKey || e.altKey || e.metaKey || inField) return null;
  const key = e.key.toLowerCase();
  if (key === "escape") return state.cancellable ? "cancel" : null;
  const arrow = ARROWS[key];
  if (arrow || key === "pageup" || key === "pagedown") {
    if (state.inPanel) return null;
    if (key === "pageup") return "zoomIn";
    if (key === "pagedown") return "zoomOut";
    return e.shiftKey ? `pan-${arrow}` : state.drawing ? null : `orbit-${arrow}`;
  }
  if (key in VIEWS) return state.drawing ? null : `view-${VIEWS[key]}`;
  return ALWAYS[key] ?? (state.drawing ? (DRAWING[key] ?? null) : null);
}

/** The keys the help card lists, in its order. */
export const SHORTCUTS: [string, string][] = [
  ["R", "Fit the view"],
  ["Double-click", "Orbit around that point"],
  ["Arrows", "Orbit the camera"],
  ["Shift+arrows", "Pan"],
  ["PgUp PgDn", "Zoom in or out"],
  ["P", "Plan view, north up"],
  ["8 2 4 6", "View from the north, south, west or east"],
  ["5 0", "View from above or below"],
  ["O", "Orthographic or perspective"],
  ["S", "Draw a section"],
  ["Shift-drag", "Straight cut; while drawing, lock 45°"],
  ["Shift+wheel", "Section width"],
  ["Enter", "Finish the section"],
  ["Backspace", "Undo the last vertex"],
  ["U", "Unfold or fold back"],
  ["X", "Clear the section"],
  ["[ ]", "Step the section back or forward"],
  ["H", "Hide or show the panel"],
  ["T", "Light or dark"],
  ["E", "Eye-dome lighting"],
  ["M", "Measure between two clicks"],
  ["Esc", "Cancel, or close a card"],
  ["?", "These shortcuts"],
];
