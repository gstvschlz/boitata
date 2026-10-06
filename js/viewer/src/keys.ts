/** What a key does in the viewer. */
export type Action = "fit" | "plan" | "draw" | "finish" | "undo" | "lock" | "unfold" | "clear" | "panel" | "theme" | "cancel" | "help";

/** The state a key's meaning depends on. */
export interface KeyState {
  /** A polyline is being drawn. */
  drawing: boolean;
  /** Something Esc closes or cancels: a drawing, a quick cut, the pick card, the help card. */
  cancellable: boolean;
}

const ALWAYS: Record<string, Action> = {
  r: "fit",
  p: "plan",
  s: "draw",
  u: "unfold",
  x: "clear",
  h: "panel",
  t: "theme",
  "?": "help",
};

const DRAWING: Record<string, Action> = { enter: "finish", backspace: "undo", shift: "lock" };

/**
 * The action of a key pressed while the viewer has focus; null leaves the key to the page. Keys with Ctrl, Alt or
 * Meta, and keys typed into a field, are never the viewer's.
 */
export function keyAction(
  e: { key: string; ctrlKey: boolean; altKey: boolean; metaKey: boolean },
  inField: boolean,
  state: KeyState,
): Action | null {
  if (e.ctrlKey || e.altKey || e.metaKey || inField) return null;
  const key = e.key.toLowerCase();
  if (key === "escape") return state.cancellable ? "cancel" : null;
  return ALWAYS[key] ?? (state.drawing ? (DRAWING[key] ?? null) : null);
}

/** The keys the help card lists, in its order. */
export const SHORTCUTS: [string, string][] = [
  ["R", "Fit the view"],
  ["P", "Plan view, north up"],
  ["S", "Draw a section"],
  ["Shift-drag", "Straight cut; while drawing, lock 45°"],
  ["Shift+wheel", "Section width"],
  ["Enter", "Finish the section"],
  ["Backspace", "Undo the last vertex"],
  ["U", "Unfold or fold back"],
  ["X", "Clear the section"],
  ["H", "Hide or show the panel"],
  ["T", "Light or dark"],
  ["Esc", "Cancel, or close a card"],
  ["?", "These shortcuts"],
];
