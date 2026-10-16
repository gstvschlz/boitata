import { describe, expect, it } from "vitest";
import { keyAction, SHORTCUTS } from "../src/keys";

const press = (key: string, mods: Partial<{ shiftKey: boolean; ctrlKey: boolean; altKey: boolean; metaKey: boolean }> = {}) => ({
  key,
  shiftKey: false,
  ctrlKey: false,
  altKey: false,
  metaKey: false,
  ...mods,
});
const idle = { drawing: false, cancellable: false };
const drawing = { drawing: true, cancellable: true };

describe("key map", () => {
  it("maps each key to its action, either case", () => {
    const expected = { r: "fit", p: "plan", s: "draw", u: "unfold", x: "clear", h: "panel", t: "theme", e: "edl", o: "ortho", "?": "help" };
    for (const [key, action] of Object.entries(expected)) {
      expect(keyAction(press(key), false, idle)).toBe(action);
      expect(keyAction(press(key.toUpperCase()), false, idle)).toBe(action);
    }
  });

  it("has no F, D or C binding", () => {
    for (const key of ["f", "d", "c"]) expect(keyAction(press(key), false, idle)).toBeNull();
  });

  it("finishes, undoes and locks 45 degrees only while drawing", () => {
    expect(keyAction(press("Enter"), false, drawing)).toBe("finish");
    expect(keyAction(press("Backspace"), false, drawing)).toBe("undo");
    expect(keyAction(press("Shift"), false, drawing)).toBe("lock");
    for (const key of ["Enter", "Backspace", "Shift"]) expect(keyAction(press(key), false, idle)).toBeNull();
  });

  it("cancels with Esc only when there is something to cancel or close", () => {
    expect(keyAction(press("Escape"), false, { drawing: false, cancellable: true })).toBe("cancel");
    expect(keyAction(press("Escape"), false, idle)).toBeNull();
  });

  it("orbits with the arrows, pans with Shift and zooms with Page Up and Down", () => {
    for (const [key, arrow] of [["ArrowLeft", "left"], ["ArrowRight", "right"], ["ArrowUp", "up"], ["ArrowDown", "down"]]) {
      expect(keyAction(press(key), false, idle)).toBe(`orbit-${arrow}`);
      expect(keyAction(press(key, { shiftKey: true }), false, idle)).toBe(`pan-${arrow}`);
    }
    expect(keyAction(press("PageUp"), false, idle)).toBe("zoomIn");
    expect(keyAction(press("PageDown"), false, idle)).toBe("zoomOut");
  });

  it("turns to the six views with the number keys laid out as a compass", () => {
    const expected = { "8": "north", "2": "south", "4": "west", "6": "east", "5": "up", "0": "down" };
    for (const [key, view] of Object.entries(expected)) expect(keyAction(press(key), false, idle)).toBe(`view-${view}`);
    for (const key of ["1", "3", "7", "9"]) expect(keyAction(press(key), false, idle)).toBeNull();
  });

  it("keeps the camera in plan while drawing: it pans and zooms but does not orbit or change view", () => {
    expect(keyAction(press("ArrowLeft"), false, drawing)).toBeNull();
    expect(keyAction(press("8"), false, drawing)).toBeNull();
    expect(keyAction(press("ArrowLeft", { shiftKey: true }), false, drawing)).toBe("pan-left");
    expect(keyAction(press("PageUp"), false, drawing)).toBe("zoomIn");
  });

  it("leaves the arrows and Page Up and Down to the panel when it has focus", () => {
    const panel = { ...idle, inPanel: true };
    for (const key of ["ArrowLeft", "ArrowUp", "PageUp", "PageDown"]) expect(keyAction(press(key), false, panel)).toBeNull();
    expect(keyAction(press("r"), false, panel)).toBe("fit");
  });

  it("leaves keys with modifiers and keys typed into fields to the page", () => {
    for (const mods of [{ ctrlKey: true }, { altKey: true }, { metaKey: true }])
      expect(keyAction(press("r", mods), false, idle)).toBeNull();
    expect(keyAction(press("r"), true, idle)).toBeNull();
    expect(keyAction(press("Escape"), true, drawing)).toBeNull();
  });

  it("lists every key it maps on the help card", () => {
    const listed = SHORTCUTS.map(([key]) => key);
    for (const key of ["R", "P", "S", "U", "X", "H", "T", "E", "O", "?", "Enter", "Backspace", "Esc", "Shift-drag", "Shift+wheel", "Arrows", "Shift+arrows", "PgUp PgDn", "8 2 4 6", "5 0", "Double-click"])
      expect(listed).toContain(key);
  });
});
