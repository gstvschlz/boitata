import { describe, expect, it } from "vitest";
import { keyAction, SHORTCUTS } from "../src/keys";

const press = (key: string, mods: Partial<{ ctrlKey: boolean; altKey: boolean; metaKey: boolean }> = {}) => ({
  key,
  ctrlKey: false,
  altKey: false,
  metaKey: false,
  ...mods,
});
const idle = { drawing: false, cancellable: false };
const drawing = { drawing: true, cancellable: true };

describe("key map", () => {
  it("maps each key to its action, either case", () => {
    const expected = { r: "fit", p: "plan", s: "draw", u: "unfold", x: "clear", h: "panel", t: "theme", "?": "help" };
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

  it("leaves keys with modifiers and keys typed into fields to the page", () => {
    for (const mods of [{ ctrlKey: true }, { altKey: true }, { metaKey: true }])
      expect(keyAction(press("r", mods), false, idle)).toBeNull();
    expect(keyAction(press("r"), true, idle)).toBeNull();
    expect(keyAction(press("Escape"), true, drawing)).toBeNull();
  });

  it("lists every key it maps on the help card", () => {
    const listed = SHORTCUTS.map(([key]) => key);
    for (const key of ["R", "P", "S", "U", "X", "H", "T", "?", "Enter", "Backspace", "Esc", "Shift-drag", "Shift+wheel"])
      expect(listed).toContain(key);
  });
});
