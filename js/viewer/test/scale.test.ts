import { describe, expect, it } from "vitest";
import { reconcileCategories } from "../src/viewer";
import type { ColumnSpec } from "../src/types";

const col = (categories: string[], colors?: string[]): ColumnSpec => ({
  name: "class",
  type: "text",
  buffer: "b",
  on: "row",
  categories,
  ...(colors ? { colors } : {}),
});

describe("reconcileCategories", () => {
  it("keeps the given order and drops nothing when every layer agrees", () => {
    const result = reconcileCategories([col(["b", "a"]), col(["b", "a"])]);
    expect(result).toEqual({ categories: ["b", "a"], colors: null });
  });

  it("uses the one scheme's order and colors when every colored column shares it", () => {
    const scheme = col(["measured", "indicated", "inferred"], ["#f00", "#0f0", "#00f"]);
    const result = reconcileCategories([scheme, scheme]);
    expect(result).toEqual({ categories: ["measured", "indicated", "inferred"], colors: ["#f00", "#0f0", "#00f"] });
  });

  it("falls back to no colors when layers disagree on the scheme", () => {
    const a = col(["x", "y"], ["#f00", "#0f0"]);
    const b = col(["y", "x"], ["#00f", "#ff0"]);
    const result = reconcileCategories([a, b]);
    expect(result.colors).toBeNull();
  });
});
