export interface Theme {
  background: string;
  panel: string;
  raised: string;
  border: string;
  text: string;
  muted: string;
  grid: string;
  box: string;
  accent: string;
  layer: string;
  halo: string;
  dark: boolean;
}

export type ThemeName = "light" | "dark";
export type ThemeSpec = "auto" | ThemeName | ({ base?: "auto" | ThemeName } & Partial<Record<keyof Theme, string>>);

export const THEMES: Record<ThemeName, Theme> = {
  light: {
    background: "#FFFFFF",
    panel: "#FFFFFF",
    raised: "#F4F6F8",
    border: "#D5DDE3",
    text: "#17303F",
    muted: "#5A6B77",
    grid: "#D5DDE3",
    box: "#5A6B77",
    accent: "#C4411F",
    layer: "#5A6B77",
    halo: "#3D4D58",
    dark: false,
  },
  dark: {
    background: "#0B151C",
    panel: "#111E27",
    raised: "#172631",
    border: "#24353F",
    text: "#E4EAEE",
    muted: "#93A3AE",
    grid: "#24353F",
    box: "#5E707C",
    accent: "#FF6A4D",
    layer: "#93A3AE",
    halo: "#B8C4CC",
    dark: true,
  },
};

export const THEME_KEYS = Object.keys(THEMES.light).filter((k) => k !== "dark") as (keyof Theme)[];

/** Theme to draw with: a built-in one, picked by the host for "auto", with a dict's overrides applied. */
export function resolveTheme(spec: ThemeSpec | undefined, hostDark: boolean): Theme {
  const pick = (name: "auto" | ThemeName | undefined): Theme =>
    ({ ...THEMES[name === "dark" || name === "light" ? name : hostDark ? "dark" : "light"] });
  if (spec === undefined || typeof spec === "string") return pick(spec);
  const theme = pick(spec.base);
  for (const key of THEME_KEYS) {
    const value = spec[key];
    if (typeof value === "string") (theme as unknown as Record<string, string>)[key] = value;
  }
  return theme;
}

/** The base a spec follows, "auto" when it tracks the host. */
export function themeBase(spec: ThemeSpec | undefined): "auto" | ThemeName {
  if (spec === undefined) return "auto";
  return typeof spec === "string" ? spec : spec.base ?? "auto";
}

/** Same spec with its base switched, keeping a dict's overrides. */
export function withBase(spec: ThemeSpec | undefined, base: ThemeName): ThemeSpec {
  return spec !== undefined && typeof spec === "object" ? { ...spec, base } : base;
}
