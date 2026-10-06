import type { ThemeSpec } from "./theme";

export type Kind = "drillholes" | "points" | "mesh" | "blocks";

export interface ColumnSpec {
  name: string;
  type: "number" | "text";
  /** Buffer key: float32 with NaN for nulls, or int32 category codes with -1 for nulls. */
  buffer: string;
  /** What a value belongs to: a row (point, block, interval), a mesh vertex or a mesh face. */
  on: "row" | "vertex" | "face";
  min?: number;
  max?: number;
  categories?: string[];
}

export interface LayerSpec {
  id: string;
  name: string;
  kind: Kind;
  representation: string;
  color: string | null;
  opacity: number;
  visible: boolean;
  values: string | null;
  /** Buffer keys of the geometry, by role (positions, triangles, rows, centers, sizes) plus block axes. */
  geometry: Record<string, string>;
  axes?: number[][];
  columns: ColumnSpec[];
}

export interface VariableSpec {
  cmap?: string | null;
  clim?: [number, number] | null;
  label?: string | null;
}

export interface SceneSpec {
  origin: [number, number, number];
  axes: [string, string, string];
  theme: ThemeSpec;
  height: number;
  view: { azimuth: number; dip: number };
  variables: Record<string, VariableSpec>;
  layers: LayerSpec[];
}

export type Buffers = Record<string, ArrayBuffer>;
