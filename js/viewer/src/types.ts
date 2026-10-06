import type { FilterState } from "./filter";
import type { SectionState } from "./section";
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
  /** Sprite diameter in CSS pixels; null for the representation's default. */
  pointSize?: number | null;
  /** Line width in CSS pixels; null for the representation's default. */
  lineWidth?: number | null;
  /** Tube or sphere radius in meters; null for a share of the layer's extent. */
  radius?: number | null;
  /** Buffer keys of the geometry, by role (positions, triangles, rows, midpoints, centers, sizes). */
  geometry: Record<string, string>;
  axes?: number[][];
  /** Drill holes: the column naming the hole and the interval columns; the depths of each row are in `depths`. */
  holes?: { hole: string; from: string | null; to: string | null };
  columns: ColumnSpec[];
  /** Initial filter: conditions on columns, combined with "and". */
  filter?: FilterState;
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
  /** What draws while the camera moves: "auto", "full", or a fixed fraction of each layer's instances. */
  motion?: "auto" | "full" | number;
  variables: Record<string, VariableSpec>;
  layers: LayerSpec[];
  /** Section to start with, in real-world coordinates; empty for none. */
  section?: SectionState | Record<string, never>;
}

export type Buffers = Record<string, ArrayBuffer>;
