import type * as THREE from "three";
import type { Box } from "../bounds";
import type { Paint } from "../paint";
import type { Buffers, LayerSpec } from "../types";
import { cells } from "./cells";
import { lines } from "./lines";
import { points } from "./points";
import { surface } from "./surface";

/** How one layer draws; each kind and representation pair registers a factory. */
export interface Representation {
  object: THREE.Object3D;
  /** Bounds of all its geometry, in local coordinates. */
  box: Box | null;
  paint(paint: Paint): void;
  setOpacity(opacity: number): void;
  /** Called before each frame with the canvas size in CSS pixels. */
  frame?(width: number, height: number): void;
  dispose(): void;
}

export type Factory = (layer: LayerSpec, buffers: Buffers) => Representation;

export const REPRESENTATIONS: Record<string, Factory> = {
  "blocks:cells": cells,
  "drillholes:lines": lines,
  "mesh:surface": surface,
  "points:points": points,
};

export function represent(layer: LayerSpec, buffers: Buffers): Representation {
  const factory = REPRESENTATIONS[`${layer.kind}:${layer.representation}`];
  if (!factory) throw new Error(`no representation ${layer.representation} for ${layer.kind}`);
  return factory(layer, buffers);
}
