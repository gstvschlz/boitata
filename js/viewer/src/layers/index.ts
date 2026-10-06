import type * as THREE from "three";
import type { Box } from "../bounds";
import type { LayerFilter } from "../filter";
import type { Paint } from "../paint";
import type { Buffers, Kind, LayerSpec } from "../types";
import { blockWireframe, cells } from "./cells";
import { lines, tubes } from "./holes";
import { blockPoints, holePoints, meshPoints, points } from "./points";
import { spheres } from "./spheres";
import { meshWireframe, surface } from "./surface";

/** How one layer draws; each kind and representation pair registers a factory. */
export interface Representation {
  object: THREE.Object3D;
  /** Bounds of all its geometry, in local coordinates. */
  box: Box | null;
  /** Instances it draws in full, which the motion budget counts. */
  instances: number;
  /** Fills its buffers from the paint and the filter's columns; a change of bounds or categories needs no repaint. */
  paint(paint: Paint): void;
  setOpacity(opacity: number): void;
  /** Draws a spatially even `fraction` of its instances while the camera moves; 1 draws them all. */
  detail?(fraction: number): void;
  /** Called before each frame with the canvas size in CSS pixels. */
  frame?(width: number, height: number): void;
  dispose(): void;
}

export type Factory = (layer: LayerSpec, buffers: Buffers, filter: LayerFilter) => Representation;

export const REPRESENTATIONS: Record<string, Factory> = {
  "drillholes:lines": lines,
  "drillholes:tubes": tubes,
  "drillholes:points": holePoints,
  "points:points": points,
  "points:spheres": spheres,
  "mesh:surface": surface,
  "mesh:wireframe": meshWireframe,
  "mesh:points": meshPoints,
  "blocks:cells": cells,
  "blocks:wireframe": blockWireframe,
  "blocks:points": blockPoints,
};

/** Representation names of a kind, the default first. */
export function representations(kind: Kind): string[] {
  return Object.keys(REPRESENTATIONS)
    .filter((key) => key.startsWith(`${kind}:`))
    .map((key) => key.slice(kind.length + 1));
}

export function represent(layer: LayerSpec, buffers: Buffers, filter: LayerFilter): Representation {
  const factory = REPRESENTATIONS[`${layer.kind}:${layer.representation}`];
  if (!factory) throw new Error(`no representation ${layer.representation} for ${layer.kind}`);
  return factory(layer, buffers, filter);
}
