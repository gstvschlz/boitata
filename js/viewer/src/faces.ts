/** All six box sides, one bit each: +x, -x, +y, -y, +z, -z along the model's axes. */
export const ALL_SIDES = 63;
/** Grids larger than this draw as whole boxes: the occupancy takes a byte per cell. */
export const MAX_CELLS = 1 << 26;

/** How a block takes part: not drawn, drawn with all its sides (cut by a section, translucent), or solid. */
export const HIDDEN = 0;
export const WHOLE = 1;
export const SOLID = 2;

/** A regular block model: cells per axis and each row's cell, `i + nx * (j + ny * k)`. */
export interface Grid {
  count: [number, number, number];
  cells: Uint32Array;
}

/**
 * Per row, the sides that show: those whose neighbour across is not a solid block. A solid block hides the side
 * of each solid neighbour, so a solid model draws only its outer shell; a whole block shows every side. 0 for a
 * hidden row. `occupied` is scratch of one byte per cell, all zero, and is left all zero.
 */
export function exposedSides(grid: Grid, state: Uint8Array, occupied: Uint8Array = new Uint8Array(cellCount(grid))): Uint8Array {
  const [nx, ny, nz] = grid.count;
  const { cells } = grid;
  const n = cells.length;
  const plane = nx * ny;
  for (let r = 0; r < n; r++) if (state[r] === SOLID) occupied[cells[r]] = 1;
  const out = new Uint8Array(n);
  for (let r = 0; r < n; r++) {
    const s = state[r];
    if (s === HIDDEN) continue;
    if (s === WHOLE) {
      out[r] = ALL_SIDES;
      continue;
    }
    const c = cells[r];
    const i = c % nx;
    const j = Math.floor(c / nx) % ny;
    const k = Math.floor(c / plane);
    let m = 0;
    if (i + 1 >= nx || !occupied[c + 1]) m |= 1;
    if (i === 0 || !occupied[c - 1]) m |= 2;
    if (j + 1 >= ny || !occupied[c + nx]) m |= 4;
    if (j === 0 || !occupied[c - nx]) m |= 8;
    if (k + 1 >= nz || !occupied[c + plane]) m |= 16;
    if (k === 0 || !occupied[c - plane]) m |= 32;
    out[r] = m;
  }
  for (let r = 0; r < n; r++) occupied[cells[r]] = 0;
  return out;
}

export const cellCount = (grid: Grid) => grid.count[0] * grid.count[1] * grid.count[2];

/** Sides drawn in all, which is the number of quads the GPU rasterizes. */
export function sideCount(sides: Uint8Array): number {
  let total = 0;
  for (let r = 0; r < sides.length; r++) for (let m = sides[r]; m; m &= m - 1) total++;
  return total;
}
