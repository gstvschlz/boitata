/** What draws while the camera moves: "auto" adapts to the frame time, "fast" a fixed fraction, "full" everything. */
export type Quality = "auto" | "fast" | "full";

export const SETTLE_MS = 150;
/** Frame time "auto" aims for while moving. */
export const TARGET_MS = 33;
/** Instances "auto" starts from, before any frame is measured. */
export const BUDGET = 250_000;
/** A layer this small always draws in full. */
export const FLOOR = 2_000;
export const MIN_FRACTION = 0.02;
export const FAST_FRACTION = 0.25;

/**
 * Permutation of `n` points (xyz triples) whose every prefix is spread evenly over space: the points are sorted
 * along a Morton curve and read in bit-reversed rank, so the first 2^k take every (n / 2^k)-th point along it.
 */
export function stratifiedOrder(xyz: ArrayLike<number>, n: number): Uint32Array {
  const order = new Uint32Array(n);
  if (n < 2) return order;
  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < n; i++)
    for (let a = 0; a < 3; a++) {
      const v = xyz[3 * i + a];
      if (v < lo[a]) lo[a] = v;
      if (v > hi[a]) hi[a] = v;
    }
  const indexBits = Math.ceil(Math.log2(n));
  const bits = Math.max(1, Math.min(10, Math.floor((52 - indexBits) / 3)));
  const cells = 2 ** bits;
  const size = 2 ** indexBits;
  const cell = (v: number, a: number) => {
    const t = hi[a] > lo[a] ? (v - lo[a]) / (hi[a] - lo[a]) : 0;
    return t >= 0 ? Math.min(cells - 1, Math.floor(t * cells)) : 0;
  };
  const keys = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const x = cell(xyz[3 * i], 0);
    const y = cell(xyz[3 * i + 1], 1);
    const z = cell(xyz[3 * i + 2], 2);
    let code = 0;
    for (let b = bits - 1; b >= 0; b--) code = code * 8 + ((x >> b) & 1) * 4 + ((y >> b) & 1) * 2 + ((z >> b) & 1);
    keys[i] = code * size + i;
  }
  keys.sort();
  let j = 0;
  for (let k = 0; k < size; k++) {
    let r = 0;
    for (let b = 0, x = k; b < indexBits; b++, x >>= 1) r = (r << 1) | (x & 1);
    if (r < n) order[j++] = keys[r] % size;
  }
  return order;
}

/** Centers of segments given as flat pairs of xyz ends. */
export function midpoints(ends: Float32Array): Float32Array {
  const n = ends.length / 6;
  const out = new Float32Array(3 * n);
  for (let i = 0; i < n; i++)
    for (let a = 0; a < 3; a++) out[3 * i + a] = (ends[6 * i + a] + ends[6 * i + 3 + a]) / 2;
  return out;
}

/** Fraction of each layer's instances drawn while moving. */
export function motionFraction(quality: Quality, fast: number, auto: number): number {
  return quality === "full" ? 1 : quality === "fast" ? fast : auto;
}

/** Instances drawn of `drawn` at `fraction`: small layers stay whole, the rest keep a prefix of their order. */
export function subsetCount(drawn: number, fraction: number): number {
  if (fraction >= 1 || drawn <= FLOOR) return drawn;
  return Math.max(FLOOR, Math.ceil(drawn * fraction));
}

/** "auto"'s first guess: the fraction that fits `instances` into the budget. */
export function startFraction(instances: number): number {
  return Math.max(MIN_FRACTION, Math.min(1, BUDGET / Math.max(instances, 1)));
}

/** Next "auto" fraction from the last frame time: shrinks when slow, grows back when there is room. */
export function adapt(fraction: number, frameMs: number): number {
  if (!(frameMs > 0)) return fraction;
  let next = fraction;
  if (frameMs > TARGET_MS * 1.2) next = fraction * Math.max(0.5, TARGET_MS / frameMs);
  else if (frameMs < TARGET_MS * 0.6) next = fraction * 1.25;
  return Math.max(MIN_FRACTION, Math.min(1, next));
}

/** Pixel ratio while moving: the pixel count drops with the instance fraction, to half the resolution at most. */
export function motionPixelRatio(devicePixelRatio: number, fraction: number): number {
  return fraction >= 1 ? devicePixelRatio : devicePixelRatio * Math.max(0.5, Math.sqrt(fraction));
}

/** Tracks whether the camera is moving: true on the first poke, false `delay` ms after the last. */
export class Settle {
  moving = false;
  private timer: ReturnType<typeof setTimeout> | undefined;

  constructor(
    private onChange: (moving: boolean) => void,
    public delay = SETTLE_MS,
  ) {}

  poke(): void {
    if (!this.moving) {
      this.moving = true;
      this.onChange(true);
    }
    clearTimeout(this.timer);
    this.timer = setTimeout(() => this.stop(), this.delay);
  }

  stop(): void {
    clearTimeout(this.timer);
    if (!this.moving) return;
    this.moving = false;
    this.onChange(false);
  }
}
