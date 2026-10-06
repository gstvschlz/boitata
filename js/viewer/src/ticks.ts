/** Step of about `span / target` rounded to 1, 2 or 5 times a power of ten. */
export function niceStep(span: number, target = 5): number {
  if (!(span > 0) || !Number.isFinite(span)) return 1;
  const raw = span / target;
  const power = 10 ** Math.floor(Math.log10(raw));
  const f = raw / power;
  return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * power;
}

/** Multiples of a nice step inside `[lo, hi]`. */
export function niceTicks(lo: number, hi: number, target = 5): number[] {
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [];
  if (hi < lo) [lo, hi] = [hi, lo];
  if (hi === lo) return [lo];
  const step = niceStep(hi - lo, target);
  const out: number[] = [];
  for (let k = Math.ceil(lo / step - 1e-9); k * step <= hi + step * 1e-9; k++) {
    out.push(Number((k * step).toPrecision(12)));
  }
  return out;
}

/** Tick label with as many decimals as `step` needs. */
export function formatTick(value: number, step: number): string {
  const decimals = step > 0 ? Math.max(0, -Math.floor(Math.log10(step) + 1e-9)) : 0;
  const text = value.toFixed(Math.min(decimals, 12));
  return /^-0(\.0*)?$/.test(text) ? text.slice(1) : text;
}

/** Labels for `ticks`, decimals from their spacing. */
export function formatTicks(ticks: number[]): string[] {
  const step = ticks.length > 1 ? ticks[1] - ticks[0] : Math.abs(ticks[0] ?? 1) || 1;
  return ticks.map((t) => formatTick(t, step));
}
