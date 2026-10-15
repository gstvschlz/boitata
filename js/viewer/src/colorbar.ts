import { categoryColor, logRange, lut, type Mapping, mapPosition, rgbToHex } from "./color";
import type { Theme } from "./theme";
import { formatLogTick, formatTicks, logTicks, niceTicks } from "./ticks";

export interface BarSpec extends Mapping {
  title: string;
  cmap: string;
  /** Categories, for a text variable. */
  categories: string[] | null;
  /** Category colors, index-matched to `categories`; null to use the built-in palette. */
  colors: string[] | null;
}

const FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif";
const W = 220;
const MAX_CATEGORIES = 12;

const esc = (s: string) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!);

/** Color bar as standalone SVG markup, so screenshots can draw it as an image. */
export function colorBarSvg(bar: BarSpec, theme: Theme, id: string): string {
  const text = (x: number, y: number, s: string, extra = "") =>
    `<text x="${x}" y="${y}" font-family="${FONT}" font-size="11" fill="${theme.muted}" ${extra}>${esc(s)}</text>`;
  const title = `<text x="10" y="18" font-family="${FONT}" font-size="12" font-weight="600" fill="${theme.text}">${esc(bar.title)}</text>`;
  let body: string;
  let height: number;
  if (bar.categories) {
    const shown = bar.categories.slice(0, MAX_CATEGORIES);
    body = shown
      .map((c, i) => {
        const y = 28 + i * 17;
        const fill = bar.colors?.[i] ?? rgbToHex(categoryColor(i));
        return `<rect x="10" y="${y}" width="11" height="11" rx="2" fill="${fill}"/>` + text(27, y + 9.5, c);
      })
      .join("");
    const more = bar.categories.length - shown.length;
    height = 34 + shown.length * 17 + (more > 0 ? 14 : 0);
    if (more > 0) body += text(10, height - 8, `+${more} more`);
  } else {
    const table = lut(bar.cmap);
    const hex = (t: number) => {
      const i = Math.min(255, Math.floor(t * 256));
      return rgbToHex([table[3 * i], table[3 * i + 1], table[3 * i + 2]]);
    };
    const x0 = 10;
    const x1 = W - 10;
    const along = (t: number) => x0 + t * (x1 - x0);
    const tick = (x: number, label: string) =>
      `<line x1="${x}" x2="${x}" y1="40" y2="44" stroke="${theme.muted}"/>` + text(x, 56, label, 'text-anchor="middle"');
    const cuts = bar.bins ?? [];
    if (cuts.length) {
      const k = cuts.length + 1;
      body =
        Array.from({ length: k }, (_, c) => {
          const fill = hex(mapPosition(c === 0 ? -Infinity : cuts[c - 1], bar));
          return `<rect x="${along(c / k)}" y="28" width="${(x1 - x0) / k}" height="12" fill="${fill}"/>`;
        }).join("") + cuts.map((c, i) => tick(along((i + 1) / k), formatLogTick(c))).join("");
    } else {
      const stops = Array.from({ length: 33 }, (_, k) => {
        const t = k / 32;
        return `<stop offset="${t.toFixed(4)}" stop-color="${hex(bar.reverse ? 1 - t : t)}"/>`;
      }).join("");
      const [lo, hi] = bar.log ? logRange(bar.lo, bar.hi) : [bar.lo, bar.hi];
      const ticks = bar.log ? logTicks(lo, hi) : niceTicks(lo, hi, 4).filter((t) => t >= lo && t <= hi);
      const labels = bar.log ? ticks.map(formatLogTick) : formatTicks(ticks);
      const plain = { lo, hi, log: bar.log };
      body =
        `<defs><linearGradient id="${id}">${stops}</linearGradient></defs>` +
        `<rect x="${x0}" y="28" width="${x1 - x0}" height="12" fill="url(#${id})"/>` +
        ticks.map((t, i) => tick(along(mapPosition(t, plain)), labels[i])).join("");
    }
    height = 64;
  }
  const bg = `<rect x="0.5" y="0.5" width="${W - 1}" height="${height - 1}" rx="5" fill="${theme.panel}" fill-opacity="0.88" stroke="${theme.border}"/>`;
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${height}" viewBox="0 0 ${W} ${height}">${bg}${title}${body}</svg>`;
}
