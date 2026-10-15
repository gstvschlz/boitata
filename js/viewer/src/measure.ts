import type { Vec3 } from "./bounds";
import type { Theme } from "./theme";
import { niceStep } from "./ticks";

/** From `a` to `b`: straight and plan distances, azimuth clockwise from north, dip in degrees positive down. */
export interface Measure {
  distance: number;
  horizontal: number;
  azimuth: number;
  dip: number;
}

export function measure(a: Vec3, b: Vec3): Measure {
  const [dx, dy, dz] = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
  const horizontal = Math.hypot(dx, dy);
  const azimuth = horizontal > 0 ? ((Math.atan2(dx, dy) * 180) / Math.PI + 360) % 360 : 0;
  return { distance: Math.hypot(horizontal, dz), horizontal, azimuth, dip: (Math.atan2(-dz, horizontal) * 180) / Math.PI };
}

/** A round length a scale bar shows, about `target` CSS pixels long at `perPixel` meters a pixel, and its length. */
export function scaleBar(perPixel: number, target = 100): { length: number; pixels: number } {
  const raw = perPixel * target;
  const length = niceStep(raw, 1);
  return { length, pixels: length / perPixel };
}

/** Meters a CSS pixel spans `distance` in front of a camera of vertical field `fov` degrees, `height` pixels tall. */
export const metersPerPixel = (distance: number, fov: number, height: number) =>
  (2 * distance * Math.tan((fov * Math.PI) / 360)) / height;

/** A scale bar as standalone SVG markup, so screenshots can draw it as an image. */
export function scaleBarSvg(pixels: number, label: string, theme: Theme): string {
  const w = Math.ceil(pixels) + 20;
  return (
    `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="30" viewBox="0 0 ${w} 30">` +
    `<rect x="0.5" y="0.5" width="${w - 1}" height="29" rx="5" fill="${theme.panel}" fill-opacity="0.88" stroke="${theme.border}"/>` +
    `<path d="M10 16v6h${pixels.toFixed(1)}v-6" fill="none" stroke="${theme.text}" stroke-width="1.5"/>` +
    `<text x="${w / 2}" y="13" text-anchor="middle" font-family="system-ui, sans-serif" font-size="11" fill="${theme.text}">${label}</text></svg>`
  );
}
