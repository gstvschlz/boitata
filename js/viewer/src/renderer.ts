import * as THREE from "three";

THREE.ColorManagement.enabled = false;

/** RGBA bytes read back from the id pass around a pixel, rows bottom up, and where that pixel falls in them. */
export interface PickWindow {
  pixels: Uint8Array;
  width: number;
  height: number;
  x: number;
  y: number;
}

/** What the viewer needs from a GPU backend; WebGL2 today. */
export interface Renderer {
  readonly canvas: HTMLCanvasElement;
  setSize(width: number, height: number, pixelRatio: number): void;
  setBackground(css: string): void;
  render(scene: THREE.Scene, camera: THREE.Camera): void;
  /**
   * Renders one frame at `scale` times the CSS size, on a transparent background if asked, and returns it as a 2D
   * canvas; the next frame redraws.
   */
  snapshot(scene: THREE.Scene, camera: THREE.Camera, scale: number, transparent?: boolean): HTMLCanvasElement;
  /**
   * Renders one frame offscreen at the CSS size, without antialiasing, and reads back the pixels within `radius` of
   * CSS pixel (x, y), y down.
   */
  pick(scene: THREE.Scene, camera: THREE.Camera, x: number, y: number, radius: number): PickWindow;
  dispose(): void;
}

export function webgl(): Renderer {
  const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: false });
  renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
  let size = { width: 1, height: 1, pixelRatio: 1 };
  return {
    canvas: renderer.domElement,
    setSize(width, height, pixelRatio) {
      size = { width, height, pixelRatio };
      renderer.setPixelRatio(pixelRatio);
      renderer.setSize(width, height);
    },
    setBackground(css) {
      renderer.setClearColor(new THREE.Color(css), 1);
    },
    render(scene, camera) {
      renderer.render(scene, camera);
    },
    snapshot(scene, camera, scale, transparent = false) {
      const limit = renderer.capabilities.maxTextureSize;
      const ratio = Math.min(scale, limit / size.width, limit / size.height);
      const alpha = renderer.getClearAlpha();
      if (transparent) renderer.setClearAlpha(0);
      renderer.setPixelRatio(ratio);
      renderer.setSize(size.width, size.height, false);
      renderer.render(scene, camera);
      const out = document.createElement("canvas");
      out.width = renderer.domElement.width;
      out.height = renderer.domElement.height;
      out.getContext("2d")!.drawImage(renderer.domElement, 0, 0);
      renderer.setClearAlpha(alpha);
      renderer.setPixelRatio(size.pixelRatio);
      renderer.setSize(size.width, size.height, false);
      return out;
    },
    pick(scene, camera, x, y, radius) {
      const width = Math.max(1, Math.round(size.width));
      const height = Math.max(1, Math.round(size.height));
      const target = new THREE.WebGLRenderTarget(width, height);
      const color = renderer.getClearColor(new THREE.Color());
      const alpha = renderer.getClearAlpha();
      renderer.setRenderTarget(target);
      renderer.setClearColor(0x000000, 0);
      renderer.clear();
      renderer.render(scene, camera);
      const px = Math.round(x);
      const py = height - 1 - Math.round(y);
      const x0 = Math.max(0, px - radius);
      const y0 = Math.max(0, py - radius);
      const w = Math.max(0, Math.min(width, px + radius + 1) - x0);
      const h = Math.max(0, Math.min(height, py + radius + 1) - y0);
      const pixels = new Uint8Array(4 * w * h);
      if (w && h) renderer.readRenderTargetPixels(target, x0, y0, w, h, pixels);
      renderer.setRenderTarget(null);
      renderer.setClearColor(color, alpha);
      target.dispose();
      return { pixels, width: w, height: h, x: px - x0, y: py - y0 };
    },
    dispose() {
      renderer.dispose();
    },
  };
}
