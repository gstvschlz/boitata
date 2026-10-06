import * as THREE from "three";

THREE.ColorManagement.enabled = false;

/** What the viewer needs from a GPU backend; WebGL2 today. */
export interface Renderer {
  readonly canvas: HTMLCanvasElement;
  setSize(width: number, height: number, pixelRatio: number): void;
  setBackground(css: string): void;
  render(scene: THREE.Scene, camera: THREE.Camera): void;
  /** Renders one frame at `scale` times the CSS size and returns it as a 2D canvas; the next frame redraws. */
  snapshot(scene: THREE.Scene, camera: THREE.Camera, scale: number): HTMLCanvasElement;
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
    snapshot(scene, camera, scale) {
      const limit = renderer.capabilities.maxTextureSize;
      const ratio = Math.min(scale, limit / size.width, limit / size.height);
      renderer.setPixelRatio(ratio);
      renderer.setSize(size.width, size.height, false);
      renderer.render(scene, camera);
      const out = document.createElement("canvas");
      out.width = renderer.domElement.width;
      out.height = renderer.domElement.height;
      out.getContext("2d")!.drawImage(renderer.domElement, 0, 0);
      renderer.setPixelRatio(size.pixelRatio);
      renderer.setSize(size.width, size.height, false);
      return out;
    },
    dispose() {
      renderer.dispose();
    },
  };
}
