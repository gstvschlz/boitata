import * as THREE from "three";

/** Neighbours sampled around each pixel, evenly around a circle. */
const SAMPLES = 8;
/** Their distance in device pixels at a pixel ratio of 1. */
const RADIUS = 1.4;
const STRENGTH = 100;

/**
 * Eye-dome lighting (Boucheny, 2009): a pass over the depth of the rendered frame that darkens each pixel by how
 * far it lies behind its neighbours, in log depth, so the edges and relief of points and surfaces read without any
 * normals. The background is copied untouched, alpha and all.
 */
export class EyeDome {
  private target = new THREE.WebGLRenderTarget(1, 1, { samples: 4, depthTexture: new THREE.DepthTexture(1, 1) });
  private scene = new THREE.Scene();
  private camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  private material = new THREE.ShaderMaterial({
    uniforms: {
      tColor: { value: null as THREE.Texture | null },
      tDepth: { value: null as THREE.Texture | null },
      uNear: { value: 0.1 },
      uFar: { value: 1 },
      uStep: { value: new THREE.Vector2() },
    },
    depthTest: false,
    depthWrite: false,
    blending: THREE.NoBlending,
    vertexShader: /* glsl */ `
      varying vec2 vUv;
      void main() {
        vUv = uv;
        gl_Position = vec4(position.xy, 0.0, 1.0);
      }`,
    fragmentShader: /* glsl */ `
      uniform sampler2D tColor;
      uniform sampler2D tDepth;
      uniform float uNear;
      uniform float uFar;
      uniform vec2 uStep;
      varying vec2 vUv;
      float logDistance(float d) {
        float z = 2.0 * d - 1.0;
        return log2(2.0 * uNear * uFar / (uFar + uNear - z * (uFar - uNear)));
      }
      void main() {
        vec4 color = texture2D(tColor, vUv);
        float d = texture2D(tDepth, vUv).r;
        if (d >= 1.0) {
          gl_FragColor = color;
          return;
        }
        float here = logDistance(d);
        float response = 0.0;
        for (int k = 0; k < ${SAMPLES}; k++) {
          float a = 6.2831853 * float(k) / ${SAMPLES}.0;
          float n = texture2D(tDepth, vUv + vec2(cos(a), sin(a)) * uStep).r;
          if (n < 1.0) response += max(0.0, here - logDistance(n));
        }
        gl_FragColor = vec4(color.rgb * exp(-${STRENGTH}.0 * response / ${SAMPLES}.0), color.a);
      }`,
  });

  constructor() {
    const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.material);
    quad.frustumCulled = false;
    this.scene.add(quad);
  }

  /** Renders `scene` offscreen, then its eye-dome-lit color to the canvas. */
  render(renderer: THREE.WebGLRenderer, scene: THREE.Scene, camera: THREE.PerspectiveCamera): void {
    const size = renderer.getDrawingBufferSize(new THREE.Vector2());
    if (this.target.width !== size.x || this.target.height !== size.y) this.target.setSize(size.x, size.y);
    renderer.setRenderTarget(this.target);
    renderer.render(scene, camera);
    renderer.setRenderTarget(null);
    const u = this.material.uniforms;
    u.tColor.value = this.target.texture;
    u.tDepth.value = this.target.depthTexture;
    u.uNear.value = camera.near;
    u.uFar.value = camera.far;
    u.uStep.value.set((RADIUS * renderer.getPixelRatio()) / size.x, (RADIUS * renderer.getPixelRatio()) / size.y);
    renderer.render(this.scene, this.camera);
  }

  dispose(): void {
    this.target.depthTexture?.dispose();
    this.target.dispose();
    this.material.dispose();
  }
}
