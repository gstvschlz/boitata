import * as THREE from "three";
import { LineMaterial } from "three/examples/jsm/lines/LineMaterial.js";

/** Uniforms every material shares; the viewer updates them once per frame. */
export const shared = {
  uLight: { value: new THREE.Vector3(0, 0, 1) },
  uPixelRatio: { value: 1 },
  /** Contrast outline of thin marks (screen-size points and lines), from the theme. */
  uHalo: { value: new THREE.Color(0.5, 0.5, 0.5) },
  /** Canvas size in CSS pixels. */
  uResolution: { value: new THREE.Vector2(1, 1) },
};

/** Width of the halo around thin marks, in CSS pixels on each side. */
export const HALO = 1;

const AMBIENT = 0.45;

/**
 * Headlight shading: a face square to the light keeps its exact color, so a block seen in plan shows its LUT entry.
 * Colors come from the `color` or `instanceColor` attribute, raw sRGB bytes.
 */
export function shadedMaterial(instanced: boolean, opacity: number): THREE.ShaderMaterial {
  const material = new THREE.ShaderMaterial({
    uniforms: { uLight: shared.uLight, uOpacity: { value: opacity } },
    vertexColors: !instanced,
    side: THREE.DoubleSide,
    vertexShader: /* glsl */ `
      varying vec3 vColor;
      varying vec3 vNormal;
      void main() {
        #ifdef USE_INSTANCING
          mat4 m = modelMatrix * instanceMatrix;
          vColor = instanceColor;
        #else
          mat4 m = modelMatrix;
          vColor = color;
        #endif
        vNormal = mat3(m) * normal;
        gl_Position = projectionMatrix * viewMatrix * m * vec4(position, 1.0);
      }`,
    fragmentShader: /* glsl */ `
      uniform vec3 uLight;
      uniform float uOpacity;
      varying vec3 vColor;
      varying vec3 vNormal;
      void main() {
        float d = abs(dot(normalize(vNormal), uLight));
        gl_FragColor = vec4(vColor * (${AMBIENT} + ${1 - AMBIENT} * d), uOpacity);
      }`,
  });
  setOpacity(material, opacity);
  return material;
}

/**
 * Round screen-size sprites of their exact color, or with `halo` the outline drawn behind them: a disc in the halo
 * color `HALO` pixels wider on each side that writes no depth, so it never covers a neighbor's color.
 */
export function pointMaterial(size: number, opacity: number, halo = false): THREE.ShaderMaterial {
  const material = new THREE.ShaderMaterial({
    uniforms: {
      uPixelRatio: shared.uPixelRatio,
      uHalo: shared.uHalo,
      uSize: { value: size + (halo ? 2 * HALO : 0) },
      uOpacity: { value: opacity },
    },
    vertexColors: true,
    vertexShader: /* glsl */ `
      uniform float uPixelRatio;
      uniform float uSize;
      varying vec3 vColor;
      void main() {
        vColor = color;
        gl_PointSize = uSize * uPixelRatio;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }`,
    fragmentShader: /* glsl */ `
      uniform vec3 uHalo;
      uniform float uOpacity;
      varying vec3 vColor;
      void main() {
        if (length(gl_PointCoord - 0.5) > 0.5) discard;
        gl_FragColor = vec4(${halo ? "uHalo" : "vColor"}, uOpacity);
      }`,
  });
  setOpacity(material, opacity, halo);
  return material;
}

/** Draw order of thin marks: halos after every opaque layer, the marks right after them. */
export const HALO_ORDER = 5;

/**
 * The 12 edges of instanced boxes as screen-space quads `width` CSS pixels wide. Instances carry a center, a size
 * along each of the model's `axes` and a color.
 */
export function boxEdgeMaterial(axes: number[][], width: number, opacity: number): THREE.ShaderMaterial {
  const [u, v, w] = axes;
  const material = new THREE.ShaderMaterial({
    uniforms: {
      uAxes: { value: new THREE.Matrix3().set(u[0], v[0], w[0], u[1], v[1], w[1], u[2], v[2], w[2]) },
      uResolution: shared.uResolution,
      uWidth: { value: width },
      uOpacity: { value: opacity },
    },
    side: THREE.DoubleSide,
    vertexShader: /* glsl */ `
      uniform mat3 uAxes;
      uniform vec2 uResolution;
      uniform float uWidth;
      attribute vec3 aStart;
      attribute vec3 aEnd;
      attribute vec2 aCorner;
      attribute vec3 iCenter;
      attribute vec3 iSize;
      attribute vec3 iColor;
      varying vec3 vColor;
      void main() {
        vColor = iColor;
        vec4 a = projectionMatrix * modelViewMatrix * vec4(iCenter + uAxes * (aStart * iSize), 1.0);
        vec4 b = projectionMatrix * modelViewMatrix * vec4(iCenter + uAxes * (aEnd * iSize), 1.0);
        vec2 d = (b.xy / b.w - a.xy / a.w) * uResolution;
        d = dot(d, d) > 1e-12 ? normalize(d) : vec2(1.0, 0.0);
        vec4 c = aCorner.x < 0.5 ? a : b;
        c.xy += vec2(-d.y, d.x) * aCorner.y * uWidth / uResolution * c.w;
        gl_Position = (a.w <= 0.0 || b.w <= 0.0) ? vec4(2.0, 2.0, 2.0, 1.0) : c;
      }`,
    fragmentShader: /* glsl */ `
      uniform float uOpacity;
      varying vec3 vColor;
      void main() {
        gl_FragColor = vec4(vColor, uOpacity);
      }`,
  });
  setOpacity(material, opacity);
  return material;
}

/** Sets a material's opacity; translucent materials and halos write no depth. */
export function setOpacity(material: THREE.ShaderMaterial | LineMaterial, opacity: number, halo = false): void {
  if (material instanceof LineMaterial) material.opacity = opacity;
  else material.uniforms.uOpacity.value = opacity;
  material.transparent = opacity < 1;
  material.depthWrite = opacity >= 1 && !halo;
}
