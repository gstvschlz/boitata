import * as THREE from "three";

/** Uniforms every material shares; the viewer updates them once per frame. */
export const shared = {
  uLight: { value: new THREE.Vector3(0, 0, 1) },
  uPixelRatio: { value: 1 },
};

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

/** Round screen-size sprites with a darker rim. */
export function pointMaterial(size: number, opacity: number): THREE.ShaderMaterial {
  const material = new THREE.ShaderMaterial({
    uniforms: { uPixelRatio: shared.uPixelRatio, uSize: { value: size }, uOpacity: { value: opacity } },
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
      uniform float uOpacity;
      varying vec3 vColor;
      void main() {
        float r = length(gl_PointCoord - 0.5) * 2.0;
        if (r > 1.0) discard;
        gl_FragColor = vec4(r > 0.72 ? vColor * 0.62 : vColor, uOpacity);
      }`,
  });
  setOpacity(material, opacity);
  return material;
}

export function setOpacity(material: THREE.ShaderMaterial, opacity: number): void {
  material.uniforms.uOpacity.value = opacity;
  material.transparent = opacity < 1;
  material.depthWrite = opacity >= 1;
}
