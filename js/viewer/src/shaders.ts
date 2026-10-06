import * as THREE from "three";
import { LineMaterial } from "three/examples/jsm/lines/LineMaterial.js";
import { COLLAPSE, FILTER_GLSL, type FilterUniforms } from "./filter";
import { MAX_SEGMENTS, SECTION_GLSL, sectionUniforms } from "./section";

/** Uniforms every material shares; the viewer updates them once per frame. */
export const shared = {
  uLight: { value: new THREE.Vector3(0, 0, 1) },
  uPixelRatio: { value: 1 },
  /** Contrast outline of thin marks (screen-size points and lines), from the theme. */
  uHalo: { value: new THREE.Color(0.5, 0.5, 0.5) },
  /** Canvas size in CSS pixels. */
  uResolution: { value: new THREE.Vector2(1, 1) },
  /** Theme accent, which outlines where a section cuts a surface. */
  uAccent: { value: new THREE.Color(1, 0.4, 0.3) },
};

/** Width of the halo around thin marks, in CSS pixels on each side. */
export const HALO = 1;

const AMBIENT = 0.45;

/** How a shaded material meets a section. */
export interface Cut {
  /** Keep or drop instances whole by their center, instead of clipping each fragment. */
  whole?: boolean;
  /** Close cut boxes with faces of their own color, unlit. */
  caps?: boolean;
  /** Outline where the slab's faces cross the surface, in the theme accent. */
  outline?: boolean;
}

/**
 * Headlight shading: a face square to the light keeps its exact color, so a block seen in plan shows its LUT entry.
 * Colors come from the `color` or `instanceColor` attribute, raw sRGB bytes. An instance failing the filter
 * collapses; a triangle with a corner failing it is discarded. A section clips fragments outside its slab, or
 * whole instances by their center; a capped box seen through its cut shows its back faces moved onto the cut
 * (depth and all) in its exact color.
 */
export function shadedMaterial(instanced: boolean, opacity: number, filter: FilterUniforms, cut: Cut = {}): THREE.ShaderMaterial {
  const material = new THREE.ShaderMaterial({
    uniforms: { ...filter, ...sectionUniforms, uLight: shared.uLight, uAccent: shared.uAccent, uOpacity: { value: opacity } },
    vertexColors: !instanced,
    side: THREE.DoubleSide,
    defines: {
      ...(cut.whole ? { SECTION_WHOLE: "" } : {}),
      ...(cut.caps ? { SECTION_CAPS: "" } : {}),
      ...(cut.outline ? { SECTION_OUTLINE: "" } : {}),
    },
    vertexShader: /* glsl */ `
      ${FILTER_GLSL}
      ${SECTION_GLSL}
      attribute vec4 aFilter;
      varying vec3 vColor;
      varying vec3 vNormal;
      varying float vPass;
      varying vec3 vWorld;
      varying vec3 vRef;
      varying vec3 vLocal;
      varying vec3 vEyeLocal;
      void main() {
        #ifdef USE_INSTANCING
          mat4 m = modelMatrix * instanceMatrix;
          vColor = instanceColor;
        #else
          mat4 m = modelMatrix;
          vColor = color;
        #endif
        vec4 world = m * vec4(position, 1.0);
        vWorld = world.xyz;
        #ifdef USE_INSTANCING
          vRef = m[3].xyz;
        #else
          vRef = world.xyz;
        #endif
        vNormal = sectionTurn(mat3(m) * normal, vRef);
        gl_Position = projectionMatrix * viewMatrix * vec4(sectionPlace(world.xyz, vRef), 1.0);
        vLocal = position;
        vEyeLocal = vec3(0.0);
        #ifdef SECTION_CAPS
          vEyeLocal = (inverse(m) * vec4(sectionUnplace(cameraPosition, vRef), 1.0)).xyz;
        #endif
        vPass = filterPass(aFilter) ? 1.0 : 0.0;
        #ifdef USE_INSTANCING
          if (vPass < 0.5) ${COLLAPSE}
        #endif
        #ifdef SECTION_WHOLE
          if (!sectionKeep(vRef)) ${COLLAPSE}
        #endif
      }`,
    fragmentShader: /* glsl */ `
      ${SECTION_GLSL}
      uniform mat4 projectionMatrix;
      uniform vec3 uLight;
      uniform vec3 uAccent;
      uniform float uOpacity;
      varying vec3 vColor;
      varying vec3 vNormal;
      varying float vPass;
      varying vec3 vWorld;
      varying vec3 vRef;
      varying vec3 vLocal;
      varying vec3 vEyeLocal;
      /** Ray parameters (eye at 0, this fragment at 1) where the ray enters and leaves segment j's stretch of the slab. */
      vec2 slabSpan(int j, vec3 eye, vec3 dir) {
        vec4 half4 = vec4(0.0, 0.0, 0.0, uSecHalf);
        vec4 planes[4] = vec4[4](uSecStart[j], uSecEnd[j], half4 - uSecSlab[j], uSecSlab[j] + half4);
        vec2 span = vec2(-1e30, 1e30);
        for (int k = 0; k < 4; k++) {
          float from = secSide(planes[k], eye);
          float along = dot(planes[k].xyz, dir);
          if (abs(along) < 1e-12) {
            if (from < 0.0) span = vec2(1.0, 0.0);
          } else if (along > 0.0) span.x = max(span.x, -from / along);
          else span.y = min(span.y, -from / along);
        }
        return span;
      }
      void main() {
        if (vPass < 0.9999) discard;
        #ifdef SECTION_CAPS
          if (!gl_FrontFacing && uSecCount > 0) {
            // the box is seen from inside where its front face was cut away: draw the cut, the nearest point of the
            // ray inside both the box and the slab, unless the front face itself lies in the slab
            vec3 eye = sectionUnplace(cameraPosition, vRef);
            vec3 dir = vWorld - eye;
            vec3 d = vLocal - vEyeLocal;
            d += vec3(1e-12) * (1.0 - step(1e-12, abs(d)));
            vec3 near = min((vec3(-0.5) - vEyeLocal) / d, (vec3(0.5) - vEyeLocal) / d);
            float enter = max(max(near.x, near.y), near.z);
            float t = 2.0;
            for (int j = 0; j < ${MAX_SEGMENTS}; j++) {
              if (j >= uSecCount) break;
              vec2 span = slabSpan(j, eye, dir);
              if (span.x > span.y || span.y < enter || span.x > 1.0) continue;
              if (span.x <= enter) discard;
              t = min(t, span.x);
            }
            if (t > 1.0) discard;
            vec4 clip = projectionMatrix * viewMatrix * vec4(sectionPlace(eye + t * dir, vRef), 1.0);
            gl_FragDepth = 0.5 * clip.z / clip.w + 0.5;
            gl_FragColor = vec4(vColor, uOpacity);
            return;
          }
          gl_FragDepth = gl_FragCoord.z;
        #endif
        #ifndef SECTION_WHOLE
          if (!sectionKeep(vWorld)) discard;
        #endif
        #ifdef SECTION_OUTLINE
          float edge = uSecCount > 0 ? sectionEdge(vWorld) : 1e30;
          float band = 1.5 * fwidth(edge);
          if (uSecCount > 0 && edge < band) {
            gl_FragColor = vec4(uAccent, 1.0);
            return;
          }
        #endif
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
export function pointMaterial(size: number, opacity: number, filter: FilterUniforms, halo = false): THREE.ShaderMaterial {
  const material = new THREE.ShaderMaterial({
    uniforms: {
      ...filter,
      ...sectionUniforms,
      uPixelRatio: shared.uPixelRatio,
      uHalo: shared.uHalo,
      uSize: { value: size + (halo ? 2 * HALO : 0) },
      uOpacity: { value: opacity },
    },
    vertexColors: true,
    vertexShader: /* glsl */ `
      ${FILTER_GLSL}
      ${SECTION_GLSL}
      attribute vec4 aFilter;
      uniform float uPixelRatio;
      uniform float uSize;
      varying vec3 vColor;
      void main() {
        vColor = color;
        gl_PointSize = uSize * uPixelRatio;
        vec3 world = (modelMatrix * vec4(position, 1.0)).xyz;
        gl_Position = projectionMatrix * viewMatrix * vec4(sectionPlace(world, world), 1.0);
        if (!filterPass(aFilter) || !sectionKeep(world)) {
          ${COLLAPSE}
          gl_PointSize = 0.0;
        }
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
export function boxEdgeMaterial(axes: number[][], width: number, opacity: number, filter: FilterUniforms): THREE.ShaderMaterial {
  const [u, v, w] = axes;
  const material = new THREE.ShaderMaterial({
    uniforms: {
      ...filter,
      ...sectionUniforms,
      uAxes: { value: new THREE.Matrix3().set(u[0], v[0], w[0], u[1], v[1], w[1], u[2], v[2], w[2]) },
      uResolution: shared.uResolution,
      uWidth: { value: width },
      uOpacity: { value: opacity },
    },
    side: THREE.DoubleSide,
    vertexShader: /* glsl */ `
      ${FILTER_GLSL}
      ${SECTION_GLSL}
      attribute vec4 aFilter;
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
      varying vec3 vWorld;
      void main() {
        vColor = iColor;
        vec3 wa = (modelMatrix * vec4(iCenter + uAxes * (aStart * iSize), 1.0)).xyz;
        vec3 wb = (modelMatrix * vec4(iCenter + uAxes * (aEnd * iSize), 1.0)).xyz;
        vWorld = aCorner.x < 0.5 ? wa : wb;
        vec4 a = projectionMatrix * viewMatrix * vec4(sectionPlace(wa, iCenter), 1.0);
        vec4 b = projectionMatrix * viewMatrix * vec4(sectionPlace(wb, iCenter), 1.0);
        vec2 d = (b.xy / b.w - a.xy / a.w) * uResolution;
        d = dot(d, d) > 1e-12 ? normalize(d) : vec2(1.0, 0.0);
        vec4 c = aCorner.x < 0.5 ? a : b;
        c.xy += vec2(-d.y, d.x) * aCorner.y * uWidth / uResolution * c.w;
        gl_Position = (a.w <= 0.0 || b.w <= 0.0) ? vec4(2.0, 2.0, 2.0, 1.0) : c;
        if (!filterPass(aFilter)) ${COLLAPSE}
      }`,
    fragmentShader: /* glsl */ `
      ${SECTION_GLSL}
      uniform float uOpacity;
      varying vec3 vColor;
      varying vec3 vWorld;
      void main() {
        if (!sectionKeep(vWorld)) discard;
        gl_FragColor = vec4(vColor, uOpacity);
      }`,
  });
  setOpacity(material, opacity);
  return material;
}

/** Instanced boxes that only write depth, so an opaque wireframe hides the edges behind it; filtered like the edges. */
export function depthMaterial(filter: FilterUniforms): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    uniforms: { ...filter, ...sectionUniforms },
    colorWrite: false,
    polygonOffset: true,
    polygonOffsetFactor: 1,
    polygonOffsetUnits: 1,
    vertexShader: /* glsl */ `
      ${FILTER_GLSL}
      ${SECTION_GLSL}
      attribute vec4 aFilter;
      varying vec3 vWorld;
      void main() {
        mat4 m = modelMatrix * instanceMatrix;
        vWorld = (m * vec4(position, 1.0)).xyz;
        gl_Position = projectionMatrix * viewMatrix * vec4(sectionPlace(vWorld, m[3].xyz), 1.0);
        if (!filterPass(aFilter)) ${COLLAPSE}
      }`,
    fragmentShader: /* glsl */ `
      ${SECTION_GLSL}
      varying vec3 vWorld;
      void main() {
        if (!sectionKeep(vWorld)) discard;
        gl_FragColor = vec4(0.0);
      }`,
  });
}

/**
 * Screen-width lines whose segments collapse when they fail the filter or when their middle lies outside a section. `paired` segments carry the values of both
 * their sides (`aFilter`, `aFilter2`): slots joined by "or" pass when either side does, the others when both do.
 */
export function lineMaterial(
  params: ConstructorParameters<typeof LineMaterial>[0],
  filter: FilterUniforms,
  paired = false,
): LineMaterial {
  const material = new LineMaterial(params);
  Object.assign(material.uniforms, filter, sectionUniforms);
  const test = paired ? "filterPass2(aFilter, aFilter2)" : "filterPass(aFilter)";
  const place = (end: string) =>
    `vec4 ${end} = viewMatrix * vec4(sectionPlace((modelMatrix * vec4(instance${end[0].toUpperCase()}${end.slice(1)}, 1.0)).xyz, secMid), 1.0);`;
  material.vertexShader = material.vertexShader
    .replace("void main() {", `${FILTER_GLSL}\n${SECTION_GLSL}\nattribute vec4 aFilter;\nattribute vec4 aFilter2;\nvoid main() {`)
    .replace(
      "vec4 start = modelViewMatrix * vec4( instanceStart, 1.0 );",
      `vec3 secMid = (modelMatrix * vec4(0.5 * (instanceStart + instanceEnd), 1.0)).xyz;\n${place("start")}`,
    )
    .replace("vec4 end = modelViewMatrix * vec4( instanceEnd, 1.0 );", place("end"))
    .replace("#include <fog_vertex>", `#include <fog_vertex>\nif (!${test} || !sectionKeep(secMid)) ${COLLAPSE}`);
  return material;
}

/** Sets a material's opacity; translucent materials and halos write no depth. */
export function setOpacity(material: THREE.ShaderMaterial | LineMaterial, opacity: number, halo = false): void {
  if (material instanceof LineMaterial) material.opacity = opacity;
  else material.uniforms.uOpacity.value = opacity;
  material.transparent = opacity < 1;
  material.depthWrite = opacity >= 1 && !halo;
}
