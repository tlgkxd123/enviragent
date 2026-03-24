import {
  ShaderMaterial,
  UniformsUtils,
  Vector2,
  Vector3,
  type IUniform,
} from 'three'
import type { PerspectiveCamera } from 'three'
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js'

/**
 * Photorealistic lens flare post-pass.
 *
 * Features (all analytical, zero extra textures):
 *  - Anamorphic horizontal streak  — wide, thin, blue-tinted (cinema anamorphic lens)
 *  - 5 aperture ghost discs        — sized and tinted like real internal lens reflections
 *  - 6-spike starburst             — rotated star pattern from aperture blades
 *  - Halo ring                     — soft diffraction ring around the source
 *  - All elements fade with source proximity to edge / off-screen
 *  - Ghosts travel along the "flare axis" (source → screen center reflection)
 *
 * The key light world direction is projected to screen UV each frame by the
 * caller via `setLightScreenUv(u, v)`. When the light is behind the camera
 * the flare is suppressed.
 */

const vertexShader = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`

const fragmentShader = /* glsl */ `
precision highp float;

uniform sampler2D tDiffuse;
uniform vec2  uResolution;
uniform vec2  uLightUv;      // screen-space UV of key light (0..1)
uniform float uIntensity;    // master intensity (0=off)
uniform float uAspect;       // width / height

varying vec2 vUv;

// Soft gaussian blob
float blob(vec2 uv, vec2 center, float radius) {
  float d = length((uv - center) * vec2(uAspect, 1.0));
  return exp(-d * d / (2.0 * radius * radius));
}

// Elongated horizontal streak (anamorphic)
float streak(vec2 uv, vec2 center, float width, float height) {
  vec2 d = uv - center;
  float h = exp(-d.x * d.x / (2.0 * width * width));
  float v = exp(-d.y * d.y / (2.0 * height * height * (1.0 / (uAspect * uAspect))));
  return h * v;
}

// N-blade starburst
float starburst(vec2 uv, vec2 center, float radius, int blades) {
  vec2 d = (uv - center) * vec2(uAspect, 1.0);
  float dist = length(d);
  float angle = atan(d.y, d.x);
  float spike = abs(cos(float(blades) * 0.5 * angle));
  float radFade = exp(-dist * dist / (2.0 * radius * radius));
  return pow(spike, 6.0) * radFade;
}

// Diffraction halo ring
float halo(vec2 uv, vec2 center, float radius, float thickness) {
  float d = length((uv - center) * vec2(uAspect, 1.0));
  float ring = exp(-pow(d - radius, 2.0) / (2.0 * thickness * thickness));
  return ring;
}

void main() {
  vec4 base = texture2D(tDiffuse, vUv);

  if (uIntensity < 0.001) {
    gl_FragColor = base;
    return;
  }

  vec2 src = uLightUv; // light position in UV space

  // Suppress if light off-screen
  float onScreen = smoothstep(0.0, 0.08, src.x)
                 * smoothstep(1.0, 0.92, src.x)
                 * smoothstep(0.0, 0.08, src.y)
                 * smoothstep(1.0, 0.92, src.y);

  if (onScreen < 0.001) {
    gl_FragColor = base;
    return;
  }

  vec3 flare = vec3(0.0);

  // ── 1. Anamorphic streak (horizontal, blue-white) ─────────────
  float s = streak(vUv, src, 0.32, 0.0014);
  // Fade the streak ends
  float streakFade = smoothstep(0.0, 0.04, vUv.x) * smoothstep(1.0, 0.96, vUv.x);
  flare += vec3(0.28, 0.62, 1.0) * s * 1.85 * streakFade;
  // Secondary dimmer streak offset slightly vertically
  float s2 = streak(vUv, src + vec2(0.0, 0.0008), 0.28, 0.0008);
  flare += vec3(0.18, 0.40, 0.95) * s2 * 0.55 * streakFade;

  // ── 2. Starburst (6-blade aperture) ──────────────────────────
  float star = starburst(vUv, src, 0.018, 6);
  flare += vec3(1.0, 0.92, 0.78) * star * 1.20;

  // ── 3. Diffraction halo ring ──────────────────────────────────
  float h = halo(vUv, src, 0.038, 0.006);
  flare += vec3(0.55, 0.72, 1.0) * h * 0.35;

  // ── 4. Aperture ghosts along flare axis ──────────────────────
  // Ghosts sit on the line from src through screen centre (0.5,0.5)
  vec2 axis = vec2(0.5) - src; // vector from src to centre

  // Ghost 1 — large, slightly warm, past centre
  vec2 g1 = src + axis * 1.18;
  flare += vec3(1.0, 0.82, 0.55) * blob(vUv, g1, 0.048) * 0.22;

  // Ghost 2 — medium, blue-violet
  vec2 g2 = src + axis * 0.72;
  flare += vec3(0.45, 0.55, 1.0) * blob(vUv, g2, 0.022) * 0.28;

  // Ghost 3 — small, green tint (chromatic aberration)
  vec2 g3 = src + axis * 1.45;
  flare += vec3(0.38, 1.0, 0.52) * blob(vUv, g3, 0.014) * 0.18;

  // Ghost 4 — tiny hot white near source
  vec2 g4 = src + axis * 0.35;
  flare += vec3(1.0, 0.98, 0.95) * blob(vUv, g4, 0.009) * 0.45;

  // Ghost 5 — wide diffuse magenta far side
  vec2 g5 = src + axis * 1.82;
  flare += vec3(1.0, 0.30, 0.72) * blob(vUv, g5, 0.062) * 0.12;

  // ── 5. Core glow at source ────────────────────────────────────
  float core = blob(vUv, src, 0.012);
  flare += vec3(1.0, 0.95, 0.85) * core * 1.8;

  // Master scale
  flare *= uIntensity * onScreen;

  // Additive blend — flares sit on top of the scene
  gl_FragColor = vec4(base.rgb + flare, 1.0);
}
`

const baseUniforms = {
  tDiffuse:    { value: null },
  uResolution: { value: new Vector2(1280, 720) },
  uLightUv:    { value: new Vector2(0.72, 0.62) },
  uIntensity:  { value: 0.88 },
  uAspect:     { value: 16 / 9 },
}

export interface LensFlarePass {
  pass: ShaderPass
  setSize: (w: number, h: number) => void
  /** Call every frame with the projected screen-UV of the key light. */
  setLightScreenUv: (u: number, v: number) => void
  setIntensity: (i: number) => void
}

export function createLensFlarePass(): LensFlarePass {
  const uniforms = UniformsUtils.clone(baseUniforms) as { [k: string]: IUniform }
  const material = new ShaderMaterial({
    uniforms,
    vertexShader,
    fragmentShader,
    depthTest: false,
    depthWrite: false,
  })
  const pass = new ShaderPass(material)

  return {
    pass,
    setSize(w: number, h: number) {
      ;(uniforms.uResolution!.value as Vector2).set(w, h)
      ;(uniforms.uAspect!.value as unknown as number)
      uniforms.uAspect!.value = w / h
    },
    setLightScreenUv(u: number, v: number) {
      ;(uniforms.uLightUv!.value as Vector2).set(u, v)
    },
    setIntensity(i: number) {
      uniforms.uIntensity!.value = i
    },
  }
}

/**
 * Project a world-space direction (normalised) to screen UV [0..1].
 * Returns null when the direction is behind the camera.
 */
export function projectDirToScreenUv(
  dir: Vector3,
  camera: PerspectiveCamera,
  target = new Vector2()
): Vector2 | null {
  // Place the "light" very far along its direction in world space
  const worldPos = dir.clone().multiplyScalar(1000)
  worldPos.project(camera) // NDC -1..1
  if (worldPos.z > 1.0) return null // behind camera
  target.set(
    (worldPos.x + 1) * 0.5,
    (1 - worldPos.y) * 0.5  // flip Y: NDC +1 = top, UV 0 = top
  )
  return target
}
