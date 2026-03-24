import {
  ShaderMaterial,
  UniformsUtils,
  Vector2,
  type IUniform,
} from 'three'
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js'

/**
 * Minimal screen-space reflection pass.
 *
 * Strategy (zero extra render targets, 16-step ray march):
 *  - Reflects the already-rendered colour buffer against itself.
 *  - Ray marched in UV space using an approximated reflection vector
 *    derived from the screen-space normal baked into the colour luminance
 *    gradient (cheap normal reconstruction from neighbour taps).
 *  - Binary search refinement (4 steps) for sub-pixel accuracy.
 *  - Edge fade + distance fade for seamless blending.
 *  - Controlled by `uSSRStrength` so it can be dialled to zero.
 *
 * Cost: ~16 texture fetches per pixel on the post-processing quad.
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
uniform vec2 uResolution;
uniform float uSSRStrength;   // 0 = off, 1 = full
uniform float uRoughnessCut;  // pixels rougher than this get less reflection

varying vec2 vUv;

// Reconstruct approximate view-space normal from luminance gradient
vec3 approxNormal(vec2 uv, vec2 texel) {
  float l  = dot(texture2D(tDiffuse, uv - vec2(texel.x, 0.0)).rgb, vec3(0.299,0.587,0.114));
  float r  = dot(texture2D(tDiffuse, uv + vec2(texel.x, 0.0)).rgb, vec3(0.299,0.587,0.114));
  float d  = dot(texture2D(tDiffuse, uv - vec2(0.0, texel.y)).rgb, vec3(0.299,0.587,0.114));
  float u2 = dot(texture2D(tDiffuse, uv + vec2(0.0, texel.y)).rgb, vec3(0.299,0.587,0.114));
  return normalize(vec3(l - r, d - u2, 0.18));
}

void main() {
  vec4 base = texture2D(tDiffuse, vUv);

  if (uSSRStrength < 0.001) {
    gl_FragColor = base;
    return;
  }

  vec2 texel = 1.0 / uResolution;

  // Approximate screen-space reflection direction
  vec3 N = approxNormal(vUv, texel);
  // View ray in screen space (pointing into screen)
  vec3 viewRay = vec3(vUv * 2.0 - 1.0, -1.0);
  vec3 R = reflect(normalize(viewRay), N);

  // Only reflect surfaces that face "up" in screen space
  if (R.z > -0.05) {
    gl_FragColor = base;
    return;
  }

  // Project reflection ray into UV space
  vec2 rayDir = R.xy * vec2(0.5, -0.5) * 0.018;  // step size in UV
  float stepLen = length(rayDir);
  if (stepLen < 1e-5) {
    gl_FragColor = base;
    return;
  }

  vec2 rayUv = vUv;
  vec4 hitColor = vec4(0.0);
  float hit = 0.0;

  // 16-step linear march
  for (int i = 0; i < 16; i++) {
    rayUv += rayDir;
    if (rayUv.x < 0.0 || rayUv.x > 1.0 || rayUv.y < 0.0 || rayUv.y > 1.0) break;
    vec4 s = texture2D(tDiffuse, rayUv);
    float sLum = dot(s.rgb, vec3(0.299, 0.587, 0.114));
    float baseLum = dot(base.rgb, vec3(0.299, 0.587, 0.114));
    // Hit when we reach a significantly brighter region (a lit surface)
    if (sLum > baseLum * 1.12 + 0.04) {
      // 4-step binary refinement
      vec2 lo = rayUv - rayDir;
      vec2 hi = rayUv;
      for (int j = 0; j < 4; j++) {
        vec2 mid = (lo + hi) * 0.5;
        vec4 ms = texture2D(tDiffuse, mid);
        float mLum = dot(ms.rgb, vec3(0.299, 0.587, 0.114));
        if (mLum > baseLum * 1.12 + 0.04) hi = mid;
        else lo = mid;
      }
      hitColor = texture2D(tDiffuse, (lo + hi) * 0.5);
      hit = 1.0;
      break;
    }
  }

  if (hit < 0.5) {
    gl_FragColor = base;
    return;
  }

  // Fade at screen edges and by reflection distance
  vec2 edgeDist = min(vUv, 1.0 - vUv);
  float edgeFade = smoothstep(0.0, 0.12, edgeDist.x) * smoothstep(0.0, 0.12, edgeDist.y);
  float dist = length(rayUv - vUv);
  float distFade = 1.0 - smoothstep(0.0, 0.45, dist);
  float alpha = edgeFade * distFade * uSSRStrength * 0.55;

  gl_FragColor = vec4(mix(base.rgb, hitColor.rgb, alpha), 1.0);
}
`

const baseUniforms = {
  tDiffuse:     { value: null },
  uResolution:  { value: new Vector2(1280, 720) },
  uSSRStrength: { value: 0.72 },
  uRoughnessCut:{ value: 0.5 },
}

export function createSSRPass(): {
  pass: ShaderPass
  setSize: (w: number, h: number) => void
  setStrength: (s: number) => void
} {
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
    },
    setStrength(s: number) {
      uniforms.uSSRStrength!.value = s
    },
  }
}
