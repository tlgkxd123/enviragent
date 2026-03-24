import {
  BackSide,
  BoxGeometry,
  Color,
  GLSL3,
  Mesh,
  ShaderMaterial,
  Vector3,
} from 'three'

/**
 * Raymarched volumetric cloud / fog volume.
 * The LLM can invoke createVolumetricCloud() from generated code.
 *
 * Implemented as a box with an inverted (BackSide) raymarching fragment shader.
 * 64-step march with FBM noise, depth-composited in additive/alpha blend.
 */

const vertGlsl = /* glsl */ `
out vec3 vWorldPos;
out vec3 vOrigin;

void main() {
  vec4 wp = modelMatrix * vec4(position, 1.0);
  vWorldPos = wp.xyz;
  vOrigin   = cameraPosition;
  gl_Position = projectionMatrix * viewMatrix * wp;
}
`

const fragGlsl = /* glsl */ `
precision highp float;

uniform vec3  uBoxMin;
uniform vec3  uBoxMax;
uniform vec3  uColor;       // cloud tint
uniform float uDensity;     // overall density scale
uniform float uTime;
uniform float uCoverage;    // 0=clear, 1=overcast
uniform vec3  uSunDir;      // normalised toward sun
uniform vec3  uSunColor;

in vec3 vWorldPos;
in vec3 vOrigin;

out vec4 fragColor;

// FBM helpers
float hash(vec3 p) {
  p = fract(p * vec3(443.8975, 397.2973, 491.1871));
  p += dot(p, p.yxz + 19.19);
  return fract((p.x + p.y) * p.z);
}

float smoothNoise(vec3 p) {
  vec3 i = floor(p);
  vec3 f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(mix(hash(i), hash(i+vec3(1,0,0)), f.x),
        mix(hash(i+vec3(0,1,0)), hash(i+vec3(1,1,0)), f.x), f.y),
    mix(mix(hash(i+vec3(0,0,1)), hash(i+vec3(1,0,1)), f.x),
        mix(hash(i+vec3(0,1,1)), hash(i+vec3(1,1,1)), f.x), f.y),
    f.z);
}

float fbm(vec3 p) {
  float v = 0.0, a = 0.5;
  for (int i = 0; i < 5; i++) {
    v += a * smoothNoise(p);
    p  = p * 2.01 + vec3(1.7, 9.2, 3.8);
    a *= 0.5;
  }
  return v;
}

float cloudDensity(vec3 p) {
  // Drift
  p.x += uTime * 0.018;
  float base = fbm(p * 0.55) - (1.0 - uCoverage) * 0.55;
  float detail = fbm(p * 1.8 + vec3(3.1, 1.7, 2.4)) * 0.28;
  return max(0.0, base + detail - 0.22) * uDensity;
}

// Slab intersection
vec2 boxIntersect(vec3 ro, vec3 rd) {
  vec3 t0 = (uBoxMin - ro) / (rd + 1e-9);
  vec3 t1 = (uBoxMax - ro) / (rd + 1e-9);
  vec3 tNear = min(t0, t1);
  vec3 tFar  = max(t0, t1);
  float tIn  = max(max(tNear.x, tNear.y), tNear.z);
  float tOut = min(min(tFar.x,  tFar.y),  tFar.z);
  return vec2(tIn, tOut);
}

void main() {
  vec3 ro = vOrigin;
  vec3 rd = normalize(vWorldPos - vOrigin);

  vec2 t = boxIntersect(ro, rd);
  if (t.x > t.y || t.y < 0.0) discard;

  float tStart = max(t.x, 0.0);
  float tEnd   = t.y;
  float span   = tEnd - tStart;
  if (span < 0.001) discard;

  const int STEPS = 48;
  float stepSize = span / float(STEPS);

  vec3  accumColor = vec3(0.0);
  float transmit   = 1.0;

  for (int i = 0; i < STEPS; i++) {
    float tt = tStart + (float(i) + 0.5) * stepSize;
    vec3  pos = ro + rd * tt;

    float d = cloudDensity(pos);
    if (d < 0.001) continue;

    // Single-scatter sun
    float sunD = cloudDensity(pos + uSunDir * 0.8);
    float shadow = exp(-sunD * 2.2);
    vec3  sunContrib = uSunColor * shadow * 1.4;

    // Ambient (sky)
    vec3  amb = uColor * 0.35;

    vec3  sampleColor = (sunContrib + amb) * uColor;
    float alpha = 1.0 - exp(-d * stepSize * 8.0);

    accumColor += transmit * alpha * sampleColor;
    transmit   *= (1.0 - alpha);
    if (transmit < 0.01) break;
  }

  float opacity = 1.0 - transmit;
  if (opacity < 0.003) discard;
  fragColor = vec4(accumColor / max(opacity, 0.001), opacity);
}
`

export interface CloudOptions {
  /** World-space centre */
  position?: [number, number, number]
  /** Box half-extents */
  size?: [number, number, number]
  /** Cloud tint colour (hex) */
  color?: number
  /** Density multiplier 0.5–4 */
  density?: number
  /** Coverage 0=clear 1=overcast */
  coverage?: number
  /** Sun direction (normalised) */
  sunDir?: [number, number, number]
  /** Sun colour (hex) */
  sunColor?: number
}

export function createVolumetricCloud(opts: CloudOptions = {}): Mesh {
  const pos     = opts.position  ?? [0, 3, 0]
  const size    = opts.size      ?? [8, 2.5, 8]
  const color   = new Color(opts.color    ?? 0xd8e8f0)
  const sunDir  = new Vector3(...(opts.sunDir  ?? [0.65, 0.55, -0.52])).normalize()
  const sunCol  = new Color(opts.sunColor ?? 0xfff2e0)

  const halfW = size[0] / 2
  const halfH = size[1] / 2
  const halfD = size[2] / 2
  const cx = pos[0], cy = pos[1], cz = pos[2]

  const mat = new ShaderMaterial({
    glslVersion: GLSL3,
    vertexShader:   vertGlsl,
    fragmentShader: fragGlsl,
    uniforms: {
      uBoxMin:   { value: new Vector3(cx - halfW, cy - halfH, cz - halfD) },
      uBoxMax:   { value: new Vector3(cx + halfW, cy + halfH, cz + halfD) },
      uColor:    { value: new Vector3(color.r, color.g, color.b) },
      uDensity:  { value: opts.density  ?? 1.2 },
      uTime:     { value: 0 },
      uCoverage: { value: opts.coverage ?? 0.55 },
      uSunDir:   { value: sunDir },
      uSunColor: { value: new Vector3(sunCol.r, sunCol.g, sunCol.b) },
    },
    side: BackSide,
    transparent: true,
    depthWrite: false,
  })

  const geo  = new BoxGeometry(size[0], size[1], size[2])
  const mesh = new Mesh(geo, mat)
  mesh.position.set(cx, cy, cz)
  mesh.renderOrder = 1
  return mesh
}

/** Call each frame to animate cloud drift. */
export function stepCloud(mesh: Mesh, dt: number): void {
  const mat = mesh.material as ShaderMaterial
  if (mat.uniforms?.uTime) {
    mat.uniforms.uTime.value += dt
  }
}
