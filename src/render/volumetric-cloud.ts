import {
  BackSide,
  Color,
  GLSL3,
  Mesh,
  ShaderMaterial,
  SphereGeometry,
  Vector3,
} from 'three'

/**
 * Volumetric raymarched cloud with realistic noise shaping.
 *
 * Noise model (Horizon Zero Dawn / Schneider-style):
 *   Base shape — Perlin-Worley blend: gradient FBM for large-scale variation,
 *   inverted Worley FBM for characteristic puffy cauliflower lobes.
 *   Detail erosion — high-frequency Worley + Perlin carves wispy edges.
 *   Height gradient — cumulus vertical profile: flat base, billowy dome,
 *   thinning at the crown.
 *
 * Lighting:   Dual-lobe Henyey-Greenstein phase (forward + backward scatter)
 *             + Beer-Powder attenuation + 6-step cone-traced sun shadow.
 * March:      72 primary steps + 6 shadow steps, jittered start.
 * Proxy:      SphereGeometry (BackSide) + analytic ray-ellipsoid intersection.
 */

const vertGlsl = /* glsl */`
out vec3 vWorldPos;
out vec3 vOrigin;

void main() {
  vec4 wp = modelMatrix * vec4(position, 1.0);
  vWorldPos = wp.xyz;
  vOrigin   = cameraPosition;
  gl_Position = projectionMatrix * viewMatrix * wp;
}
`

const fragGlsl = /* glsl */`
precision highp float;

uniform vec3  uCenter;
uniform vec3  uRadii;
uniform vec3  uColor;
uniform float uDensity;
uniform float uTime;
uniform float uCoverage;
uniform vec3  uSunDir;
uniform vec3  uSunColor;

in vec3 vWorldPos;
in vec3 vOrigin;
out vec4 fragColor;

// ── Hashes ───────────────────────────────────────────────────────────────────
float hash13(vec3 p) {
  p = fract(p * vec3(443.8975, 397.2973, 491.1871));
  p += dot(p, p.yxz + 19.19);
  return fract((p.x + p.y) * p.z);
}
vec3 hash33(vec3 p) {
  p = vec3(dot(p, vec3(127.1,311.7,74.7)),
           dot(p, vec3(269.5,183.3,246.1)),
           dot(p, vec3(113.5,271.9,124.6)));
  return fract(sin(p) * 43758.5453) * 2.0 - 1.0;
}

// ── Gradient (Perlin-style) noise ────────────────────────────────────────────
float gradientNoise(vec3 p) {
  vec3 i = floor(p);
  vec3 f = fract(p);
  vec3 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0); // quintic Hermite
  return mix(mix(mix(dot(hash33(i+vec3(0,0,0)), f-vec3(0,0,0)),
                     dot(hash33(i+vec3(1,0,0)), f-vec3(1,0,0)), u.x),
                 mix(dot(hash33(i+vec3(0,1,0)), f-vec3(0,1,0)),
                     dot(hash33(i+vec3(1,1,0)), f-vec3(1,1,0)), u.x), u.y),
             mix(mix(dot(hash33(i+vec3(0,0,1)), f-vec3(0,0,1)),
                     dot(hash33(i+vec3(1,0,1)), f-vec3(1,0,1)), u.x),
                 mix(dot(hash33(i+vec3(0,1,1)), f-vec3(0,1,1)),
                     dot(hash33(i+vec3(1,1,1)), f-vec3(1,1,1)), u.x), u.y), u.z);
}

// ── Worley (cellular) noise ──────────────────────────────────────────────────
// Returns F1 (distance to nearest feature point) — creates the puffy cell shapes
float worleyNoise(vec3 p) {
  vec3 i = floor(p);
  vec3 f = fract(p);
  float minDist = 1.0;
  for (int x = -1; x <= 1; x++)
  for (int y = -1; y <= 1; y++)
  for (int z = -1; z <= 1; z++) {
    vec3 neighbor = vec3(float(x), float(y), float(z));
    vec3 point    = hash33(i + neighbor) * 0.5 + 0.5;
    float d       = length(neighbor + point - f);
    minDist       = min(minDist, d);
  }
  return minDist;
}

// ── FBM variants ─────────────────────────────────────────────────────────────
float perlinFBM(vec3 p, int octaves) {
  float v = 0.0, a = 0.5, freq = 1.0;
  for (int i = 0; i < 7; i++) {
    if (i >= octaves) break;
    v += a * gradientNoise(p * freq);
    freq *= 2.0;
    a    *= 0.5;
  }
  return v;
}

float worleyFBM(vec3 p, int octaves) {
  float v = 0.0, a = 0.5, freq = 1.0;
  for (int i = 0; i < 4; i++) {
    if (i >= octaves) break;
    v += a * worleyNoise(p * freq);
    freq *= 2.0;
    a    *= 0.5;
  }
  return v;
}

// ── Ray-ellipsoid intersection ───────────────────────────────────────────────
vec2 ellipsoidIntersect(vec3 ro, vec3 rd) {
  vec3 oc = (ro - uCenter) / uRadii;
  vec3 dc =  rd             / uRadii;
  float a = dot(dc, dc);
  float b = 2.0 * dot(oc, dc);
  float c = dot(oc, oc) - 1.0;
  float disc = b * b - 4.0 * a * c;
  if (disc < 0.0) return vec2(-1.0);
  float sq = sqrt(disc);
  return vec2((-b - sq) / (2.0 * a),
              (-b + sq) / (2.0 * a));
}

// ── Height fraction inside the cloud (0 = bottom, 1 = top) ──────────────────
float heightFrac(vec3 p) {
  float bottom = uCenter.y - uRadii.y;
  float top    = uCenter.y + uRadii.y;
  return clamp((p.y - bottom) / max(top - bottom, 0.001), 0.0, 1.0);
}

// Cumulus vertical profile: flat base, round top, thinning at crown
float heightGradient(float h) {
  float base  = smoothstep(0.0, 0.15, h);   // flat bottom cutoff
  float round = 1.0 - smoothstep(0.6, 1.0, h); // rounded falloff at top
  return base * round;
}

// ── Ellipsoidal envelope (radial fade) ───────────────────────────────────────
float envelope(vec3 p) {
  vec3 q = (p - uCenter) / uRadii;
  float d = dot(q, q);
  return 1.0 - smoothstep(0.4, 1.0, d);
}

// Utility: remap value from [lo,hi] to [newLo,newHi]
float remap(float v, float lo, float hi, float newLo, float newHi) {
  return newLo + (clamp(v, lo, hi) - lo) / max(hi - lo, 0.0001) * (newHi - newLo);
}

// ── Cloud density ────────────────────────────────────────────────────────────
float cloudDensity(vec3 p) {
  float env = envelope(p);
  if (env < 0.001) return 0.0;

  float hf = heightFrac(p);
  float hg = heightGradient(hf);
  if (hg < 0.001) return 0.0;

  // Wind-animated coordinate
  vec3 wind = vec3(uTime * 0.022, 0.0, uTime * 0.011);

  // ── Base shape: Perlin-Worley blend at low frequency ──
  // Large-scale Perlin gives overall shape variation
  vec3 baseCoord = p * 0.32 + wind;
  float pn = perlinFBM(baseCoord, 5) * 0.5 + 0.5; // remap [-1,1] → [0,1]

  // Worley at the same scale creates the characteristic puffy cells
  float wn = 1.0 - worleyFBM(baseCoord * 1.2, 3);

  // Perlin-Worley: Perlin shapes the large form, Worley carves puffy lobes
  float baseShape = remap(pn, wn * 0.4, 1.0, 0.0, 1.0);
  baseShape *= hg;

  // Coverage threshold
  float coverageThresh = 1.0 - uCoverage;
  float baseDensity = remap(baseShape, coverageThresh, 1.0, 0.0, 1.0);
  baseDensity = clamp(baseDensity, 0.0, 1.0);
  if (baseDensity < 0.001) return 0.0;

  // ── Detail erosion: high-freq Worley carves wispy edges ──
  vec3 detailCoord = p * 1.8 + wind * 1.5;
  float detWorley  = worleyFBM(detailCoord, 3);
  float detPerlin  = (perlinFBM(detailCoord * 1.3, 3) * 0.5 + 0.5) * 0.3;
  float detailNoise = detWorley * 0.7 + detPerlin;

  // Erode more at edges (low baseDensity) and near top
  float erosion = mix(detailNoise * 0.35, detailNoise * 0.7, 1.0 - baseDensity);
  float finalDens = clamp(baseDensity - erosion, 0.0, 1.0);

  return finalDens * env * uDensity;
}

// ── Henyey-Greenstein phase function ─────────────────────────────────────────
float hgPhase(float cosTheta, float g) {
  float g2 = g * g;
  return (1.0 - g2) / (4.0 * 3.14159265 * pow(max(0.001, 1.0 + g2 - 2.0 * g * cosTheta), 1.5));
}

// Dual-lobe phase: strong forward + soft backward scatter (silver lining effect)
float dualPhase(float cosTheta) {
  return mix(hgPhase(cosTheta, 0.6), hgPhase(cosTheta, -0.25), 0.25);
}

// ── Beer-Powder approximation ────────────────────────────────────────────────
// Beer's law + powder effect for realistic brightness in thick vs thin cloud
float beerPowder(float opticalDepth) {
  float beer   = exp(-opticalDepth);
  float powder = 1.0 - exp(-opticalDepth * 2.0);
  return beer * mix(1.0, powder, 0.6);
}

// ── 6-step cone shadow ray toward sun ────────────────────────────────────────
float sunShadow(vec3 p) {
  float shadow = 0.0;
  float stepLen = dot(uRadii, vec3(0.333)) * 0.4;
  for (int i = 1; i <= 6; i++) {
    vec3 sp = p + uSunDir * (float(i) * stepLen);
    shadow += cloudDensity(sp) * stepLen;
  }
  return shadow;
}

void main() {
  vec3 ro = vOrigin;
  vec3 rd = normalize(vWorldPos - vOrigin);

  vec2 t = ellipsoidIntersect(ro, rd);
  if (t.x < 0.0 && t.y < 0.0) discard;

  float tMin = max(0.0, t.x);
  float tMax = t.y;
  if (tMax <= tMin) discard;

  float totalDist = tMax - tMin;
  const int STEPS = 72;
  float stepSize  = totalDist / float(STEPS);

  float jitter = hash13(vec3(gl_FragCoord.xy * 0.01, uTime)) * stepSize;

  vec3  accumColor = vec3(0.0);
  float transmit   = 1.0;
  float cosTheta   = dot(rd, uSunDir);
  float phase      = dualPhase(cosTheta);

  for (int i = 0; i < STEPS; i++) {
    if (transmit < 0.01) break;

    float tt = tMin + jitter + float(i) * stepSize;
    if (tt > tMax) break;
    vec3 pos = ro + rd * tt;

    float dens = cloudDensity(pos);
    if (dens < 0.001) continue;

    float sampleAlpha = 1.0 - exp(-dens * stepSize * 4.5);

    float shadow   = sunShadow(pos);
    float sunAtten = beerPowder(shadow * 2.5);
    vec3  sunLight = uSunColor * sunAtten * phase * 1.2;

    // Height-based ambient: bluish from below, bright white from above
    float hf     = heightFrac(pos);
    float upDot  = clamp(dot(normalize(pos - uCenter), vec3(0,1,0)), 0.0, 1.0);
    vec3  ambient = mix(vec3(0.5, 0.55, 0.7), vec3(0.9, 0.93, 1.0), upDot * hf) * 0.38;

    vec3  sampleColor = uColor * (sunLight + ambient);
    accumColor += transmit * sampleAlpha * sampleColor;
    transmit   *= (1.0 - sampleAlpha);
  }

  vec2 tEdge   = ellipsoidIntersect(ro, rd);
  float edgeFade = smoothstep(0.0, stepSize * 4.0, tEdge.y - tEdge.x);
  float opacity  = (1.0 - transmit) * edgeFade;

  if (opacity < 0.004) discard;
  fragColor = vec4(accumColor / max(opacity, 0.001), opacity);
}
`

export interface VolumetricCloudOpts {
  position?:  [number, number, number]
  size?:      [number, number, number]
  color?:     number
  density?:   number
  coverage?:  number
  sunDir?:    [number, number, number]
  sunColor?:  number
}

/**
 * Create a volumetric raymarched cloud.
 * Proxy geometry is a SphereGeometry (BackSide) whose radius = max(size)/2.
 * The GLSL shader performs analytic ray-ellipsoid intersection so the
 * rendered shape is always a smooth ellipsoid regardless of view angle.
 */
export function createVolumetricCloud(opts: VolumetricCloudOpts = {}): Mesh {
  const pos    = opts.position ?? [0, 3, 0]
  const size   = opts.size    ?? [4, 2, 4]
  const color  = new Color(opts.color ?? 0xffffff)
  const sunDir = new Vector3(...(opts.sunDir ?? [0.65, 0.55, -0.52])).normalize()
  const sunCol = new Color(opts.sunColor ?? 0xfff2e0)

  const center = new Vector3(pos[0], pos[1], pos[2])
  const radii  = new Vector3(size[0] / 2, size[1] / 2, size[2] / 2)

  const mat = new ShaderMaterial({
    glslVersion: GLSL3,
    vertexShader:   vertGlsl,
    fragmentShader: fragGlsl,
    uniforms: {
      uCenter:   { value: center },
      uRadii:    { value: radii },
      uColor:    { value: new Vector3(color.r, color.g, color.b) },
    uDensity:  { value: opts.density  ?? 0.85 },
    uTime:     { value: 0 },
    uCoverage: { value: opts.coverage ?? 0.42 },
      uSunDir:   { value: sunDir },
      uSunColor: { value: new Vector3(sunCol.r, sunCol.g, sunCol.b) },
    },
    side:        BackSide,
    transparent: true,
    depthWrite:  false,
  })

  // Proxy sphere radius = largest semi-axis + 2% padding so rays always enter
  const proxyR = Math.max(radii.x, radii.y, radii.z) * 1.02
  const geo    = new SphereGeometry(proxyR, 32, 24)
  const mesh   = new Mesh(geo, mat)
  mesh.position.copy(center)
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
