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
 * Truly volumetric raymarched cloud.
 *
 * Proxy geometry: SphereGeometry (BackSide) — gives a round silhouette from
 * every angle, unlike the old BoxGeometry which always showed cube edges.
 *
 * Ray bounds: analytic ray-ellipsoid intersection in world space.
 * Density:    FBM noise with smooth ellipsoidal envelope — density tapers to
 *             zero before the geometric boundary, eliminating hard cutoff.
 * Lighting:   Henyey-Greenstein phase function + 6-step cone-traced sun
 *             shadow ray for volumetric self-shadowing and god-ray feel.
 * March:      64 primary steps + 6 shadow steps. Step size adapts to ellipsoid
 *             diameter so cost is constant regardless of cloud scale.
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

// Ellipsoid centre + semi-axes (world space)
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

// ── Hash / noise ──────────────────────────────────────────────────────────────
float hash(vec3 p) {
  p = fract(p * vec3(443.8975, 397.2973, 491.1871));
  p += dot(p, p.yxz + 19.19);
  return fract((p.x + p.y) * p.z);
}

float smoothNoise(vec3 p) {
  vec3 i = floor(p), f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(mix(hash(i),           hash(i+vec3(1,0,0)), f.x),
        mix(hash(i+vec3(0,1,0)),hash(i+vec3(1,1,0)),f.x), f.y),
    mix(mix(hash(i+vec3(0,0,1)),hash(i+vec3(1,0,1)),f.x),
        mix(hash(i+vec3(0,1,1)),hash(i+vec3(1,1,1)),f.x), f.y), f.z);
}

float fbm(vec3 p) {
  float v = 0.0, a = 0.5;
  for (int i = 0; i < 6; i++) {
    v += a * smoothNoise(p);
    p  = p * 2.03 + vec3(1.7, 9.2, 3.8);
    a *= 0.48;
  }
  return v;
}

// ── Ray-ellipsoid intersection ─────────────────────────────────────────────────
// Returns vec2(tMin, tMax) in ray parameter space, or vec2(-1) if no hit.
vec2 ellipsoidIntersect(vec3 ro, vec3 rd) {
  // Transform ray into unit-sphere space of the ellipsoid
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

// ── Smooth ellipsoidal envelope ───────────────────────────────────────────────
// Returns 1 at centre, fades to 0 at the ellipsoid surface.
float envelope(vec3 p) {
  vec3 q = (p - uCenter) / uRadii;
  float d = dot(q, q); // 0 at centre, 1 at surface
  return 1.0 - smoothstep(0.5, 1.0, d);
}

// ── Cloud density at world point p ───────────────────────────────────────────
float cloudDensity(vec3 p) {
  float env = envelope(p);
  if (env < 0.001) return 0.0;

  // Animated drift
  vec3 q = p * 0.45;
  q.x += uTime * 0.018;
  q.z += uTime * 0.009;

  // Base large-scale shape
  float base = fbm(q) - (1.0 - uCoverage) * 0.6;

  // Fine detail layer (cheaper, fewer octaves)
  float detail = fbm(p * 1.1 + vec3(uTime * 0.012, 0.0, uTime * 0.007)) * 0.25;

  float d = clamp(base + detail, 0.0, 1.0) * env;
  return d * uDensity;
}

// ── Henyey-Greenstein phase function ─────────────────────────────────────────
float hg(float cosTheta, float g) {
  float g2 = g * g;
  return (1.0 - g2) / (4.0 * 3.14159265 * pow(max(0.0, 1.0 + g2 - 2.0 * g * cosTheta), 1.5));
}

// ── 6-step cone shadow ray toward sun ────────────────────────────────────────
float sunShadow(vec3 p) {
  float shadow = 0.0;
  float step   = dot(uRadii, vec3(0.333)) * 0.4; // ~40% of mean radius / step
  for (int i = 1; i <= 6; i++) {
    vec3 sp = p + uSunDir * (float(i) * step);
    shadow += cloudDensity(sp) * step;
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

  // March step sized to ellipsoid diameter
  float totalDist  = tMax - tMin;
  const int STEPS  = 64;
  float stepSize   = totalDist / float(STEPS);

  // Jitter start to hide banding
  float jitter = hash(vec3(gl_FragCoord.xy * 0.01, uTime)) * stepSize;

  // Proper front-to-back transmittance (avoids summed HDR blowout + bloom white-out)
  vec3  accumColor = vec3(0.0);
  float transmit   = 1.0;
  float cosTheta   = dot(rd, uSunDir);
  float phase      = mix(hg(cosTheta, 0.5), hg(cosTheta, -0.2), 0.3);

  for (int i = 0; i < STEPS; i++) {
    if (transmit < 0.02) break;

    float tt  = tMin + jitter + float(i) * stepSize;
    if (tt > tMax) break;
    vec3 pos  = ro + rd * tt;

    float dens = cloudDensity(pos);
    if (dens < 0.001) continue;

    float sampleAlpha = 1.0 - exp(-dens * stepSize * 4.0);

    float shadow   = sunShadow(pos);
    float sunAtten = exp(-shadow * 2.5);
    vec3  sunLight = uSunColor * sunAtten * phase * 1.15;

    float upDot  = clamp(dot(normalize(pos - uCenter), vec3(0,1,0)), 0.0, 1.0);
    vec3  ambient = mix(vec3(0.55, 0.60, 0.72), vec3(0.92, 0.95, 1.0), upDot) * 0.35;

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
