/** Three.js injects projectionMatrix, modelViewMatrix, normalMatrix, etc. */

export const orbitalVertexShader = /* glsl */ `
uniform float uDispScale;

in float aDensity;
in float aPhase;

out vec3 vNormal;
out float vDensity;
out float vPhase;
out vec3 vViewPos;
out vec3 vWorldPos;
out vec3 vWorldNormal;

void main() {
  vec3 dir = normalize(position);
  float disp = uDispScale * aDensity;
  vec3 displaced = dir * (1.0 + disp);
  vec4 mvPosition = modelViewMatrix * vec4(displaced, 1.0);
  vViewPos = mvPosition.xyz;
  vNormal = normalize(normalMatrix * normalize(displaced));
  vec4 wp = modelMatrix * vec4(displaced, 1.0);
  vWorldPos = wp.xyz;
  vWorldNormal = normalize(mat3(modelMatrix) * normalize(displaced));
  vDensity = aDensity;
  vPhase = aPhase;
  gl_Position = projectionMatrix * mvPosition;
}
`

export const orbitalFragmentShader = /* glsl */ `
precision highp float;

out vec4 fragColor;

uniform vec3 uAccent;
uniform float uTime;
uniform samplerCube envMap;
uniform float envMapIntensity;
uniform float uBandStrength;
uniform float uSaturation;
// Roughness drives how sharp reflections appear (0=mirror, 1=diffuse)
uniform float uRoughness;

in vec3 vNormal;
in float vDensity;
in float vPhase;
in vec3 vViewPos;
in vec3 vWorldPos;
in vec3 vWorldNormal;

vec3 hsv2rgb(vec3 c) {
  vec3 p = abs(fract(c.xxx + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0);
  vec3 rgb = clamp(p - 1.0, 0.0, 1.0);
  return c.z * mix(vec3(1.0), rgb, c.y);
}

// GGX / Trowbridge-Reitz NDF
float D_GGX(float NdotH, float roughness) {
  float a = roughness * roughness;
  float a2 = a * a;
  float d = NdotH * NdotH * (a2 - 1.0) + 1.0;
  return a2 / (3.14159265 * d * d + 1e-6);
}

// Smith GGX geometry term (combined view+light)
float G_Smith(float NdotV, float NdotL, float roughness) {
  float a = roughness * roughness;
  float gv = NdotL * (NdotV * (1.0 - a) + a);
  float gl = NdotV * (NdotL * (1.0 - a) + a);
  return 0.5 / (gv + gl + 1e-6);
}

// Schlick Fresnel
vec3 F_Schlick(float VdotH, vec3 F0) {
  return F0 + (1.0 - F0) * pow(1.0 - VdotH, 5.0);
}

// Sample env at two roughness levels and interpolate (fakes glossy cone)
// Uses standard GLSL ES 3.0 textureLod (WebGL2, no extension needed)
vec3 sampleEnvGlossy(vec3 R, float roughness) {
  float mip0 = roughness * 4.0;
  float mip1 = mip0 + 1.5;
  vec3 s0 = textureLod(envMap, R, mip0).rgb;
  vec3 s1 = textureLod(envMap, R, mip1).rgb;
  return mix(s0, s1, roughness * 0.6);
}

void main() {
  vec3 N  = normalize(vNormal);
  vec3 Nw = normalize(vWorldNormal);
  vec3 Vw = normalize(cameraPosition - vWorldPos);
  vec3 V  = normalize(-vViewPos);

  // Orbital base colour
  float hue = fract(vPhase / 6.28318530718 + 0.08);
  float d = clamp(vDensity, 0.0, 1.0);
  float q = floor(d * 9.0) / 9.0;
  float dMix = mix(d, q, uBandStrength * 0.55);
  float logD = log(max(d, 0.0001));
  float ring = fract(logD * 3.5);
  float ringMul = mix(1.0, 0.88 + 0.24 * ring, uBandStrength * 0.45);
  dMix *= ringMul;
  float satBoost = smoothstep(0.08, 0.42, vDensity);
  float S = mix(0.22, 0.76, satBoost) * uSaturation;
  float hsvVal = 0.18 + 0.78 * dMix;
  vec3 base = hsv2rgb(vec3(hue, S, hsvVal));

  // Orbital surface is dielectric-like with tinted F0
  float metallic = 0.18;
  float roughness = clamp(uRoughness, 0.04, 1.0);
  vec3 F0 = mix(vec3(0.06), base, metallic);

  // Key light (matches studio env key direction)
  vec3 L = normalize(vec3(0.65, 0.55, -0.52));
  vec3 H = normalize(V + L);
  float NdotL = max(dot(N, L), 0.0);
  float NdotV = max(dot(N, V), 1e-4);
  float NdotH = max(dot(N, H), 0.0);
  float VdotH = max(dot(V, H), 0.0);

  // GGX specular lobe
  float D = D_GGX(NdotH, roughness);
  float G = G_Smith(NdotV, NdotL, roughness);
  vec3  F = F_Schlick(VdotH, F0);
  vec3 spec = D * G * F * NdotL * 3.14159265;

  // Diffuse (Lambertian, energy-conserving)
  vec3 kD = (1.0 - F) * (1.0 - metallic);
  vec3 diffuse = kD * base * NdotL * 1.8;

  // Ambient from env irradiance (low-mip = pre-integrated diffuse)
  vec3 irradiance = textureLod(envMap, Nw, 6.0).rgb * envMapIntensity;
  vec3 ambient = kD * base * irradiance * 1.4;

  vec3 col = ambient + diffuse + spec * uAccent;

  // ── Glossy environment reflection (Schlick Fresnel gating) ──────
  vec3 Rw = reflect(-Vw, Nw);
  vec3 envColor = sampleEnvGlossy(Rw, roughness) * envMapIntensity;
  vec3 Fenv = F_Schlick(NdotV, F0);
  // Smooth transition: glancing = near-mirror, direct = diffuse-mix
  float fresnelEnv = Fenv.r * 0.5 + Fenv.g * 0.3 + Fenv.b * 0.2;
  float envWeight = mix(0.08, 0.88, pow(fresnelEnv, 1.4)) * (1.0 - roughness * 0.7);
  col = mix(col, envColor, clamp(envWeight, 0.0, 1.0));

  // Accent rim on specular peak
  col += uAccent * pow(max(dot(N, H), 0.0), 96.0) * 0.22 * fresnelEnv;

  // Subtle breathing glow
  float pulse = 0.5 + 0.5 * sin(uTime * 0.65);
  col += base * 0.04 * pulse;

  fragColor = vec4(col, 1.0);
}
`
