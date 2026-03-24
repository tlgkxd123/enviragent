/**
 * Defaults for orbital ShaderMaterial and psiToColor — keep GLSL literals in shaders.ts in sync.
 */
export const ORBITAL_VISUAL = {
  uBandStrength: 0.42,
  uSaturation: 1.0,
  /** Matches fract(vPhase / TWO_PI + offset) in fragment shader */
  phaseHueOffset: 0.08,
  bandQuantSteps: 9,
  /** mix(d, floor(d*9)/9, uBandStrength * bandMix) */
  bandMix: 0.55,
  /** mix(1, 0.88 + 0.24*ring, uBandStrength * logRingBlend) */
  logRingScale: 3.5,
  logRingBlend: 0.45,
  satLow: 0.22,
  satHigh: 0.76,
  densitySatStart: 0.08,
  densitySatEnd: 0.42,
  valBase: 0.18,
  valRange: 0.78,
} as const
