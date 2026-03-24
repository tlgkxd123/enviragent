import {
  DataTexture,
  FloatType,
  LinearFilter,
  LinearMipmapLinearFilter,
  PMREMGenerator,
  RGBAFormat,
  SRGBColorSpace,
} from 'three'
import type { WebGLRenderer, Texture } from 'three'

/**
 * LLM-driven procedural environment generator.
 *
 * The user describes an environment in natural language (e.g. "deep space nebula",
 * "golden hour desert", "underwater bioluminescent cave").
 *
 * 1. Sends the description to an OpenAI-compatible /chat/completions endpoint.
 * 2. The model returns a JSON config describing up to 4 HDR light lobes + sky gradient.
 * 3. A 4096×2048 Float32 equirect is synthesised from that config and baked to PMREM.
 *
 * API key is session-only — never persisted to localStorage or sent anywhere except
 * the configured endpoint.
 */

export interface EnvLight {
  /** World direction the light comes FROM (Y-up, need not be normalised) */
  direction: [number, number, number]
  /** Linear HDR colour — values can exceed 1 for bright sources */
  color: [number, number, number]
  /** Lobe sharpness power: 1 = broad area, 18 = sharp sun */
  sharpness: number
  /** Peak intensity multiplier */
  intensity: number
}

export interface EnvConfig {
  /** Linear HDR sky colour at zenith */
  skyZenith: [number, number, number]
  /** Linear HDR sky colour at horizon */
  skyHorizon: [number, number, number]
  /** Linear HDR ground bounce */
  ground: [number, number, number]
  /** 1–4 directional light sources */
  lights: EnvLight[]
  /** Subtle noise shimmer strength 0–1 */
  shimmer: number
}

const SYSTEM_PROMPT = `You are a physically-based lighting designer for a 3D renderer.
When given a scene description, respond with ONLY a valid JSON object (no markdown fences, no explanation) matching:
{
  "skyZenith":  [r, g, b],  // linear HDR, zenith sky, values 0..0.08
  "skyHorizon": [r, g, b],  // linear HDR, horizon sky, values 0..0.05
  "ground":     [r, g, b],  // linear HDR, ground bounce, values 0..0.03
  "lights": [
    {
      "direction":  [x, y, z],  // normalised Y-up world direction FROM which light arrives
      "color":      [r, g, b],  // linear HDR, bright sources can exceed 1 (e.g. sun=[3.2,2.6,1.6])
      "sharpness":  number,     // lobe power: sun=18, softbox=3, area=1.5, candle=6
      "intensity":  number      // peak multiplier, typical 0.5..20
    }
  ],
  "shimmer": number  // subtle noise shimmer 0..1
}
Use 1–4 lights. Key light should be brightest. Return ONLY the JSON.`

export async function generateEnvFromDescription(
  description: string,
  apiKey: string,
  model = 'gpt-4o-mini',
  baseUrl = 'https://api.openai.com/v1'
): Promise<EnvConfig> {
  const res = await fetch(`${baseUrl}/chat/completions`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model,
      temperature: 0.7,
      max_tokens: 600,
      messages: [
        { role: 'system', content: SYSTEM_PROMPT },
        { role: 'user', content: description },
      ],
    }),
  })

  if (!res.ok) {
    const text = await res.text()
    throw new Error(`API error ${res.status}: ${text}`)
  }

  const json = await res.json() as { choices?: Array<{ message?: { content?: string } }> }
  const content = json.choices?.[0]?.message?.content ?? ''

  // Strip possible markdown code fences the model may emit despite instructions
  const cleaned = content
    .replace(/^```(?:json)?\s*/i, '')
    .replace(/\s*```\s*$/m, '')
    .trim()

  let config: EnvConfig
  try {
    config = JSON.parse(cleaned) as EnvConfig
  } catch {
    throw new Error(`Model returned invalid JSON:\n${content}`)
  }

  return config
}

/**
 * Synthesise a 4096×2048 HDR equirect from an EnvConfig,
 * bake to PMREM, and return the env cubemap texture.
 */
export function buildEnvFromConfig(
  config: EnvConfig,
  renderer: WebGLRenderer
): { envMap: Texture; pmrem: PMREMGenerator } {
  const W = 4096
  const H = 2048
  const data = new Float32Array(W * H * 4)

  const lights = (config.lights ?? []).slice(0, 4).map((l) => {
    const len = Math.hypot(l.direction[0], l.direction[1], l.direction[2]) || 1
    return {
      dx: l.direction[0] / len,
      dy: l.direction[1] / len,
      dz: l.direction[2] / len,
      r: l.color[0],
      g: l.color[1],
      b: l.color[2],
      sharp: Math.max(1, l.sharpness ?? 4),
      int: Math.max(0, l.intensity ?? 1),
    }
  })

  const sz = config.skyZenith  ?? [0.004, 0.005, 0.009]
  const sh = config.skyHorizon ?? [0.002, 0.003, 0.005]
  const gr = config.ground     ?? [0.008, 0.006, 0.003]
  const shimmer = config.shimmer ?? 0

  for (let y = 0; y < H; y++) {
    const v     = y / (H - 1)
    const theta = v * Math.PI
    const sinT  = Math.sin(theta)
    const cosT  = Math.cos(theta)

    for (let x = 0; x < W; x++) {
      const u   = x / (W - 1)
      const phi = u * 2 * Math.PI - Math.PI
      const dx  = sinT * Math.cos(phi)
      const dy  = cosT
      const dz  = sinT * Math.sin(phi)

      const upness   = Math.max(0,  dy)
      const downness = Math.max(0, -dy)
      const horiz    = 1 - Math.abs(dy)

      // Sky gradient
      let r = sz[0] * upness + sh[0] * horiz * 0.6
      let g = sz[1] * upness + sh[1] * horiz * 0.6
      let b = sz[2] * upness + sh[2] * horiz * 0.6

      // Ground bounce
      r += gr[0] * downness
      g += gr[1] * downness
      b += gr[2] * downness

      // Light lobes
      for (const l of lights) {
        const dot  = Math.max(0, dx * l.dx + dy * l.dy + dz * l.dz)
        const lobe = Math.pow(dot, l.sharp) * l.int
        r += lobe * l.r
        g += lobe * l.g
        b += lobe * l.b
      }

      // Shimmer noise
      if (shimmer > 0) {
        const n = Math.sin(phi * 7.3 + theta * 5.1) * Math.cos(phi * 3.7 - theta * 9.2)
        const s = Math.max(0, n) * 0.006 * shimmer
        r += s * 0.6; g += s * 0.4; b += s * 1.2
      }

      const i = (y * W + x) * 4
      data[i] = r; data[i + 1] = g; data[i + 2] = b; data[i + 3] = 1.0
    }
  }

  const tex = new DataTexture(data, W, H, RGBAFormat, FloatType)
  tex.colorSpace = SRGBColorSpace
  tex.minFilter  = LinearMipmapLinearFilter
  tex.magFilter  = LinearFilter
  tex.generateMipmaps = true
  tex.needsUpdate = true

  const pmrem  = new PMREMGenerator(renderer)
  pmrem.compileEquirectangularShader()
  const envMap = pmrem.fromEquirectangular(tex).texture
  tex.dispose()

  return { envMap, pmrem }
}

// ── Built-in presets (no API key required) ──────────────────────────────────

export const ENV_PRESETS: Record<string, EnvConfig> = {
  'Studio (default)': {
    skyZenith:  [0.004, 0.005, 0.009],
    skyHorizon: [0.002, 0.003, 0.005],
    ground:     [0.018, 0.012, 0.006],
    shimmer: 0,
    lights: [
      { direction: [ 0.65,  0.55, -0.52], color: [3.20, 2.55, 1.60], sharpness: 3,  intensity: 6.2 },
      { direction: [-0.72,  0.48, -0.50], color: [0.82, 0.94, 1.60], sharpness: 2.5,intensity: 2.8 },
      { direction: [-0.05,  0.62,  0.78], color: [0.30, 0.90, 1.20], sharpness: 6,  intensity: 4.5 },
      { direction: [ 0.0,  -1.0,   0.0],  color: [0.55, 0.32, 0.08], sharpness: 1.6,intensity: 1.2 },
    ],
  },
  'Deep space nebula': {
    skyZenith:  [0.002, 0.003, 0.012],
    skyHorizon: [0.001, 0.002, 0.008],
    ground:     [0.001, 0.001, 0.003],
    shimmer: 0.9,
    lights: [
      { direction: [ 0.4,  0.7, -0.6], color: [2.8, 2.2, 1.4], sharpness: 16, intensity: 14 },
      { direction: [-0.8,  0.3,  0.5], color: [0.4, 0.6, 2.2], sharpness: 2,  intensity: 1.8 },
      { direction: [ 0.1, -0.9,  0.4], color: [0.6, 0.2, 1.4], sharpness: 1,  intensity: 0.8 },
    ],
  },
  'Golden hour desert': {
    skyZenith:  [0.018, 0.024, 0.048],
    skyHorizon: [0.048, 0.028, 0.008],
    ground:     [0.022, 0.014, 0.004],
    shimmer: 0.1,
    lights: [
      { direction: [ 0.70,  0.18, -0.69], color: [3.8, 2.2, 0.8], sharpness: 14, intensity: 18 },
      { direction: [-0.50,  0.60,  0.60], color: [0.9, 0.95, 1.6], sharpness: 1,  intensity: 1.2 },
    ],
  },
  'Underwater bioluminescent': {
    skyZenith:  [0.001, 0.008, 0.018],
    skyHorizon: [0.000, 0.004, 0.012],
    ground:     [0.000, 0.002, 0.006],
    shimmer: 0.6,
    lights: [
      { direction: [ 0.2,  0.8, -0.6], color: [0.2, 1.8, 1.4], sharpness: 4, intensity: 6 },
      { direction: [-0.6,  0.4,  0.7], color: [0.4, 0.4, 2.2], sharpness: 3, intensity: 3 },
      { direction: [ 0.8, -0.2, -0.6], color: [0.2, 1.2, 0.6], sharpness: 5, intensity: 2 },
    ],
  },
  'Neon noir city': {
    skyZenith:  [0.003, 0.002, 0.008],
    skyHorizon: [0.006, 0.002, 0.010],
    ground:     [0.004, 0.002, 0.006],
    shimmer: 0.3,
    lights: [
      { direction: [ 0.5,  0.7, -0.5], color: [0.4, 0.2, 2.4], sharpness: 3,  intensity: 5  },
      { direction: [-0.6,  0.5,  0.6], color: [2.4, 0.1, 0.4], sharpness: 2,  intensity: 3  },
      { direction: [ 0.0,  0.9,  0.4], color: [0.1, 1.8, 0.8], sharpness: 4,  intensity: 2  },
    ],
  },
}
