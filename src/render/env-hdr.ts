import {
  DataTexture,
  FloatType,
  LinearFilter,
  LinearMipmapLinearFilter,
  PMREMGenerator,
  RGBAFormat,
  SRGBColorSpace,
  Texture,
} from 'three'
import type { WebGLRenderer } from 'three'

/**
 * Build a 4096×2048 HDR equirectangular environment in a Float32 CPU buffer,
 * then bake it to a PMREM cubemap via PMREMGenerator.
 *
 * The environment is a "photographer's studio" with:
 *   - Warm key softbox  (upper-right)
 *   - Cool fill panel   (upper-left)
 *   - Ground bounce     (bottom)
 *   - Teal rim strip    (back-upper)
 *   - Deep space gradient background with subtle nebula shimmer
 *
 * All computed analytically — zero GPU draw calls during bake input.
 * The PMREM output drives both scene.environment and the orbital shader envMap,
 * giving effectively 8K-quality filtered reflections at one-time bake cost.
 */
export function buildStudioEnvMap(renderer: WebGLRenderer): {
  envMap: Texture
  pmrem: PMREMGenerator
} {
  const W = 4096
  const H = 2048
  const data = new Float32Array(W * H * 4)

  for (let y = 0; y < H; y++) {
    const v = y / (H - 1)             // 0 = top, 1 = bottom
    const theta = v * Math.PI         // polar: 0=north pole, π=south
    const sinT = Math.sin(theta)
    const cosT = Math.cos(theta)

    for (let x = 0; x < W; x++) {
      const u = x / (W - 1)           // 0..1 azimuth
      const phi = u * 2 * Math.PI - Math.PI  // -π..π
      const sinP = Math.sin(phi)
      const cosP = Math.cos(phi)

      // World direction (Y-up)
      const dx = sinT * cosP
      const dy = cosT
      const dz = sinT * sinP

      let r = 0, g = 0, b = 0

      // ── Background gradient (deep space, very dark) ──────────────
      const upness = Math.max(0, dy)          // 0..1 above horizon
      const downness = Math.max(0, -dy)
      r += 0.004 + 0.012 * upness
      g += 0.005 + 0.010 * upness
      b += 0.009 + 0.022 * upness
      // Warm ground
      r += 0.018 * downness
      g += 0.012 * downness
      b += 0.006 * downness

      // ── Key light: warm softbox upper-right-front ─────────────────
      // Direction: (0.65, 0.55, -0.52) normalised
      const kx = 0.65, ky = 0.55, kz = -0.52
      const kLen = Math.hypot(kx, ky, kz)
      const dot_k = Math.max(0, dx * kx / kLen + dy * ky / kLen + dz * kz / kLen)
      // Softbox: wide lobe (power 3) + broad falloff
      const key = Math.pow(dot_k, 3) * 6.2
      r += key * 3.20
      g += key * 2.55
      b += key * 1.60
      // Inner hot spot
      const keyHot = Math.pow(dot_k, 18) * 22.0
      r += keyHot * 3.8
      g += keyHot * 3.2
      b += keyHot * 2.2

      // ── Fill light: cool blue-white panel upper-left ──────────────
      const fx = -0.72, fy = 0.48, fz = -0.50
      const fLen = Math.hypot(fx, fy, fz)
      const dot_f = Math.max(0, dx * fx / fLen + dy * fy / fLen + dz * fz / fLen)
      const fill = Math.pow(dot_f, 2.5) * 2.8
      r += fill * 0.82
      g += fill * 0.94
      b += fill * 1.60

      // ── Rim light: teal strip back-upper ──────────────────────────
      const rx2 = -0.05, ry2 = 0.62, rz2 = 0.78
      const rLen = Math.hypot(rx2, ry2, rz2)
      const dot_r = Math.max(0, dx * rx2 / rLen + dy * ry2 / rLen + dz * rz2 / rLen)
      const rim = Math.pow(dot_r, 6) * 4.5
      r += rim * 0.30
      g += rim * 0.90
      b += rim * 1.20

      // ── Ground bounce: warm amber below horizon ───────────────────
      const gb = Math.pow(downness, 1.6) * 1.2
      r += gb * 0.55
      g += gb * 0.32
      b += gb * 0.08

      // ── Subtle nebula shimmer (hashed noise) ─────────────────────
      const nx = Math.sin(phi * 7.3 + theta * 5.1) * Math.cos(phi * 3.7 - theta * 9.2)
      const shimmer = Math.max(0, nx) * 0.006
      r += shimmer * 0.6
      g += shimmer * 0.4
      b += shimmer * 1.2

      const i = (y * W + x) * 4
      data[i]     = r
      data[i + 1] = g
      data[i + 2] = b
      data[i + 3] = 1.0
    }
  }

  const tex = new DataTexture(data, W, H, RGBAFormat, FloatType)
  tex.colorSpace = SRGBColorSpace
  tex.minFilter = LinearMipmapLinearFilter
  tex.magFilter = LinearFilter
  tex.generateMipmaps = true
  tex.needsUpdate = true

  const pmrem = new PMREMGenerator(renderer)
  pmrem.compileEquirectangularShader()
  const envMap = pmrem.fromEquirectangular(tex).texture
  tex.dispose()

  return { envMap, pmrem }
}
