import * as THREE from 'three'

// ═══════════════════════════════════════════════════════════════════════════════
// Gerstner-Wave Ocean Mesh
//
// Real 3D vertex displacement — each frame, multiple Gerstner wave components
// move every vertex horizontally and vertically to create sharp crests with
// wide troughs. Normals are computed analytically from wave derivatives.
// Foam is derived from the surface Jacobian (convergence → whitecaps).
// Material: MeshPhysicalMaterial with transmission + IOR for translucent water.
// ═══════════════════════════════════════════════════════════════════════════════

// ── Gerstner wave math ──────────────────────────────────────────────────────

interface GerstnerWave {
  dir: [number, number]   // normalized direction on XZ plane
  amplitude: number       // vertical amplitude
  frequency: number       // spatial frequency (2π / wavelength)
  speed: number           // phase speed
  steepness: number       // Q factor (0 = sine, 1 = max Gerstner sharpness)
}

function buildWaveSet(
  count: number,
  baseAmplitude: number,
  baseWavelength: number,
  choppiness: number,
  seed: number,
): GerstnerWave[] {
  const waves: GerstnerWave[] = []
  let rng = seed | 0 || 137
  const lcg = () => { rng = (rng * 1103515245 + 12345) & 0x7fffffff; return rng / 0x7fffffff }

  for (let i = 0; i < count; i++) {
    const ratio = 1 + i * 0.6
    const wavelength = baseWavelength / ratio
    const amp = baseAmplitude / (ratio * ratio) * (0.7 + lcg() * 0.6)
    const freq = (2 * Math.PI) / wavelength
    const angle = (lcg() - 0.5) * Math.PI * 1.2 + (lcg() - 0.5) * 0.4
    const dx = Math.cos(angle)
    const dz = Math.sin(angle)
    const speed = Math.sqrt(9.81 / freq) // deep-water dispersion
    const Q = Math.min(choppiness / (freq * amp * count), 1.0)
    waves.push({ dir: [dx, dz], amplitude: amp, frequency: freq, speed, steepness: Q })
  }
  return waves
}

// ── Mesh update ─────────────────────────────────────────────────────────────

function updateGerstnerMesh(
  geo: THREE.PlaneGeometry,
  waves: GerstnerWave[],
  t: number,
  basePositions: Float32Array,
  Nx: number,
  Nz: number,
  _Lx: number,
  _Lz: number,
  foamEnabled: boolean,
) {
  const pos = geo.attributes.position as THREE.BufferAttribute
  const nor = geo.attributes.normal as THREE.BufferAttribute
  const col = foamEnabled ? geo.attributes.color as THREE.BufferAttribute : null

  const pArr = pos.array as Float32Array
  const nArr = nor.array as Float32Array
  const cArr = col ? col.array as Float32Array : null

  const nWaves = waves.length

  for (let iz = 0; iz <= Nz; iz++) {
    for (let ix = 0; ix <= Nx; ix++) {
      const idx = iz * (Nx + 1) + ix
      const i3 = idx * 3

      const bx = basePositions[i3]
      const bz = basePositions[i3 + 2]

      let dx = 0, dy = 0, dz = 0
      let ddx_dx = 0, ddx_dz = 0, ddz_dx = 0, ddz_dz = 0
      let ddy_dx = 0, ddy_dz = 0

      for (let w = 0; w < nWaves; w++) {
        const wv = waves[w]
        const Dx = wv.dir[0], Dz = wv.dir[1]
        const A = wv.amplitude, f = wv.frequency, Q = wv.steepness
        const phase = f * (Dx * bx + Dz * bz) - wv.speed * f * t
        const s = Math.sin(phase)
        const c = Math.cos(phase)

        dx -= Q * A * Dx * s
        dz -= Q * A * Dz * s
        dy += A * c

        const fA = f * A
        const fAc = fA * c
        const fAs = fA * s

        ddx_dx -= Q * Dx * Dx * fAc
        ddx_dz -= Q * Dx * Dz * fAc
        ddz_dx -= Q * Dz * Dx * fAc
        ddz_dz -= Q * Dz * Dz * fAc
        ddy_dx -= Dx * fAs
        ddy_dz -= Dz * fAs
      }

      pArr[i3] = bx + dx
      pArr[i3 + 1] = dy
      pArr[i3 + 2] = bz + dz

      // Analytical normal from Gerstner partial derivatives
      const Jxx = 1 + ddx_dx
      const Jzz = 1 + ddz_dz
      const Jxz = ddx_dz
      const Jzx = ddz_dx

      const nx = -ddy_dx
      const nz = -ddy_dz
      const ny = 1.0
      const nLen = Math.sqrt(nx * nx + ny * ny + nz * nz)
      nArr[i3] = nx / nLen
      nArr[i3 + 1] = ny / nLen
      nArr[i3 + 2] = nz / nLen

      if (cArr) {
        const J = Jxx * Jzz - Jxz * Jzx
        const foam = Math.max(0, Math.min(1, 1 - J))
        const base = 0.02
        cArr[i3] = base + foam * 0.95
        cArr[i3 + 1] = base + foam * 0.97
        cArr[i3 + 2] = base + foam * 1.0
      }
    }
  }

  pos.needsUpdate = true
  nor.needsUpdate = true
  if (col) col.needsUpdate = true
}

// ── Public API ──────────────────────────────────────────────────────────────

export interface WaterOptions {
  size?: [number, number]
  resolution?: number
  color?: number
  wave_scale?: number
  choppiness?: number
  wave_count?: number
  speed?: number
  foam?: boolean
  seed?: number
  opacity?: number
}

export interface WaterHandle {
  mesh: THREE.Mesh
  update: (dt: number, t: number) => void
  dispose: () => void
}

export function createFFTWater(opts: WaterOptions = {}): WaterHandle {
  const [Lx, Lz] = opts.size ?? [40, 40]
  const res = Math.min(512, Math.max(32, opts.resolution ?? 128))
  const Nx = res, Nz = res
  const waveScale = opts.wave_scale ?? 0.4
  const choppiness = opts.choppiness ?? 0.8
  const waveCount = Math.min(32, Math.max(3, opts.wave_count ?? 12))
  const globalSpeed = opts.speed ?? 1.0
  const foamEnabled = opts.foam !== false
  const waterColor = opts.color ?? 0x006994
  const opacity = opts.opacity ?? 0.85
  const seed = opts.seed ?? 42

  const baseWavelength = Math.max(Lx, Lz) * 0.6
  const waves = buildWaveSet(waveCount, waveScale, baseWavelength, choppiness, seed)

  const geo = new THREE.PlaneGeometry(Lx, Lz, Nx, Nz)
  geo.rotateX(-Math.PI / 2)

  const basePositions = new Float32Array(geo.attributes.position.count * 3)
  basePositions.set(geo.attributes.position.array as Float32Array)

  if (foamEnabled) {
    const colors = new Float32Array(geo.attributes.position.count * 3)
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3))
  }

  ;(geo.attributes.position as THREE.BufferAttribute).usage = THREE.DynamicDrawUsage
  ;(geo.attributes.normal as THREE.BufferAttribute).usage = THREE.DynamicDrawUsage

  const mat = new THREE.MeshPhysicalMaterial({
    color: waterColor,
    transparent: true,
    opacity,
    roughness: 0.05,
    metalness: 0.0,
    transmission: 0.6,
    ior: 1.33,
    thickness: 2.0,
    envMapIntensity: 1.2,
    side: THREE.DoubleSide,
    flatShading: false,
    vertexColors: foamEnabled,
  })

  const mesh = new THREE.Mesh(geo, mat)
  mesh.receiveShadow = true
  mesh.castShadow = false

  const handle: WaterHandle = {
    mesh,
    update(_dt: number, t: number) {
      updateGerstnerMesh(geo, waves, t * globalSpeed, basePositions, Nx, Nz, Lx, Lz, foamEnabled)
    },
    dispose() {
      geo.dispose()
      mat.dispose()
    },
  }

  handle.update(0, 0)
  return handle
}
