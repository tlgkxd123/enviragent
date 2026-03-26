import * as THREE from 'three'

// ═══════════════════════════════════════════════════════════════════════════════
// Advanced procedural object generators
//
// Algorithms:
//   Terrain    – multi-octave domain-warped FBM → particle-based hydraulic
//                erosion (Beyer 2015) with sediment transport + thermal talus
//   NoiseSphere – 3D gradient noise, ridged multi-fractal, domain warping
//   Scatter    – Poisson disk sampling (Bridson 2007) for blue-noise placement
//   Crystals   – hexagonal prism growth from Voronoi seed competition
//   Rocks      – subdivided icosahedra + 3D multi-scale noise displacement
//                  with erosion smoothing on high-curvature vertices
// ═══════════════════════════════════════════════════════════════════════════════

// ── Seeded PRNG (xoshiro128**) ──────────────────────────────────────────────

function xoshiro128ss(seed: number): () => number {
  let s0 = seed >>> 0 || 1
  let s1 = (seed * 2654435761) >>> 0 || 1
  let s2 = (seed * 2246822519) >>> 0 || 1
  let s3 = (seed * 3266489917) >>> 0 || 1
  return () => {
    const r = Math.imul(s1 * 5, 7) >>> 0
    const t = (s1 << 9) >>> 0
    s2 ^= s0; s3 ^= s1; s1 ^= s2; s0 ^= s3
    s2 ^= t; s3 = ((s3 << 11) | (s3 >>> 21)) >>> 0
    return (r >>> 0) / 4294967296
  }
}

// ── 3D gradient noise with analytical derivatives ───────────────────────────

const GRAD3 = [
  1, 1, 0, -1, 1, 0, 1, -1, 0, -1, -1, 0,
  1, 0, 1, -1, 0, 1, 1, 0, -1, -1, 0, -1,
  0, 1, 1, 0, -1, 1, 0, 1, -1, 0, -1, -1,
]

function buildPerm(seed: number): Uint8Array {
  const p = new Uint8Array(512)
  const rng = xoshiro128ss(seed)
  for (let i = 0; i < 256; i++) p[i] = i
  for (let i = 255; i > 0; i--) {
    const j = (rng() * (i + 1)) >>> 0
    const tmp = p[i]; p[i] = p[j]; p[j] = tmp
  }
  for (let i = 0; i < 256; i++) p[i + 256] = p[i]
  return p
}

function smoothstep(t: number): number { return t * t * t * (t * (t * 6 - 15) + 10) }
function smoothstepDeriv(t: number): number { return 30 * t * t * (t * (t - 2) + 1) }

interface NoiseResult { value: number; dx: number; dy: number; dz: number }

function noise3d(x: number, y: number, z: number, perm: Uint8Array): NoiseResult {
  const ix = Math.floor(x), iy = Math.floor(y), iz = Math.floor(z)
  const fx = x - ix, fy = y - iy, fz = z - iz
  const X = ix & 255, Y = iy & 255, Z = iz & 255

  const ux = smoothstep(fx), uy = smoothstep(fy), uz = smoothstep(fz)
  const dux = smoothstepDeriv(fx), duy = smoothstepDeriv(fy), duz = smoothstepDeriv(fz)

  function grad(hash: number, gx: number, gy: number, gz: number): number {
    const h = (hash & 11) * 3
    return GRAD3[h] * gx + GRAD3[h + 1] * gy + GRAD3[h + 2] * gz
  }

  const a = perm[X] + Y, aa = perm[a] + Z, ab = perm[a + 1] + Z
  const b = perm[X + 1] + Y, ba = perm[b] + Z, bb = perm[b + 1] + Z

  const g000 = grad(perm[aa], fx, fy, fz)
  const g100 = grad(perm[ba], fx - 1, fy, fz)
  const g010 = grad(perm[ab], fx, fy - 1, fz)
  const g110 = grad(perm[bb], fx - 1, fy - 1, fz)
  const g001 = grad(perm[aa + 1], fx, fy, fz - 1)
  const g101 = grad(perm[ba + 1], fx - 1, fy, fz - 1)
  const g011 = grad(perm[ab + 1], fx, fy - 1, fz - 1)
  const g111 = grad(perm[bb + 1], fx - 1, fy - 1, fz - 1)

  const x00 = g000 + ux * (g100 - g000), x10 = g010 + ux * (g110 - g010)
  const x01 = g001 + ux * (g101 - g001), x11 = g011 + ux * (g111 - g011)
  const xy0 = x00 + uy * (x10 - x00), xy1 = x01 + uy * (x11 - x01)
  const value = xy0 + uz * (xy1 - xy0)

  const dx00 = (g100 - g000), dx10 = (g110 - g010)
  const dx01 = (g101 - g001), dx11 = (g111 - g011)
  const dxy0 = dx00 + uy * (dx10 - dx00), dxy1 = dx01 + uy * (dx11 - dx01)
  const dx = dux * (dxy0 + uz * (dxy1 - dxy0))

  const dy = duy * ((x10 - x00) + uz * ((x11 - x01) - (x10 - x00)))
  const dz = duz * (xy1 - xy0)

  return { value, dx, dy, dz }
}

// ── Multi-octave FBM with domain warping ────────────────────────────────────

interface FbmOpts {
  octaves: number
  lacunarity: number
  gain: number
  warpStrength: number
  ridged: boolean
}

function fbm3(x: number, y: number, z: number, perm: Uint8Array, opts: FbmOpts): NoiseResult {
  let { octaves, lacunarity, gain, warpStrength, ridged } = opts

  if (warpStrength > 0) {
    const w1 = noise3d(x + 5.2, y + 1.3, z + 7.8, perm)
    const w2 = noise3d(x + 1.7, y + 9.2, z + 3.4, perm)
    const w3 = noise3d(x + 8.3, y + 2.8, z + 6.1, perm)
    x += w1.value * warpStrength
    y += w2.value * warpStrength
    z += w3.value * warpStrength
  }

  let value = 0, amplitude = 1, frequency = 1, maxAmp = 0
  let dx = 0, dy = 0, dz = 0

  for (let i = 0; i < octaves; i++) {
    const n = noise3d(x * frequency, y * frequency, z * frequency, perm)
    let v = n.value
    if (ridged) v = 1 - Math.abs(v)
    value += v * amplitude
    dx += n.dx * amplitude * frequency
    dy += n.dy * amplitude * frequency
    dz += n.dz * amplitude * frequency
    maxAmp += amplitude
    amplitude *= gain
    frequency *= lacunarity
  }

  const s = 1 / maxAmp
  return { value: value * s, dx: dx * s, dy: dy * s, dz: dz * s }
}

function fbm2Terrain(
  nx: number, nz: number, perm: Uint8Array, opts: FbmOpts
): { height: number; gx: number; gz: number } {
  const n = fbm3(nx, 0, nz, perm, opts)
  return { height: n.value, gx: n.dx, gz: n.dz }
}

// ── Hydraulic erosion (Beyer 2015 particle method) ──────────────────────────

interface ErosionOpts {
  iterations: number
  inertia: number
  sedimentCapacity: number
  minSedimentCapacity: number
  erosionRate: number
  depositionRate: number
  evaporationRate: number
  gravity: number
  maxLifetime: number
  erosionRadius: number
}

const DEFAULT_EROSION: ErosionOpts = {
  iterations: 50000,
  inertia: 0.05,
  sedimentCapacity: 4,
  minSedimentCapacity: 0.01,
  erosionRate: 0.3,
  depositionRate: 0.3,
  evaporationRate: 0.01,
  gravity: 4,
  maxLifetime: 30,
  erosionRadius: 3,
}

function hydraulicErosion(
  heights: Float32Array, w: number, h: number,
  opts: ErosionOpts, rng: () => number
): void {
  const { iterations, inertia, sedimentCapacity, minSedimentCapacity,
    erosionRate, depositionRate, evaporationRate, gravity, maxLifetime, erosionRadius } = opts

  const erodeR = Math.ceil(erosionRadius)

  function heightAndGrad(px: number, pz: number) {
    const ix = Math.floor(px), iz = Math.floor(pz)
    const fx = px - ix, fz = pz - iz
    if (ix < 0 || ix >= w - 1 || iz < 0 || iz >= h - 1) {
      return { h: 0, gx: 0, gz: 0 }
    }
    const h00 = heights[iz * w + ix]
    const h10 = heights[iz * w + ix + 1]
    const h01 = heights[(iz + 1) * w + ix]
    const h11 = heights[(iz + 1) * w + ix + 1]
    const height = h00 * (1 - fx) * (1 - fz) + h10 * fx * (1 - fz) + h01 * (1 - fx) * fz + h11 * fx * fz
    const gx = (h10 - h00) * (1 - fz) + (h11 - h01) * fz
    const gz = (h01 - h00) * (1 - fx) + (h11 - h10) * fx
    return { h: height, gx, gz }
  }

  function deposit(px: number, pz: number, amount: number) {
    const ix = Math.floor(px), iz = Math.floor(pz)
    if (ix < 0 || ix >= w - 1 || iz < 0 || iz >= h - 1) return
    const fx = px - ix, fz = pz - iz
    heights[iz * w + ix] += amount * (1 - fx) * (1 - fz)
    heights[iz * w + ix + 1] += amount * fx * (1 - fz)
    heights[(iz + 1) * w + ix] += amount * (1 - fx) * fz
    heights[(iz + 1) * w + ix + 1] += amount * fx * fz
  }

  function erode(px: number, pz: number, amount: number) {
    const cx = Math.floor(px), cz = Math.floor(pz)
    let totalWeight = 0
    const weights: { idx: number; w: number }[] = []

    for (let dz = -erodeR; dz <= erodeR; dz++) {
      for (let dx = -erodeR; dx <= erodeR; dx++) {
        const gx = cx + dx, gz = cz + dz
        if (gx < 0 || gx >= w || gz < 0 || gz >= h) continue
        const dist = Math.sqrt((gx - px) ** 2 + (gz - pz) ** 2)
        if (dist > erosionRadius) continue
        const weight = Math.max(0, erosionRadius - dist)
        totalWeight += weight
        weights.push({ idx: gz * w + gx, w: weight })
      }
    }

    if (totalWeight === 0) return
    for (const e of weights) {
      heights[e.idx] -= amount * (e.w / totalWeight)
    }
  }

  for (let iter = 0; iter < iterations; iter++) {
    let px = rng() * (w - 2) + 0.5
    let pz = rng() * (h - 2) + 0.5
    let dirX = 0, dirZ = 0
    let speed = 1, water = 1, sediment = 0

    for (let step = 0; step < maxLifetime; step++) {
      const { h: curH, gx, gz } = heightAndGrad(px, pz)

      dirX = dirX * inertia - gx * (1 - inertia)
      dirZ = dirZ * inertia - gz * (1 - inertia)
      const len = Math.sqrt(dirX * dirX + dirZ * dirZ)
      if (len > 0) { dirX /= len; dirZ /= len }

      const newX = px + dirX
      const newZ = pz + dirZ

      if (newX < 1 || newX >= w - 2 || newZ < 1 || newZ >= h - 2) break

      const { h: newH } = heightAndGrad(newX, newZ)
      const deltaH = newH - curH

      const cap = Math.max(-deltaH * speed * water * sedimentCapacity, minSedimentCapacity)

      if (sediment > cap || deltaH > 0) {
        const depositAmt = deltaH > 0
          ? Math.min(deltaH, sediment)
          : (sediment - cap) * depositionRate
        sediment -= depositAmt
        deposit(px, pz, depositAmt)
      } else {
        const erodeAmt = Math.min((cap - sediment) * erosionRate, -deltaH)
        sediment += erodeAmt
        erode(px, pz, erodeAmt)
      }

      speed = Math.sqrt(Math.max(speed * speed + deltaH * gravity, 0.001))
      water *= (1 - evaporationRate)
      px = newX
      pz = newZ
    }
  }
}

// ── Thermal erosion (talus angle) ───────────────────────────────────────────

function thermalErosion(
  heights: Float32Array, w: number, h: number,
  talusAngle: number, iterations: number
): void {
  const maxDelta = Math.tan(talusAngle)
  const DX = [-1, 0, 1, 0, -1, 1, 1, -1]
  const DZ = [0, -1, 0, 1, -1, -1, 1, 1]

  for (let iter = 0; iter < iterations; iter++) {
    for (let z = 1; z < h - 1; z++) {
      for (let x = 1; x < w - 1; x++) {
        const idx = z * w + x
        const ch = heights[idx]
        let maxSlope = 0, totalSlope = 0
        const slopes: number[] = []
        const indices: number[] = []

        for (let d = 0; d < 8; d++) {
          const ni = (z + DZ[d]) * w + (x + DX[d])
          const diff = ch - heights[ni]
          if (diff > maxDelta) {
            slopes.push(diff - maxDelta)
            indices.push(ni)
            totalSlope += diff - maxDelta
            if (diff > maxSlope) maxSlope = diff
          }
        }

        if (totalSlope > 0) {
          const move = (maxSlope - maxDelta) * 0.5
          for (let d = 0; d < slopes.length; d++) {
            heights[indices[d]] += move * (slopes[d] / totalSlope)
          }
          heights[idx] -= move
        }
      }
    }
  }
}

// ── Poisson disk sampling (Bridson 2007) ────────────────────────────────────

interface PoissonPoint { x: number; z: number }

function poissonDiskSampling(
  width: number, depth: number, minDist: number,
  rng: () => number, maxSamples = 30
): PoissonPoint[] {
  const cellSize = minDist / Math.SQRT2
  const gridW = Math.ceil(width / cellSize)
  const gridH = Math.ceil(depth / cellSize)
  const grid: (PoissonPoint | null)[] = new Array(gridW * gridH).fill(null)
  const points: PoissonPoint[] = []
  const active: number[] = []

  function gridIdx(x: number, z: number): number {
    const gx = Math.floor(x / cellSize)
    const gz = Math.floor(z / cellSize)
    return gz * gridW + gx
  }

  function tooClose(x: number, z: number): boolean {
    const gx = Math.floor(x / cellSize)
    const gz = Math.floor(z / cellSize)
    for (let dz = -2; dz <= 2; dz++) {
      for (let dx = -2; dx <= 2; dx++) {
        const nx = gx + dx, nz = gz + dz
        if (nx < 0 || nx >= gridW || nz < 0 || nz >= gridH) continue
        const p = grid[nz * gridW + nx]
        if (p && (p.x - x) ** 2 + (p.z - z) ** 2 < minDist * minDist) return true
      }
    }
    return false
  }

  const x0 = rng() * width, z0 = rng() * depth
  const p0: PoissonPoint = { x: x0, z: z0 }
  points.push(p0)
  active.push(0)
  grid[gridIdx(x0, z0)] = p0

  while (active.length > 0) {
    const ai = (rng() * active.length) >>> 0
    const pidx = active[ai]
    const base = points[pidx]
    let found = false

    for (let k = 0; k < maxSamples; k++) {
      const angle = rng() * Math.PI * 2
      const r = minDist + rng() * minDist
      const nx = base.x + Math.cos(angle) * r
      const nz = base.z + Math.sin(angle) * r
      if (nx < 0 || nx >= width || nz < 0 || nz >= depth) continue
      if (tooClose(nx, nz)) continue

      const np: PoissonPoint = { x: nx, z: nz }
      points.push(np)
      active.push(points.length - 1)
      grid[gridIdx(nx, nz)] = np
      found = true
      break
    }

    if (!found) active.splice(ai, 1)
  }

  return points
}

// ── Public: Terrain ─────────────────────────────────────────────────────────

export interface TerrainOptions {
  fbmOctaves?: number
  fbmLacunarity?: number
  fbmGain?: number
  warpStrength?: number
  ridged?: boolean
  erosionIterations?: number
  erosionEnabled?: boolean
  thermalIterations?: number
  talusAngle?: number
}

export function createTerrainGeometry(
  width: number,
  depth: number,
  segX: number,
  segZ: number,
  heightScale: number,
  seed: number,
  opts: TerrainOptions = {}
): THREE.BufferGeometry {
  const perm = buildPerm(seed)
  const rng = xoshiro128ss(seed + 777)
  const fbmOpts: FbmOpts = {
    octaves: opts.fbmOctaves ?? 6,
    lacunarity: opts.fbmLacunarity ?? 2.0,
    gain: opts.fbmGain ?? 0.5,
    warpStrength: opts.warpStrength ?? 0.4,
    ridged: opts.ridged ?? false,
  }

  const gw = segX + 1, gh = segZ + 1
  const heights = new Float32Array(gw * gh)

  for (let z = 0; z < gh; z++) {
    for (let x = 0; x < gw; x++) {
      const nx = (x / segX) * 3.0
      const nz = (z / segZ) * 3.0
      const { height } = fbm2Terrain(nx, nz, perm, fbmOpts)
      heights[z * gw + x] = height * heightScale
    }
  }

  if (opts.erosionEnabled !== false && (opts.erosionIterations ?? 0) > 0) {
    const erosionIter = Math.min(opts.erosionIterations ?? 30000, 200000)
    hydraulicErosion(heights, gw, gh, {
      ...DEFAULT_EROSION,
      iterations: erosionIter,
    }, rng)
  }

  if ((opts.thermalIterations ?? 0) > 0) {
    thermalErosion(heights, gw, gh, opts.talusAngle ?? 0.65, opts.thermalIterations!)
  }

  const geo = new THREE.PlaneGeometry(width, depth, segX, segZ)
  geo.rotateX(-Math.PI / 2)
  const pos = geo.attributes.position as THREE.BufferAttribute
  for (let z = 0; z < gh; z++) {
    for (let x = 0; x < gw; x++) {
      const vi = z * gw + x
      pos.setY(vi, heights[vi])
    }
  }
  pos.needsUpdate = true
  geo.computeVertexNormals()

  // Vertex colors: slope-based (grass on flat, rock on steep)
  const normals = geo.attributes.normal as THREE.BufferAttribute
  const colors = new Float32Array(pos.count * 3)
  const grassCol = new THREE.Color(0x4a7a3a)
  const rockCol = new THREE.Color(0x8a7a6a)
  const snowCol = new THREE.Color(0xe8e8e8)
  for (let i = 0; i < pos.count; i++) {
    const ny = normals.getY(i)
    const h = pos.getY(i) / heightScale
    const slope = 1 - ny
    const c = new THREE.Color()
    if (h > 0.7) {
      c.lerpColors(rockCol, snowCol, Math.min((h - 0.7) / 0.3, 1))
    } else {
      c.lerpColors(grassCol, rockCol, Math.min(slope * 4, 1))
    }
    colors[i * 3] = c.r; colors[i * 3 + 1] = c.g; colors[i * 3 + 2] = c.b
  }
  geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3))

  return geo
}

// ── Public: Noise Sphere (ridged multi-fractal + domain warping) ────────────

export interface NoiseSphereOptions {
  fbmOctaves?: number
  fbmLacunarity?: number
  fbmGain?: number
  warpStrength?: number
  ridged?: boolean
}

export function createNoiseSphereGeometry(
  radius: number,
  widthSeg: number,
  heightSeg: number,
  disp: number,
  seed: number,
  opts: NoiseSphereOptions = {}
): THREE.BufferGeometry {
  const perm = buildPerm(seed)
  const fbmOpts: FbmOpts = {
    octaves: opts.fbmOctaves ?? 5,
    lacunarity: opts.fbmLacunarity ?? 2.2,
    gain: opts.fbmGain ?? 0.5,
    warpStrength: opts.warpStrength ?? 0.5,
    ridged: opts.ridged ?? true,
  }

  const geo = new THREE.SphereGeometry(radius, widthSeg, heightSeg)
  const pos = geo.attributes.position as THREE.BufferAttribute

  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i)
    const len = Math.sqrt(x * x + y * y + z * z) || 1
    const nx = x / len, ny = y / len, nz = z / len

    const n = fbm3(nx * 2.5, ny * 2.5, nz * 2.5, perm, fbmOpts)
    const d = 1 + (n.value - 0.3) * 2 * disp

    pos.setX(i, nx * radius * d)
    pos.setY(i, ny * radius * d)
    pos.setZ(i, nz * radius * d)
  }

  pos.needsUpdate = true
  geo.computeVertexNormals()
  return geo
}

// ── Public: Scatter (Poisson disk sampling) ─────────────────────────────────

export function createScatterGroup(
  base: 'sphere' | 'box' | 'cone' | 'tetrahedron' | 'octahedron',
  count: number,
  spread: number,
  sizeMin: number,
  sizeMax: number,
  seed: number,
  matFactory: () => THREE.MeshPhysicalMaterial
): THREE.Group {
  const g = new THREE.Group()
  const rng = xoshiro128ss(seed)

  const minDist = (spread * 2) / Math.sqrt(count * 1.5)
  const samples = poissonDiskSampling(spread * 2, spread * 2, minDist, rng)

  // Use InstancedMesh for performance when count > 10
  const mkGeo = (s: number) => {
    switch (base) {
      case 'box': return new THREE.BoxGeometry(s, s * (0.6 + rng() * 0.8), s * (0.7 + rng() * 0.5))
      case 'cone': return new THREE.ConeGeometry(s * 0.45, s * 1.2, 5 + Math.floor(rng() * 6))
      case 'tetrahedron': return new THREE.TetrahedronGeometry(s, 0)
      case 'octahedron': return new THREE.OctahedronGeometry(s, 0)
      default: return new THREE.SphereGeometry(s * 0.5, 12, 10)
    }
  }

  const actualCount = Math.min(count, samples.length)

  if (actualCount > 10 && base !== 'box') {
    // InstancedMesh path for GPU efficiency
    const avgSize = (sizeMin + sizeMax) / 2
    const geo = mkGeo(avgSize)
    const mat = matFactory()
    const instMesh = new THREE.InstancedMesh(geo, mat, actualCount)
    const dummy = new THREE.Object3D()
    const color = new THREE.Color()
    const baseColor = mat.color.clone()
    const hsl = { h: 0, s: 0, l: 0 }
    baseColor.getHSL(hsl)

    for (let i = 0; i < actualCount; i++) {
      const p = samples[i]
      const s = sizeMin + rng() * (sizeMax - sizeMin)
      const scaleRatio = s / avgSize

      dummy.position.set(
        p.x - spread,
        rng() * s * 0.15,
        p.z - spread
      )
      dummy.rotation.set(rng() * 6, rng() * 6, rng() * 6)
      dummy.scale.setScalar(scaleRatio * (0.75 + rng() * 0.55))
      dummy.updateMatrix()
      instMesh.setMatrixAt(i, dummy.matrix)

      color.setHSL(hsl.h + (rng() - 0.5) * 0.05, hsl.s, hsl.l * (0.85 + rng() * 0.3))
      instMesh.setColorAt(i, color)
    }
    instMesh.instanceMatrix.needsUpdate = true
    if (instMesh.instanceColor) instMesh.instanceColor.needsUpdate = true
    instMesh.castShadow = true
    instMesh.receiveShadow = true
    g.add(instMesh)
  } else {
    for (let i = 0; i < actualCount; i++) {
      const s = sizeMin + rng() * (sizeMax - sizeMin)
      const mesh = new THREE.Mesh(mkGeo(s), matFactory())
      const p = samples[i]
      mesh.position.set(p.x - spread, rng() * s * 0.15, p.z - spread)
      mesh.rotation.set(rng() * 6, rng() * 6, rng() * 6)
      mesh.castShadow = true
      mesh.receiveShadow = true
      g.add(mesh)
    }
  }

  return g
}

// ── Public: Crystal Cluster (hexagonal prisms + competitive growth) ─────────

function createHexPrismGeometry(
  radius: number, height: number, facets: number
): THREE.BufferGeometry {
  const geo = new THREE.CylinderGeometry(radius, radius * 0.85, height, facets, 1, false)
  // Taper the top slightly for a natural crystal termination
  const pos = geo.attributes.position as THREE.BufferAttribute
  for (let i = 0; i < pos.count; i++) {
    const y = pos.getY(i)
    if (y > height * 0.3) {
      const t = (y - height * 0.3) / (height * 0.7)
      const taper = 1 - t * 0.45
      pos.setX(i, pos.getX(i) * taper)
      pos.setZ(i, pos.getZ(i) * taper)
    }
  }
  pos.needsUpdate = true
  geo.computeVertexNormals()
  return geo
}

export function createCrystalClusterGroup(
  count: number,
  spread: number,
  heightScale: number,
  seed: number,
  matFactory: () => THREE.MeshPhysicalMaterial
): THREE.Group {
  const g = new THREE.Group()
  const rng = xoshiro128ss(seed)

  // Voronoi seed competition: place crystal bases, then grow
  const seeds: { x: number; z: number; energy: number }[] = []
  for (let i = 0; i < count; i++) {
    const angle = rng() * Math.PI * 2
    const dist = Math.pow(rng(), 0.6) * spread * 0.85
    seeds.push({
      x: Math.cos(angle) * dist,
      z: Math.sin(angle) * dist,
      energy: 0.3 + rng() * 0.7,
    })
  }

  // Competitive growth: crystals near others grow shorter
  for (let i = 0; i < seeds.length; i++) {
    for (let j = i + 1; j < seeds.length; j++) {
      const dx = seeds[i].x - seeds[j].x
      const dz = seeds[i].z - seeds[j].z
      const dist = Math.sqrt(dx * dx + dz * dz)
      const threshold = spread * 0.3
      if (dist < threshold) {
        const suppression = 1 - dist / threshold
        seeds[i].energy *= (1 - suppression * 0.4)
        seeds[j].energy *= (1 - suppression * 0.4)
      }
    }
  }

  for (const s of seeds) {
    const h = (0.4 + s.energy * 1.6) * heightScale
    const r = (0.06 + rng() * 0.14) * heightScale * (0.7 + s.energy * 0.3)
    const facets = rng() > 0.3 ? 6 : (rng() > 0.5 ? 8 : 5) // Hex dominant
    const geo = createHexPrismGeometry(r, h, facets)
    const mesh = new THREE.Mesh(geo, matFactory())
    mesh.position.set(s.x, h * 0.35, s.z)
    // Slight tilt from vertical for natural look
    mesh.rotation.set(
      (rng() - 0.5) * 0.3,
      rng() * Math.PI * 2,
      (rng() - 0.5) * 0.3
    )
    mesh.castShadow = true
    mesh.receiveShadow = true
    g.add(mesh)
  }

  return g
}

// ── Public: Rock Field (3D noise-displaced icosahedra + erosion smoothing) ──

function createErodedRockGeometry(
  baseRadius: number, seed: number, rng: () => number
): THREE.BufferGeometry {
  const detail = rng() > 0.4 ? 2 : 1
  const geo = new THREE.IcosahedronGeometry(baseRadius, detail)
  const pos = geo.attributes.position as THREE.BufferAttribute
  const perm = buildPerm(seed)

  // Multi-scale 3D noise displacement for realistic rock surface
  const fbmOpts: FbmOpts = {
    octaves: 4,
    lacunarity: 2.3,
    gain: 0.45,
    warpStrength: 0.2,
    ridged: rng() > 0.5,
  }

  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i)
    const len = Math.sqrt(x * x + y * y + z * z) || 1
    const nx = x / len, ny = y / len, nz = z / len

    const n = fbm3(nx * 3, ny * 3, nz * 3, perm, fbmOpts)
    const d = 1 + (n.value - 0.5) * 0.4

    // Flatten bottom slightly for stable resting
    const yFactor = ny < -0.3 ? 0.7 + (ny + 0.3) * 0.3 : 1
    pos.setX(i, nx * baseRadius * d * (0.7 + rng() * 0.6))
    pos.setY(i, ny * baseRadius * d * yFactor * (0.8 + rng() * 0.4))
    pos.setZ(i, nz * baseRadius * d * (0.7 + rng() * 0.6))
  }

  pos.needsUpdate = true

  // Erosion pass: smooth high-curvature vertices (Laplacian smoothing)
  const adjacency = buildAdjacency(geo)
  laplacianSmooth(geo, adjacency, 2, 0.3)

  geo.computeVertexNormals()
  return geo
}

function buildAdjacency(geo: THREE.BufferGeometry): Map<number, Set<number>> {
  const adj = new Map<number, Set<number>>()
  const index = geo.index!
  for (let i = 0; i < index.count; i += 3) {
    const a = index.getX(i), b = index.getX(i + 1), c = index.getX(i + 2)
    for (const [v1, v2] of [[a, b], [b, c], [c, a]]) {
      if (!adj.has(v1)) adj.set(v1, new Set())
      if (!adj.has(v2)) adj.set(v2, new Set())
      adj.get(v1)!.add(v2)
      adj.get(v2)!.add(v1)
    }
  }
  return adj
}

function laplacianSmooth(
  geo: THREE.BufferGeometry, adj: Map<number, Set<number>>,
  iterations: number, factor: number
): void {
  const pos = geo.attributes.position as THREE.BufferAttribute
  for (let iter = 0; iter < iterations; iter++) {
    const newPos = new Float32Array(pos.count * 3)
    for (let i = 0; i < pos.count; i++) {
      const neighbors = adj.get(i)
      if (!neighbors || neighbors.size === 0) {
        newPos[i * 3] = pos.getX(i)
        newPos[i * 3 + 1] = pos.getY(i)
        newPos[i * 3 + 2] = pos.getZ(i)
        continue
      }
      let ax = 0, ay = 0, az = 0
      for (const ni of neighbors) {
        ax += pos.getX(ni); ay += pos.getY(ni); az += pos.getZ(ni)
      }
      const n = neighbors.size
      newPos[i * 3] = pos.getX(i) * (1 - factor) + (ax / n) * factor
      newPos[i * 3 + 1] = pos.getY(i) * (1 - factor) + (ay / n) * factor
      newPos[i * 3 + 2] = pos.getZ(i) * (1 - factor) + (az / n) * factor
    }
    for (let i = 0; i < pos.count; i++) {
      pos.setXYZ(i, newPos[i * 3], newPos[i * 3 + 1], newPos[i * 3 + 2])
    }
  }
  pos.needsUpdate = true
}

export function createRockFieldGroup(
  count: number,
  spread: number,
  sizeMin: number,
  sizeMax: number,
  seed: number,
  matFactory: () => THREE.MeshPhysicalMaterial
): THREE.Group {
  const g = new THREE.Group()
  const rng = xoshiro128ss(seed)

  // Poisson disk sampling for natural rock placement
  const minDist = spread * 2 / Math.sqrt(count * 2)
  const samples = poissonDiskSampling(spread * 2, spread * 2, minDist, rng)
  const actualCount = Math.min(count, samples.length)

  for (let i = 0; i < actualCount; i++) {
    const s = sizeMin + rng() * (sizeMax - sizeMin)
    const rockSeed = seed + i * 137
    const geo = createErodedRockGeometry(s, rockSeed, rng)
    const mesh = new THREE.Mesh(geo, matFactory())
    const p = samples[i]
    mesh.position.set(p.x - spread, s * 0.15, p.z - spread)
    mesh.rotation.set(rng() * 0.5, rng() * Math.PI * 2, rng() * 0.5)
    mesh.castShadow = true
    mesh.receiveShadow = true
    g.add(mesh)
  }

  return g
}
