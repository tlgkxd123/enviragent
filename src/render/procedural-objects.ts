import * as THREE from 'three'

/** Deterministic PRNG from seed (Mulberry32). */
function mulberry32(seed: number): () => number {
  let t = seed >>> 0
  return () => {
    t += 0x6d2b79f5
    let r = Math.imul(t ^ (t >>> 15), 1 | t)
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r)
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296
  }
}

function hash3(ix: number, iy: number, iz: number, seed: number): number {
  let h = (ix * 374761393 + iy * 668265263 + iz * 1274126177 + seed * 1442695041) >>> 0
  h = Math.imul(h ^ (h >>> 13), 1274126177)
  return (h ^ (h >>> 16)) / 4294967296
}

/** Value noise on a coarse grid, smoothstep interpolated (cheap terrain). */
function fbm2(x: number, z: number, seed: number): number {
  const s = 4
  const x0 = Math.floor(x * s)
  const z0 = Math.floor(z * s)
  const fx = x * s - x0
  const fz = z * s - z0
  const u = fx * fx * (3 - 2 * fx)
  const v = fz * fz * (3 - 2 * fz)
  const h00 = hash3(x0, z0, 0, seed)
  const h10 = hash3(x0 + 1, z0, 0, seed)
  const h01 = hash3(x0, z0 + 1, 0, seed)
  const h11 = hash3(x0 + 1, z0 + 1, 0, seed)
  const hx0 = h00 + u * (h10 - h00)
  const hx1 = h01 + u * (h11 - h01)
  return hx0 + v * (hx1 - hx0)
}

export function createTerrainGeometry(
  width: number,
  depth: number,
  segX: number,
  segZ: number,
  heightScale: number,
  seed: number
): THREE.BufferGeometry {
  const geo = new THREE.PlaneGeometry(width, depth, segX, segZ)
  geo.rotateX(-Math.PI / 2)
  const pos = geo.attributes.position as THREE.BufferAttribute
  const w2 = width / 2
  const d2 = depth / 2
  for (let i = 0; i < pos.count; i++) {
    const px = pos.getX(i)
    const pz = pos.getZ(i)
    const nx = (px + w2) / width
    const nz = (pz + d2) / depth
    const n =
      fbm2(nx * 2.2, nz * 2.2, seed) * 0.65 +
      fbm2(nx * 5, nz * 5, seed + 17) * 0.25 +
      fbm2(nx * 12, nz * 12, seed + 41) * 0.1
    pos.setY(i, n * heightScale)
  }
  pos.needsUpdate = true
  geo.computeVertexNormals()
  return geo
}

export function createNoiseSphereGeometry(
  radius: number,
  widthSeg: number,
  heightSeg: number,
  disp: number,
  seed: number
): THREE.BufferGeometry {
  const geo = new THREE.SphereGeometry(radius, widthSeg, heightSeg)
  const pos = geo.attributes.position as THREE.BufferAttribute
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i)
    const y = pos.getY(i)
    const z = pos.getZ(i)
    const len = Math.sqrt(x * x + y * y + z * z) || 1
    const nx = x / len
    const ny = y / len
    const nz = z / len
    const n =
      fbm2(nx * 3 + 0.5, nz * 3 + 0.5, seed) * 0.7 +
      fbm2(ny * 4 + 0.2, nx * 4, seed + 9) * 0.3
    const d = 1 + (n - 0.45) * 2 * disp
    pos.setX(i, nx * radius * d)
    pos.setY(i, ny * radius * d)
    pos.setZ(i, nz * radius * d)
  }
  pos.needsUpdate = true
  geo.computeVertexNormals()
  return geo
}

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
  const rnd = mulberry32(seed)
  const mkGeo = (s: number) => {
    switch (base) {
      case 'box': return new THREE.BoxGeometry(s, s * (0.6 + rnd() * 0.8), s * (0.7 + rnd() * 0.5))
      case 'cone': return new THREE.ConeGeometry(s * 0.45, s * 1.2, 5 + Math.floor(rnd() * 6))
      case 'tetrahedron': return new THREE.TetrahedronGeometry(s, 0)
      case 'octahedron': return new THREE.OctahedronGeometry(s, 0)
      default: return new THREE.SphereGeometry(s * 0.5, 12, 10)
    }
  }
  for (let i = 0; i < count; i++) {
    const s = sizeMin + rnd() * (sizeMax - sizeMin)
    const mesh = new THREE.Mesh(mkGeo(s), matFactory())
    const r = Math.pow(rnd(), 0.55) * spread
    const th = rnd() * Math.PI * 2
    const ph = Math.acos(2 * rnd() - 1)
    mesh.position.set(r * Math.sin(ph) * Math.cos(th), r * Math.cos(ph) * 0.85, r * Math.sin(ph) * Math.sin(th))
    mesh.rotation.set(rnd() * 6, rnd() * 6, rnd() * 6)
    mesh.castShadow = true
    mesh.receiveShadow = true
    g.add(mesh)
  }
  return g
}

export function createCrystalClusterGroup(
  count: number,
  spread: number,
  heightScale: number,
  seed: number,
  matFactory: () => THREE.MeshPhysicalMaterial
): THREE.Group {
  const g = new THREE.Group()
  const rnd = mulberry32(seed)
  for (let i = 0; i < count; i++) {
    const h = (0.4 + rnd() * 1.8) * heightScale
    const r = (0.08 + rnd() * 0.2) * heightScale
    const mesh = new THREE.Mesh(
      new THREE.ConeGeometry(r, h, 5 + Math.floor(rnd() * 4)),
      matFactory()
    )
    const d = rnd() * spread * 0.85
    const ang = rnd() * Math.PI * 2
    mesh.position.set(Math.cos(ang) * d, h * 0.35, Math.sin(ang) * d)
    mesh.rotation.set(rnd() * 0.4, rnd() * Math.PI * 2, rnd() * 0.4)
    mesh.castShadow = true
    mesh.receiveShadow = true
    g.add(mesh)
  }
  return g
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
  const rnd = mulberry32(seed)
  for (let i = 0; i < count; i++) {
    const s = sizeMin + rnd() * (sizeMax - sizeMin)
    const detail = rnd() > 0.55 ? 1 : 0
    const mesh = new THREE.Mesh(new THREE.IcosahedronGeometry(s, detail), matFactory())
    mesh.scale.setScalar(0.75 + rnd() * 0.55)
    const d = Math.pow(rnd(), 0.5) * spread
    const ang = rnd() * Math.PI * 2
    mesh.position.set(Math.cos(ang) * d, s * 0.2, Math.sin(ang) * d)
    mesh.rotation.set(rnd() * 3, rnd() * 3, rnd() * 3)
    mesh.castShadow = true
    mesh.receiveShadow = true
    g.add(mesh)
  }
  return g
}
