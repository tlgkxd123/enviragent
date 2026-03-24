import * as THREE from 'three'

// ═══════════════════════════════════════════════════════════════════════════════
// Procedural tree generator
//
// Algorithms:
//   1. Space Colonization (Runions et al. 2007) for phototropic crown shape
//   2. da Vinci pipe model for branch radius taper (r_parent² = Σ r_child²)
//   3. Golden-angle phyllotaxis for leaf placement
//   4. Catmull–Rom spline interpolation for smooth branch geometry
//   5. InstancedMesh leaves for GPU-efficient foliage
//   6. Seeded PRNG (xoshiro128**) for reproducible results
// ═══════════════════════════════════════════════════════════════════════════════

// ── Seeded PRNG ──────────────────────────────────────────────────────────────

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

// ── Vector helpers ───────────────────────────────────────────────────────────

const _v = new THREE.Vector3()

function randomPointInEllipsoid(
  rng: () => number,
  rx: number, ry: number, rz: number,
  center: THREE.Vector3
): THREE.Vector3 {
  while (true) {
    const x = (rng() * 2 - 1)
    const y = (rng() * 2 - 1)
    const z = (rng() * 2 - 1)
    if (x * x + y * y + z * z <= 1) {
      return new THREE.Vector3(
        x * rx + center.x,
        y * ry + center.y,
        z * rz + center.z
      )
    }
  }
}

// ── Space Colonization ──────────────────────────────────────────────────────

interface TreeNode {
  pos: THREE.Vector3
  parent: number     // index of parent, -1 for root
  children: number[]
  radius: number
  depth: number
}

interface SpaceColonizationOpts {
  attractorCount: number
  crownCenter: THREE.Vector3
  crownRadiusX: number
  crownRadiusY: number
  crownRadiusZ: number
  influenceRadius: number
  killRadius: number
  stepSize: number
  maxIterations: number
  trunkHeight: number
  trunkSteps: number
  rng: () => number
  tropism: THREE.Vector3   // gravity / phototropism bias
}

function spaceColonization(opts: SpaceColonizationOpts): TreeNode[] {
  const {
    attractorCount, crownCenter, crownRadiusX, crownRadiusY, crownRadiusZ,
    influenceRadius, killRadius, stepSize, maxIterations, trunkHeight, trunkSteps, rng, tropism
  } = opts

  const nodes: TreeNode[] = []

  // Build trunk as straight segments from origin upward
  for (let i = 0; i <= trunkSteps; i++) {
    const t = i / trunkSteps
    const y = t * trunkHeight
    nodes.push({
      pos: new THREE.Vector3(0, y, 0),
      parent: i === 0 ? -1 : i - 1,
      children: [],
      radius: 0,
      depth: i,
    })
    if (i > 0) nodes[i - 1].children.push(i)
  }

  // Scatter attractors in ellipsoidal crown volume
  const attractors: THREE.Vector3[] = []
  for (let i = 0; i < attractorCount; i++) {
    attractors.push(randomPointInEllipsoid(rng, crownRadiusX, crownRadiusY, crownRadiusZ, crownCenter))
  }
  const alive = new Uint8Array(attractors.length).fill(1)

  // Iterative growth
  for (let iter = 0; iter < maxIterations; iter++) {
    // Find closest node for each attractor
    const influence = new Map<number, THREE.Vector3>()

    for (let ai = 0; ai < attractors.length; ai++) {
      if (!alive[ai]) continue
      const a = attractors[ai]
      let closestIdx = -1
      let closestDist = Infinity
      for (let ni = 0; ni < nodes.length; ni++) {
        const d = a.distanceTo(nodes[ni].pos)
        if (d < closestDist) { closestDist = d; closestIdx = ni }
      }
      if (closestIdx < 0 || closestDist > influenceRadius) continue
      if (closestDist < killRadius) { alive[ai] = 0; continue }

      if (!influence.has(closestIdx)) influence.set(closestIdx, new THREE.Vector3())
      _v.copy(a).sub(nodes[closestIdx].pos).normalize()
      influence.get(closestIdx)!.add(_v)
    }

    if (influence.size === 0) break

    const newNodes: TreeNode[] = []
    for (const [ni, dir] of influence) {
      dir.normalize()
      // Add tropism (gravitropism / phototropism)
      dir.add(tropism).normalize()

      const newPos = nodes[ni].pos.clone().addScaledVector(dir, stepSize)
      const newIdx = nodes.length + newNodes.length
      const node: TreeNode = {
        pos: newPos,
        parent: ni,
        children: [],
        radius: 0,
        depth: nodes[ni].depth + 1,
      }
      nodes[ni].children.push(newIdx)
      newNodes.push(node)
    }
    nodes.push(...newNodes)
  }

  return nodes
}

// ── da Vinci pipe-model radius ──────────────────────────────────────────────

function computeRadii(nodes: TreeNode[], baseRadius: number, exponent: number): void {
  const leafRadius = baseRadius * 0.015
  function walk(i: number): number {
    const n = nodes[i]
    if (n.children.length === 0) {
      n.radius = leafRadius
      return leafRadius
    }
    let sumPow = 0
    for (const ci of n.children) {
      const cr = walk(ci)
      sumPow += Math.pow(cr, exponent)
    }
    n.radius = Math.pow(sumPow, 1 / exponent)
    return n.radius
  }
  walk(0)
  // Ensure base radius matches requested size
  const scale = baseRadius / Math.max(nodes[0].radius, 0.001)
  for (const n of nodes) n.radius *= scale
}

// ── Catmull–Rom spline for smooth branches ──────────────────────────────────

function catmullRomPoint(
  p0: THREE.Vector3, p1: THREE.Vector3, p2: THREE.Vector3, p3: THREE.Vector3,
  t: number, tension: number
): THREE.Vector3 {
  const s = (1 - tension) * 0.5
  const t2 = t * t, t3 = t2 * t
  const h1 = -s * t3 + 2 * s * t2 - s * t
  const h2 = (2 - s) * t3 + (s - 3) * t2 + 1
  const h3 = (s - 2) * t3 + (3 - 2 * s) * t2 + s * t
  const h4 = s * t3 - s * t2
  return new THREE.Vector3(
    h1 * p0.x + h2 * p1.x + h3 * p2.x + h4 * p3.x,
    h1 * p0.y + h2 * p1.y + h3 * p2.y + h4 * p3.y,
    h1 * p0.z + h2 * p1.z + h3 * p2.z + h4 * p3.z,
  )
}

// ── Branch mesh builder (tapered tubes along splines) ───────────────────────

function buildBranchGeometry(
  nodes: TreeNode[],
  radialSegments: number,
  lengthSubdivisions: number,
  rng: () => number
): THREE.BufferGeometry {
  const positions: number[] = []
  const normals: number[] = []
  const uvs: number[] = []
  const indices: number[] = []
  let vertexOffset = 0

  function getChain(startIdx: number): number[] {
    const chain: number[] = [startIdx]
    let cur = startIdx
    while (nodes[cur].children.length > 0) {
      // Follow the thickest child (main continuation)
      let best = nodes[cur].children[0]
      let bestR = nodes[best].radius
      for (let i = 1; i < nodes[cur].children.length; i++) {
        const ci = nodes[cur].children[i]
        if (nodes[ci].radius > bestR) { best = ci; bestR = nodes[ci].radius }
      }
      chain.push(best)
      cur = best
    }
    return chain
  }

  function addTube(chain: number[]): void {
    if (chain.length < 2) return
    const pts = chain.map(i => nodes[i].pos)
    const radii = chain.map(i => nodes[i].radius)

    const steps = Math.max(chain.length - 1, 1) * lengthSubdivisions
    const ringCount = steps + 1
    const baseVert = vertexOffset

    for (let s = 0; s <= steps; s++) {
      const t = s / steps
      const segF = t * (chain.length - 1)
      const segI = Math.min(Math.floor(segF), chain.length - 2)
      const segT = segF - segI

      // Catmull–Rom control points with clamped endpoints
      const i0 = Math.max(segI - 1, 0)
      const i1 = segI
      const i2 = Math.min(segI + 1, pts.length - 1)
      const i3 = Math.min(segI + 2, pts.length - 1)

      const center = catmullRomPoint(pts[i0], pts[i1], pts[i2], pts[i3], segT, 0.5)
      const radius = THREE.MathUtils.lerp(radii[i1], radii[i2], segT)

      // Tangent for frame
      const tangent = new THREE.Vector3()
      if (s < steps) {
        const nextT = (s + 1) / steps
        const nSegF = nextT * (chain.length - 1)
        const nSegI = Math.min(Math.floor(nSegF), chain.length - 2)
        const nSegT = nSegF - nSegI
        const n0 = Math.max(nSegI - 1, 0)
        const n1 = nSegI
        const n2 = Math.min(nSegI + 1, pts.length - 1)
        const n3 = Math.min(nSegI + 2, pts.length - 1)
        const nextPt = catmullRomPoint(pts[n0], pts[n1], pts[n2], pts[n3], nSegT, 0.5)
        tangent.copy(nextPt).sub(center).normalize()
      } else {
        tangent.copy(center).sub(pts[pts.length - 2]).normalize()
      }

      // Build orthonormal frame
      const up = Math.abs(tangent.y) < 0.99 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0)
      const biNormal = new THREE.Vector3().crossVectors(tangent, up).normalize()
      const normal = new THREE.Vector3().crossVectors(biNormal, tangent).normalize()

      // Ring of vertices with bark noise
      for (let r = 0; r < radialSegments; r++) {
        const angle = (r / radialSegments) * Math.PI * 2
        const cos = Math.cos(angle)
        const sin = Math.sin(angle)

        // Bark displacement: subtle noise
        const barkNoise = 1 + (rng() - 0.5) * 0.06

        const px = center.x + (cos * biNormal.x + sin * normal.x) * radius * barkNoise
        const py = center.y + (cos * biNormal.y + sin * normal.y) * radius * barkNoise
        const pz = center.z + (cos * biNormal.z + sin * normal.z) * radius * barkNoise

        positions.push(px, py, pz)
        const nx = cos * biNormal.x + sin * normal.x
        const ny = cos * biNormal.y + sin * normal.y
        const nz = cos * biNormal.z + sin * normal.z
        const nLen = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1
        normals.push(nx / nLen, ny / nLen, nz / nLen)
        uvs.push(r / radialSegments, t)
      }
    }

    // Indices
    for (let s = 0; s < steps; s++) {
      for (let r = 0; r < radialSegments; r++) {
        const a = baseVert + s * radialSegments + r
        const b = baseVert + s * radialSegments + (r + 1) % radialSegments
        const c = baseVert + (s + 1) * radialSegments + (r + 1) % radialSegments
        const d = baseVert + (s + 1) * radialSegments + r
        indices.push(a, b, c, a, c, d)
      }
    }

    vertexOffset += ringCount * radialSegments
  }

  // Build chains: from root follow thickest child, then recurse branches
  const visited = new Set<number>()
  function emitChains(startIdx: number): void {
    const chain = getChain(startIdx)
    for (const i of chain) visited.add(i)
    addTube(chain)

    for (const i of chain) {
      for (const ci of nodes[i].children) {
        if (!visited.has(ci)) emitChains(ci)
      }
    }
  }
  emitChains(0)

  const geo = new THREE.BufferGeometry()
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
  geo.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3))
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2))
  geo.setIndex(indices)
  geo.computeBoundingSphere()
  return geo
}

// ── Leaf placement with golden-angle phyllotaxis ────────────────────────────

const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5))

function collectLeafPositions(
  nodes: TreeNode[],
  rng: () => number,
  leafDensity: number,
  minDepth: number
): { position: THREE.Vector3; normal: THREE.Vector3 }[] {
  const leaves: { position: THREE.Vector3; normal: THREE.Vector3 }[] = []
  for (let i = 0; i < nodes.length; i++) {
    const n = nodes[i]
    if (n.children.length > 0 && n.depth < minDepth) continue
    // Terminal nodes and thin branches get leaves
    if (n.children.length > 0 && n.radius > nodes[0].radius * 0.08) continue

    const count = Math.max(1, Math.round(leafDensity * (0.5 + rng() * 0.5)))
    for (let li = 0; li < count; li++) {
      const angle = li * GOLDEN_ANGLE + rng() * 0.3
      const spread = n.radius * 3 + rng() * 0.15
      const dir = new THREE.Vector3(
        Math.cos(angle) * spread,
        (rng() - 0.3) * spread * 0.5,
        Math.sin(angle) * spread
      )
      const pos = n.pos.clone().add(dir)
      // Normal roughly pointing outward+up
      const norm = dir.clone().normalize().lerp(new THREE.Vector3(0, 1, 0), 0.4).normalize()
      leaves.push({ position: pos, normal: norm })
    }
  }
  return leaves
}

// ── Instanced leaf mesh ─────────────────────────────────────────────────────

function buildLeafInstancedMesh(
  leaves: { position: THREE.Vector3; normal: THREE.Vector3 }[],
  leafSize: number,
  leafColor: THREE.Color,
  leafColorVariation: number,
  rng: () => number
): THREE.InstancedMesh {
  // Double-sided quad geometry for a single leaf
  const leafGeo = new THREE.PlaneGeometry(leafSize, leafSize * 1.6, 1, 1)

  const mat = new THREE.MeshPhysicalMaterial({
    color: leafColor,
    roughness: 0.75,
    metalness: 0.0,
    side: THREE.DoubleSide,
    transparent: true,
    alphaTest: 0.3,
  })

  const count = Math.min(leaves.length, 65536)
  const mesh = new THREE.InstancedMesh(leafGeo, mat, count)

  const dummy = new THREE.Object3D()
  const color = new THREE.Color()
  const hsl = { h: 0, s: 0, l: 0 }
  leafColor.getHSL(hsl)

  for (let i = 0; i < count; i++) {
    const leaf = leaves[i]
    dummy.position.copy(leaf.position)

    // Orient leaf to face roughly along normal with random twist
    dummy.lookAt(leaf.position.clone().add(leaf.normal))
    dummy.rotateZ(rng() * Math.PI * 2)
    dummy.rotateX((rng() - 0.5) * 0.6)

    const s = 0.6 + rng() * 0.8
    dummy.scale.set(s, s, s)
    dummy.updateMatrix()
    mesh.setMatrixAt(i, dummy.matrix)

    // Color variation
    color.setHSL(
      hsl.h + (rng() - 0.5) * leafColorVariation,
      hsl.s * (0.8 + rng() * 0.4),
      hsl.l * (0.7 + rng() * 0.6)
    )
    mesh.setColorAt(i, color)
  }

  mesh.instanceMatrix.needsUpdate = true
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true
  return mesh
}

// ── Public API ──────────────────────────────────────────────────────────────

export interface TreeOptions {
  seed?: number
  trunkHeight?: number
  trunkRadius?: number
  /** [rx, ry, rz] ellipsoidal crown shape */
  crownSize?: [number, number, number]
  crownOffset?: [number, number, number]
  attractorCount?: number
  branchStepSize?: number
  maxIterations?: number
  /** da Vinci exponent (2.0 = classical, higher = thinner branches) */
  pipeExponent?: number
  radialSegments?: number
  leafDensity?: number
  leafSize?: number
  barkColor?: number
  leafColor?: number
  leafColorVariation?: number
  /** 0 = no wind sway, 1 = moderate */
  windStrength?: number
  /** Additional tropism bias [x,y,z] — positive y = phototropism */
  tropism?: [number, number, number]
  position?: [number, number, number]
}

export interface TreeResult {
  group: THREE.Group
  /** Call each frame for wind sway animation */
  update?: (dt: number, t: number) => void
}

export function generateTree(opts: TreeOptions = {}): TreeResult {
  const seed            = opts.seed ?? 42
  const rng             = xoshiro128ss(seed)
  const trunkHeight     = opts.trunkHeight ?? 2.5
  const trunkRadius     = opts.trunkRadius ?? 0.12
  const crownSize       = opts.crownSize ?? [1.8, 2.0, 1.8]
  const crownOff        = opts.crownOffset ?? [0, 0, 0]
  const attractorCount  = opts.attractorCount ?? 800
  const branchStepSize  = opts.branchStepSize ?? 0.18
  const maxIterations   = opts.maxIterations ?? 120
  const pipeExponent    = opts.pipeExponent ?? 2.2
  const radialSegments  = opts.radialSegments ?? 8
  const leafDensity     = opts.leafDensity ?? 4
  const leafSize        = opts.leafSize ?? 0.12
  const barkColor       = opts.barkColor ?? 0x5c3a1e
  const leafColor       = opts.leafColor ?? 0x3a7d2e
  const leafColorVar    = opts.leafColorVariation ?? 0.08
  const windStrength    = opts.windStrength ?? 0.4
  const tropism         = new THREE.Vector3(...(opts.tropism ?? [0, 0.12, 0]))
  const position        = opts.position ?? [0, 0, 0]

  const crownCenter = new THREE.Vector3(
    crownOff[0],
    trunkHeight + crownSize[1] * 0.3 + crownOff[1],
    crownOff[2]
  )

  // 1. Space colonization
  const influenceRadius = Math.max(...crownSize) * 1.5
  const killRadius      = branchStepSize * 1.8

  const nodes = spaceColonization({
    attractorCount,
    crownCenter,
    crownRadiusX: crownSize[0],
    crownRadiusY: crownSize[1],
    crownRadiusZ: crownSize[2],
    influenceRadius,
    killRadius,
    stepSize: branchStepSize,
    maxIterations,
    trunkHeight,
    trunkSteps: Math.max(4, Math.ceil(trunkHeight / (branchStepSize * 0.8))),
    rng,
    tropism,
  })

  // 2. da Vinci radii
  computeRadii(nodes, trunkRadius, pipeExponent)

  // 3. Branch geometry (tapered Catmull–Rom tubes)
  const branchGeo = buildBranchGeometry(nodes, radialSegments, 2, rng)
  const barkMat = new THREE.MeshPhysicalMaterial({
    color: new THREE.Color(barkColor),
    roughness: 0.92,
    metalness: 0.0,
  })
  const branchMesh = new THREE.Mesh(branchGeo, barkMat)
  branchMesh.castShadow = true
  branchMesh.receiveShadow = true

  // 4. Leaf placement (golden-angle phyllotaxis on terminal branches)
  const minLeafDepth = Math.floor(nodes.length > 0 ? nodes[nodes.length - 1].depth * 0.4 : 3)
  const leafPositions = collectLeafPositions(nodes, rng, leafDensity, minLeafDepth)
  const leafMesh = buildLeafInstancedMesh(
    leafPositions, leafSize, new THREE.Color(leafColor), leafColorVar, rng
  )
  leafMesh.castShadow = true

  // 5. Assemble group
  const group = new THREE.Group()
  group.add(branchMesh)
  group.add(leafMesh)
  group.position.set(position[0], position[1], position[2])

  // 6. Wind sway (vertex-based would be expensive; use group oscillation)
  let update: ((dt: number, t: number) => void) | undefined
  if (windStrength > 0) {
    const windPhase = rng() * Math.PI * 2
    update = (_dt: number, t: number) => {
      const sway = Math.sin(t * 1.2 + windPhase) * windStrength * 0.015
      const sway2 = Math.sin(t * 0.7 + windPhase * 1.3) * windStrength * 0.008
      leafMesh.rotation.z = sway
      leafMesh.rotation.x = sway2
    }
  }

  return { group, update }
}
