import {
  Box3,
  BufferAttribute,
  BufferGeometry,
  Color,
  Mesh,
  Object3D,
  MeshPhysicalMaterial,
  Points,
  PointsMaterial,
  Vector3,
} from 'three'
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js'
import { MeshSurfaceSampler } from 'three/addons/math/MeshSurfaceSampler.js'
import type { Complex } from '../physics/complex'
import { cAbsSq, cArg } from '../physics/complex'
import { ORBITAL_VISUAL } from './orbital-visual'
import type { OrbitalParams } from '../physics/state'
import { psiSuperposition } from '../physics/state'
import { radialShellRadius } from '../physics/hydrogen'

const TWO_PI = Math.PI * 2
const _tmp = new Vector3()

export interface ImportedModelHandles {
  mesh: Mesh
  mergedGeometry: BufferGeometry
  points: Points
  particleCount: number
}

/** Center group and scale so its bounding box fits ~2 * targetRadius. */
export function normalizeGroupToFit(group: Object3D, targetRadius = 1.05): void {
  const box = new Box3().setFromObject(group)
  const center = box.getCenter(new Vector3())
  const size = box.getSize(new Vector3())
  const maxDim = Math.max(size.x, size.y, size.z, 1e-8)
  const s = (2 * targetRadius) / maxDim
  group.position.sub(center)
  group.scale.multiplyScalar(s)
}

export function mergeWorldMeshes(root: Object3D): BufferGeometry {
  root.updateMatrixWorld(true)
  const parts: BufferGeometry[] = []
  root.traverse((obj: Object3D) => {
    if (obj instanceof Mesh && obj.geometry) {
      const g = obj.geometry.clone()
      g.applyMatrix4(obj.matrixWorld)
      parts.push(g)
    }
  })
  if (parts.length === 0) {
    return new BufferGeometry()
  }
  const merged = mergeGeometries(parts, false)
  if (!merged) return new BufferGeometry()
  if (!merged.getAttribute('normal')) {
    merged.computeVertexNormals()
  }
  return merged
}

function dirToAngles(x: number, y: number, z: number): { theta: number; phi: number } {
  const len = Math.hypot(x, y, z) || 1e-10
  const uz = z / len
  const theta = Math.acos(Math.min(1, Math.max(-1, uz)))
  const phi = Math.atan2(y, x)
  return { theta, phi }
}

function fract(x: number): number {
  return x - Math.floor(x)
}

function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - edge0) / (edge1 - edge0)))
  return t * t * (3 - 2 * t)
}

/** Matches orbital fragment: quantized bands, log rings, density-weighted saturation. */
function psiToColor(psi: Complex, densityNorm: number): Color {
  const d = Math.min(1, Math.max(0, densityNorm))
  const hueRaw = cArg(psi) / TWO_PI + ORBITAL_VISUAL.phaseHueOffset
  const hue = hueRaw - Math.floor(hueRaw)
  const k = ORBITAL_VISUAL.uBandStrength * ORBITAL_VISUAL.bandMix
  const q = Math.floor(d * ORBITAL_VISUAL.bandQuantSteps) / ORBITAL_VISUAL.bandQuantSteps
  let dMix = d * (1 - k) + q * k
  const logD = Math.log(Math.max(d, 1e-4))
  const ring = fract(logD * ORBITAL_VISUAL.logRingScale)
  const a = ORBITAL_VISUAL.uBandStrength * ORBITAL_VISUAL.logRingBlend
  const ringMul = (1 - a) + a * (0.88 + 0.24 * ring)
  dMix *= ringMul

  const satBoost = smoothstep(
    ORBITAL_VISUAL.densitySatStart,
    ORBITAL_VISUAL.densitySatEnd,
    densityNorm
  )
  const S =
    (ORBITAL_VISUAL.satLow + (ORBITAL_VISUAL.satHigh - ORBITAL_VISUAL.satLow) * satBoost) *
    ORBITAL_VISUAL.uSaturation
  const V = ORBITAL_VISUAL.valBase + ORBITAL_VISUAL.valRange * dMix
  return new Color().setHSL(hue, S, V)
}

export function updateWavefunctionOnGeometry(
  geometry: BufferGeometry,
  primary: OrbitalParams,
  secondary: OrbitalParams | null,
  mix: number,
  evolve: boolean,
  time: number
): void {
  const posAttr = geometry.getAttribute('position') as BufferAttribute
  const n = posAttr.count
  const arr = posAttr.array as Float32Array
  const densities = new Float32Array(n)
  let maxP = 0
  const nMax = Math.max(primary.n, secondary?.n ?? primary.n)
  const rEval = radialShellRadius(nMax)

  for (let i = 0; i < n; i++) {
    const x = arr[i * 3]
    const y = arr[i * 3 + 1]
    const z = arr[i * 3 + 2]
    const { theta, phi } = dirToAngles(x, y, z)
    const psi = psiSuperposition(primary, secondary, mix, rEval, theta, phi, time, evolve)
    const p = cAbsSq(psi)
    densities[i] = p
    if (p > maxP) maxP = p
  }
  if (maxP < 1e-24) maxP = 1

  let colors = geometry.getAttribute('color') as BufferAttribute | undefined
  if (!colors || colors.count !== n) {
    const c = new Float32Array(n * 3)
    geometry.setAttribute('color', new BufferAttribute(c, 3))
    colors = geometry.getAttribute('color') as BufferAttribute
  }
  const cArr = colors.array as Float32Array
  for (let i = 0; i < n; i++) {
    const x = arr[i * 3]
    const y = arr[i * 3 + 1]
    const z = arr[i * 3 + 2]
    const { theta, phi } = dirToAngles(x, y, z)
    const psi = psiSuperposition(primary, secondary, mix, rEval, theta, phi, time, evolve)
    const col = psiToColor(psi, densities[i]! / maxP)
    cArr[i * 3] = col.r
    cArr[i * 3 + 1] = col.g
    cArr[i * 3 + 2] = col.b
  }
  colors.needsUpdate = true
}

export function buildParticleCloud(
  surfaceMesh: Mesh,
  particleCount: number,
  primary: OrbitalParams,
  secondary: OrbitalParams | null,
  mix: number,
  evolve: boolean,
  time: number
): Points {
  const sampler = new MeshSurfaceSampler(surfaceMesh).build()
  const positions = new Float32Array(particleCount * 3)
  const colors = new Float32Array(particleCount * 3)
  for (let i = 0; i < particleCount; i++) {
    sampler.sample(_tmp)
    positions[i * 3] = _tmp.x
    positions[i * 3 + 1] = _tmp.y
    positions[i * 3 + 2] = _tmp.z
  }
  updateParticleColors(
    positions,
    colors,
    particleCount,
    primary,
    secondary,
    mix,
    evolve,
    time
  )
  const geom = new BufferGeometry()
  geom.setAttribute('position', new BufferAttribute(positions, 3))
  geom.setAttribute('color', new BufferAttribute(colors, 3))
  const mat = new PointsMaterial({
    size: 0.022,
    vertexColors: true,
    transparent: true,
    opacity: 0.92,
    depthWrite: true,
    sizeAttenuation: true,
  })
  return new Points(geom, mat)
}

export function updateParticleColors(
  positions: Float32Array,
  colors: Float32Array,
  particleCount: number,
  primary: OrbitalParams,
  secondary: OrbitalParams | null,
  mix: number,
  evolve: boolean,
  time: number
): void {
  const nMax = Math.max(primary.n, secondary?.n ?? primary.n)
  const rEval = radialShellRadius(nMax)
  let maxP = 0
  const dens = new Float32Array(particleCount)
  for (let i = 0; i < particleCount; i++) {
    const x = positions[i * 3]!
    const y = positions[i * 3 + 1]!
    const z = positions[i * 3 + 2]!
    const { theta, phi } = dirToAngles(x, y, z)
    const psi = psiSuperposition(primary, secondary, mix, rEval, theta, phi, time, evolve)
    const p = cAbsSq(psi)
    dens[i] = p
    if (p > maxP) maxP = p
  }
  if (maxP < 1e-24) maxP = 1
  for (let i = 0; i < particleCount; i++) {
    const x = positions[i * 3]!
    const y = positions[i * 3 + 1]!
    const z = positions[i * 3 + 2]!
    const { theta, phi } = dirToAngles(x, y, z)
    const psi = psiSuperposition(primary, secondary, mix, rEval, theta, phi, time, evolve)
    const col = psiToColor(psi, dens[i]! / maxP)
    colors[i * 3] = col.r
    colors[i * 3 + 1] = col.g
    colors[i * 3 + 2] = col.b
  }
}

export function createImportedModelMesh(
  mergedGeometry: BufferGeometry,
  envIntensity = 0.95
): Mesh {
  const mat = new MeshPhysicalMaterial({
    color: 0xffffff,
    vertexColors: true,
    metalness: 0.32,
    roughness: 0.42,
    envMapIntensity: envIntensity,
    clearcoat: 0.18,
    clearcoatRoughness: 0.35,
  })
  const mesh = new Mesh(mergedGeometry, mat)
  mesh.castShadow = true
  mesh.receiveShadow = true
  return mesh
}

export function disposeImported(handles: ImportedModelHandles | null): void {
  if (!handles) return
  handles.mergedGeometry.dispose()
  handles.mesh.geometry.dispose()
  const mm = handles.mesh.material
  if (!Array.isArray(mm)) mm.dispose()
  handles.points.geometry.dispose()
  const pm = handles.points.material
  if (!Array.isArray(pm)) pm.dispose()
}
