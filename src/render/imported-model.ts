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

export function buildParticleCloud(
  surfaceMesh: Mesh,
  particleCount: number
): Points {
  const sampler = new MeshSurfaceSampler(surfaceMesh).build()
  const positions = new Float32Array(particleCount * 3)
  const colors = new Float32Array(particleCount * 3)
  const baseColor = new Color(0x88bbff)
  for (let i = 0; i < particleCount; i++) {
    sampler.sample(_tmp)
    positions[i * 3] = _tmp.x
    positions[i * 3 + 1] = _tmp.y
    positions[i * 3 + 2] = _tmp.z
    const h = 0.55 + Math.random() * 0.15
    const s = 0.5 + Math.random() * 0.3
    const l = 0.4 + Math.random() * 0.3
    baseColor.setHSL(h, s, l)
    colors[i * 3] = baseColor.r
    colors[i * 3 + 1] = baseColor.g
    colors[i * 3 + 2] = baseColor.b
  }
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

export function createImportedModelMesh(
  mergedGeometry: BufferGeometry,
  envIntensity = 0.95
): Mesh {
  const mat = new MeshPhysicalMaterial({
    color: 0xffffff,
    vertexColors: false,
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
