import { BufferGeometry, Group, IcosahedronGeometry, Mesh, MeshBasicMaterial, Vector3 } from 'three'

const _p = new Vector3()

/**
 * Multi-scale radial displacement on the unit sphere (macro lobes + octaves + fine ridges).
 * Tuned to stay mostly convex-avoiding self-intersections at icosahedron detail 5.
 */
function radialDisplacement(p: Vector3): number {
  const x = p.x
  const y = p.y
  const z = p.z
  const az = Math.atan2(y, x)
  const pol = Math.acos(Math.max(-1, Math.min(1, z)))

  let s = 0

  s += 0.11 * Math.sin(pol * 5) * Math.cos(az * 3)
  s += 0.075 * Math.cos(pol * 7) * Math.sin(az * 5)
  s += 0.05 * Math.sin(pol * 3 + az * 4)

  let amp = 0.095
  let f = 3.6
  let xx = x
  let yy = y
  let zz = z
  for (let o = 0; o < 5; o++) {
    s += amp * Math.sin(f * xx * 1.08 + yy * 0.42 + zz * 0.31)
    s += amp * 0.62 * Math.sin(f * yy * 1.05 + zz * 0.55 + xx * 0.28)
    s += amp * 0.45 * Math.cos(f * zz * 1.02 + xx * 0.33 + yy * 0.41)
    amp *= 0.5
    f *= 2.05
    const nx = xx * 0.91 + yy * 0.12
    const ny = yy * 0.88 - xx * 0.09
    const nz = zz * 0.94 + xx * 0.05
    xx = nx
    yy = ny
    zz = nz
  }

  s += 0.028 * Math.sin((x + y * 1.7 + z * 0.73) * 31)
  s += 0.018 * Math.sin((y * 1.2 + z * 1.9 + x * 0.4) * 37)

  return Math.max(-0.22, Math.min(0.28, s))
}

/**
 * ~20k faces: icosahedron detail 5, vertices displaced along normals for a radiolarian-like shell.
 */
export function createRadiolarianGeometry(): BufferGeometry {
  const geo = new IcosahedronGeometry(1, 5)
  const pos = geo.attributes.position.array as Float32Array
  const n = pos.length / 3
  for (let i = 0; i < n; i++) {
    _p.set(pos[i * 3]!, pos[i * 3 + 1]!, pos[i * 3 + 2]!)
    _p.normalize()
    const r = 1 + radialDisplacement(_p)
    _p.multiplyScalar(r)
    pos[i * 3] = _p.x
    pos[i * 3 + 1] = _p.y
    pos[i * 3 + 2] = _p.z
  }
  geo.attributes.position.needsUpdate = true
  geo.computeVertexNormals()
  return geo
}

/** Single-mesh group for the same import path as OBJ/FBX. */
export function createRadiolarianGroup(): Group {
  const geometry = createRadiolarianGeometry()
  const mesh = new Mesh(
    geometry,
    new MeshBasicMaterial({ color: 0xffffff, wireframe: false })
  )
  const group = new Group()
  group.add(mesh)
  return group
}
