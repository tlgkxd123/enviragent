import * as THREE from 'three'
import type { Scene, PerspectiveCamera, WebGLRenderer } from 'three'
import { createVolumetricCloud, stepCloud as _stepCloud } from './volumetric-cloud'
import { buildEnvFromConfig } from './env-codegen'
import type { EnvConfig } from './env-codegen'
import { buildProceduralTexture } from './procedural-texture'
import type { ProceduralTextureHandle } from './procedural-texture'
import {
  createCrystalClusterGroup,
  createNoiseSphereGeometry,
  createRockFieldGroup,
  createScatterGroup,
  createTerrainGeometry,
} from './procedural-objects'
import type { TerrainOptions, NoiseSphereOptions } from './procedural-objects'
import { generateTree } from './procedural-tree'
import type { TreeOptions } from './procedural-tree'
import { createFFTWater } from './fft-water'
import type { WaterOptions } from './fft-water'
import { createLandscape } from './terrain'
import type { LandscapeOptions } from './terrain'
import { Sky } from 'three/addons/objects/Sky.js'
import { createCurlSwarm } from './curl-swarm'
import type { CurlSwarmOptions } from './curl-swarm'

/**
 * Tool definitions + executors for the agentic scene builder.
 * Each tool has a JSON Schema (for OpenAI function-calling) and an executor.
 * Scene objects are registered by name so later tool calls can reference them.
 */

export interface ToolContext {
  scene: Scene
  camera: PerspectiveCamera
  renderer: WebGLRenderer
  applyEnvMap: (t: THREE.Texture) => void
  registry: Map<string, THREE.Object3D>
  frameFns: Array<(dt: number, t: number) => void>
  added: THREE.Object3D[]
  /** Renders the current frame and returns a data-URL screenshot (JPEG). */
  captureFrame?: () => string | null
}

export interface ToolResult {
  ok: boolean
  message: string
}

function track(ctx: ToolContext, name: string, obj: THREE.Object3D): void {
  ctx.scene.add(obj)
  ctx.added.push(obj)
  if (name) ctx.registry.set(name, obj)
}

// ── Singleton removal (auto-replace previous water/terrain) ─────────────────

type FrameFn = (dt: number, t: number) => void
const _singletons: Record<string, { name: string; updateFn: FrameFn | null }> = {}

function removePrevious(ctx: ToolContext, tag: string) {
  const prev = _singletons[tag]
  if (!prev) return
  const obj = ctx.registry.get(prev.name)
  if (obj) {
    ctx.scene.remove(obj)
    disposeObjectResources(obj)
    ctx.registry.delete(prev.name)
    const idx = ctx.added.indexOf(obj)
    if (idx >= 0) ctx.added.splice(idx, 1)
  }
  if (prev.updateFn) {
    const fi = ctx.frameFns.indexOf(prev.updateFn)
    if (fi >= 0) ctx.frameFns.splice(fi, 1)
  }
  delete _singletons[tag]
}

// ── Executors ────────────────────────────────────────────────────────────────

function makeGeometryFromToolArgs(a: Record<string, unknown>): THREE.BufferGeometry {
  const p = (a.geometry_params as number[]) ?? []
  switch (String(a.geometry ?? 'sphere').toLowerCase()) {
    case 'sphere':      return new THREE.SphereGeometry(p[0] ?? 1, p[1] ?? 48, p[2] ?? 48)
    case 'box':         return new THREE.BoxGeometry(p[0] ?? 1, p[1] ?? 1, p[2] ?? 1)
    case 'cylinder':    return new THREE.CylinderGeometry(p[0] ?? 0.5, p[1] ?? 0.5, p[2] ?? 1, p[3] ?? 32)
    case 'torus':       return new THREE.TorusGeometry(p[0] ?? 1, p[1] ?? 0.3, p[2] ?? 16, p[3] ?? 100)
    case 'torus_knot':  return new THREE.TorusKnotGeometry(p[0] ?? 0.8, p[1] ?? 0.25, p[2] ?? 128, p[3] ?? 16)
    case 'cone':        return new THREE.ConeGeometry(p[0] ?? 0.5, p[1] ?? 1, p[2] ?? 32)
    case 'octahedron':  return new THREE.OctahedronGeometry(p[0] ?? 1, Math.min(p[1] ?? 0, 5))
    case 'icosahedron': return new THREE.IcosahedronGeometry(p[0] ?? 1, Math.min(p[1] ?? 0, 5))
    case 'tetrahedron': return new THREE.TetrahedronGeometry(p[0] ?? 1, p[1] ?? 0)
    case 'plane':       return new THREE.PlaneGeometry(p[0] ?? 2, p[1] ?? 2, p[2] ?? 1, p[3] ?? 1)
    case 'ring':        return new THREE.RingGeometry(p[0] ?? 0.5, p[1] ?? 1, p[2] ?? 32)
    default:            return new THREE.SphereGeometry(1, 32, 32)
  }
}

function makePhysicalMaterial(a: Record<string, unknown>): THREE.MeshPhysicalMaterial {
  const opts: THREE.MeshPhysicalMaterialParameters = {
    color: new THREE.Color(String(a.color ?? '#aaaaff')),
    metalness: Number(a.metalness ?? 0.1),
    roughness: Number(a.roughness ?? 0.4),
    wireframe: Boolean(a.wireframe ?? false),
    envMapIntensity: Number(a.env_map_intensity ?? 1.0),
  }
  if (Number(a.transmission ?? 0) > 0) {
    opts.transmission = Number(a.transmission)
    opts.thickness = Number(a.thickness ?? 1.0)
  }
  if (a.emissive) {
    opts.emissive = new THREE.Color(String(a.emissive))
    opts.emissiveIntensity = Number(a.emissive_intensity ?? 1.0)
  }
  return new THREE.MeshPhysicalMaterial(opts)
}

function execAddProceduralMesh(ctx: ToolContext, a: Record<string, unknown>): ToolResult {
  const preset = String(a.preset ?? 'noise_sphere').toLowerCase()
  const seed = Number(a.seed ?? 1337) | 0
  const name = String(a.name ?? 'proc')
  const pos = (a.position as number[]) ?? [0, 0, 0]
  const rot = a.rotation as number[] | undefined
  const sc = a.scale

  const mat = () => makePhysicalMaterial(a)

  let root: THREE.Object3D

  switch (preset) {
    case 'terrain': {
      const w = Number(a.terrain_width ?? 14)
      const d = Number(a.terrain_depth ?? 14)
      const sx = Math.min(256, Math.max(8, Math.floor(Number(a.terrain_segments_x ?? 64))))
      const sz = Math.min(256, Math.max(8, Math.floor(Number(a.terrain_segments_z ?? 64))))
      const h = Number(a.height_scale ?? 1.8)
      const tOpts: TerrainOptions = {
        fbmOctaves:       a.fbm_octaves != null ? Number(a.fbm_octaves) : undefined,
        fbmLacunarity:    a.fbm_lacunarity != null ? Number(a.fbm_lacunarity) : undefined,
        fbmGain:          a.fbm_gain != null ? Number(a.fbm_gain) : undefined,
        warpStrength:     a.warp_strength != null ? Number(a.warp_strength) : undefined,
        ridged:           a.ridged != null ? Boolean(a.ridged) : undefined,
        erosionEnabled:   a.erosion_enabled != null ? Boolean(a.erosion_enabled) : true,
        erosionIterations:a.erosion_iterations != null ? Number(a.erosion_iterations) : 30000,
        thermalIterations:a.thermal_iterations != null ? Number(a.thermal_iterations) : undefined,
        talusAngle:       a.talus_angle != null ? Number(a.talus_angle) : undefined,
      }
      const geo = createTerrainGeometry(w, d, sx, sz, h, seed, tOpts)
      const m = mat()
      m.vertexColors = true
      root = new THREE.Mesh(geo, m)
      break
    }
    case 'noise_sphere': {
      const r = Number(a.radius ?? 1)
      const ws = Math.min(256, Math.max(16, Math.floor(Number(a.width_segments ?? 48))))
      const hs = Math.min(256, Math.max(16, Math.floor(Number(a.height_segments ?? 48))))
      const disp = Number(a.displacement ?? 0.35)
      const nsOpts: NoiseSphereOptions = {
        fbmOctaves:    a.fbm_octaves != null ? Number(a.fbm_octaves) : undefined,
        fbmLacunarity: a.fbm_lacunarity != null ? Number(a.fbm_lacunarity) : undefined,
        fbmGain:       a.fbm_gain != null ? Number(a.fbm_gain) : undefined,
        warpStrength:  a.warp_strength != null ? Number(a.warp_strength) : undefined,
        ridged:        a.ridged != null ? Boolean(a.ridged) : undefined,
      }
      const geo = createNoiseSphereGeometry(r, ws, hs, disp, seed, nsOpts)
      root = new THREE.Mesh(geo, mat())
      break
    }
    case 'scatter': {
      const base = String(a.scatter_geometry ?? 'sphere').toLowerCase() as
        'sphere' | 'box' | 'cone' | 'tetrahedron' | 'octahedron'
      const count = Math.min(400, Math.max(1, Math.floor(Number(a.instance_count ?? 24))))
      const spread = Number(a.spread ?? 4)
      const smin = Number(a.size_min ?? 0.08)
      const smax = Number(a.size_max ?? 0.45)
      root = createScatterGroup(
        ['sphere', 'box', 'cone', 'tetrahedron', 'octahedron'].includes(base) ? base : 'sphere',
        count,
        spread,
        smin,
        smax,
        seed,
        mat
      )
      break
    }
    case 'crystal_cluster': {
      const count = Math.min(200, Math.max(3, Math.floor(Number(a.crystal_count ?? 28))))
      const spread = Number(a.spread ?? 2.2)
      const hs = Number(a.height_scale ?? 1)
      root = createCrystalClusterGroup(count, spread, hs, seed, mat)
      break
    }
    case 'rock_field': {
      const count = Math.min(300, Math.max(2, Math.floor(Number(a.rock_count ?? 40))))
      const spread = Number(a.spread ?? 5)
      const smin = Number(a.size_min ?? 0.12)
      const smax = Number(a.size_max ?? 0.55)
      root = createRockFieldGroup(count, spread, smin, smax, seed, mat)
      break
    }
    default:
      return { ok: false, message: `Unknown procedural preset "${preset}".` }
  }

  root.position.set(pos[0], pos[1], pos[2])
  if (rot) root.rotation.set(rot[0], rot[1], rot[2])
  if (sc !== undefined) {
    if (Array.isArray(sc)) root.scale.set(sc[0], sc[1], sc[2])
    else root.scale.setScalar(Number(sc))
  }
  root.traverse((o) => {
    if (o instanceof THREE.Mesh) {
      o.castShadow = Boolean(a.cast_shadow ?? true)
      o.receiveShadow = true
    }
  })

  track(ctx, name, root)
  return { ok: true, message: `Procedural "${name}" (${preset}, seed ${seed}) added.` }
}

function execAddMesh(ctx: ToolContext, a: Record<string, unknown>): ToolResult {
  const geo = makeGeometryFromToolArgs(a)
  const mat  = makePhysicalMaterial(a)
  const mesh = new THREE.Mesh(geo, mat)
  const pos = (a.position as number[]) ?? [0, 0, 0]
  mesh.position.set(pos[0], pos[1], pos[2])
  const rot = a.rotation as number[] | undefined
  if (rot) mesh.rotation.set(rot[0], rot[1], rot[2])
  const sc = a.scale
  if (sc !== undefined) {
    if (Array.isArray(sc)) mesh.scale.set(sc[0], sc[1], sc[2])
    else mesh.scale.setScalar(Number(sc))
  }
  mesh.castShadow    = Boolean(a.cast_shadow ?? true)
  mesh.receiveShadow = true
  track(ctx, String(a.name ?? ''), mesh)
  return { ok: true, message: `Mesh "${a.name}" (${a.geometry}) added at [${pos}].` }
}

function execAddLight(ctx: ToolContext, a: Record<string, unknown>): ToolResult {
  const col = new THREE.Color(String(a.color ?? '#ffffff'))
  const int = Number(a.intensity ?? 1)
  let light: THREE.Light
  switch (String(a.type ?? 'point').toLowerCase()) {
    case 'point':       light = new THREE.PointLight(col, int, Number(a.distance ?? 20)); break
    case 'spot':        light = new THREE.SpotLight(col, int, Number(a.distance ?? 30)); break
    case 'directional': light = new THREE.DirectionalLight(col, int); break
    case 'ambient':     light = new THREE.AmbientLight(col, int); break
    case 'hemisphere':  light = new THREE.HemisphereLight(col, new THREE.Color('#101018'), int); break
    default:            light = new THREE.PointLight(col, int, 20)
  }
  const pos = a.position as number[] | undefined
  if (pos) light.position.set(pos[0], pos[1], pos[2])
  if (a.cast_shadow && light instanceof THREE.DirectionalLight) {
    light.castShadow = true
    light.shadow.mapSize.set(1024, 1024)
  }
  track(ctx, String(a.name ?? ''), light)
  return { ok: true, message: `Light "${a.name}" (${a.type}, ${int}) added.` }
}

function execAddCloud(ctx: ToolContext, a: Record<string, unknown>): ToolResult {
  const pos = a.position as number[] ?? [0, 4, 0]
  const sz  = a.size     as number[] ?? [8, 2.5, 8]
  const mesh = createVolumetricCloud({
    position:  [pos[0], pos[1], pos[2]],
    size:      [sz[0],  sz[1],  sz[2]],
    color:     new THREE.Color(String(a.color ?? '#d8e8f0')).getHex(),
    density:   Number(a.density  ?? 0.85),
    coverage:  Number(a.coverage ?? 0.42),
    sunDir:    a.sun_dir  as [number,number,number] | undefined,
    sunColor:  a.sun_color ? new THREE.Color(String(a.sun_color)).getHex() : undefined,
  })
  ctx.scene.add(mesh)
  ctx.added.push(mesh)
  ctx.registry.set(String(a.name ?? ''), mesh)
  ctx.frameFns.push((dt) => _stepCloud(mesh, dt))
  return { ok: true, message: `Cloud "${a.name}" added at [${pos}], size [${sz}].` }
}

function execAddParticles(ctx: ToolContext, a: Record<string, unknown>): ToolResult {
  const N      = Math.min(80000, Math.max(100, Number(a.count ?? 8000)))
  const spread = Number(a.spread ?? 3)
  const flatY  = Number(a.flatten_y ?? 1.0)
  const center = (a.center as number[]) ?? [0, 1.5, 0]
  const hr     = (a.color_hue_range as number[]) ?? [0.5, 0.85]
  const pos = new Float32Array(N * 3)
  const col = new Float32Array(N * 3)
  for (let i = 0; i < N; i++) {
    const r     = Math.pow(Math.random(), 0.45) * spread
    const theta = Math.random() * Math.PI * 2
    const phi   = Math.acos(2 * Math.random() - 1)
    pos[i*3]   = center[0] + r * Math.sin(phi) * Math.cos(theta)
    pos[i*3+1] = center[1] + r * Math.cos(phi) * flatY
    pos[i*3+2] = center[2] + r * Math.sin(phi) * Math.sin(theta)
    const c = new THREE.Color().setHSL(hr[0] + Math.random() * (hr[1] - hr[0]), 0.85, 0.45 + Math.random() * 0.3)
    col[i*3] = c.r; col[i*3+1] = c.g; col[i*3+2] = c.b
  }
  const geo = new THREE.BufferGeometry()
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3))
  geo.setAttribute('color',    new THREE.BufferAttribute(col, 3))
  const mat = new THREE.PointsMaterial({
    size: Number(a.size ?? 0.025), vertexColors: true,
    transparent: true, opacity: Number(a.opacity ?? 0.8), sizeAttenuation: true,
  })
  const pts = new THREE.Points(geo, mat)
  track(ctx, String(a.name ?? ''), pts)
  return { ok: true, message: `Particle system "${a.name}" with ${N} particles added.` }
}

function execAddCurlSwarm(ctx: ToolContext, a: Record<string, unknown>): ToolResult {
  const name = String(a.name ?? 'swarm')
  const pos = (a.position as number[]) ?? [0, 0, 0]
  const rot = a.rotation as number[] | undefined
  const sc = a.scale

  const opts: CurlSwarmOptions = {
    count: Number(a.count ?? 10000),
    radius: Number(a.radius ?? 5.0),
    color1: a.color1 ? new THREE.Color(String(a.color1)).getHex() : undefined,
    color2: a.color2 ? new THREE.Color(String(a.color2)).getHex() : undefined,
    sizeMin: Number(a.size_min ?? 0.02),
    sizeMax: Number(a.size_max ?? 0.08),
    speed: Number(a.speed ?? 0.2),
    scale: Number(a.scale_noise ?? 0.5),
    swirl: Number(a.swirl ?? 1.5),
  }

  const swarm = createCurlSwarm(opts)
  swarm.position.set(pos[0], pos[1], pos[2])
  if (rot) swarm.rotation.set(rot[0], rot[1], rot[2])
  if (sc != null) {
    if (Array.isArray(sc)) swarm.scale.set(sc[0], sc[1], sc[2])
    else swarm.scale.setScalar(Number(sc))
  }

  track(ctx, name, swarm)
  
  if (swarm.userData.update) {
    ctx.frameFns.push(swarm.userData.update)
  }

  return { ok: true, message: `Added Curl Noise Swarm '${name}' with ${opts.count} particles.` }
}

function execAnimate(ctx: ToolContext, a: Record<string, unknown>): ToolResult {
  const name = String(a.name ?? '')
  const obj  = ctx.registry.get(name)
  if (!obj) return { ok: false, message: `Object "${name}" not found. Use a name you previously created.` }
  const baseY   = obj.position.y
  const basePos = obj.position.clone()
  let   orbitAngle = 0
  ctx.frameFns.push((dt, t) => {
    const rs = a.rotate_speed as number[] | undefined
    if (rs) { obj.rotation.x += rs[0]*dt; obj.rotation.y += rs[1]*dt; obj.rotation.z += rs[2]*dt }
    const bobA = Number(a.bob_amplitude ?? 0)
    if (bobA) obj.position.y = baseY + Math.sin(t * Number(a.bob_speed ?? 1)) * bobA
    const orbitR = Number(a.orbit_radius ?? 0)
    if (orbitR) {
      orbitAngle += Number(a.orbit_speed ?? 0.5) * dt
      const ax = String(a.orbit_axis ?? 'y')
      if (ax === 'y') { obj.position.x = basePos.x + Math.cos(orbitAngle)*orbitR; obj.position.z = basePos.z + Math.sin(orbitAngle)*orbitR }
      else if (ax === 'x') { obj.position.y = basePos.y + Math.cos(orbitAngle)*orbitR; obj.position.z = basePos.z + Math.sin(orbitAngle)*orbitR }
      else { obj.position.x = basePos.x + Math.cos(orbitAngle)*orbitR; obj.position.y = basePos.y + Math.sin(orbitAngle)*orbitR }
    }
    const ps = Number(a.pulse_scale ?? 0)
    if (ps) obj.scale.setScalar(1 + Math.sin(t * Number(a.pulse_speed ?? 2)) * ps)
  })
  return { ok: true, message: `Animation applied to "${name}".` }
}

function execSetCamera(ctx: ToolContext, a: Record<string, unknown>): ToolResult {
  const pos = (a.position as number[]) ?? [3.2, 1.45, 3.6]
  const lat = (a.look_at  as number[]) ?? [0, 0, 0]
  const dur = Number(a.animate_duration ?? 0)
  if (a.fov) { ctx.camera.fov = Number(a.fov); ctx.camera.updateProjectionMatrix() }
  if (dur > 0) {
    const startP = ctx.camera.position.clone()
    const endP   = new THREE.Vector3(pos[0], pos[1], pos[2])
    const tgt    = new THREE.Vector3(lat[0], lat[1], lat[2])
    let elapsed  = 0
    const fn = (dt: number) => {
      elapsed = Math.min(elapsed + dt, dur)
      const e = elapsed / dur
      const ease = e < 0.5 ? 2*e*e : -1+(4-2*e)*e
      ctx.camera.position.lerpVectors(startP, endP, ease)
      ctx.camera.lookAt(tgt)
      if (elapsed >= dur) ctx.frameFns.splice(ctx.frameFns.indexOf(fn), 1)
    }
    ctx.frameFns.push(fn)
  } else {
    ctx.camera.position.set(pos[0], pos[1], pos[2])
    ctx.camera.lookAt(lat[0], lat[1], lat[2])
  }
  return { ok: true, message: `Camera set to [${pos}], looking at [${lat}]${dur ? ` over ${dur}s` : ''}.` }
}

function execSetEnvironment(ctx: ToolContext, a: Record<string, unknown>): ToolResult {
  try {
    const cfg = a as unknown as EnvConfig
    const { envMap } = buildEnvFromConfig(cfg, ctx.renderer)
    ctx.applyEnvMap(envMap)
    return { ok: true, message: `Environment applied (${cfg.lights?.length ?? 0} lights).` }
  } catch (e) {
    return { ok: false, message: `Environment error: ${String(e)}` }
  }
}

function execSetBackground(ctx: ToolContext, a: Record<string, unknown>): ToolResult {
  ctx.scene.background = new THREE.Color(String(a.color ?? '#040506'))
  return { ok: true, message: `Background set to ${a.color}.` }
}

function execAddSky(ctx: ToolContext, a: Record<string, unknown>): ToolResult {
  removePrevious(ctx, 'sky')

  const name = String(a.name ?? 'sky')
  const elevation = Number(a.sun_elevation ?? 45)
  const azimuth   = Number(a.sun_azimuth ?? 180)
  const turbidity  = Number(a.turbidity ?? 2)
  const rayleigh   = Number(a.rayleigh ?? 1)
  const mieCoeff   = Number(a.mie_coefficient ?? 0.005)
  const mieDir     = Number(a.mie_directional_g ?? 0.8)
  const skyScale   = Number(a.scale ?? 1000)

  const sky = new Sky()
  sky.scale.setScalar(skyScale)

  const skyUniforms = sky.material.uniforms
  skyUniforms['turbidity'].value = turbidity
  skyUniforms['rayleigh'].value = rayleigh
  skyUniforms['mieCoefficient'].value = mieCoeff
  skyUniforms['mieDirectionalG'].value = mieDir

  const phi = THREE.MathUtils.degToRad(90 - elevation)
  const theta = THREE.MathUtils.degToRad(azimuth)
  const sunPos = new THREE.Vector3().setFromSphericalCoords(1, phi, theta)
  skyUniforms['sunPosition'].value.copy(sunPos)

  track(ctx, name, sky)
  _singletons['sky'] = { name, updateFn: null }

  // Update directional lights in the scene to match sun direction
  const sunWorldPos = sunPos.clone().multiplyScalar(100)
  ctx.scene.traverse((child) => {
    if (child instanceof THREE.DirectionalLight) {
      child.position.copy(sunWorldPos)
    }
  })

  // Generate PMREM environment map from the sky for reflections
  const pmrem = new THREE.PMREMGenerator(ctx.renderer)
  pmrem.compileCubemapShader()
  const renderTarget = pmrem.fromScene(sky as unknown as THREE.Scene)
  ctx.applyEnvMap(renderTarget.texture)
  ctx.scene.background = renderTarget.texture
  pmrem.dispose()

  const elStr = elevation.toFixed(1)
  const azStr = azimuth.toFixed(1)
  return {
    ok: true,
    message: `Sky "${name}" added (sun elevation ${elStr}°, azimuth ${azStr}°, turbidity ${turbidity}, rayleigh ${rayleigh}). Environment map + background updated for reflections.`,
  }
}

function execListObjects(ctx: ToolContext, _a: Record<string, unknown>): ToolResult {
  const names = [...ctx.registry.keys()]
  return { ok: true, message: names.length ? `Objects in scene: ${names.join(', ')}.` : 'No objects added yet.' }
}

function execRemoveObject(ctx: ToolContext, a: Record<string, unknown>): ToolResult {
  const name = String(a.name ?? '')
  const obj  = ctx.registry.get(name)
  if (!obj) return { ok: false, message: `Object "${name}" not found.` }
  ctx.scene.remove(obj)
  ctx.added.splice(ctx.added.indexOf(obj), 1)
  ctx.registry.delete(name)
  obj.traverse((c) => {
    if (c instanceof THREE.Mesh) { c.geometry?.dispose(); const m = c.material; if (Array.isArray(m)) m.forEach(x=>x.dispose()); else (m as THREE.Material)?.dispose() }
  })
  return { ok: true, message: `Object "${name}" removed.` }
}

function disposeObjectResources(obj: THREE.Object3D): void {
  obj.traverse((c) => {
    if (c instanceof THREE.Mesh || c instanceof THREE.Points || c instanceof THREE.Line) {
      c.geometry?.dispose()
      const m = c.material
      if (Array.isArray(m)) m.forEach((x) => (x as THREE.Material).dispose())
      else (m as THREE.Material | undefined)?.dispose()
    }
  })
}

// ── Procedural texture registry (disposed on reset_scene) ───────────────────
const _proceduralTextures = new Set<ProceduralTextureHandle>()

function execGenerateTree(ctx: ToolContext, a: Record<string, unknown>): ToolResult {
  const name = String(a.name ?? `tree_${Date.now()}`)
  const pos  = (a.position as number[]) ?? [0, -1.35, 0]

  const treeOpts: TreeOptions = {
    position:           [pos[0], pos[1], pos[2]],
    seed:               Number(a.seed ?? Math.floor(Math.random() * 99999)),
    trunkHeight:        Number(a.trunk_height  ?? 2.5),
    trunkRadius:        Number(a.trunk_radius  ?? 0.12),
    crownSize:          (a.crown_size  as [number, number, number]) ?? undefined,
    crownOffset:        (a.crown_offset as [number, number, number]) ?? undefined,
    attractorCount:     a.attractor_count != null ? Number(a.attractor_count) : undefined,
    branchStepSize:     a.branch_step   != null ? Number(a.branch_step)      : undefined,
    maxIterations:      a.max_iterations != null ? Number(a.max_iterations)   : undefined,
    pipeExponent:       a.pipe_exponent  != null ? Number(a.pipe_exponent)    : undefined,
    radialSegments:     a.radial_segments != null ? Number(a.radial_segments) : undefined,
    leafDensity:        a.leaf_density   != null ? Number(a.leaf_density)     : undefined,
    leafSize:           a.leaf_size      != null ? Number(a.leaf_size)        : undefined,
    barkColor:          a.bark_color  ? new THREE.Color(String(a.bark_color)).getHex()  : undefined,
    leafColor:          a.leaf_color  ? new THREE.Color(String(a.leaf_color)).getHex()  : undefined,
    leafColorVariation: a.leaf_color_variation != null ? Number(a.leaf_color_variation) : undefined,
    windStrength:       a.wind_strength != null ? Number(a.wind_strength) : undefined,
    tropism:            (a.tropism as [number, number, number]) ?? undefined,
  }

  const { group, update } = generateTree(treeOpts)

  if (a.scale != null) {
    const s = a.scale
    if (typeof s === 'number') group.scale.setScalar(s)
    else if (Array.isArray(s)) group.scale.set(Number(s[0] ?? 1), Number(s[1] ?? 1), Number(s[2] ?? 1))
  }
  if (a.rotation) {
    const r = a.rotation as number[]
    group.rotation.set(r[0] ?? 0, r[1] ?? 0, r[2] ?? 0)
  }

  track(ctx, name, group)
  if (update) ctx.frameFns.push(update)

  const nodeCount = group.children.length
  return {
    ok: true,
    message: `Tree "${name}" generated (space-colonization + da Vinci taper, ${nodeCount} meshes). Base at [${pos.join(',')}].`,
  }
}

// ── Edge-midpoint subdivision (preserves shared edges) ───────────────────────

function subdivideGeometry(geo: THREE.BufferGeometry): THREE.BufferGeometry {
  let src = geo
  if (!src.index) {
    const n = src.attributes.position.count
    const trivIdx = new Uint32Array(n)
    for (let i = 0; i < n; i++) trivIdx[i] = i
    src = src.clone()
    src.setIndex(new THREE.BufferAttribute(trivIdx, 1))
  }

  const oldPos = (src.attributes.position as THREE.BufferAttribute).array as Float32Array
  const oldUv = src.attributes.uv
    ? (src.attributes.uv as THREE.BufferAttribute).array as Float32Array : undefined
  const oldCol = src.attributes.color
    ? (src.attributes.color as THREE.BufferAttribute).array as Float32Array : undefined
  const colSize = src.attributes.color ? (src.attributes.color as THREE.BufferAttribute).itemSize : 3
  const indices = src.index!.array as Uint16Array | Uint32Array
  const nV = src.attributes.position.count
  const nTri = indices.length / 3

  const edgeMid = new Map<number, number>()
  let next = nV
  function mid(a: number, b: number): number {
    const key = Math.min(a, b) * 1000000 + Math.max(a, b)
    let m = edgeMid.get(key)
    if (m === undefined) { m = next++; edgeMid.set(key, m) }
    return m
  }
  for (let t = 0; t < nTri; t++) {
    const i0 = indices[t * 3], i1 = indices[t * 3 + 1], i2 = indices[t * 3 + 2]
    mid(i0, i1); mid(i1, i2); mid(i2, i0)
  }

  const totV = next
  const newPos = new Float32Array(totV * 3)
  newPos.set(oldPos.subarray(0, nV * 3))
  let newUv: Float32Array | undefined
  if (oldUv) { newUv = new Float32Array(totV * 2); newUv.set(oldUv.subarray(0, nV * 2)) }
  let newCol: Float32Array | undefined
  if (oldCol) { newCol = new Float32Array(totV * colSize); newCol.set(oldCol.subarray(0, nV * colSize)) }

  for (const [key, mi] of edgeMid) {
    const a = Math.floor(key / 1000000), b = key % 1000000
    newPos[mi * 3]     = (oldPos[a * 3]     + oldPos[b * 3])     / 2
    newPos[mi * 3 + 1] = (oldPos[a * 3 + 1] + oldPos[b * 3 + 1]) / 2
    newPos[mi * 3 + 2] = (oldPos[a * 3 + 2] + oldPos[b * 3 + 2]) / 2
    if (newUv && oldUv) {
      newUv[mi * 2]     = (oldUv[a * 2]     + oldUv[b * 2])     / 2
      newUv[mi * 2 + 1] = (oldUv[a * 2 + 1] + oldUv[b * 2 + 1]) / 2
    }
    if (newCol && oldCol) {
      for (let c = 0; c < colSize; c++)
        newCol[mi * colSize + c] = (oldCol[a * colSize + c] + oldCol[b * colSize + c]) / 2
    }
  }

  const newIdx = new Uint32Array(nTri * 4 * 3)
  let ii = 0
  for (let t = 0; t < nTri; t++) {
    const v0 = indices[t * 3], v1 = indices[t * 3 + 1], v2 = indices[t * 3 + 2]
    const m01 = mid(v0, v1), m12 = mid(v1, v2), m20 = mid(v2, v0)
    newIdx[ii++] = v0;  newIdx[ii++] = m01; newIdx[ii++] = m20
    newIdx[ii++] = m01; newIdx[ii++] = v1;  newIdx[ii++] = m12
    newIdx[ii++] = m20; newIdx[ii++] = m12; newIdx[ii++] = v2
    newIdx[ii++] = m01; newIdx[ii++] = m12; newIdx[ii++] = m20
  }

  const out = new THREE.BufferGeometry()
  out.setAttribute('position', new THREE.BufferAttribute(newPos, 3))
  if (newUv) out.setAttribute('uv', new THREE.BufferAttribute(newUv, 2))
  if (newCol) out.setAttribute('color', new THREE.BufferAttribute(newCol, colSize))
  out.setIndex(new THREE.BufferAttribute(newIdx, 1))
  out.computeVertexNormals()
  return out
}

function execTessellateMesh(ctx: ToolContext, a: Record<string, unknown>): ToolResult {
  const name = String(a.name ?? '')
  const obj = ctx.registry.get(name)
  if (!obj) return { ok: false, message: `Object "${name}" not found.` }
  const levels = Math.min(4, Math.max(1, Number(a.levels ?? 1)))
  let mesh: THREE.Mesh | undefined
  obj.traverse((c) => { if (c instanceof THREE.Mesh && !mesh) mesh = c })
  if (!mesh) return { ok: false, message: `No mesh found under "${name}".` }
  const before = mesh.geometry.attributes.position.count
  let geo = mesh.geometry
  for (let i = 0; i < levels; i++) geo = subdivideGeometry(geo)
  mesh.geometry.dispose()
  mesh.geometry = geo
  return { ok: true, message: `Tessellated "${name}" ×${levels}: ${before} → ${geo.attributes.position.count} vertices (${(geo.index?.count ?? 0) / 3} triangles).` }
}

function execAddWater(ctx: ToolContext, a: Record<string, unknown>): ToolResult {
  removePrevious(ctx, 'water')
  const name = String(a.name ?? 'water')
  const pos = (a.position as number[]) ?? [0, -1.35, 0]
  const wopts: WaterOptions = {
    size: (a.size as [number, number]) ?? [40, 40],
    resolution: a.resolution != null ? Number(a.resolution) : undefined,
    color: a.color ? new THREE.Color(String(a.color)).getHex() : undefined,
    wave_scale: a.wave_scale != null ? Number(a.wave_scale) : undefined,
    choppiness: a.choppiness != null ? Number(a.choppiness) : undefined,
    wave_count: a.wave_count != null ? Number(a.wave_count) : undefined,
    speed: a.speed != null ? Number(a.speed) : undefined,
    foam: a.foam != null ? Boolean(a.foam) : undefined,
    seed: a.seed != null ? Number(a.seed) : undefined,
    opacity: a.opacity != null ? Number(a.opacity) : undefined,
  }
  const handle = createFFTWater(wopts)
  handle.mesh.position.set(pos[0], pos[1], pos[2])
  track(ctx, name, handle.mesh)
  ctx.frameFns.push(handle.update)
  _singletons['water'] = { name, updateFn: handle.update }
  const sz = wopts.size ?? [40, 40]
  const r = wopts.resolution ?? 128
  return { ok: true, message: `3D ocean "${name}" added (${sz[0]}×${sz[1]}, ${r}×${r} vertices, ${wopts.wave_count ?? 12} Gerstner waves, choppiness ${wopts.choppiness ?? 0.8}, foam ${wopts.foam !== false ? 'on' : 'off'}).` }
}

function execAddTerrain(ctx: ToolContext, a: Record<string, unknown>): ToolResult {
  removePrevious(ctx, 'terrain')
  const name = String(a.name ?? 'terrain')
  const pos = (a.position as number[]) ?? [0, -2, 0]
  const lopts: LandscapeOptions = {
    size: (a.size as [number, number]) ?? undefined,
    resolution: a.resolution != null ? Number(a.resolution) : undefined,
    biome: a.biome != null ? String(a.biome) : undefined,
    height_scale: a.height_scale != null ? Number(a.height_scale) : undefined,
    seed: a.seed != null ? Number(a.seed) : undefined,
    octaves: a.octaves != null ? Number(a.octaves) : undefined,
    persistence: a.persistence != null ? Number(a.persistence) : undefined,
    lacunarity: a.lacunarity != null ? Number(a.lacunarity) : undefined,
    ridge_fraction: a.ridge_fraction != null ? Number(a.ridge_fraction) : undefined,
    snow_line: a.snow_line != null ? Number(a.snow_line) : undefined,
    tree_line: a.tree_line != null ? Number(a.tree_line) : undefined,
    tex_scale: a.tex_scale != null ? Number(a.tex_scale) : undefined,
  }
  const handle = createLandscape(lopts)
  handle.mesh.position.set(pos[0], pos[1], pos[2])
  track(ctx, name, handle.mesh)
  _singletons['terrain'] = { name, updateFn: null }
  const biome = lopts.biome ?? 'mountains'
  const sz = lopts.size ?? [40, 40]
  const r = lopts.resolution ?? 128
  return { ok: true, message: `Landscape "${name}" added (${sz[0]}×${sz[1]}, ${r}×${r} vertices, biome="${biome}", height ${handle.heightMin.toFixed(1)}→${handle.heightMax.toFixed(1)}, textured with grass/rock/dirt/snow blended by height+slope).` }
}

function execResetScene(ctx: ToolContext, _a: Record<string, unknown>): ToolResult {
  const n = ctx.added.length
  for (const obj of [...ctx.added]) {
    ctx.scene.remove(obj)
    disposeObjectResources(obj)
  }
  ctx.added.length = 0
  ctx.frameFns.length = 0
  ctx.registry.clear()
  for (const h of _proceduralTextures) h.dispose()
  _proceduralTextures.clear()
  for (const k of Object.keys(_singletons)) delete _singletons[k]
  return { ok: true, message: n ? `Scene reset: removed ${n} top-level object(s).` : 'Scene was already empty.' }
}

function execGenTexture(ctx: ToolContext, a: Record<string, unknown>): ToolResult {
  const domain = String(a.domain ?? 'cellular') as 'cellular' | 'material' | 'surface'
  const preset = String(a.preset ?? 'voronoi_membrane')
  const resolution = Math.min(4096, Math.max(64, Number(a.resolution ?? 512)))
  const params = (a.params ?? {}) as Record<string, number>
  const bake = Boolean(a.bake ?? false)
  const animate = !bake && a.animate !== false
  const mapSlot = String(a.map_slot ?? 'map') as
    'map' | 'emissiveMap' | 'normalMap' | 'alphaMap' | 'roughnessMap'

  let handle: ProceduralTextureHandle
  try {
    handle = buildProceduralTexture(ctx.renderer, { domain, preset, resolution, params, animate, bake })
  } catch (e) {
    return { ok: false, message: `gen_texture error: ${String(e)}` }
  }
  _proceduralTextures.add(handle)

  const tex = handle.texture
  if (mapSlot !== 'normalMap') {
    tex.colorSpace = THREE.SRGBColorSpace
  }

  const targetName = String(a.target_name ?? '').trim()
  if (targetName) {
    const obj = ctx.registry.get(targetName)
    if (!obj) return { ok: false, message: `gen_texture: object "${targetName}" not found.` }
    let foundMesh: THREE.Mesh | undefined
    obj.traverse((c) => { if (c instanceof THREE.Mesh && !foundMesh) foundMesh = c })
    if (!foundMesh) return { ok: false, message: `gen_texture: no mesh found under "${targetName}".` }
    const mat = foundMesh.material as THREE.MeshStandardMaterial | THREE.MeshPhysicalMaterial
    if (mat && mapSlot in mat) {
      (mat as unknown as Record<string, unknown>)[mapSlot] = tex
      mat.needsUpdate = true
    }
    const appliedMaps: string[] = [mapSlot]
    if (domain === 'surface' && mat) {
      if (handle.normalMap) {
        mat.normalMap = handle.normalMap
        handle.normalMap.colorSpace = THREE.LinearSRGBColorSpace
        mat.normalScale = new THREE.Vector2(1, 1)
        appliedMaps.push('normalMap')
      }
      if (handle.ormMap) {
        handle.ormMap.colorSpace = THREE.LinearSRGBColorSpace
        mat.aoMap = handle.ormMap
        mat.roughnessMap = handle.ormMap
        mat.metalnessMap = handle.ormMap
        mat.roughness = 1.0
        mat.metalness = 1.0
        appliedMaps.push('aoMap', 'roughnessMap', 'metalnessMap')
      }
      if (handle.emissiveMap) {
        mat.emissiveMap = handle.emissiveMap
        mat.emissive = new THREE.Color(1, 1, 1)
        mat.emissiveIntensity = 1.0
        appliedMaps.push('emissiveMap')
      }
      mat.needsUpdate = true
    }
    if (animate) ctx.frameFns.push((_dt, t) => handle.updateTime(t))
    return { ok: true, message: `Procedural texture (${domain}/${preset} @${resolution}px) applied to "${targetName}" [${appliedMaps.join(', ')}].` }
  }

  // No target — create a new display plane
  const name = String(a.name ?? `tex_${preset}`)
  const geo  = new THREE.PlaneGeometry(2, 2)
  const planeMat = new THREE.MeshStandardMaterial({ [mapSlot]: tex, side: THREE.DoubleSide })
  if (mapSlot === 'emissiveMap') {
    planeMat.emissive = new THREE.Color(1, 1, 1)
    planeMat.emissiveIntensity = 1
  }
  if (domain === 'surface') {
    if (handle.normalMap) {
      planeMat.normalMap = handle.normalMap
      handle.normalMap.colorSpace = THREE.LinearSRGBColorSpace
    }
    if (handle.ormMap) {
      handle.ormMap.colorSpace = THREE.LinearSRGBColorSpace
      planeMat.aoMap = handle.ormMap
      planeMat.roughnessMap = handle.ormMap
      planeMat.metalnessMap = handle.ormMap
      planeMat.roughness = 1.0
      planeMat.metalness = 1.0
    }
    if (handle.emissiveMap) {
      planeMat.emissiveMap = handle.emissiveMap
      planeMat.emissive = new THREE.Color(1, 1, 1)
      planeMat.emissiveIntensity = 1.0
    }
  }
  const mesh = new THREE.Mesh(geo, planeMat)
  const pos = (a.position as number[]) ?? [0, 1.5, 0]
  mesh.position.set(pos[0] ?? 0, pos[1] ?? 1.5, pos[2] ?? 0)
  track(ctx, name, mesh)

  if (animate) ctx.frameFns.push((_dt, t) => handle.updateTime(t))
  return { ok: true, message: `Procedural texture plane "${name}" created (${domain}/${preset} @${resolution}px${domain === 'surface' ? ', full PBR maps' : ''}, animate=${animate}).` }
}

function makeDataTexture(
  w: number,
  h: number,
  fn: (x: number, y: number, w: number, h: number) => { r: number; g: number; b: number; a: number }
): THREE.DataTexture {
  const data = new Uint8Array(w * h * 4)
  for (let y2 = 0; y2 < h; y2++) {
    for (let x2 = 0; x2 < w; x2++) {
      const px = fn(x2, y2, w, h)
      const i = (y2 * w + x2) * 4
      data[i]     = Math.max(0, Math.min(255, Math.round(px.r)))
      data[i + 1] = Math.max(0, Math.min(255, Math.round(px.g)))
      data[i + 2] = Math.max(0, Math.min(255, Math.round(px.b)))
      data[i + 3] = Math.max(0, Math.min(255, Math.round(px.a)))
    }
  }
  const tex = new THREE.DataTexture(data, w, h)
  tex.needsUpdate = true
  return tex
}

function parseShaderUniforms(raw: unknown): Record<string, THREE.IUniform> {
  if (!raw || typeof raw !== 'object') return {}
  const out: Record<string, THREE.IUniform> = {}
  for (const [k, v] of Object.entries(raw as Record<string, unknown>)) {
    if (!v || typeof v !== 'object') continue
    const u = v as Record<string, unknown>
    const t = String(u.type ?? 'float')
    const val = u.value
    switch (t) {
      case 'float':
        out[k] = { value: Number(val ?? 0) }
        break
      case 'int':
        out[k] = { value: Math.floor(Number(val ?? 0)) }
        break
      case 'vec2': {
        const ar = (val as number[]) ?? [0, 0]
        out[k] = { value: new THREE.Vector2(ar[0], ar[1]) }
        break
      }
      case 'vec3': {
        const ar = (val as number[]) ?? [0, 0, 0]
        out[k] = { value: new THREE.Vector3(ar[0], ar[1], ar[2]) }
        break
      }
      case 'vec4': {
        const ar = (val as number[]) ?? [0, 0, 0, 1]
        out[k] = { value: new THREE.Vector4(ar[0], ar[1], ar[2], ar[3]) }
        break
      }
      case 'color':
        out[k] = { value: new THREE.Color(String(val ?? '#ffffff')) }
        break
      default:
        out[k] = { value: Number(val ?? 0) }
    }
  }
  return out
}

function shaderSide(a: Record<string, unknown>): THREE.Side {
  const s = String(a.side ?? 'front').toLowerCase()
  if (s === 'double') return THREE.DoubleSide
  if (s === 'back') return THREE.BackSide
  return THREE.FrontSide
}

function execExecThreejsCode(ctx: ToolContext, a: Record<string, unknown>): ToolResult {
  const code = String(a.code ?? '')
  if (code.length < 1) return { ok: false, message: 'Empty code.' }
  if (code.length > 100_000) return { ok: false, message: 'Code exceeds 100k characters.' }

  const api = {
    scene: ctx.scene,
    camera: ctx.camera,
    renderer: ctx.renderer,
    add(obj: THREE.Object3D, name?: string) {
      ctx.scene.add(obj)
      ctx.added.push(obj)
      if (name !== undefined && name !== '') ctx.registry.set(String(name), obj)
    },
    remove(obj: THREE.Object3D) {
      ctx.scene.remove(obj)
      const ix = ctx.added.indexOf(obj)
      if (ix !== -1) ctx.added.splice(ix, 1)
      for (const [k, v] of [...ctx.registry.entries()]) {
        if (v === obj) ctx.registry.delete(k)
      }
    },
    get(name: string) {
      return ctx.registry.get(name) ?? null
    },
    cloud(opts: Parameters<typeof createVolumetricCloud>[0]) {
      const mesh = createVolumetricCloud(opts)
      ctx.scene.add(mesh)
      ctx.added.push(mesh)
      ctx.frameFns.push((dt) => _stepCloud(mesh, dt))
      return mesh
    },
    stepCloud: _stepCloud,
    texture: makeDataTexture,
    env(config: EnvConfig) {
      const { envMap } = buildEnvFromConfig(config, ctx.renderer)
      ctx.applyEnvMap(envMap)
    },
    onFrame(fn: (dt: number, t: number) => void) {
      ctx.frameFns.push(fn)
    },
    setBackground(hexOrString: number | string) {
      if (typeof hexOrString === 'string') ctx.scene.background = new THREE.Color(hexOrString)
      else ctx.scene.background = new THREE.Color(hexOrString)
    },
  }

  try {
    // eslint-disable-next-line no-new-func
    const fn = new Function('THREE', 'api', `"use strict";\n${code}`)
    fn(THREE, api)
  } catch (e) {
    return { ok: false, message: `exec_threejs_code: ${String(e)}` }
  }
  return { ok: true, message: 'JavaScript executed (THREE + api).' }
}

/**
 * Three.js ShaderMaterial with glslVersion: GLSL3 already injects these as
 * built-in attributes/uniforms. If the LLM re-declares them the GLSL compiler
 * throws "redefinition". Strip them out before compilation.
 */
const THREE_BUILTIN_VERTEX_DECLS = [
  // attributes
  /^\s*in\s+vec3\s+position\s*;\s*$/m,
  /^\s*in\s+vec2\s+uv\s*;\s*$/m,
  /^\s*in\s+vec3\s+normal\s*;\s*$/m,
  /^\s*in\s+vec2\s+uv2\s*;\s*$/m,
  /^\s*in\s+vec4\s+color\s*;\s*$/m,
  /^\s*attribute\s+vec3\s+position\s*;\s*$/m,
  /^\s*attribute\s+vec2\s+uv\s*;\s*$/m,
  /^\s*attribute\s+vec3\s+normal\s*;\s*$/m,
  // built-in uniforms
  /^\s*uniform\s+mat4\s+modelMatrix\s*;\s*$/m,
  /^\s*uniform\s+mat4\s+modelViewMatrix\s*;\s*$/m,
  /^\s*uniform\s+mat4\s+projectionMatrix\s*;\s*$/m,
  /^\s*uniform\s+mat4\s+viewMatrix\s*;\s*$/m,
  /^\s*uniform\s+mat3\s+normalMatrix\s*;\s*$/m,
  /^\s*uniform\s+vec3\s+cameraPosition\s*;\s*$/m,
]

function sanitizeVertexShader(src: string): string {
  let out = src
  for (const re of THREE_BUILTIN_VERTEX_DECLS) out = out.replace(re, '')
  return out
}

function execGenShaderCode(ctx: ToolContext, a: Record<string, unknown>): ToolResult {
  const vs = sanitizeVertexShader(String(a.vertex_shader ?? ''))
  const fs = String(a.fragment_shader ?? '')
  if (vs.length < 8) return { ok: false, message: 'vertex_shader is required.' }
  if (fs.length < 8) return { ok: false, message: 'fragment_shader is required.' }
  if (vs.length > 120_000 || fs.length > 120_000) return { ok: false, message: 'Shader source too long.' }

  let uniforms = parseShaderUniforms(a.uniforms)
  if (Boolean(a.use_scene_environment) && ctx.scene.environment) {
    uniforms = { ...uniforms, envMap: { value: ctx.scene.environment } }
  }

  const mat = new THREE.ShaderMaterial({
    vertexShader: vs,
    fragmentShader: fs,
    uniforms,
    glslVersion: THREE.GLSL3,
    side: shaderSide(a),
  })
  if (Boolean(a.wireframe)) mat.wireframe = true

  const target = String(a.target_name ?? '').trim()
  if (target) {
    const obj = ctx.registry.get(target)
    if (!obj) return { ok: false, message: `Unknown object "${target}".` }
    let foundMesh: THREE.Mesh | undefined
    obj.traverse((c) => {
      if (c instanceof THREE.Mesh && !foundMesh) foundMesh = c
    })
    if (!foundMesh) return { ok: false, message: `No mesh found under "${target}".` }
    const old = foundMesh.material
    if (Array.isArray(old)) old.forEach((m) => m.dispose())
    else old?.dispose()
    foundMesh.material = mat
    if (mat.uniforms.uTime !== undefined && a.animate_u_time !== false) {
      ctx.frameFns.push((_dt, t) => {
        const u = mat.uniforms.uTime
        if (u) u.value = t
      })
    }
    return { ok: true, message: `Shader material applied to "${target}".` }
  }

  const geo = makeGeometryFromToolArgs(a)
  const mesh = new THREE.Mesh(geo, mat)
  const pos = (a.position as number[]) ?? [0, 1.2, 0]
  mesh.position.set(pos[0], pos[1], pos[2])
  const rot = a.rotation as number[] | undefined
  if (rot) mesh.rotation.set(rot[0], rot[1], rot[2])
  const sc = a.scale
  if (sc !== undefined) {
    if (Array.isArray(sc)) mesh.scale.set(sc[0], sc[1], sc[2])
    else mesh.scale.setScalar(Number(sc))
  }
  mesh.castShadow = Boolean(a.cast_shadow ?? true)
  mesh.receiveShadow = true
  track(ctx, String(a.name ?? 'shaderMesh'), mesh)

  if (mat.uniforms.uTime !== undefined && a.animate_u_time !== false) {
    ctx.frameFns.push((_dt, t) => {
      const u = mat.uniforms.uTime
      if (u) u.value = t
    })
  }

  return { ok: true, message: `Mesh "${String(a.name ?? 'shaderMesh')}" created with ShaderMaterial (GLSL3).` }
}

// ── Tool registry ─────────────────────────────────────────────────────────────

export interface SceneTool {
  schema: object
  execute: (ctx: ToolContext, args: Record<string, unknown>) => ToolResult
}

const GEO_ENUM = ['sphere','box','cylinder','torus','torus_knot','cone','octahedron','icosahedron','tetrahedron','plane','ring']
const VEC3 = { type: 'array', items: { type: 'number' }, minItems: 3, maxItems: 3 }

export const SCENE_TOOLS: Record<string, SceneTool> = {
  add_mesh: {
    execute: execAddMesh,
    schema: {
      type: 'function',
      function: {
        name: 'add_mesh',
        description: 'Add a PBR 3D mesh. Use transmission>0 for glass. Use emissive for glowing objects.',
        parameters: {
          type: 'object',
          required: ['name','geometry','position'],
          properties: {
            name:              { type: 'string' },
            geometry:          { type: 'string', enum: GEO_ENUM },
            geometry_params:   { type: 'array', items: { type: 'number' }, description: 'Passed to THREE geometry constructor in order' },
            color:             { type: 'string', description: 'CSS hex color e.g. #ff8844' },
            metalness:         { type: 'number', minimum: 0, maximum: 1 },
            roughness:         { type: 'number', minimum: 0, maximum: 1 },
            transmission:      { type: 'number', minimum: 0, maximum: 1, description: '>0 = glass/transparent' },
            thickness:         { type: 'number' },
            emissive:          { type: 'string' },
            emissive_intensity:{ type: 'number' },
            env_map_intensity: { type: 'number' },
            wireframe:         { type: 'boolean' },
            position:          { ...VEC3 },
            rotation:          { ...VEC3, description: 'Euler XYZ radians' },
            scale:             { description: 'Uniform number or [x,y,z]' },
            cast_shadow:       { type: 'boolean' },
          },
        },
      },
    },
  },
  add_procedural_mesh: {
    execute: execAddProceduralMesh,
    schema: {
      type: 'function',
      function: {
        name: 'add_procedural_mesh',
        description:
          'Advanced procedural geometry (seeded). terrain=domain-warped FBM + hydraulic/thermal erosion + slope-based vertex colors; noise_sphere=3D ridged multi-fractal + domain warping; scatter=Poisson-disk blue-noise + InstancedMesh; crystal_cluster=hex prisms + Voronoi competition; rock_field=3D-noise icosahedra + Laplacian erosion + Poisson placement.',
        parameters: {
          type: 'object',
          required: ['name', 'preset', 'position'],
          properties: {
            name: { type: 'string' },
            preset: {
              type: 'string',
              enum: ['terrain', 'noise_sphere', 'scatter', 'crystal_cluster', 'rock_field'],
              description:
                'terrain=eroded heightmap; noise_sphere=ridged organic blob; scatter=Poisson-distributed meshes; crystal_cluster=hex prism growth; rock_field=eroded rocks',
            },
            seed: { type: 'integer', description: 'Random seed for reproducible variation' },
            position: { ...VEC3 },
            rotation: { ...VEC3, description: 'Euler XYZ radians' },
            scale: { description: 'Uniform number or [x,y,z]' },
            color: { type: 'string' },
            metalness: { type: 'number', minimum: 0, maximum: 1 },
            roughness: { type: 'number', minimum: 0, maximum: 1 },
            wireframe: { type: 'boolean' },
            cast_shadow: { type: 'boolean' },
            // Terrain params
            terrain_width: { type: 'number' },
            terrain_depth: { type: 'number' },
            terrain_segments_x: { type: 'integer', description: 'Grid resolution X (max 256, default 64)' },
            terrain_segments_z: { type: 'integer', description: 'Grid resolution Z (max 256, default 64)' },
            height_scale: { type: 'number', description: 'Max vertical displacement' },
            // Advanced noise params (terrain + noise_sphere)
            fbm_octaves: { type: 'integer', description: 'Noise octaves 1-8 (default 6 terrain, 5 sphere)' },
            fbm_lacunarity: { type: 'number', description: 'Frequency multiplier per octave (default 2.0)' },
            fbm_gain: { type: 'number', description: 'Amplitude multiplier per octave (default 0.5)' },
            warp_strength: { type: 'number', description: 'Domain warping 0-2 (default 0.4 terrain, 0.5 sphere)' },
            ridged: { type: 'boolean', description: 'Use ridged multi-fractal for sharp ridges/veins' },
            // Erosion params (terrain)
            erosion_enabled: { type: 'boolean', description: 'Run hydraulic erosion simulation (default true)' },
            erosion_iterations: { type: 'integer', description: 'Erosion droplet count (default 30000, max 200000)' },
            thermal_iterations: { type: 'integer', description: 'Thermal weathering passes (0=disabled)' },
            talus_angle: { type: 'number', description: 'Talus angle in radians for thermal erosion (default 0.65)' },
            // Noise sphere params
            radius: { type: 'number', description: 'Base radius for noise_sphere' },
            width_segments: { type: 'integer' },
            height_segments: { type: 'integer' },
            displacement: { type: 'number', description: 'Surface noise strength for noise_sphere' },
            // Scatter params
            scatter_geometry: {
              type: 'string',
              enum: ['sphere', 'box', 'cone', 'tetrahedron', 'octahedron'],
            },
            instance_count: { type: 'integer', description: 'Number of scattered meshes' },
            spread: { type: 'number', description: 'Placement radius for scatter / crystals / rocks' },
            size_min: { type: 'number' },
            size_max: { type: 'number' },
            crystal_count: { type: 'integer' },
            rock_count: { type: 'integer' },
            emissive: { type: 'string' },
            emissive_intensity: { type: 'number' },
            transmission: { type: 'number' },
            thickness: { type: 'number' },
            env_map_intensity: { type: 'number' },
          },
        },
      },
    },
  },
  add_light: {
    execute: execAddLight,
    schema: {
      type: 'function',
      function: {
        name: 'add_light',
        description: 'Add a light source. Types: point, spot, directional, ambient, hemisphere.',
        parameters: {
          type: 'object',
          required: ['name','type'],
          properties: {
            name:        { type: 'string' },
            type:        { type: 'string', enum: ['point','spot','directional','ambient','hemisphere'] },
            color:       { type: 'string' },
            intensity:   { type: 'number' },
            position:    { ...VEC3 },
            distance:    { type: 'number' },
            cast_shadow: { type: 'boolean' },
          },
        },
      },
    },
  },
  add_cloud: {
    execute: execAddCloud,
    schema: {
      type: 'function',
      function: {
        name: 'add_cloud',
        description:
          'Add a raymarched volumetric cloud (ellipsoid). Keep density 0.6–1.2 and coverage 0.35–0.55 for wispy clouds; higher values look solid and blow out with bloom. Stage floor top is near y≈-1.35 — place ground props above that.',
        parameters: {
          type: 'object',
          required: ['name','position','size'],
          properties: {
            name:      { type: 'string' },
            position:  { ...VEC3 },
            size:      { ...VEC3, description: '[width, height, depth]' },
            color:     { type: 'string', description: 'Cloud tint hex' },
            density:   { type: 'number', description: '0.5–3' },
            coverage:  { type: 'number', description: '0=clear, 1=overcast' },
            sun_dir:   { ...VEC3 },
            sun_color: { type: 'string' },
          },
        },
      },
    },
  },
  add_particles: {
    execute: execAddParticles,
    schema: {
      type: 'function',
      function: {
        name: 'add_particles',
        description: 'Add a vertex-coloured particle point cloud. Good for nebulas, dust, sparks.',
        parameters: {
          type: 'object',
          required: ['name','count'],
          properties: {
            name:             { type: 'string' },
            count:            { type: 'integer', description: 'Max 80000' },
            spread:           { type: 'number' },
            center:           { ...VEC3 },
            color_hue_range:  { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2, description: '[minHue, maxHue] 0-1' },
            size:             { type: 'number', description: 'Point size in world units' },
            opacity:          { type: 'number', minimum: 0, maximum: 1 },
            flatten_y:        { type: 'number', description: 'Y-axis flattening factor, 1=sphere, 0.2=disc' },
          },
        },
      },
    },
  },
  add_curl_swarm: {
    execute: execAddCurlSwarm,
    schema: {
      type: 'function',
      function: {
        name: 'add_curl_swarm',
        description: 'Add an advanced GPU-based Curl Noise Particle Swarm. Uses analytical curl noise in the vertex shader to advect instanced geometry for fluid-like swirling motions.',
        parameters: {
          type: 'object',
          properties: {
            name:       { type: 'string' },
            count:      { type: 'number', description: 'Number of particles, e.g. 10000 to 50000' },
            radius:     { type: 'number', description: 'Spawn radius' },
            color1:     { type: 'string', description: 'CSS hex color 1' },
            color2:     { type: 'string', description: 'CSS hex color 2' },
            size_min:   { type: 'number' },
            size_max:   { type: 'number' },
            speed:      { type: 'number', description: 'Speed of noise evolution' },
            scale_noise:{ type: 'number', description: 'Scale of the curl noise' },
            swirl:      { type: 'number', description: 'Intensity of the curl displacement' },
            position:   { ...VEC3 },
            rotation:   { ...VEC3 },
            scale:      { type: ['number', 'array'], items: { type: 'number' } },
          },
        },
      },
    },
  },
  animate_object: {
    execute: execAnimate,
    schema: {
      type: 'function',
      function: {
        name: 'animate_object',
        description: 'Attach a continuous animation to a named object (rotate, bob, orbit, pulse).',
        parameters: {
          type: 'object',
          required: ['name'],
          properties: {
            name:           { type: 'string' },
            rotate_speed:   { ...VEC3, description: 'Radians/sec for each axis [x,y,z]' },
            bob_amplitude:  { type: 'number', description: 'Y bob height in world units' },
            bob_speed:      { type: 'number' },
            orbit_radius:   { type: 'number' },
            orbit_speed:    { type: 'number', description: 'Radians/sec' },
            orbit_axis:     { type: 'string', enum: ['x','y','z'] },
            pulse_scale:    { type: 'number', description: 'Scale oscillation amplitude' },
            pulse_speed:    { type: 'number' },
          },
        },
      },
    },
  },
  set_camera: {
    execute: execSetCamera,
    schema: {
      type: 'function',
      function: {
        name: 'set_camera',
        description: 'Move and aim the camera, optionally with a smooth animated transition.',
        parameters: {
          type: 'object',
          required: ['position','look_at'],
          properties: {
            position:          { ...VEC3 },
            look_at:           { ...VEC3 },
            fov:               { type: 'number', description: 'Field of view in degrees' },
            animate_duration:  { type: 'number', description: 'Transition duration in seconds (0=instant)' },
          },
        },
      },
    },
  },
  set_environment: {
    execute: execSetEnvironment,
    schema: {
      type: 'function',
      function: {
        name: 'set_environment',
        description: 'Replace the HDR environment map with a physically-based lighting setup.',
        parameters: {
          type: 'object',
          required: ['lights'],
          properties: {
            skyZenith:  { ...VEC3, description: 'Linear HDR zenith sky RGB, values ~0..0.08' },
            skyHorizon: { ...VEC3 },
            ground:     { ...VEC3, description: 'Ground bounce colour' },
            shimmer:    { type: 'number', minimum: 0, maximum: 1 },
            lights: {
              type: 'array',
              items: {
                type: 'object',
                properties: {
                  direction:  { ...VEC3 },
                  color:      { ...VEC3, description: 'Linear HDR RGB, key light ~[3,2.5,1.5]' },
                  sharpness:  { type: 'number', description: 'Lobe power: 1=broad, 18=sun' },
                  intensity:  { type: 'number' },
                },
              },
            },
          },
        },
      },
    },
  },
  set_background: {
    execute: execSetBackground,
    schema: {
      type: 'function',
      function: {
        name: 'set_background',
        description: 'Set the scene background color.',
        parameters: {
          type: 'object',
          required: ['color'],
          properties: {
            color: { type: 'string', description: 'CSS hex color e.g. #040506' },
          },
        },
      },
    },
  },
  exec_threejs_code: {
    execute: execExecThreejsCode,
    schema: {
      type: 'function',
      function: {
        name: 'exec_threejs_code',
        description:
          'Execute a JavaScript snippet with THREE and api (sandboxed). Body only — no import/export. api.add(obj, name?), api.remove(obj), api.get(name), api.cloud(opts), api.stepCloud, api.texture(w,h,fn), api.env(config), api.onFrame(fn), api.setBackground(hex|"#rrggbb"). Use for custom graphs, curves, Groups, BufferGeometry, anything not covered by other tools.',
        parameters: {
          type: 'object',
          required: ['code'],
          properties: {
            code: {
              type: 'string',
              description:
                'JavaScript statements. Runs as new Function("THREE","api", code). Max ~100k chars.',
            },
          },
        },
      },
    },
  },
  gen_shader_code: {
    execute: execGenShaderCode,
    schema: {
      type: 'function',
      function: {
        name: 'gen_shader_code',
        description:
          'Apply WebGL2 GLSL ES 3.0 shaders (THREE.GLSL3): vertex must use in/out; fragment must declare `out vec4 fragColor` and use in vars matching vertex outs. Optional target_name to replace material on existing mesh; otherwise creates new mesh (geometry + name + position). Uniforms JSON: { uTime: { type: "float", value: 0 } }. If use_scene_environment true, adds envMap from scene. uTime auto-animates with scene time when animate_u_time is true. NEVER redeclare built-in Three.js vertex attributes or uniforms (position, uv, normal, modelViewMatrix, projectionMatrix, modelMatrix, viewMatrix, normalMatrix, cameraPosition) — they are already provided and redeclaring them causes a shader compile error.',
        parameters: {
          type: 'object',
          required: ['vertex_shader', 'fragment_shader'],
          properties: {
            vertex_shader: { type: 'string' },
            fragment_shader: { type: 'string' },
            uniforms: {
              type: 'object',
              description:
                'Map of uniform name to { type: float|int|vec2|vec3|vec4|color, value: number or array or hex string }',
            },
            target_name: {
              type: 'string',
              description: 'Registered object name whose first Mesh gets this material',
            },
            name: { type: 'string', description: 'Registry name when creating a new mesh' },
            geometry: { type: 'string', enum: GEO_ENUM },
            geometry_params: { type: 'array', items: { type: 'number' } },
            position: { ...VEC3 },
            rotation: { ...VEC3 },
            scale: { description: 'number or [x,y,z]' },
            use_scene_environment: { type: 'boolean' },
            animate_u_time: { type: 'boolean', description: 'If uniforms include uTime, drive it from scene clock (default true)' },
            side: { type: 'string', enum: ['front', 'back', 'double'] },
            wireframe: { type: 'boolean' },
            cast_shadow: { type: 'boolean' },
          },
        },
      },
    },
  },
  list_objects: {
    execute: execListObjects,
    schema: {
      type: 'function',
      function: {
        name: 'list_objects',
        description: 'List all named objects currently in the generated scene.',
        parameters: { type: 'object', properties: {} },
      },
    },
  },
  remove_object: {
    execute: execRemoveObject,
    schema: {
      type: 'function',
      function: {
        name: 'remove_object',
        description: 'Remove a named object from the scene.',
        parameters: {
          type: 'object',
          required: ['name'],
          properties: {
            name: { type: 'string' },
          },
        },
      },
    },
  },
  gen_texture: {
    execute: execGenTexture,
    schema: {
      type: 'function',
      function: {
        name: 'gen_texture',
        description:
          'Generate a physically-based GPU procedural texture (WebGL2 GLSL ES 3.0) and apply it to a named mesh or create a new display plane. ' +
          'domain="cellular": presets voronoi_membrane|reaction_diffusion|cytoskeleton|mitochondria. ' +
          'domain="material": presets crystal_lattice|thin_film|grain_boundary|dislocation_field. ' +
          'domain="surface": HYPERREAL PBR materials — auto-generates albedo + analytical normal map + ORM (AO/Roughness/Metalness) maps from a single preset. ' +
          'Presets: weathered_metal (params: scale,scratch_density,grime) | marble (scale,vein_intensity,vein_freq,color_temp) | rough_stone (scale,crack_depth,weathering,mineral_variation) | aged_wood (scale,ring_freq,grain_strength,age) | rust_iron (scale,corrosion,pitting,clean_patches) | cracked_earth (scale,crack_width,dryness,dust_color) | concrete (scale,aggregate,crack_density,staining) | lava (scale,crack_glow,coolness,flow_speed; also generates emissive map). ' +
          'Surface textures use gradient noise with analytical derivatives for perfect normal maps, multi-layer composition (scratches+pitting+grime, veins+zones, etc.), and Voronoi crack networks. Default bake=true (zero per-frame cost). ' +
          'Set animate=true to re-render the texture every frame with uTime. Use map_slot to choose which material channel to fill (surface domain auto-applies all PBR maps regardless).',
        parameters: {
          type: 'object',
          required: ['domain', 'preset'],
          properties: {
            domain:      { type: 'string', enum: ['cellular', 'material', 'surface'], description: 'Texture domain. Use "surface" for hyperreal PBR materials (auto-generates normal+roughness+AO+metalness maps).' },
            preset:      { type: 'string', description: 'Preset name — see tool description for full list.' },
            resolution:  { type: 'number', description: '64, 128, 256, 512 (default), 1024, 2048, or 4096. Higher resolutions with bake=true have zero per-frame cost.' },
            bake:        { type: 'boolean', description: 'Render once at full resolution and never re-render (zero per-frame cost). Enables mipmaps. Best for static or slow-changing textures at 1024-4096px. Default false.' },
            params:      { type: 'object', description: 'Preset-specific numeric parameters as a flat object.' },
            animate:     { type: 'boolean', description: 'Re-render texture every frame with uTime. Default true.' },
            target_name: { type: 'string', description: 'Apply texture to this named mesh instead of creating a new plane.' },
            map_slot:    { type: 'string', enum: ['map','emissiveMap','normalMap','alphaMap','roughnessMap'], description: 'Material slot. Default: map.' },
            name:        { type: 'string', description: 'Name of the new display plane (when no target_name).' },
            position:    { type: 'array', items: { type: 'number' }, description: '[x,y,z] position of new plane.' },
          },
        },
      },
    },
  },
  generate_tree: {
    execute: execGenerateTree,
    schema: {
      type: 'function',
      function: {
        name: 'generate_tree',
        description:
          'Generate a botanically-accurate procedural tree using space-colonization growth, da Vinci pipe-model taper, Catmull-Rom spline branches, golden-angle phyllotactic leaf placement, and InstancedMesh foliage. ' +
          'Much more realistic than scatter-of-cones. Each seed gives a unique tree. Stage floor is y≈-1.35.',
        parameters: {
          type: 'object',
          required: ['name', 'position'],
          properties: {
            name:                { type: 'string' },
            position:            { ...VEC3, description: 'Base of the trunk [x,y,z]. Floor is y≈-1.35.' },
            seed:                { type: 'integer', description: 'Random seed for reproducible shape.' },
            trunk_height:        { type: 'number', description: 'Trunk height before crown (default 2.5).' },
            trunk_radius:        { type: 'number', description: 'Base trunk radius (default 0.12).' },
            crown_size:          { ...VEC3, description: 'Ellipsoidal crown [rx, ry, rz] (default [1.8, 2, 1.8]).' },
            crown_offset:        { ...VEC3, description: 'Shift crown center from trunk top.' },
            attractor_count:     { type: 'integer', description: 'Space-colonization attractors (default 800). More = denser crown.' },
            branch_step:         { type: 'number', description: 'Branch growth step size (default 0.18). Smaller = smoother.' },
            max_iterations:      { type: 'integer', description: 'Growth iterations (default 120).' },
            pipe_exponent:       { type: 'number', description: 'da Vinci exponent (default 2.2). 2=classical, higher=thinner sub-branches.' },
            radial_segments:     { type: 'integer', description: 'Tube cross-section segments (default 8).' },
            leaf_density:        { type: 'number', description: 'Leaves per terminal node (default 4).' },
            leaf_size:           { type: 'number', description: 'Individual leaf size (default 0.12).' },
            bark_color:          { type: 'string', description: 'Hex color for bark (default #5c3a1e).' },
            leaf_color:          { type: 'string', description: 'Hex color for leaves (default #3a7d2e).' },
            leaf_color_variation:{ type: 'number', description: 'Hue variation across leaves 0-0.3 (default 0.08).' },
            wind_strength:       { type: 'number', description: '0=static, 0.4=gentle breeze (default 0.4).' },
            tropism:             { ...VEC3, description: 'Growth bias [x,y,z]. Positive y=phototropism (default [0,0.12,0]).' },
            scale:               { description: 'Uniform number or [x,y,z].' },
            rotation:            { ...VEC3, description: 'Euler XYZ radians.' },
          },
        },
      },
    },
  },
  tessellate_mesh: {
    execute: execTessellateMesh,
    schema: {
      type: 'function',
      function: {
        name: 'tessellate_mesh',
        description:
          'Subdivide a named mesh into higher resolution geometry using edge-midpoint tessellation. ' +
          'Each level quadruples the triangle count. Preserves UVs, vertex colors, and shared edges. ' +
          'Useful before applying displacement maps, procedural deformation, or for smoother silhouettes.',
        parameters: {
          type: 'object',
          required: ['name'],
          properties: {
            name: { type: 'string', description: 'Name of the mesh to tessellate.' },
            levels: { type: 'number', description: 'Subdivision levels (1-4, default 1). Each level ×4 triangles.' },
          },
        },
      },
    },
  },
  add_water: {
    execute: execAddWater,
    schema: {
      type: 'function',
      function: {
        name: 'add_water',
        description:
          'Add a 3D ocean mesh with real Gerstner wave vertex displacement. ' +
          'Vertices move every frame — sharp crests, wide troughs, foam from Jacobian whitecaps. ' +
          'MeshPhysicalMaterial with transmission/IOR 1.33 for translucent water. Default y=-1.35 (stage floor). ' +
          'Tune wave_scale (amplitude), choppiness (horizontal pull 0-1.5), wave_count (complexity 3-32), resolution (mesh detail 32-512).',
        parameters: {
          type: 'object',
          properties: {
            name:        { type: 'string', description: 'Name for the water object (default "water").' },
            size:        { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2, description: 'World [width, depth] (default [40,40]).' },
            position:    { ...VEC3, description: 'World position [x,y,z] (default [0,-1.35,0]).' },
            resolution:  { type: 'number', description: 'Mesh grid resolution per axis (default 128, max 512). Higher = smoother waves but more CPU.' },
            color:       { type: 'string', description: 'Water body color hex (default "#006994").' },
            wave_scale:  { type: 'number', description: 'Wave amplitude (default 0.4). Higher = taller waves.' },
            choppiness:  { type: 'number', description: 'Gerstner horizontal displacement factor 0-1.5 (default 0.8). Higher = sharper crests.' },
            wave_count:  { type: 'number', description: 'Number of overlapping wave components 3-32 (default 12). More = richer detail.' },
            speed:       { type: 'number', description: 'Wave animation speed multiplier (default 1.0).' },
            foam:        { type: 'boolean', description: 'Enable Jacobian-based foam/whitecap vertex colors (default true).' },
            seed:        { type: 'number', description: 'Random seed for wave directions (default 42).' },
            opacity:     { type: 'number', description: 'Water opacity 0-1 (default 0.85).' },
          },
        },
      },
    },
  },
  add_terrain: {
    execute: execAddTerrain,
    schema: {
      type: 'function',
      function: {
        name: 'add_terrain',
        description:
          'Add a 3D landscape mesh with procedural heightmap and multi-texture material. ' +
          'Uses multi-octave gradient noise + ridged noise for realistic terrain. ' +
          'Custom shader blends 4 photo textures (grass/rock/dirt/snow) based on height band and surface steepness. ' +
          'Biome presets: mountains, rolling_hills, desert, arctic, volcanic, canyon, plateau. ' +
          'Auto-removes previous terrain when called again. Default y=-2.',
        parameters: {
          type: 'object',
          properties: {
            name:           { type: 'string', description: 'Name for the terrain object (default "terrain").' },
            size:           { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2, description: 'World [width, depth] (default [40,40]).' },
            position:       { ...VEC3, description: 'World position [x,y,z] (default [0,-2,0]).' },
            resolution:     { type: 'number', description: 'Mesh grid resolution per axis (default 128, max 512).' },
            biome:          { type: 'string', description: 'Biome preset: mountains, rolling_hills, desert, arctic, volcanic, canyon, plateau (default "mountains").' },
            height_scale:   { type: 'number', description: 'Vertical height multiplier (default depends on biome, ~6-10 for mountains).' },
            seed:           { type: 'number', description: 'Random seed for heightmap (default 42).' },
            octaves:        { type: 'number', description: 'Noise octaves 1-10 (default ~7). More = finer detail.' },
            persistence:    { type: 'number', description: 'Noise persistence/gain 0-1 (default ~0.48). Higher = rougher.' },
            ridge_fraction: { type: 'number', description: 'Blend between smooth FBM and ridged noise 0-1 (default ~0.4). Higher = sharper ridges/peaks.' },
            snow_line:      { type: 'number', description: 'Normalized height 0-1 where snow starts (default ~0.75).' },
            tree_line:      { type: 'number', description: 'Normalized height 0-1 where grass→rock transition (default ~0.45).' },
            tex_scale:      { type: 'number', description: 'Texture tiling density (default 0.25). Higher = finer texture repeat.' },
          },
        },
      },
    },
  },
  add_sky: {
    execute: execAddSky,
    schema: {
      type: 'function',
      function: {
        name: 'add_sky',
        description:
          'Add a physically-based atmospheric sky using Preetham scattering model. ' +
          'Creates a sky dome, positions the sun, and generates an environment map for reflections on all objects. ' +
          'Also sets the scene background to the sky. Auto-removes previous sky. ' +
          'Use sun_elevation (0°=horizon, 90°=zenith) and sun_azimuth (0°=north, 180°=south) to position the sun. ' +
          'For golden hour use elevation ~5-15. For noon use ~60-90. For sunset use ~1-5.',
        parameters: {
          type: 'object',
          properties: {
            name:              { type: 'string', description: 'Name for the sky object (default "sky").' },
            sun_elevation:     { type: 'number', description: 'Sun elevation angle in degrees (default 45). 0=horizon, 90=directly overhead.' },
            sun_azimuth:       { type: 'number', description: 'Sun azimuth angle in degrees (default 180). 0=north, 90=east, 180=south, 270=west.' },
            turbidity:         { type: 'number', description: 'Atmospheric turbidity 1-20 (default 2). Higher = hazier, more yellow/orange sky.' },
            rayleigh:          { type: 'number', description: 'Rayleigh scattering coefficient (default 1). Higher = bluer sky.' },
            mie_coefficient:   { type: 'number', description: 'Mie scattering coefficient (default 0.005). Higher = more glow around sun.' },
            mie_directional_g: { type: 'number', description: 'Mie scattering directionality 0-1 (default 0.8). Higher = tighter sun glow.' },
            scale:             { type: 'number', description: 'Sky dome scale (default 1000).' },
          },
        },
      },
    },
  },
  reset_scene: {
    execute: execResetScene,
    schema: {
      type: 'function',
      function: {
        name: 'reset_scene',
        description:
          'Remove all objects added by the scene agent (generated meshes, lights, clouds, etc.), clear animations, and empty the name registry.',
        parameters: { type: 'object', properties: {} },
      },
    },
  },
}

// ── Dynamic tool registry ───────────────────────────────────────────────────

/** Dynamically-registered tools created at runtime via create_tool. */
const DYNAMIC_TOOLS: Record<string, { execute: (ctx: ToolContext, a: Record<string, unknown>) => ToolResult; schema: object }> = {}

/**
 * create_tool executor.
 * The agent provides:
 *   - tool_name: snake_case identifier
 *   - description: what the tool does
 *   - parameters_schema: JSON Schema object for the parameters
 *   - body: JS function body string. Receives (ctx, args, THREE) and must return { ok, message }.
 *           ctx is ToolContext, args is the parsed arguments object, THREE is the three.js namespace.
 */
function execCreateTool(_ctx: ToolContext, a: Record<string, unknown>): ToolResult {
  const toolName = String(a.tool_name ?? '').replace(/[^a-z0-9_]/gi, '_').toLowerCase()
  if (!toolName) return { ok: false, message: 'tool_name is required' }
  if (toolName in SCENE_TOOLS) return { ok: false, message: `Cannot override built-in tool: ${toolName}` }

  const body = String(a.body ?? '')
  if (!body) return { ok: false, message: 'body is required' }

  const description = String(a.description ?? `Dynamic tool: ${toolName}`)
  const paramsSchema = (a.parameters_schema as object | undefined) ?? { type: 'object', properties: {} }

  // fn receives (ctx, args, THREE, track) where track(ctx, name, obj) registers+adds an object
  type TrackFn = (ctx: ToolContext, name: string, obj: THREE.Object3D) => void
  type DynFn = (ctx: ToolContext, args: Record<string, unknown>, THREE: typeof import('three'), track: TrackFn) => ToolResult
  let fn: DynFn
  try {
    // eslint-disable-next-line no-new-func
    fn = new Function('ctx', 'args', 'THREE', 'track', body) as DynFn
  } catch (e) {
    return { ok: false, message: `Syntax error in tool body: ${String(e)}` }
  }

  DYNAMIC_TOOLS[toolName] = {
    execute: (c, a2) => {
      try {
        return fn(c, a2, THREE, track)
      } catch (e) {
        return { ok: false, message: `Runtime error in ${toolName}: ${String(e)}` }
      }
    },
    schema: {
      type: 'function',
      function: {
        name: toolName,
        description,
        parameters: paramsSchema,
      },
    },
  }

  return { ok: true, message: `Tool "${toolName}" registered. You can now call it in subsequent turns.` }
}

/** Return all tool schemas for the OpenAI tools array (built-ins + dynamic). */
export function getToolSchemas(): object[] {
  return [
    ...Object.values(SCENE_TOOLS).map((t) => t.schema),
    ...Object.values(DYNAMIC_TOOLS).map((t) => t.schema),
    CREATE_TOOL_ENTRY.schema,
  ]
}

/** Dispatch a tool call by name (built-ins, dynamic, and create_tool itself). */
export function dispatchTool(
  name: string,
  args: Record<string, unknown>,
  ctx: ToolContext
): ToolResult {
  if (name === 'create_tool') return execCreateTool(ctx, args)
  const tool = SCENE_TOOLS[name] ?? DYNAMIC_TOOLS[name]
  if (!tool) return { ok: false, message: `Unknown tool: ${name}` }
  try {
    return tool.execute(ctx, args)
  } catch (e) {
    return { ok: false, message: `Tool error: ${String(e)}` }
  }
}

/** Schema entry for create_tool itself (kept separate so it's always last). */
const CREATE_TOOL_ENTRY = {
  schema: {
    type: 'function',
    function: {
      name: 'create_tool',
      description:
        'Define and register a brand-new tool at runtime. After registration the tool is immediately available for you to call in subsequent rounds. Use this to build reusable helpers (e.g. spawn_asteroid, add_neon_ring) that you can call many times with different args.',
      parameters: {
        type: 'object',
        required: ['tool_name', 'description', 'body'],
        properties: {
          tool_name: {
            type: 'string',
            description: 'snake_case name for the new tool (e.g. "spawn_comet"). Must be unique.',
          },
          description: {
            type: 'string',
            description: 'What this tool does — shown to the model in future rounds.',
          },
          parameters_schema: {
            type: 'object',
            description:
              'JSON Schema object describing the parameters the new tool accepts. E.g. { "type": "object", "properties": { "count": { "type": "number" } } }',
          },
          body: {
            type: 'string',
            description:
              'JavaScript function body (NOT an arrow function — just the statements). Receives three arguments: ctx (ToolContext with .scene, .camera, .renderer, .registry, .frameFns, .added), args (parsed parameter object), THREE (the three.js namespace), track (function(ctx,name,obj) to register+add objects). Must return { ok: boolean, message: string }.',
          },
        },
      },
    },
  },
}
