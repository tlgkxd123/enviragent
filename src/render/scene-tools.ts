import * as THREE from 'three'
import type { Scene, PerspectiveCamera, WebGLRenderer } from 'three'
import { createVolumetricCloud, stepCloud as _stepCloud } from './volumetric-cloud'
import { buildEnvFromConfig } from './env-codegen'
import type { EnvConfig } from './env-codegen'

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

// ── Executors ────────────────────────────────────────────────────────────────

function execAddMesh(ctx: ToolContext, a: Record<string, unknown>): ToolResult {
  const p = (a.geometry_params as number[]) ?? []
  let geo: THREE.BufferGeometry
  switch (String(a.geometry ?? 'sphere').toLowerCase()) {
    case 'sphere':      geo = new THREE.SphereGeometry(p[0]??1, p[1]??48, p[2]??48); break
    case 'box':         geo = new THREE.BoxGeometry(p[0]??1, p[1]??1, p[2]??1); break
    case 'cylinder':    geo = new THREE.CylinderGeometry(p[0]??0.5, p[1]??0.5, p[2]??1, p[3]??32); break
    case 'torus':       geo = new THREE.TorusGeometry(p[0]??1, p[1]??0.3, p[2]??16, p[3]??100); break
    case 'torus_knot':  geo = new THREE.TorusKnotGeometry(p[0]??0.8, p[1]??0.25, p[2]??128, p[3]??16); break
    case 'cone':        geo = new THREE.ConeGeometry(p[0]??0.5, p[1]??1, p[2]??32); break
    case 'octahedron':  geo = new THREE.OctahedronGeometry(p[0]??1, Math.min(p[1]??0, 5)); break
    case 'icosahedron': geo = new THREE.IcosahedronGeometry(p[0]??1, Math.min(p[1]??0, 5)); break
    case 'tetrahedron': geo = new THREE.TetrahedronGeometry(p[0]??1, p[1]??0); break
    case 'plane':       geo = new THREE.PlaneGeometry(p[0]??2, p[1]??2, p[2]??1, p[3]??1); break
    case 'ring':        geo = new THREE.RingGeometry(p[0]??0.5, p[1]??1, p[2]??32); break
    default:            geo = new THREE.SphereGeometry(1, 32, 32)
  }
  const opts: THREE.MeshPhysicalMaterialParameters = {
    color:     new THREE.Color(String(a.color ?? '#aaaaff')),
    metalness: Number(a.metalness ?? 0.1),
    roughness: Number(a.roughness ?? 0.4),
    wireframe: Boolean(a.wireframe ?? false),
    envMapIntensity: Number(a.env_map_intensity ?? 1.0),
  }
  if (Number(a.transmission ?? 0) > 0) {
    opts.transmission = Number(a.transmission)
    opts.thickness    = Number(a.thickness ?? 1.0)
  }
  if (a.emissive) {
    opts.emissive          = new THREE.Color(String(a.emissive))
    opts.emissiveIntensity = Number(a.emissive_intensity ?? 1.0)
  }
  const mat  = new THREE.MeshPhysicalMaterial(opts)
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
    density:   Number(a.density  ?? 1.2),
    coverage:  Number(a.coverage ?? 0.55),
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
        description: 'Add a raymarched volumetric cloud / fog volume. Auto-animates drift.',
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
}

/** Return all tool schemas for the OpenAI tools array. */
export function getToolSchemas(): object[] {
  return Object.values(SCENE_TOOLS).map((t) => t.schema)
}

/** Dispatch a tool call by name. */
export function dispatchTool(
  name: string,
  args: Record<string, unknown>,
  ctx: ToolContext
): ToolResult {
  const tool = SCENE_TOOLS[name]
  if (!tool) return { ok: false, message: `Unknown tool: ${name}` }
  try {
    return tool.execute(ctx, args)
  } catch (e) {
    return { ok: false, message: `Tool error: ${String(e)}` }
  }
}
