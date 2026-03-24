import * as THREE from 'three'
import type { Scene, PerspectiveCamera, WebGLRenderer } from 'three'
import { createVolumetricCloud, stepCloud } from './volumetric-cloud'
import { buildEnvFromConfig } from './env-codegen'
import type { EnvConfig } from './env-codegen'

/**
 * LLM-driven scene code generator.
 *
 * The model receives a system prompt describing the full Three.js + scene API
 * available, then returns a self-contained JS function body.
 * We wrap it in new Function('THREE','api', code) and call it.
 *
 * Generated code can:
 *  - Create any Three.js geometry / material / mesh / group / light
 *  - Create volumetric clouds via api.cloud(opts)
 *  - Create procedural textures via api.texture(w, h, fn)
 *  - Set camera position / target
 *  - Apply a new HDR env via api.env(config)
 *  - Register per-frame callbacks via api.onFrame(fn)
 */

export interface SceneCodegenContext {
  scene: Scene
  camera: PerspectiveCamera
  renderer: WebGLRenderer
  applyEnvMap: (envMap: THREE.Texture) => void
  onLog: (msg: string) => void
  onError: (msg: string) => void
}

export interface GeneratedScene {
  objects: THREE.Object3D[]
  frameFns: Array<(dt: number, t: number) => void>
  dispose: () => void
}

const SYSTEM_PROMPT = `You are a creative Three.js scene programmer for a quantum physics renderer.
Respond with ONLY a JavaScript function body (no function keyword, no markdown fences, no imports).
The code runs inside: new Function('THREE', 'api', <your code>)

Available globals:
  THREE  - full three.js r183 namespace

Available api:
  api.scene      - THREE.Scene
  api.camera     - THREE.PerspectiveCamera
  api.renderer   - THREE.WebGLRenderer
  api.add(obj)   - add Object3D to scene (tracked for cleanup)
  api.remove(obj)
  api.cloud(opts) - raymarched volumetric cloud, returns Mesh (auto-added)
    opts: { position:[x,y,z], size:[w,h,d], color:0xHEX, density:1.2, coverage:0.55, sunDir:[x,y,z], sunColor:0xHEX }
  api.stepCloud(mesh, dt) - call in onFrame to animate cloud drift
  api.texture(w, h, fn)  - procedural DataTexture
    fn(x,y,w,h) returns {r,g,b,a} each 0-255
  api.env(config) - apply new HDR environment
    config: { skyZenith:[r,g,b], skyHorizon:[r,g,b], ground:[r,g,b], lights:[{direction,color,sharpness,intensity}], shimmer:0-1 }
  api.onFrame(fn) - fn(dt, totalTime) called every frame
  api.log(msg)    - show status message
  api.setBackground(hex)
  api.camera.position.set(x,y,z)
  api.camera.lookAt(x,y,z)

Rules:
- Use api.add() for everything. Never scene.add directly.
- For volumetric clouds use api.cloud() only.
- For animations use api.onFrame().
- To step clouds each frame: api.onFrame((dt,t) => api.stepCloud(cloudMesh, dt))
- No import/require/fetch/eval/document/window.
- Max ~200k vertices total.
- Physically plausible PBR values.
- Return nothing.

Example:
  const geo = new THREE.OctahedronGeometry(0.6, 2)
  const mat = new THREE.MeshPhysicalMaterial({color:0x88ccff,transmission:0.92,roughness:0.0,thickness:1.2})
  const mesh = new THREE.Mesh(geo, mat)
  mesh.position.set(0, 1.5, 0)
  api.add(mesh)
  api.onFrame((dt,t) => { mesh.rotation.y += dt*0.6; mesh.position.y = 1.5+Math.sin(t*0.8)*0.18 })
  api.log('Crystal added')`

export async function generateSceneCode(
  description: string,
  apiKey: string,
  model = 'gpt-4o',
  baseUrl = 'https://api.openai.com/v1'
): Promise<string> {
  const res = await fetch(`${baseUrl}/chat/completions`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model,
      temperature: 0.8,
      max_tokens: 2048,
      messages: [
        { role: 'system', content: SYSTEM_PROMPT },
        { role: 'user', content: description },
      ],
    }),
  })

  if (!res.ok) {
    const text = await res.text()
    throw new Error(`API error ${res.status}: ${text}`)
  }

  const json = await res.json() as { choices?: Array<{ message?: { content?: string } }> }
  const content = json.choices?.[0]?.message?.content ?? ''
  return content
    .replace(/^```(?:javascript|js)?\s*/i, '')
    .replace(/\s*```\s*$/m, '')
    .trim()
}

export function executeSceneCode(
  code: string,
  ctx: SceneCodegenContext
): GeneratedScene {
  const addedObjects: THREE.Object3D[] = []
  const frameFns: Array<(dt: number, t: number) => void> = []

  function makeTexture(
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

  const api = {
    scene:    ctx.scene,
    camera:   ctx.camera,
    renderer: ctx.renderer,

    add(obj: THREE.Object3D) {
      ctx.scene.add(obj)
      addedObjects.push(obj)
    },

    remove(obj: THREE.Object3D) {
      ctx.scene.remove(obj)
      const idx = addedObjects.indexOf(obj)
      if (idx !== -1) addedObjects.splice(idx, 1)
    },

    cloud(opts: Parameters<typeof createVolumetricCloud>[0]) {
      const mesh = createVolumetricCloud(opts)
      ctx.scene.add(mesh)
      addedObjects.push(mesh)
      return mesh
    },

    stepCloud,

    texture: makeTexture,

    env(config: EnvConfig) {
      const { envMap } = buildEnvFromConfig(config, ctx.renderer)
      ctx.applyEnvMap(envMap)
    },

    onFrame(fn: (dt: number, t: number) => void) {
      frameFns.push(fn)
    },

    log(msg: unknown) {
      ctx.onLog(String(msg))
    },

    setBackground(hex: number) {
      ctx.scene.background = new THREE.Color(hex)
    },
  }

  try {
    // eslint-disable-next-line no-new-func
    const fn = new Function('THREE', 'api', `"use strict";
${code}`)
    fn(THREE, api)
  } catch (e) {
    ctx.onError(`Execution error: ${String(e)}`)
  }

  function disposeObj(obj: THREE.Object3D) {
    obj.traverse((child) => {
      const c = child as THREE.Mesh
      if (c.isMesh || (child as THREE.Points).isPoints) {
        c.geometry?.dispose()
        const m = c.material
        if (Array.isArray(m)) m.forEach((x) => (x as THREE.Material).dispose())
        else (m as THREE.Material)?.dispose()
      }
    })
  }

  return {
    objects: addedObjects,
    frameFns,
    dispose() {
      for (const obj of [...addedObjects]) {
        ctx.scene.remove(obj)
        disposeObj(obj)
      }
      addedObjects.length = 0
      frameFns.length = 0
    },
  }
}

// Built-in snippets — no API key needed
export const CODE_SNIPPETS: Record<string, string> = {
  'Floating crystal': [
    'const geo = new THREE.OctahedronGeometry(0.7, 2)',
    'const mat = new THREE.MeshPhysicalMaterial({color:0x88ccff,metalness:0.05,roughness:0.0,transmission:0.94,thickness:1.2,envMapIntensity:1.4})',
    'const mesh = new THREE.Mesh(geo, mat)',
    'mesh.position.set(0, 2.2, 0)',
    'api.add(mesh)',
    'api.onFrame((dt,t) => { mesh.rotation.y += dt*0.55; mesh.rotation.x = Math.sin(t*0.3)*0.18; mesh.position.y = 2.2+Math.sin(t*0.8)*0.22 })',
    "api.log('Floating crystal added')",
  ].join('\n'),

  'Volumetric nebula clouds': [
    'const c1 = api.cloud({position:[0,4,0], size:[10,3,10], color:0xc0d8ff, density:1.0, coverage:0.6, sunDir:[0.65,0.55,-0.52], sunColor:0xfff4e0})',
    'const c2 = api.cloud({position:[3,5.5,-2], size:[6,2,6], color:0xffd0e8, density:0.8, coverage:0.45})',
    'const c3 = api.cloud({position:[-4,3.5,2], size:[7,2.5,5], color:0xd0f0ff, density:0.9, coverage:0.5})',
    'api.onFrame((dt) => { api.stepCloud(c1,dt); api.stepCloud(c2,dt); api.stepCloud(c3,dt) })',
    "api.log('Nebula clouds added')",
  ].join('\n'),

  'Particle nebula': [
    'const N = 18000',
    'const positions = new Float32Array(N*3)',
    'const colors = new Float32Array(N*3)',
    'for(let i=0;i<N;i++){',
    '  const r=Math.pow(Math.random(),0.45)*5',
    '  const theta=Math.random()*Math.PI*2',
    '  const phi=Math.acos(2*Math.random()-1)',
    '  positions[i*3]=r*Math.sin(phi)*Math.cos(theta)',
    '  positions[i*3+1]=r*Math.cos(phi)*0.4+1.5',
    '  positions[i*3+2]=r*Math.sin(phi)*Math.sin(theta)',
    '  const hue=0.55+Math.random()*0.3',
    '  const c=new THREE.Color().setHSL(hue,0.8,0.5+Math.random()*0.3)',
    '  colors[i*3]=c.r; colors[i*3+1]=c.g; colors[i*3+2]=c.b',
    '}',
    'const geo=new THREE.BufferGeometry()',
    'geo.setAttribute("position",new THREE.BufferAttribute(positions,3))',
    'geo.setAttribute("color",new THREE.BufferAttribute(colors,3))',
    'const mat=new THREE.PointsMaterial({size:0.022,vertexColors:true,transparent:true,opacity:0.72,sizeAttenuation:true})',
    'const pts=new THREE.Points(geo,mat)',
    'api.add(pts)',
    'api.onFrame((dt,t)=>{ pts.rotation.y+=dt*0.04 })',
    "api.log('Particle nebula added')",
  ].join('\n'),

  'Procedural texture sphere': [
    'const tex = api.texture(256, 256, (x,y,w,h) => {',
    '  const u=x/w*Math.PI*8, v=y/h*Math.PI*8',
    '  const n=(Math.sin(u+Math.cos(v))*Math.cos(u-Math.sin(v))+1)*0.5',
    '  const hue=n*0.4+0.55',
    '  const c=new THREE.Color().setHSL(hue,0.8,0.45+n*0.3)',
    '  return {r:c.r*255,g:c.g*255,b:c.b*255,a:255}',
    '})',
    'tex.wrapS=tex.wrapT=THREE.RepeatWrapping',
    'const geo=new THREE.SphereGeometry(1.1,64,64)',
    'const mat=new THREE.MeshPhysicalMaterial({map:tex,metalness:0.2,roughness:0.35,envMapIntensity:0.9})',
    'const mesh=new THREE.Mesh(geo,mat)',
    'mesh.position.set(0,1.5,0)',
    'api.add(mesh)',
    'api.onFrame((dt,t)=>{ mesh.rotation.y+=dt*0.25 })',
    "api.log('Procedural texture sphere added')",
  ].join('\n'),

  'Orbiting light rig': [
    'const lights = []',
    'const colors = [0xff4488, 0x44aaff, 0x44ff88]',
    'const pivots = []',
    'for(let i=0;i<3;i++){',
    '  const pivot=new THREE.Object3D()',
    '  const light=new THREE.PointLight(colors[i], 2.5, 12)',
    '  light.position.set(3.5, 0.8, 0)',
    '  const sphere=new THREE.Mesh(new THREE.SphereGeometry(0.08,8,8), new THREE.MeshBasicMaterial({color:colors[i]}))',
    '  sphere.position.copy(light.position)',
    '  pivot.add(light)',
    '  pivot.add(sphere)',
    '  pivot.rotation.y = (i/3)*Math.PI*2',
    '  pivots.push(pivot)',
    '  api.add(pivot)',
    '}',
    'api.onFrame((dt,t)=>{ for(const p of pivots) p.rotation.y+=dt*0.55 })',
    "api.log('Orbiting light rig added')",
  ].join('\n'),

  'Cinematic camera sweep': [
    'let elapsed = 0',
    'const startPos = api.camera.position.clone()',
    'const endPos = new THREE.Vector3(0, 3.5, 6)',
    'const duration = 4.0',
    'api.onFrame((dt) => {',
    '  elapsed = Math.min(elapsed + dt, duration)',
    '  const t = elapsed / duration',
    '  const ease = t<0.5 ? 2*t*t : -1+(4-2*t)*t',
    '  api.camera.position.lerpVectors(startPos, endPos, ease)',
    '  api.camera.lookAt(0, 0.5, 0)',
    '})',
    "api.log('Camera sweep started')",
  ].join('\n'),
}
