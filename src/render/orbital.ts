import {
  ACESFilmicToneMapping,
  BoxHelper,
  BufferAttribute,
  Color,
  CylinderGeometry,
  DirectionalLight,
  DoubleSide,
  GLSL3,
  Group,
  Mesh,
  MeshPhysicalMaterial,
  Object3D,
  PerspectiveCamera,
  PCFSoftShadowMap,
  PlaneGeometry,
  Scene,
  ShaderMaterial,
  SphereGeometry,
  TorusGeometry,
  Vector2,
  Vector3,
  WebGLRenderer,
  Raycaster,
} from 'three'
import type { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js'
import { FBXLoader } from 'three/addons/loaders/FBXLoader.js'
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { TransformControls } from 'three/addons/controls/TransformControls.js'
import { buildStudioEnvMap } from './env-hdr'
import { cAbsSq, cArg } from '../physics/complex'
import { localSphereToPhysicsAngles, radialShellRadius } from '../physics/hydrogen'
import type { OrbitalParams } from '../physics/state'
import { psiSuperposition } from '../physics/state'
import { ORBITAL_VISUAL } from './orbital-visual'
import { orbitalFragmentShader, orbitalVertexShader } from './shaders'
import { createOrbitalComposer } from './postfx'
import type { LensFlarePass } from './lens-flare-pass'
import type { AtomHalftoneUniforms } from './atom-halftone'
import {
  buildParticleCloud,
  createImportedModelMesh,
  disposeImported,
  mergeWorldMeshes,
  normalizeGroupToFit,
  updateParticleColors,
  updateWavefunctionOnGeometry,
  type ImportedModelHandles,
} from './imported-model'
import { createRadiolarianGroup } from './procedural-radiolarian'
import { createLiquidSystem, type LiquidOptions } from './liquid-sim'

const tmp = new Vector3()

export type VisualizationMode = 'orbital' | 'model'

export type TransformGizmoMode = 'translate' | 'rotate' | 'scale'
export type TransformGizmoTarget = 'orbital' | 'import' | 'selected'

export interface ImportedVisualOptions {
  showMesh: boolean
  particles: boolean
  particleCount: number
}

export interface AtomHalftoneOptions {
  enabled: boolean
  /** Dot spacing in CSS pixels (logical). */
  cellSize: number
  /** Blend toward halftone, 0–1. */
  strength: number
}

export interface MeshSelectionInfo {
  key: string
  displayName: string
}

export interface OrbitalScene {
  scene: Scene
  camera: PerspectiveCamera
  renderer: WebGLRenderer
  composer: EffectComposer
  atomHalftoneUniforms: AtomHalftoneUniforms
  controls: OrbitControls
  mesh: Mesh
  material: ShaderMaterial
  importRoot: Group
  updateOrbital: (
    primary: OrbitalParams,
    secondary: OrbitalParams | null,
    mix: number,
    evolve: boolean,
    time: number
  ) => void
  setVisualizationMode: (mode: VisualizationMode) => void
  loadObjFile: (file: File) => Promise<void>
  loadFbxFile: (file: File) => Promise<void>
  /** Detailed procedural icosphere mesh (radiolarian-style shell). */
  loadProceduralRadiolarian: () => Promise<void>
  clearImportedModel: () => void
  setImportedVisuals: (opts: ImportedVisualOptions) => void
  setModelStatusHandler: (handler: ((message: string) => void) | null) => void
  /** Re-apply ψ coloring to imported mesh/particles from last orbital parameters. */
  syncImportedWavefunction: () => void
  transformControls: TransformControls
  setTransformGizmoVisible: (visible: boolean) => void
  setTransformGizmoTarget: (target: TransformGizmoTarget) => void
  setTransformGizmoMode: (mode: TransformGizmoMode) => void
  hasImportedModel: () => boolean
  setMeshSelectionEnabled: (enabled: boolean) => void
  selectMeshAt: (clientX: number, clientY: number) => boolean
  getSelectedMeshInfo: () => MeshSelectionInfo | null
  setLiquidOptions: (opts: Partial<LiquidOptions>) => void
  step: (dt: number) => void
  setAtomHalftone: (opts: Partial<AtomHalftoneOptions>) => void
  setSize: (w: number, h: number) => void
  lensFlare: LensFlarePass
  applyEnvMap: (envMap: import('three').Texture) => void
  dispose: () => void
}

function disposeObject3D(obj: Group | Mesh) {
  obj.traverse((child) => {
    if (child instanceof Mesh) {
      child.geometry.dispose()
      const m = child.material
      if (Array.isArray(m)) m.forEach((x) => x.dispose())
      else m.dispose()
    }
  })
}

export function createOrbitalScene(canvas: HTMLCanvasElement): OrbitalScene {
  const scene = new Scene()
  scene.background = new Color(0x040506)

  const camera = new PerspectiveCamera(40, 1, 0.05, 220)
  camera.position.set(3.2, 1.45, 3.6)

  const renderer = new WebGLRenderer({
    canvas,
    antialias: true,
    alpha: false,
    powerPreference: 'high-performance',
  })
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
  renderer.outputColorSpace = 'srgb'
  renderer.toneMapping = ACESFilmicToneMapping
  renderer.toneMappingExposure = 0.92
  renderer.shadowMap.enabled = true
  renderer.shadowMap.type = PCFSoftShadowMap

  const { envMap, pmrem: pmremGenerator } = buildStudioEnvMap(renderer)
  scene.environment = envMap

  const sun = new DirectionalLight(0xfff2e6, 2.1)
  sun.position.set(6.5, 12, 4.2)
  sun.castShadow = true
  sun.shadow.mapSize.set(2048, 2048)
  sun.shadow.camera.near = 0.5
  sun.shadow.camera.far = 80
  sun.shadow.camera.left = -28
  sun.shadow.camera.right = 28
  sun.shadow.camera.top = 28
  sun.shadow.camera.bottom = -28
  sun.shadow.bias = -0.00015
  scene.add(sun)

  const fill = new DirectionalLight(0x8ec5ff, 0.35)
  fill.position.set(-8, 4, -6)
  scene.add(fill)

  const rim = new DirectionalLight(0xb8c8ff, 0.22)
  rim.position.set(-5.5, 2.2, -9)
  scene.add(rim)

  const controls = new OrbitControls(camera, canvas)
  controls.enableDamping = true
  controls.dampingFactor = 0.06
  controls.minDistance = 1.4
  controls.maxDistance = 28
  controls.target.set(0, 0.15, 0)

  const { composer, atomHalftoneUniforms, ssrSetSize, lensFlare } = createOrbitalComposer(renderer, scene, camera)

  let atomHalftoneState: AtomHalftoneOptions = {
    enabled: false,
    cellSize: 8,
    strength: 0.92,
  }

  function syncAtomHalftoneBg() {
    const bg = scene.background as Color
    atomHalftoneUniforms.bgColor.value.set(bg.r, bg.g, bg.b)
  }

  function applyAtomHalftoneUniforms() {
    const { enabled, cellSize, strength } = atomHalftoneState
    atomHalftoneUniforms.cellSize.value = Math.min(24, Math.max(2, cellSize))
    atomHalftoneUniforms.mixAmount.value = enabled ? Math.min(1, Math.max(0, strength)) : 0
    syncAtomHalftoneBg()
  }

  function setAtomHalftone(opts: Partial<AtomHalftoneOptions>) {
    atomHalftoneState = { ...atomHalftoneState, ...opts }
    applyAtomHalftoneUniforms()
  }

  applyAtomHalftoneUniforms()

  const stage = new Group()
  scene.add(stage)

  const selectableMeshes = new Map<string, { mesh: Mesh; displayName: string }>()
  let selectedKey: string | null = null

  function registerSelectableMesh(key: string, targetMesh: Mesh, displayName: string) {
    selectableMeshes.set(key, { mesh: targetMesh, displayName })
  }

  function unregisterSelectableMesh(key: string) {
    selectableMeshes.delete(key)
    if (selectedKey === key) {
      selectedKey = null
    }
  }

  const floorGeo = new PlaneGeometry(160, 160, 1, 1)
  const floorMat = new MeshPhysicalMaterial({
    color: 0x0c0d11,
    metalness: 0.94,
    roughness: 0.32,
    envMapIntensity: 1.05,
    clearcoat: 0.22,
    clearcoatRoughness: 0.45,
  })
  const floor = new Mesh(floorGeo, floorMat)
  floor.rotation.x = -Math.PI / 2
  floor.position.y = -1.35
  floor.receiveShadow = true
  stage.add(floor)
  registerSelectableMesh('stage:floor', floor, 'Floor')

  const ringInnerGeo = new TorusGeometry(5.4, 0.075, 48, 320)
  const ringMat = new MeshPhysicalMaterial({
    color: 0x1c1814,
    metalness: 0.93,
    roughness: 0.2,
    envMapIntensity: 1.15,
    clearcoat: 0.55,
    clearcoatRoughness: 0.12,
  })
  const ringInner = new Mesh(ringInnerGeo, ringMat)
  ringInner.rotation.x = Math.PI / 2
  ringInner.position.y = -0.55
  ringInner.castShadow = true
  ringInner.receiveShadow = true
  stage.add(ringInner)
  registerSelectableMesh('stage:ringInner', ringInner, 'Inner ring')

  const ringOuterGeo = new TorusGeometry(9.2, 0.05, 40, 280)
  const ringOuterMat = new MeshPhysicalMaterial({
    color: 0x14161a,
    metalness: 0.9,
    roughness: 0.28,
    envMapIntensity: 0.95,
    clearcoat: 0.35,
    clearcoatRoughness: 0.2,
  })
  const ringOuter = new Mesh(ringOuterGeo, ringOuterMat)
  ringOuter.rotation.x = Math.PI / 2
  ringOuter.position.y = -0.72
  ringOuter.castShadow = true
  ringOuter.receiveShadow = true
  stage.add(ringOuter)
  registerSelectableMesh('stage:ringOuter', ringOuter, 'Outer ring')

  const plinthGeo = new CylinderGeometry(2.35, 2.55, 0.22, 72, 1)
  const plinthMat = new MeshPhysicalMaterial({
    color: 0x101216,
    metalness: 0.86,
    roughness: 0.38,
    envMapIntensity: 0.9,
  })
  const plinth = new Mesh(plinthGeo, plinthMat)
  plinth.position.y = -0.78
  plinth.castShadow = true
  plinth.receiveShadow = true
  stage.add(plinth)
  registerSelectableMesh('stage:plinth', plinth, 'Plinth')

  const segments = 96
  const geometry = new SphereGeometry(1, segments, segments)
  const count = geometry.attributes.position.count

  const densityAttr = new Float32Array(count)
  const phaseAttr = new Float32Array(count)

  geometry.setAttribute('aDensity', new BufferAttribute(densityAttr, 1))
  geometry.setAttribute('aPhase', new BufferAttribute(phaseAttr, 1))

  const material = new ShaderMaterial({
    uniforms: {
      uDispScale: { value: 0.55 },
      uTime: { value: 0 },
      uAccent: { value: new Color(0xffb84d) },
      envMap: { value: envMap },
      envMapIntensity: { value: 1.15 },
      uBandStrength: { value: ORBITAL_VISUAL.uBandStrength },
      uSaturation: { value: ORBITAL_VISUAL.uSaturation },
      uRoughness: { value: 0.18 },
    },
    glslVersion: GLSL3,
    vertexShader: orbitalVertexShader,
    fragmentShader: orbitalFragmentShader,
    side: DoubleSide,
  })

  const mesh = new Mesh(geometry, material)
  mesh.rotation.x = Math.PI / 2
  mesh.position.y = 0.12
  mesh.castShadow = false
  scene.add(mesh)
  registerSelectableMesh('orbital', mesh, 'Hydrogen orbital')

  const importRoot = new Group()
  importRoot.position.y = 0.12
  scene.add(importRoot)

  const liquid = createLiquidSystem(560, 1.25)
  scene.add(liquid.points)

  const transformControls = new TransformControls(camera, canvas)
  transformControls.size = 0.88
  scene.add(transformControls.getHelper())
  transformControls.addEventListener('dragging-changed', (e) => {
    controls.enabled = !(e as unknown as { value: boolean }).value
  })
  transformControls.attach(mesh)

  let gizmoVisible = true
  let transformTarget: TransformGizmoTarget = 'orbital'
  let meshSelectionEnabled = true  // matches #meshSelectEnabled checked default in HTML
  const raycaster = new Raycaster()
  const pointerNdc = new Vector2()
  const selectedHelper = new BoxHelper(mesh, 0x7ec7ff)
  selectedHelper.visible = false
  scene.add(selectedHelper)

  let liquidOptions: LiquidOptions = {
    enabled: false,
    particleCount: 560,
    viscosity: 0.22,
  }

  function resolveTransformObject(): Object3D {
    if (transformTarget === 'selected' && selectedKey) {
      const entry = selectableMeshes.get(selectedKey)
      if (entry && entry.mesh.visible) return entry.mesh
    }
    if (transformTarget === 'import' && imported) return importRoot
    return mesh
  }

  function applyTransformGizmo() {
    if (!gizmoVisible) {
      transformControls.detach()
      return
    }
    transformControls.attach(resolveTransformObject())
  }

  function setTransformGizmoVisible(visible: boolean) {
    gizmoVisible = visible
    applyTransformGizmo()
  }

  function setTransformGizmoTarget(target: TransformGizmoTarget) {
    if (target === 'selected' && !selectedKey) {
      transformTarget = 'orbital'
      applyTransformGizmo()
      return
    }
    if (target === 'import' && !imported) transformTarget = 'orbital'
    else transformTarget = target
    applyTransformGizmo()
  }

  function setTransformGizmoMode(mode: TransformGizmoMode) {
    transformControls.setMode(mode)
  }

  function hasImportedModel() {
    return imported !== null
  }

  function setMeshSelectionEnabled(enabled: boolean) {
    meshSelectionEnabled = enabled
    if (!enabled) {
      selectedHelper.visible = false
      if (transformTarget === 'selected') setTransformGizmoTarget('orbital')
    }
  }

  function getSelectedMeshInfo(): MeshSelectionInfo | null {
    if (!selectedKey) return null
    const entry = selectableMeshes.get(selectedKey)
    if (!entry) return null
    return { key: selectedKey, displayName: entry.displayName }
  }

  function updateSelectionHelper() {
    if (!meshSelectionEnabled || !selectedKey) {
      selectedHelper.visible = false
      return
    }
    const entry = selectableMeshes.get(selectedKey)
    if (!entry || !entry.mesh.visible) {
      selectedHelper.visible = false
      return
    }
    selectedHelper.visible = true
    selectedHelper.setFromObject(entry.mesh)
    selectedHelper.update()
  }

  // Collect all Object3D nodes belonging to the TransformControls helper so we
  // can exclude them from raycasting (otherwise the gizmo axes intercept clicks).
  const tcHelperNodes = new Set<Object3D>()
  transformControls.getHelper().traverse((o) => tcHelperNodes.add(o))

  function selectMeshAt(clientX: number, clientY: number): boolean {
    if (!meshSelectionEnabled) return false
    const rect = canvas.getBoundingClientRect()
    if (rect.width <= 0 || rect.height <= 0) return false
    pointerNdc.x = ((clientX - rect.left) / rect.width) * 2 - 1
    pointerNdc.y = -((clientY - rect.top) / rect.height) * 2 + 1
    raycaster.setFromCamera(pointerNdc, camera)
    const meshes = Array.from(selectableMeshes.values())
      .map((x) => x.mesh)
      .filter((m) => m.visible && !tcHelperNodes.has(m))
    // Use recursive=true so imported Groups are traversed, but filter out
    // non-selectable children by checking against the selectableMeshes set.
    const hits = raycaster.intersectObjects(meshes, true)
    if (hits.length === 0) return false
    // Walk up to find the registered selectable ancestor
    let hitObj: Object3D | null = hits[0]!.object
    let selected: [string, { mesh: Mesh; displayName: string }] | undefined
    while (hitObj) {
      selected = Array.from(selectableMeshes.entries()).find(([, v]) => v.mesh === hitObj)
      if (selected) break
      hitObj = hitObj.parent
    }
    if (!selected) return false
    selectedKey = selected[0]
    updateSelectionHelper()
    // Auto-attach gizmo to the newly selected mesh so user doesn't need to
    // manually switch the dropdown target to "Selected mesh".
    if (gizmoVisible) {
      transformTarget = 'selected'
      applyTransformGizmo()
    }
    setModelStatus(`Selected: ${selected[1].displayName}`)
    return true
  }

  function setLiquidOptions(opts: Partial<LiquidOptions>) {
    liquidOptions = { ...liquidOptions, ...opts }
    liquid.setEnabled(liquidOptions.enabled)
    liquid.setParticleCount(liquidOptions.particleCount)
    liquid.setViscosity(liquidOptions.viscosity)
  }

  function step(dt: number) {
    liquid.step(dt)
    updateSelectionHelper()
  }

  setLiquidOptions(liquidOptions)

  let vizMode: VisualizationMode = 'orbital'
  let imported: ImportedModelHandles | null = null
  let importedVisuals: ImportedVisualOptions = {
    showMesh: true,
    particles: true,
    particleCount: 8000,
  }

  let lastOrbital: {
    primary: OrbitalParams
    secondary: OrbitalParams | null
    mix: number
    evolve: boolean
    time: number
  } = {
    primary: { n: 2, l: 1, m: 0 },
    secondary: null,
    mix: 0,
    evolve: false,
    time: 0,
  }

  let modelStatusHandler: ((message: string) => void) | null = null

  function setModelStatus(message: string) {
    modelStatusHandler?.(message)
  }

  function syncImportedWavefunction() {
    if (!imported) return
    const { primary, secondary, mix, evolve, time } = lastOrbital
    updateWavefunctionOnGeometry(
      imported.mergedGeometry,
      primary,
      secondary,
      mix,
      evolve,
      time
    )
    const posAttr = imported.points.geometry.getAttribute('position') as BufferAttribute
    const colAttr = imported.points.geometry.getAttribute('color') as BufferAttribute
    updateParticleColors(
      posAttr.array as Float32Array,
      colAttr.array as Float32Array,
      imported.particleCount,
      primary,
      secondary,
      mix,
      evolve,
      time
    )
    colAttr.needsUpdate = true
  }

  function syncVisibility() {
    mesh.visible = vizMode === 'orbital'
    const hasModel = imported !== null
    importRoot.visible = vizMode === 'model' && hasModel
    if (imported) {
      imported.mesh.visible = importedVisuals.showMesh
      imported.points.visible = importedVisuals.particles
    }
  }

  function rebuildParticles() {
    if (!imported) return
    const { mesh, particleCount } = imported
    importRoot.remove(imported.points)
    imported.points.geometry.dispose()
    const pm = imported.points.material
    if (!Array.isArray(pm)) pm.dispose()
    const pts = buildParticleCloud(
      mesh,
      particleCount,
      lastOrbital.primary,
      lastOrbital.secondary,
      lastOrbital.mix,
      lastOrbital.evolve,
      lastOrbital.time
    )
    importRoot.add(pts)
    imported.points = pts
  }

  function clearImportedModel() {
    if (!imported) return
    unregisterSelectableMesh('import:mesh')
    importRoot.remove(imported.mesh)
    importRoot.remove(imported.points)
    disposeImported(imported)
    imported = null
    if (transformTarget === 'import') transformTarget = 'orbital'
    syncVisibility()
    applyTransformGizmo()
  }

  async function finishImport(raw: Group, statusLabel = 'Loaded') {
    clearImportedModel()
    normalizeGroupToFit(raw, 1.05)
    const merged = mergeWorldMeshes(raw)
    raw.traverse((o) => {
      if (o instanceof Mesh) {
        o.geometry.dispose()
        const m = o.material
        if (Array.isArray(m)) m.forEach((x) => x.dispose())
        else m.dispose()
      }
    })

    const pos = merged.getAttribute('position')
    if (!pos || pos.count === 0) {
      throw new Error('File contains no mesh geometry.')
    }

    const meshImp = createImportedModelMesh(merged)
    const pc = Math.min(
      50_000,
      Math.max(500, Math.floor(importedVisuals.particleCount))
    )
    const pts = buildParticleCloud(
      meshImp,
      pc,
      lastOrbital.primary,
      lastOrbital.secondary,
      lastOrbital.mix,
      lastOrbital.evolve,
      lastOrbital.time
    )
    importRoot.add(meshImp)
    importRoot.add(pts)
    imported = {
      mesh: meshImp,
      mergedGeometry: merged,
      points: pts,
      particleCount: pc,
    }
    registerSelectableMesh('import:mesh', meshImp, statusLabel)
    importedVisuals = { ...importedVisuals, particleCount: pc }
    syncVisibility()
    setModelStatus(
      `${statusLabel}: ${pos.count.toLocaleString()} verts · ${pc.toLocaleString()} particles`
    )
    syncImportedWavefunction()
    applyTransformGizmo()
  }

  async function loadObjFile(file: File) {
    setModelStatus('Loading OBJ…')
    const loader = new OBJLoader()
    const url = URL.createObjectURL(file)
    try {
      const group = await loader.loadAsync(url)
      await finishImport(group)
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Failed to load OBJ'
      setModelStatus(msg)
      console.error(e)
    } finally {
      URL.revokeObjectURL(url)
    }
  }

  async function loadFbxFile(file: File) {
    setModelStatus('Loading FBX…')
    const loader = new FBXLoader()
    const url = URL.createObjectURL(file)
    try {
      const group = await loader.loadAsync(url)
      await finishImport(group)
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Failed to load FBX'
      setModelStatus(msg)
      console.error(e)
    } finally {
      URL.revokeObjectURL(url)
    }
  }

  async function loadProceduralRadiolarian() {
    setModelStatus('Building radiolarian shell…')
    await Promise.resolve()
    try {
      await finishImport(createRadiolarianGroup(), 'Radiolarian shell')
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Procedural build failed'
      setModelStatus(msg)
      console.error(e)
    }
  }

  function setVisualizationMode(mode: VisualizationMode) {
    vizMode = mode
    syncVisibility()
  }

  function setImportedVisuals(opts: ImportedVisualOptions) {
    importedVisuals = { ...opts }
    if (!imported) {
      syncVisibility()
      return
    }
    imported.mesh.visible = opts.showMesh
    imported.points.visible = opts.particles
    const target = Math.min(50_000, Math.max(500, Math.floor(opts.particleCount)))
    if (target !== imported.particleCount) {
      imported.particleCount = target
      rebuildParticles()
    }
    syncVisibility()
  }

  function updateOrbital(
    primary: OrbitalParams,
    secondary: OrbitalParams | null,
    mix: number,
    evolve: boolean,
    time: number
  ) {
    lastOrbital = { primary, secondary, mix, evolve, time }

    if (vizMode === 'model') {
      if (imported) {
        syncImportedWavefunction()
      }
      return
    }

    const pos = geometry.attributes.position
    const nMax = Math.max(primary.n, secondary?.n ?? primary.n)
    const rEval = radialShellRadius(nMax)

    let maxP = 0
    const rawDens = new Float32Array(count)
    const rawPhase = new Float32Array(count)

    for (let i = 0; i < count; i++) {
      tmp.fromBufferAttribute(pos, i)
      const { theta, phi } = localSphereToPhysicsAngles(tmp.x, tmp.y, tmp.z)
      const psi = psiSuperposition(primary, secondary, mix, rEval, theta, phi, time, evolve)
      const p = cAbsSq(psi)
      rawDens[i] = p
      rawPhase[i] = cArg(psi)
      if (p > maxP) maxP = p
    }
    if (maxP < 1e-20) maxP = 1

    for (let i = 0; i < count; i++) {
      densityAttr[i] = rawDens[i] / maxP
      phaseAttr[i] = rawPhase[i]
    }

    geometry.attributes.aDensity.needsUpdate = true
    geometry.attributes.aPhase.needsUpdate = true
  }

  function setSize(w: number, h: number) {
    camera.aspect = w / h
    camera.updateProjectionMatrix()
    renderer.setSize(w, h, false)
    composer.setSize(w, h)
    atomHalftoneUniforms.resolution.value.set(w, h)
    ssrSetSize(w, h)
    lensFlare.setSize(w, h)
  }

  // Track the currently active env map so we can dispose the old one on swap.
  let activeEnvMap = envMap

  function applyEnvMap(newEnvMap: import('three').Texture) {
    const old = activeEnvMap
    activeEnvMap = newEnvMap
    scene.environment = newEnvMap
    material.uniforms.envMap!.value = newEnvMap
    // Update all MeshPhysicalMaterial instances on the stage
    stage.traverse((child) => {
      if (child instanceof Mesh) {
        const m = child.material
        if (!Array.isArray(m) && 'envMap' in m) {
          ;(m as { envMap: unknown }).envMap = newEnvMap
          m.needsUpdate = true
        }
      }
    })
    if (old !== envMap) old.dispose() // don't dispose the original baked-in map
  }

  function dispose() {
    controls.dispose()
    transformControls.dispose()
    composer.dispose()
    liquid.dispose()
    selectedHelper.geometry.dispose()
    const hm = selectedHelper.material
    if (!Array.isArray(hm)) hm.dispose()
    geometry.dispose()
    material.dispose()
    clearImportedModel()
    pmremGenerator.dispose()
    envMap.dispose()
    if (activeEnvMap !== envMap) activeEnvMap.dispose()
    disposeObject3D(stage)
    sun.shadow.map?.dispose()
    renderer.dispose()
  }

  syncVisibility()

  return {
    scene,
    camera,
    renderer,
    composer,
    atomHalftoneUniforms,
    controls,
    mesh,
    material,
    importRoot,
    updateOrbital,
    setVisualizationMode,
    loadObjFile,
    loadFbxFile,
    loadProceduralRadiolarian,
    clearImportedModel,
    setImportedVisuals,
    setModelStatusHandler(handler: ((message: string) => void) | null) {
      modelStatusHandler = handler
    },
    syncImportedWavefunction,
    transformControls,
    setTransformGizmoVisible,
    setTransformGizmoTarget,
    setTransformGizmoMode,
    hasImportedModel,
    setMeshSelectionEnabled,
    selectMeshAt,
    getSelectedMeshInfo,
    setLiquidOptions,
    step,
    setAtomHalftone,
    setSize,
    lensFlare,
    applyEnvMap,
    dispose,
  }
}
