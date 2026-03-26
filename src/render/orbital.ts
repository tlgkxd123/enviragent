import {
  ACESFilmicToneMapping,
  BoxHelper,
  Color,
  CylinderGeometry,
  DirectionalLight,
  Group,
  Mesh,
  MeshPhysicalMaterial,
  Object3D,
  PerspectiveCamera,
  PCFSoftShadowMap,
  PlaneGeometry,
  Scene,
  Vector2,
  WebGLRenderer,
  Raycaster,
} from 'three'
import type { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js'
import { FBXLoader } from 'three/addons/loaders/FBXLoader.js'
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { TransformControls } from 'three/addons/controls/TransformControls.js'
import { buildStudioEnvMap } from './env-hdr'
import { createOrbitalComposer } from './postfx'
import type { LensFlarePass } from './lens-flare-pass'
import type { AtomHalftoneUniforms } from './atom-halftone'
import {
  buildParticleCloud,
  createImportedModelMesh,
  disposeImported,
  mergeWorldMeshes,
  normalizeGroupToFit,
  type ImportedModelHandles,
} from './imported-model'
import { createRadiolarianGroup } from './procedural-radiolarian'
import { createLiquidSystem, type LiquidOptions } from './liquid-sim'

export type TransformGizmoMode = 'translate' | 'rotate' | 'scale'
export type TransformGizmoTarget = 'import' | 'selected'

export interface ImportedVisualOptions {
  showMesh: boolean
  particles: boolean
  particleCount: number
}

export interface AtomHalftoneOptions {
  enabled: boolean
  cellSize: number
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
  importRoot: Group
  loadObjFile: (file: File) => Promise<void>
  loadFbxFile: (file: File) => Promise<void>
  loadProceduralRadiolarian: () => Promise<void>
  clearImportedModel: () => void
  setImportedVisuals: (opts: ImportedVisualOptions) => void
  setModelStatusHandler: (handler: ((message: string) => void) | null) => void
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

  const camera = new PerspectiveCamera(40, 1, 0.001, 1e7)
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
  controls.minDistance = 0.0005
  controls.maxDistance = 1e6
  controls.zoomSpeed = 1.2
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

  let gizmoVisible = true
  let transformTarget: TransformGizmoTarget = 'selected'
  let meshSelectionEnabled = true
  const raycaster = new Raycaster()
  const pointerNdc = new Vector2()
  const selectedHelper = new BoxHelper(new Mesh(), 0x7ec7ff)
  selectedHelper.visible = false
  scene.add(selectedHelper)

  let liquidOptions: LiquidOptions = {
    enabled: false,
    particleCount: 560,
    viscosity: 0.22,
  }

  function resolveTransformObject(): Object3D | null {
    if (transformTarget === 'selected' && selectedKey) {
      const entry = selectableMeshes.get(selectedKey)
      if (entry && entry.mesh.visible) return entry.mesh
    }
    if (transformTarget === 'import' && imported) return importRoot
    return null
  }

  function applyTransformGizmo() {
    if (!gizmoVisible) {
      transformControls.detach()
      return
    }
    const obj = resolveTransformObject()
    if (obj) transformControls.attach(obj)
    else transformControls.detach()
  }

  function setTransformGizmoVisible(visible: boolean) {
    gizmoVisible = visible
    applyTransformGizmo()
  }

  function setTransformGizmoTarget(target: TransformGizmoTarget) {
    if (target === 'selected' && !selectedKey) {
      transformTarget = 'selected'
      applyTransformGizmo()
      return
    }
    if (target === 'import' && !imported) transformTarget = 'selected'
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
    const hits = raycaster.intersectObjects(meshes, true)
    if (hits.length === 0) return false
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
    const dist = camera.position.distanceTo(controls.target)
    camera.near = Math.max(0.0001, dist * 0.0001)
    camera.far  = Math.max(1e5, dist * 1e4)
    camera.updateProjectionMatrix()
  }

  setLiquidOptions(liquidOptions)

  let imported: ImportedModelHandles | null = null
  let importedVisuals: ImportedVisualOptions = {
    showMesh: true,
    particles: true,
    particleCount: 8000,
  }

  let modelStatusHandler: ((message: string) => void) | null = null

  function setModelStatus(message: string) {
    modelStatusHandler?.(message)
  }

  function syncVisibility() {
    const hasModel = imported !== null
    importRoot.visible = hasModel
    if (imported) {
      imported.mesh.visible = importedVisuals.showMesh
      imported.points.visible = importedVisuals.particles
    }
  }

  function rebuildParticles() {
    if (!imported) return
    const { mesh: impMesh, particleCount } = imported
    importRoot.remove(imported.points)
    imported.points.geometry.dispose()
    const pm = imported.points.material
    if (!Array.isArray(pm)) pm.dispose()
    const pts = buildParticleCloud(impMesh, particleCount)
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
    const pts = buildParticleCloud(meshImp, pc)
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

  function setSize(w: number, h: number) {
    camera.aspect = w / h
    camera.updateProjectionMatrix()
    renderer.setSize(w, h, false)
    composer.setSize(w, h)
    atomHalftoneUniforms.resolution.value.set(w, h)
    ssrSetSize(w, h)
    lensFlare.setSize(w, h)
  }

  let activeEnvMap = envMap

  function applyEnvMap(newEnvMap: import('three').Texture) {
    const old = activeEnvMap
    activeEnvMap = newEnvMap
    scene.environment = newEnvMap
    stage.traverse((child) => {
      if (child instanceof Mesh) {
        const m = child.material
        if (!Array.isArray(m) && 'envMap' in m) {
          ;(m as { envMap: unknown }).envMap = newEnvMap
          m.needsUpdate = true
        }
      }
    })
    if (old !== envMap) old.dispose()
  }

  function dispose() {
    controls.dispose()
    transformControls.dispose()
    composer.dispose()
    liquid.dispose()
    selectedHelper.geometry.dispose()
    const hm = selectedHelper.material
    if (!Array.isArray(hm)) hm.dispose()
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
    importRoot,
    loadObjFile,
    loadFbxFile,
    loadProceduralRadiolarian,
    clearImportedModel,
    setImportedVisuals,
    setModelStatusHandler(handler: ((message: string) => void) | null) {
      modelStatusHandler = handler
    },
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
