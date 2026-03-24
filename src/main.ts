import './style.css'
import { createOrbitalScene } from './render/orbital'
import { bindDisplayUi } from './ui/display-bindings'
import { bindGizmoUi } from './ui/gizmo-bindings'
import { bindLiquidUi } from './ui/liquid-bindings'
import { bindModelUi } from './ui/model-bindings'
import { bindUi } from './ui/bindings'
import { updateQuantumReadout } from './ui/quantum-readout'
import { bindEnvGenUi } from './ui/env-gen-bindings'
import { bindChatbotUi } from './ui/chatbot-bindings'
import { projectDirToScreenUv } from './render/lens-flare-pass'
import { Vector3 } from 'three'

const _keyLightDir = new Vector3(0.65, 0.55, -0.52).normalize()

const canvas = document.getElementById('canvas') as HTMLCanvasElement
const orbital = createOrbitalScene(canvas)

orbital.setModelStatusHandler((msg) => {
  const el = document.getElementById('modelStatus')
  if (el) el.textContent = msg
})

function resize() {
  orbital.setSize(window.innerWidth, window.innerHeight)
}
resize()
window.addEventListener('resize', resize)

bindUi((state) => {
  orbital.updateOrbital(
    state.primary,
    state.secondary,
    state.mix,
    state.evolve,
    state.time
  )
  updateQuantumReadout(
    state.primary,
    state.secondary,
    state.secondary !== null
  )
})

const gizmoUi = bindGizmoUi(orbital)
bindDisplayUi(orbital)
bindLiquidUi(orbital)
bindEnvGenUi({
  renderer: orbital.renderer,
  scene: orbital.scene,
  applyEnvMap: (envMap) => orbital.applyEnvMap(envMap),
})

const sceneGen = bindChatbotUi(
  {
    scene: orbital.scene,
    camera: orbital.camera,
    renderer: orbital.renderer,
    applyEnvMap: (envMap) => orbital.applyEnvMap(envMap),
  },
  () => (document.getElementById('chatModel') as HTMLInputElement)?.value ?? 'gpt-4o',
  () => (document.getElementById('chatApiKey') as HTMLInputElement)?.value ?? '',
  () => (document.getElementById('chatBaseUrl') as HTMLInputElement)?.value ?? ''
)

const meshSelectEnabled = document.getElementById('meshSelectEnabled') as HTMLInputElement
orbital.setMeshSelectionEnabled(meshSelectEnabled.checked)
meshSelectEnabled.addEventListener('change', () => {
  orbital.setMeshSelectionEnabled(meshSelectEnabled.checked)
})

canvas.addEventListener('pointerdown', (e) => {
  if (e.button !== 0) return
  orbital.selectMeshAt(e.clientX, e.clientY)
})

bindModelUi((m) => {
  orbital.setVisualizationMode(m.viewMode)
  orbital.setImportedVisuals({
    showMesh: m.showImportedMesh,
    particles: m.particlesEnabled,
    particleCount: m.particleCount,
  })
  orbital.syncImportedWavefunction()
})

document.getElementById('fileObj')?.addEventListener('change', async (e) => {
  const input = e.target as HTMLInputElement
  const f = input.files?.[0]
  if (!f) return
  await orbital.loadObjFile(f)
  gizmoUi.syncImportOption()
  input.value = ''
})

document.getElementById('fileFbx')?.addEventListener('change', async (e) => {
  const input = e.target as HTMLInputElement
  const f = input.files?.[0]
  if (!f) return
  await orbital.loadFbxFile(f)
  gizmoUi.syncImportOption()
  input.value = ''
})

document.getElementById('clearModel')?.addEventListener('click', () => {
  orbital.clearImportedModel()
  gizmoUi.syncImportOption()
  const el = document.getElementById('modelStatus')
  if (el) el.textContent = 'Import cleared.'
})

document.getElementById('proceduralModel')?.addEventListener('click', async () => {
  await orbital.loadProceduralRadiolarian()
  gizmoUi.syncImportOption()
  const viewMode = document.getElementById('viewMode') as HTMLSelectElement
  viewMode.value = 'model'
  viewMode.dispatchEvent(new Event('change'))
})

const resetCam = document.getElementById('resetCam') as HTMLButtonElement
resetCam.addEventListener('click', () => {
  orbital.camera.position.set(3.2, 1.45, 3.6)
  orbital.controls.target.set(0, 0.15, 0)
  orbital.controls.update()
})

let lastTs = performance.now()
let totalTime = 0

function animate(ts: number) {
  const dt = (ts - lastTs) * 0.001
  lastTs = ts
  totalTime += dt
  orbital.material.uniforms.uTime.value = ts * 0.001
  orbital.step(dt)
  orbital.controls.update()
  sceneGen.step(dt, totalTime)

  // Project key light to screen UV and feed to lens flare pass
  const uv = projectDirToScreenUv(_keyLightDir, orbital.camera)
  if (uv) {
    orbital.lensFlare.setLightScreenUv(uv.x, uv.y)
    orbital.lensFlare.setIntensity(0.88)
  } else {
    orbital.lensFlare.setIntensity(0) // light behind camera — hide flare
  }

  orbital.composer.render()
  requestAnimationFrame(animate)
}
requestAnimationFrame(animate)
