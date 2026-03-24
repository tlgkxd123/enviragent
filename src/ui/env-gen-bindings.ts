import {
  generateEnvFromDescription,
  buildEnvFromConfig,
  ENV_PRESETS,
} from '../render/env-codegen'
import type { WebGLRenderer } from 'three'
import type { Scene, Texture } from 'three'

export interface EnvGenApi {
  renderer: WebGLRenderer
  scene: Scene
  /** Called with the new envMap texture after generation. */
  applyEnvMap: (envMap: Texture) => void
}

export function bindEnvGenUi(api: EnvGenApi): void {
  const apiKeyInput   = document.getElementById('envApiKey')     as HTMLInputElement
  const modelInput    = document.getElementById('envModel')      as HTMLInputElement
  const promptInput   = document.getElementById('envPrompt')     as HTMLTextAreaElement
  const presetSelect  = document.getElementById('envPreset')     as HTMLSelectElement
  const generateBtn   = document.getElementById('envGenerate')   as HTMLButtonElement
  const presetBtn     = document.getElementById('envPresetApply')as HTMLButtonElement
  const statusEl      = document.getElementById('envGenStatus')  as HTMLElement

  // Populate preset dropdown
  for (const name of Object.keys(ENV_PRESETS)) {
    const opt = document.createElement('option')
    opt.value = name
    opt.textContent = name
    presetSelect.appendChild(opt)
  }

  function setStatus(msg: string, isError = false) {
    statusEl.textContent = msg
    statusEl.style.color = isError ? '#ff6b6b' : '#8a8880'
  }

  function applyConfig(config: Parameters<typeof buildEnvFromConfig>[0]) {
    setStatus('Building environment map…')
    try {
      const { envMap } = buildEnvFromConfig(config, api.renderer)
      api.scene.environment = envMap
      api.applyEnvMap(envMap)
      setStatus('Environment applied.')
    } catch (e) {
      setStatus(`Build error: ${String(e)}`, true)
    }
  }

  // Apply preset
  presetBtn.addEventListener('click', () => {
    const cfg = ENV_PRESETS[presetSelect.value]
    if (!cfg) return
    applyConfig(cfg)
  })

  // Generate via LLM
  generateBtn.addEventListener('click', async () => {
    const description = promptInput.value.trim()
    if (!description) {
      setStatus('Enter a scene description first.', true)
      return
    }
    const apiKey = apiKeyInput.value.trim()
    if (!apiKey) {
      setStatus('Enter an OpenAI API key.', true)
      return
    }
    const model = modelInput.value.trim() || 'gpt-4o-mini'

    generateBtn.disabled = true
    setStatus('Asking model…')
    try {
      const config = await generateEnvFromDescription(description, apiKey, model)
      applyConfig(config)
    } catch (e) {
      setStatus(`Error: ${String(e)}`, true)
    } finally {
      generateBtn.disabled = false
    }
  })

  setStatus('Choose a preset or describe an environment.')
}
