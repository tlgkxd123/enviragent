import { generateSceneCode, executeSceneCode, CODE_SNIPPETS } from '../render/scene-codegen'
import type { SceneCodegenContext, GeneratedScene } from '../render/scene-codegen'

/**
 * Binds the scene code-generator UI panel.
 *
 * Manages:
 *  - Snippet dropdown + apply
 *  - Code editor <textarea>
 *  - LLM generate button (description → code → display in editor)
 *  - Run button (execute current editor code)
 *  - Clear button (dispose all generated objects)
 *  - Per-frame tick forwarding
 */
export function bindSceneGenUi(
  ctx: SceneCodegenContext,
  getModel: () => string,
  getApiKey: () => string
): { step: (dt: number, t: number) => void } {
  const snippetSelect = document.getElementById('sgSnippet')     as HTMLSelectElement
  const snippetApply  = document.getElementById('sgSnippetApply')as HTMLButtonElement
  const descInput     = document.getElementById('sgDescription') as HTMLTextAreaElement
  const generateBtn   = document.getElementById('sgGenerate')    as HTMLButtonElement
  const codeEditor    = document.getElementById('sgCode')        as HTMLTextAreaElement
  const runBtn        = document.getElementById('sgRun')         as HTMLButtonElement
  const clearBtn      = document.getElementById('sgClear')       as HTMLButtonElement
  const statusEl      = document.getElementById('sgStatus')      as HTMLElement

  // Populate snippet dropdown
  for (const name of Object.keys(CODE_SNIPPETS)) {
    const opt = document.createElement('option')
    opt.value = name
    opt.textContent = name
    snippetSelect.appendChild(opt)
  }

  let current: GeneratedScene | null = null

  function setStatus(msg: string, isError = false) {
    statusEl.textContent = msg
    statusEl.style.color = isError ? '#ff6b6b' : '#8a8880'
  }

  function clearGenerated() {
    if (current) {
      current.dispose()
      current = null
    }
    setStatus('Scene cleared.')
  }

  function runCode(code: string) {
    clearGenerated()
    if (!code.trim()) { setStatus('No code to run.', true); return }
    setStatus('Running…')
    current = executeSceneCode(code, {
      ...ctx,
      onLog:   (msg) => setStatus(msg),
      onError: (msg) => setStatus(msg, true),
    })
    const count = current.objects.length
    if (count > 0) setStatus(`Done — ${count} object(s) added.`)
  }

  // Snippet → load into editor
  snippetApply.addEventListener('click', () => {
    const code = CODE_SNIPPETS[snippetSelect.value]
    if (code) {
      codeEditor.value = code
      setStatus('Snippet loaded — click Run to execute.')
    }
  })

  // LLM generate → put code in editor
  generateBtn.addEventListener('click', async () => {
    const description = descInput.value.trim()
    if (!description) { setStatus('Enter a description first.', true); return }
    const apiKey = getApiKey()
    if (!apiKey) { setStatus('Enter an API key in the Environment panel.', true); return }
    const model = getModel() || 'gpt-4o'

    generateBtn.disabled = true
    setStatus('Generating code…')
    try {
      const code = await generateSceneCode(description, apiKey, model)
      codeEditor.value = code
      setStatus('Code generated — review then click Run.')
    } catch (e) {
      setStatus(`Generation error: ${String(e)}`, true)
    } finally {
      generateBtn.disabled = false
    }
  })

  // Run button → execute editor contents
  runBtn.addEventListener('click', () => {
    runCode(codeEditor.value)
  })

  // Clear button
  clearBtn.addEventListener('click', clearGenerated)

  // Forward animation frames to generated scene
  function step(dt: number, t: number) {
    if (!current) return
    for (const fn of current.frameFns) {
      try { fn(dt, t) } catch { /* ignore per-frame errors */ }
    }
  }

  setStatus('Choose a snippet or describe a scene.')
  return { step }
}
