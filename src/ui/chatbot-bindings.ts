import { runSceneAgent } from '../render/scene-agent'
import type { ToolContext } from '../render/scene-tools'
import * as THREE from 'three'

/**
 * Chat-style UI for the agentic scene builder.
 *
 * A single input + send button drives the agent loop.
 * Tool calls are shown inline as collapsible log entries.
 * The user can send follow-up messages to refine the scene
 * (history is preserved across turns within the session).
 * "Clear scene" disposes all generated objects.
 */

export function bindChatbotUi(
  ctx: Omit<ToolContext, 'registry' | 'frameFns' | 'added'>,
  getModel: () => string,
  getApiKey: () => string,
  getBaseUrl: () => string = () => ''
): { step: (dt: number, t: number) => void } {
  const chatLog    = document.getElementById('chatLog')    as HTMLDivElement
  const chatInput  = document.getElementById('chatInput')  as HTMLTextAreaElement
  const sendBtn    = document.getElementById('chatSend')   as HTMLButtonElement
  const clearBtn   = document.getElementById('chatClear')  as HTMLButtonElement
  const statusEl   = document.getElementById('chatStatus') as HTMLElement

  // Per-session state
  const registry = new Map<string, THREE.Object3D>()
  const frameFns: Array<(dt: number, t: number) => void> = []
  const added: THREE.Object3D[] = []

  const toolCtx: ToolContext = { ...ctx, registry, frameFns, added }

  let abortCtrl: AbortController | null = null
  let running = false

  function setStatus(msg: string, isError = false) {
    statusEl.textContent = msg
    statusEl.style.color = isError ? '#ff6b6b' : '#8a8880'
  }

  function scrollBottom() {
    chatLog.scrollTop = chatLog.scrollHeight
  }

  function appendBubble(role: 'user' | 'assistant', content: string) {
    const wrap = document.createElement('div')
    wrap.className = `chat-bubble chat-bubble--${role}`
    wrap.textContent = content
    chatLog.appendChild(wrap)
    scrollBottom()
  }

  let streamBubble: HTMLDivElement | null = null

  function beginAssistantStream() {
    streamBubble = null
  }

  function appendAssistantDelta(delta: string) {
    if (!streamBubble) {
      streamBubble = document.createElement('div')
      streamBubble.className = 'chat-bubble chat-bubble--assistant'
      streamBubble.textContent = ''
      chatLog.appendChild(streamBubble)
    }
    streamBubble.textContent += delta
    scrollBottom()
  }

  function appendToolEntry(name: string, ok: boolean, message: string) {
    const wrap = document.createElement('div')
    wrap.className = 'chat-tool'
    const icon = ok ? '✓' : '✗'
    wrap.innerHTML = `<span class="chat-tool__icon chat-tool__icon--${ok ? 'ok' : 'err'}">${icon}</span>
      <span class="chat-tool__name">${name}</span>
      <span class="chat-tool__msg">${message}</span>`
    chatLog.appendChild(wrap)
    scrollBottom()
  }

  function clearScene() {
    for (const obj of [...added]) {
      ctx.scene.remove(obj)
      obj.traverse((c) => {
        const m = c as THREE.Mesh
        if (m.isMesh) {
          m.geometry?.dispose()
          const mat = m.material
          if (Array.isArray(mat)) mat.forEach((x) => (x as THREE.Material).dispose())
          else (mat as THREE.Material)?.dispose()
        }
      })
    }
    added.length = 0
    frameFns.length = 0
    registry.clear()

    const note = document.createElement('div')
    note.className = 'chat-divider'
    note.textContent = '— scene cleared —'
    chatLog.appendChild(note)
    scrollBottom()
    setStatus('Scene cleared.')
  }

  async function send() {
    const text = chatInput.value.trim()
    if (!text || running) return

    const apiKey = getApiKey()
    if (!apiKey) {
      setStatus('Enter an API key above.', true)
      return
    }

    const baseUrl = getBaseUrl().trim() || 'https://api.openai.com/v1'
    chatInput.value = ''
    running = true
    sendBtn.disabled = true
    abortCtrl = new AbortController()

    await runSceneAgent(
      text,
      toolCtx,
      { apiKey, model: getModel() || 'gpt-4o', baseUrl },
      {
        signal: abortCtrl.signal,
        onMessage(role, content) {
          if (role === 'user') appendBubble(role, content)
        },
        onAssistantStreamBegin: beginAssistantStream,
        onAssistantDelta: appendAssistantDelta,
        onToolCall(name, _args, result) { appendToolEntry(name, result.ok, result.message) },
        onStatus(msg) { setStatus(msg) },
        onDone() {
          running = false
          sendBtn.disabled = false
          abortCtrl = null
          streamBubble = null
        },
      }
    )
  }

  sendBtn.addEventListener('click', () => { void send() })
  chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); void send() }
  })
  clearBtn.addEventListener('click', clearScene)

  setStatus('Describe a scene and press Send (or Enter).')

  return {
    step(dt: number, t: number) {
      for (const fn of frameFns) {
        try { fn(dt, t) } catch { /* ignore */ }
      }
    },
  }
}
