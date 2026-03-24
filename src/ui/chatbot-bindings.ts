import { runSceneAgent } from '../render/scene-agent'
import type { ToolContext } from '../render/scene-tools'
import { parseThinkingDisplay } from './thinking-parse'
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
  getBaseUrl: () => string = () => '',
  getReasoningEffort: () => string = () => 'none'
): { step: (dt: number, t: number) => void } {
  const chatLog    = document.getElementById('chatLog')    as HTMLDivElement
  const chatInput  = document.getElementById('chatInput')  as HTMLTextAreaElement
  const sendBtn    = document.getElementById('chatSend')   as HTMLButtonElement
  const clearBtn   = document.getElementById('chatClear')  as HTMLButtonElement
  const statusEl   = document.getElementById('chatStatus') as HTMLElement
  const sseBadgeEl = document.getElementById('chatSseBadge') as HTMLElement | null
  const apiKeyEl     = document.getElementById('chatApiKey')    as HTMLInputElement
  const baseUrlEl    = document.getElementById('chatBaseUrl')   as HTMLInputElement
  const modelEl      = document.getElementById('chatModel')     as HTMLInputElement
  const reasoningEl  = document.getElementById('chatReasoning') as HTMLSelectElement

  // ── localStorage persistence ────────────────────────────────────────────
  const LS_API_KEY   = 'subatomic_chat_api_key'
  const LS_BASE_URL  = 'subatomic_chat_base_url'
  const LS_MODEL     = 'subatomic_chat_model'
  const LS_REASONING = 'subatomic_chat_reasoning'

  function loadFromStorage() {
    const savedKey       = localStorage.getItem(LS_API_KEY)
    const savedBaseUrl   = localStorage.getItem(LS_BASE_URL)
    const savedModel     = localStorage.getItem(LS_MODEL)
    const savedReasoning = localStorage.getItem(LS_REASONING)
    if (savedKey       && apiKeyEl)    apiKeyEl.value    = savedKey
    if (savedBaseUrl   && baseUrlEl)   baseUrlEl.value   = savedBaseUrl
    if (savedModel     && modelEl)     modelEl.value     = savedModel
    if (savedReasoning && reasoningEl) reasoningEl.value = savedReasoning
  }

  function saveToStorage() {
    if (apiKeyEl)    localStorage.setItem(LS_API_KEY,   apiKeyEl.value)
    if (baseUrlEl)   localStorage.setItem(LS_BASE_URL,  baseUrlEl.value)
    if (modelEl)     localStorage.setItem(LS_MODEL,     modelEl.value)
    if (reasoningEl) localStorage.setItem(LS_REASONING, reasoningEl.value)
  }

  loadFromStorage()
  apiKeyEl?.addEventListener('change', saveToStorage)
  baseUrlEl?.addEventListener('change', saveToStorage)
  modelEl?.addEventListener('change', saveToStorage)
  reasoningEl?.addEventListener('change', saveToStorage)
  // Also save on blur so pasting without typing still persists
  apiKeyEl?.addEventListener('blur', saveToStorage)
  baseUrlEl?.addEventListener('blur', saveToStorage)
  modelEl?.addEventListener('blur', saveToStorage)

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

  /** If no CoT (tags / API reasoning) starts within this window, hide the Thinking wait and stream main text only. */
  const COT_WAIT_MS = 15_000

  let streamBubble: HTMLDivElement | null = null
  let streamRawAccum = ''
  let streamReasoningAccum = ''
  let streamToolPreview = ''
  /** True while assistant SSE is in progress for the current assistant turn. */
  let assistantStreaming = false
  let cotStreamStartMs = 0
  /** Set when COT_WAIT_MS elapsed with no thinking content yet — keep generating in the main bubble only. */
  let cotWaitSkipped = false
  let cotWaitPollTimer: ReturnType<typeof setInterval> | null = null

  /** Unicode-safe length / slice for streamed text (for-of over code points). */
  function charLen(s: string): number {
    let n = 0
    for (const _ of s) n++
    return n
  }
  function sliceChars(s: string, n: number): string {
    if (n <= 0) return ''
    let i = 0
    let out = ''
    for (const ch of s) {
      if (i >= n) break
      out += ch
      i++
    }
    return out
  }

  /** Typewriter reveal: targets grow from SSE; display catches up char-by-char (adaptive speed for large chunks). */
  let typeTargetMain = ''
  let typeTargetThinking = ''
  let typeRevealMain = 0
  let typeRevealThinking = 0
  let prevMainTargetKey: '' | 'pending' | 'tool' | 'text' = ''
  let typewriterRaf: number | null = null
  let lastShowMainPending = false
  let lastInThinking = false

  function cancelTypewriter() {
    if (typewriterRaf !== null) {
      cancelAnimationFrame(typewriterRaf)
      typewriterRaf = null
    }
  }

  function stepForLag(lag: number): number {
    if (lag <= 0) return 0
    if (lag > 200) return Math.min(32, Math.ceil(lag / 12))
    if (lag > 80) return 12
    if (lag > 30) return 5
    return 2
  }

  function applyTypewriterSlice() {
    if (!streamBubble) return
    const pre = streamBubble.querySelector('.chat-thinking__body') as HTMLPreElement
    const main = streamBubble.querySelector('.chat-bubble__text') as HTMLDivElement
    main.textContent = sliceChars(typeTargetMain, typeRevealMain)
    main.classList.toggle('chat-bubble__text--pending', lastShowMainPending)
    pre.textContent = sliceChars(typeTargetThinking, typeRevealThinking)
    if (lastInThinking) {
      requestAnimationFrame(() => {
        pre.scrollTop = pre.scrollHeight
      })
    }
    scrollBottom()
  }

  function runTypewriterTick() {
    typewriterRaf = null
    const tm = charLen(typeTargetMain)
    const tt = charLen(typeTargetThinking)
    if (typeRevealMain < tm) {
      typeRevealMain = Math.min(tm, typeRevealMain + stepForLag(tm - typeRevealMain))
    }
    if (typeRevealThinking < tt) {
      typeRevealThinking = Math.min(tt, typeRevealThinking + stepForLag(tt - typeRevealThinking))
    }
    applyTypewriterSlice()
    if (typeRevealMain < tm || typeRevealThinking < tt) {
      typewriterRaf = requestAnimationFrame(runTypewriterTick)
    }
  }

  function scheduleTypewriter() {
    if (typewriterRaf !== null) return
    typewriterRaf = requestAnimationFrame(runTypewriterTick)
  }

  function clearCotWaitPoll() {
    if (cotWaitPollTimer !== null) {
      clearInterval(cotWaitPollTimer)
      cotWaitPollTimer = null
    }
  }

  function reasoningEffortActive(): boolean {
    const r = getReasoningEffort().trim() || 'none'
    return r !== 'none'
  }

  function ensureAssistantStreamBubble(): HTMLDivElement {
    if (!streamBubble) {
      streamBubble = document.createElement('div')
      streamBubble.className = 'chat-bubble chat-bubble--assistant'

      const details = document.createElement('details')
      details.className = 'chat-thinking'

      const summary = document.createElement('summary')
      summary.className = 'chat-thinking__summary'
      summary.textContent = 'Thinking'

      const pre = document.createElement('pre')
      pre.className = 'chat-thinking__body'

      details.appendChild(summary)
      details.appendChild(pre)

      const main = document.createElement('div')
      main.className = 'chat-bubble__text'

      streamBubble.appendChild(details)
      streamBubble.appendChild(main)
      chatLog.appendChild(streamBubble)
    }
    return streamBubble
  }

  function updateAssistantStreamBubble() {
    const bubble = ensureAssistantStreamBubble()
    const details = bubble.querySelector('details.chat-thinking') as HTMLDetailsElement
    const summary = bubble.querySelector('.chat-thinking__summary') as HTMLElement

    const { visible, thinking, inThinking } = parseThinkingDisplay(streamRawAccum)
    let combinedThinking = thinking
    if (streamReasoningAccum.length > 0) {
      combinedThinking = combinedThinking
        ? `${combinedThinking}\n\n──\n${streamReasoningAccum}`
        : streamReasoningAccum
    }

    const anyCot =
      inThinking || combinedThinking.length > 0

    if (
      assistantStreaming &&
      !anyCot &&
      Date.now() - cotStreamStartMs >= COT_WAIT_MS
    ) {
      cotWaitSkipped = true
    }

    const placeholder =
      assistantStreaming &&
      reasoningEffortActive() &&
      !cotWaitSkipped &&
      combinedThinking.length === 0 &&
      !inThinking
        ? 'Waiting for reasoning / output…'
        : ''

    const thinkingTarget = combinedThinking || placeholder

    const noTokensYet =
      !streamRawAccum.trim() &&
      !streamReasoningAccum.trim() &&
      !streamToolPreview.trim()
    const showMainPending =
      assistantStreaming && noTokensYet
    const hasVisible = visible.trim().length > 0
    const showToolPreview =
      !hasVisible && streamToolPreview.trim().length > 0
    const mainTarget = showMainPending
      ? 'Receiving stream…'
      : showToolPreview
        ? streamToolPreview
        : visible

    const mainTargetKey: '' | 'pending' | 'tool' | 'text' =
      showMainPending ? 'pending' : showToolPreview ? 'tool' : 'text'
    if (prevMainTargetKey !== mainTargetKey) {
      typeRevealMain = 0
    }
    prevMainTargetKey = mainTargetKey

    typeTargetMain = mainTarget
    typeTargetThinking = thinkingTarget

    typeRevealMain = Math.min(typeRevealMain, charLen(typeTargetMain))
    typeRevealThinking = Math.min(typeRevealThinking, charLen(typeTargetThinking))

    lastShowMainPending = showMainPending
    lastInThinking = inThinking

    const showThinking =
      inThinking ||
      combinedThinking.length > 0 ||
      placeholder.length > 0 ||
      (assistantStreaming && reasoningEffortActive() && !cotWaitSkipped)

    details.hidden = !showThinking
    details.classList.toggle('chat-thinking--streaming', inThinking)
    if (summary) summary.textContent = inThinking ? 'Thinking (streaming…)' : 'Thinking'
    if (showThinking) details.open = true

    scheduleTypewriter()
    scrollBottom()
  }

  function beginAssistantStream() {
    streamBubble = null
    streamRawAccum = ''
    streamReasoningAccum = ''
    streamToolPreview = ''
    assistantStreaming = true
    cotStreamStartMs = Date.now()
    cotWaitSkipped = false
    cancelTypewriter()
    typeTargetMain = ''
    typeTargetThinking = ''
    typeRevealMain = 0
    typeRevealThinking = 0
    prevMainTargetKey = ''
    clearCotWaitPoll()
    cotWaitPollTimer = setInterval(() => {
      if (!assistantStreaming) {
        clearCotWaitPoll()
        return
      }
      updateAssistantStreamBubble()
    }, 400)
    if (sseBadgeEl) sseBadgeEl.hidden = true
    updateAssistantStreamBubble()
    requestAnimationFrame(() => {
      streamBubble?.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
    })
  }

  function appendAssistantDelta(delta: string) {
    streamRawAccum += delta
    const bubble = ensureAssistantStreamBubble()
    bubble.classList.add('chat-bubble--sse')
    if (sseBadgeEl) sseBadgeEl.hidden = false
    updateAssistantStreamBubble()
  }

  function appendAssistantReasoningDelta(delta: string) {
    streamReasoningAccum += delta
    const bubble = ensureAssistantStreamBubble()
    bubble.classList.add('chat-bubble--sse')
    if (sseBadgeEl) sseBadgeEl.hidden = false
    updateAssistantStreamBubble()
  }

  function appendAssistantToolPreviewDelta(preview: string) {
    streamToolPreview = preview
    const bubble = ensureAssistantStreamBubble()
    bubble.classList.add('chat-bubble--sse')
    if (sseBadgeEl) sseBadgeEl.hidden = false
    updateAssistantStreamBubble()
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

    const reasoningEffort = getReasoningEffort().trim() || 'none'
    await runSceneAgent(
      text,
      toolCtx,
      { apiKey, model: getModel() || 'gpt-4o', baseUrl, reasoningEffort },
      {
        signal: abortCtrl.signal,
        onMessage(role, content) {
          if (role === 'user') appendBubble(role, content)
        },
        onAssistantStreamBegin: beginAssistantStream,
        onAssistantStreamEnd() {
          assistantStreaming = false
          clearCotWaitPoll()
          if (sseBadgeEl) sseBadgeEl.hidden = true
          if (streamBubble) streamBubble.classList.remove('chat-bubble--sse')
          const hadRenderableAssistantText =
            streamRawAccum.trim().length > 0 || streamReasoningAccum.trim().length > 0
          streamToolPreview = ''
          if (!hadRenderableAssistantText) {
            cancelTypewriter()
            streamBubble?.remove()
            streamBubble = null
            return
          }
          updateAssistantStreamBubble()
          cancelTypewriter()
          typeRevealMain = charLen(typeTargetMain)
          typeRevealThinking = charLen(typeTargetThinking)
          applyTypewriterSlice()
        },
        onAssistantDelta: appendAssistantDelta,
        onAssistantReasoningDelta: appendAssistantReasoningDelta,
        onAssistantToolPreviewDelta: appendAssistantToolPreviewDelta,
        onToolCall(name, _args, result) { appendToolEntry(name, result.ok, result.message) },
        onStatus(msg) { setStatus(msg) },
        onDone() {
          running = false
          sendBtn.disabled = false
          abortCtrl = null
          assistantStreaming = false
          clearCotWaitPoll()
          cancelTypewriter()
          streamToolPreview = ''
          streamBubble = null
          if (sseBadgeEl) sseBadgeEl.hidden = true
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
