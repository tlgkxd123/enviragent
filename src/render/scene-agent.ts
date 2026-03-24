import { readSseJsonLines, type SseJsonMeta } from '../sse/read-sse'
import { stripThinkingTags } from '../ui/thinking-parse'
import { getToolSchemas, dispatchTool } from './scene-tools'
import type { ToolContext, ToolResult } from './scene-tools'

/**
 * Agentic scene-building loop using OpenAI function-calling.
 *
 * Uses streaming (SSE) for chat completions: assistant text streams live;
 * tool_calls are accumulated from deltas, then executed.
 */

/** Default max time between SSE chunks before treating the stream as stuck (reasoning models can go silent for minutes). */
export const DEFAULT_SSE_IDLE_TIMEOUT_MS = 600_000

export interface AgentOptions {
  apiKey: string
  model?: string
  baseUrl?: string
  maxRounds?: number
  /** OpenAI o-series reasoning effort: 'low' | 'medium' | 'high'.
   *  When set (and not 'none'), `temperature` is omitted from the request
   *  (o-series models reject temperature) and `reasoning_effort` is added. */
  reasoningEffort?: string
  /** Max milliseconds without any SSE body chunk before aborting streaming (then non-streaming fallback). Default 10 minutes. */
  sseIdleTimeoutMs?: number
}

export type MessageRole = 'user' | 'assistant' | 'tool' | 'system'

export interface ChatMessage {
  role: MessageRole
  content: string
  tool_call_id?: string
  tool_calls?: ToolCallDef[]
  name?: string
}

interface ToolCallDef {
  id: string
  type: 'function'
  function: { name: string; arguments: string }
}

interface ApiMessage {
  role: string
  content: string | null
  tool_calls?: ToolCallDef[] | unknown[]
  tool_call_id?: string
  name?: string
}

const SYSTEM = `You are an expert Three.js scene artist building a live 3D quantum physics visualization.
You MUST use the provided tools to add meshes, lights, volumetric clouds, particle systems, animate objects,
set the camera, and change the environment. Do not describe a scene in words only — always call tools to
actually build it. Build incrementally with multiple tool calls when useful.
The stage floor mesh top surface is near y ≈ -1.35; place trees, rocks, and props on the ground with their base at or slightly above that Y (not floating mid-air). For scatter/forest presets, set base height and spread so instances sit on the floor. Use add_procedural_mesh with higher segments (e.g. 48+) and MeshPhysicalMaterial for readable detail; avoid placing only default cones without position sanity checks.
For clouds: prefer moderate density and coverage so they stay wispy — extreme values look like a solid white blob under bloom/post-processing.
Use add_procedural_mesh for dynamic content: terrain heightmaps, organic noise spheres, scattered props (forests of primitives), crystal clusters, rock fields — vary the seed for different layouts.
Use exec_threejs_code for arbitrary Three.js when other tools are not enough (custom BufferGeometry, curves, helpers).
Use gen_texture to apply physically-based GPU procedural textures: domain=atomic (orbital_density, orbital_phase, interference, radial_probability, electron_cloud), domain=cellular (voronoi_membrane, reaction_diffusion, cytoskeleton, mitochondria), domain=material (crystal_lattice, thin_film, grain_boundary, dislocation_field). Set animate=true for live animation. Apply to mesh via target_name+map_slot, or omit to create a display plane.
Use gen_shader_code for custom GLSL ES 3.0 materials (GLSL3: out fragColor, in/out varyings); pass uniforms as JSON with type/value. IMPORTANT: Never redeclare Three.js built-in vertex attributes/uniforms (position, uv, normal, modelViewMatrix, projectionMatrix, modelMatrix, viewMatrix, normalMatrix, cameraPosition) — they are already injected by Three.js and redeclaring them causes a GLSL compile error.
Use reset_scene to wipe all AI-generated content and start over.
Use create_tool to define reusable custom tools (e.g. spawn_asteroid, add_neon_ring) that accept parameters — then call them multiple times. The body receives (ctx, args, THREE, track); use track(ctx, name, obj) to add objects to the scene.
After calling tools, briefly describe what you created in plain text. Be creative and cinematic.
You may put private step-by-step planning in <thinking>...</thinking> XML tags; the UI shows that in a Thinking panel and omits it from API history. Write the user-facing summary outside those tags.
Do not ask clarifying questions — just build based on the description. Call list_objects if you
need to know what exists. When you are satisfied with the scene, give a short summary and stop.`

function normalizeBaseUrl(url: string): string {
  return url.replace(/\/+$/, '')
}

function normalizeToolCalls(raw: unknown): ToolCallDef[] | undefined {
  if (!Array.isArray(raw) || raw.length === 0) return undefined
  const out: ToolCallDef[] = []
  for (let i = 0; i < raw.length; i++) {
    const tc = raw[i]
    if (!tc || typeof tc !== 'object') continue
    const t = tc as Record<string, unknown>
    const id = String(t.id ?? `call_${i}`)
    const fn = t.function
    if (!fn || typeof fn !== 'object') continue
    const f = fn as Record<string, unknown>
    const name = String(f.name ?? '')
    if (!name) continue
    let args = f.arguments
    if (args !== undefined && typeof args === 'object' && args !== null) {
      args = JSON.stringify(args)
    } else if (typeof args !== 'string') {
      args = '{}'
    }
    out.push({ id, type: 'function', function: { name, arguments: args as string } })
  }
  return out.length ? out : undefined
}

function normalizeAssistantMessage(msg: ApiMessage): ApiMessage {
  const tc = normalizeToolCalls(msg.tool_calls)
  if (tc) return { ...msg, tool_calls: tc }
  return msg
}

/** Strip <thinking> blocks before sending assistant turns back to the API. */
function stripAssistantForHistory(msg: ApiMessage): ApiMessage {
  if (msg.role !== 'assistant') return msg
  if (msg.content == null || typeof msg.content !== 'string') return msg
  const s = stripThinkingTags(msg.content)
  return { ...msg, content: s.length ? s : null }
}

interface StreamAcc {
  content: string
  /** Provider reasoning stream (o-series, Anthropic proxies, etc.) — not sent back as assistant content. */
  reasoning: string
  toolCalls: Map<number, { id: string; name: string; arguments: string }>
}

function createStreamAcc(): StreamAcc {
  return { content: '', reasoning: '', toolCalls: new Map() }
}

/** OpenAI-compatible text; some gateways send multimodal `content` arrays. */
function normalizeDeltaContent(delta: Record<string, unknown>): string {
  const c = delta.content
  if (typeof c === 'string') return c
  if (Array.isArray(c)) {
    return c
      .map((part: unknown) => {
        if (typeof part === 'string') return part
        if (part && typeof part === 'object' && part !== null) {
          const p = part as Record<string, unknown>
          if (typeof p.text === 'string') return p.text
        }
        return ''
      })
      .join('')
  }
  return ''
}

function pickReasoningChunk(delta: Record<string, unknown>): string {
  const v = delta.reasoning_content ?? delta.reasoning ?? delta.thinking
  return typeof v === 'string' && v.length > 0 ? v : ''
}

/** Named SSE `event:` lines that carry chain-of-thought (gateway-specific). */
function sseEventIsReasoning(ev: string): boolean {
  const e = ev.toLowerCase()
  return (
    e === 'reasoning' ||
    e === 'reasoning_delta' ||
    e === 'reasoning_summary' ||
    e === 'thinking' ||
    e === 'cot' ||
    e === 'chain_of_thought' ||
    e === 'reasoning_chunk'
  )
}

function extractSseReasoningPayload(obj: Record<string, unknown>): string {
  if (typeof obj.text === 'string' && obj.text.length > 0) return obj.text
  if (typeof obj.reasoning === 'string') return obj.reasoning
  if (typeof obj.thinking === 'string') return obj.thinking
  if (typeof obj.content === 'string') return obj.content
  if (typeof obj.delta === 'string') return obj.delta
  const choices = obj.choices as Array<Record<string, unknown>> | undefined
  const choice = choices?.[0]
  if (!choice) return ''
  const d = choice.delta as Record<string, unknown> | undefined
  if (d) {
    const r = pickReasoningChunk(d)
    if (r.length > 0) return r
    const c = normalizeDeltaContent(d)
    if (c.length > 0) return c
  }
  return ''
}

/**
 * Map one SSE `data:` JSON + optional `event:` name to an OpenAI-style delta for applyStreamDelta.
 */
function routeSseChunkToDelta(
  obj: Record<string, unknown>,
  meta: SseJsonMeta
): Record<string, unknown> | null {
  if (meta.event && sseEventIsReasoning(meta.event)) {
    const t = extractSseReasoningPayload(obj)
    if (t.length > 0) return { reasoning_content: t }
    return null
  }

  const typ = typeof obj.type === 'string' ? obj.type.toLowerCase() : ''
  if (typ === 'reasoning' || typ === 'thinking') {
    const t = extractSseReasoningPayload(obj)
    if (t.length > 0) return { reasoning_content: t }
  }

  const choices = obj.choices as Array<Record<string, unknown>> | undefined
  const choice = choices?.[0]
  if (!choice) return null
  let delta = choice.delta as Record<string, unknown> | undefined
  if (!delta || Object.keys(delta).length === 0) {
    const msg = choice.message as Record<string, unknown> | undefined
    if (msg && typeof msg.content === 'string' && msg.content.length > 0) {
      delta = { content: msg.content }
    }
  }
  if (delta && Object.keys(delta).length > 0) return delta
  return null
}

function applyStreamDelta(
  acc: StreamAcc,
  delta: Record<string, unknown>
): { contentDelta: string; reasoningDelta: string } {
  const out = { contentDelta: '', reasoningDelta: '' }
  const c = normalizeDeltaContent(delta)
  if (c.length > 0) {
    acc.content += c
    out.contentDelta = c
  }
  const r = pickReasoningChunk(delta)
  if (r.length > 0) {
    acc.reasoning += r
    out.reasoningDelta = r
  }
  if (!Array.isArray(delta.tool_calls)) return out
  for (const raw of delta.tool_calls) {
    if (!raw || typeof raw !== 'object') continue
    const t = raw as Record<string, unknown>
    const idx = Number(t.index ?? 0)
    if (!acc.toolCalls.has(idx)) {
      acc.toolCalls.set(idx, { id: '', name: '', arguments: '' })
    }
    const cur = acc.toolCalls.get(idx)!
    if (t.id) cur.id = String(t.id)
    const fn = t.function
    if (fn && typeof fn === 'object') {
      const f = fn as Record<string, unknown>
      if (f.name) cur.name = String(f.name)
      if (f.arguments !== undefined) cur.arguments += String(f.arguments)
    }
  }
  return out
}

function streamAccToMessage(acc: StreamAcc): ApiMessage {
  const tool_calls: ToolCallDef[] = []
  const indices = [...acc.toolCalls.keys()].sort((a, b) => a - b)
  for (const idx of indices) {
    const tc = acc.toolCalls.get(idx)!
    if (!tc.name) continue
    const id = tc.id || `call_${idx}`
    tool_calls.push({
      id,
      type: 'function',
      function: { name: tc.name, arguments: tc.arguments || '{}' },
    })
  }
  return {
    role: 'assistant',
    content: acc.content.length ? acc.content : null,
    ...(tool_calls.length ? { tool_calls } : {}),
  }
}

/** Compact live preview of streamed tool calls for the UI. */
function streamAccToToolPreview(acc: StreamAcc): string {
  const indices = [...acc.toolCalls.keys()].sort((a, b) => a - b)
  const parts: string[] = []
  for (const idx of indices) {
    const tc = acc.toolCalls.get(idx)!
    const name = tc.name || 'choosing_tool'
    const args = tc.arguments.trim()
    if (!args) {
      parts.push(`Planning ${name}…`)
      continue
    }
    const compactArgs = args.replace(/\s+/g, ' ').slice(0, 180)
    parts.push(`Planning ${name}(${compactArgs}${args.length > 180 ? '…' : ''})`)
  }
  return parts.join('\n\n')
}

async function fetchChatCompletionStreaming(
  endpoint: string,
  body: Record<string, unknown>,
  headers: HeadersInit,
  signal: AbortSignal | undefined,
  idleTimeoutMs: number,
  onDelta: (delta: Record<string, unknown>) => void
): Promise<void> {
  const streamHeaders = new Headers(headers)
  streamHeaders.set('Accept', 'text/event-stream')
  const res = await fetch(endpoint, {
    method: 'POST',
    signal,
    headers: streamHeaders,
    body: JSON.stringify({ ...body, stream: true }),
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`${res.status}: ${text}`)
  }
  await readSseJsonLines(res, {
    signal,
    idleTimeoutMs,
    onJson: (obj, meta) => {
      const delta = routeSseChunkToDelta(obj, meta)
      if (delta) onDelta(delta)
    },
  })
}

async function fetchChatCompletionJson(
  endpoint: string,
  body: Record<string, unknown>,
  headers: HeadersInit,
  signal: AbortSignal | undefined
): Promise<ApiMessage> {
  const res = await fetch(endpoint, {
    method: 'POST',
    signal,
    headers,
    body: JSON.stringify({ ...body, stream: false }),
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`${res.status}: ${text}`)
  }
  const json = (await res.json()) as {
    choices: Array<{ message: ApiMessage }>
  }
  const message = json.choices?.[0]?.message
  if (!message) throw new Error('Empty response from model')
  return normalizeAssistantMessage(message)
}

export async function runSceneAgent(
  userMessage: string,
  ctx: ToolContext,
  opts: AgentOptions,
  callbacks: {
    onMessage: (role: 'user' | 'assistant', content: string) => void
    onAssistantStreamBegin?: () => void
    /** Fires after each successful SSE stream completes (before tool execution). */
    onAssistantStreamEnd?: () => void
    onAssistantDelta?: (delta: string) => void
    /** Provider-native reasoning tokens (reasoning_content / reasoning / thinking). */
    onAssistantReasoningDelta?: (delta: string) => void
    /** Live partial tool-call preview during SSE when the model streams function arguments first. */
    onAssistantToolPreviewDelta?: (preview: string) => void
    onToolCall: (name: string, args: Record<string, unknown>, result: ToolResult) => void
    onStatus: (msg: string) => void
    onDone: () => void
    signal?: AbortSignal
  }
): Promise<void> {
  const {
    apiKey,
    model = 'gpt-4o',
    baseUrl = 'https://api.openai.com/v1',
    maxRounds = 12,
    reasoningEffort,
    sseIdleTimeoutMs = DEFAULT_SSE_IDLE_TIMEOUT_MS,
  } = opts
  const useReasoning = !!reasoningEffort && reasoningEffort !== 'none'

  const endpoint = `${normalizeBaseUrl(baseUrl)}/chat/completions`
  // tools is rebuilt each round so dynamically created tools are included
  const headers = {
    'Content-Type': 'application/json',
    Authorization: `Bearer ${apiKey}`,
  }

  const history: ApiMessage[] = [
    { role: 'system',  content: SYSTEM },
    { role: 'user',    content: userMessage },
  ]

  callbacks.onMessage('user', userMessage)

  async function completeOneRound(
    body: Record<string, unknown>,
    wantRequired: boolean,
    streamStep: number
  ): Promise<ApiMessage | null> {
    callbacks.onAssistantStreamBegin?.()
    callbacks.onStatus(`Streaming… (step ${streamStep})`)

    const runStream = async (b: Record<string, unknown>): Promise<ApiMessage> => {
      const acc = createStreamAcc()
      let lastToolPreview = ''
      await fetchChatCompletionStreaming(
        endpoint,
        b,
        headers,
        callbacks.signal,
        sseIdleTimeoutMs,
        (delta) => {
          const { contentDelta, reasoningDelta } = applyStreamDelta(acc, delta)
          if (contentDelta) callbacks.onAssistantDelta?.(contentDelta)
          if (reasoningDelta) callbacks.onAssistantReasoningDelta?.(reasoningDelta)
          const toolPreview = streamAccToToolPreview(acc)
          if (toolPreview && toolPreview !== lastToolPreview) {
            lastToolPreview = toolPreview
            callbacks.onAssistantToolPreviewDelta?.(toolPreview)
          }
        }
      )
      callbacks.onAssistantStreamEnd?.()
      if (callbacks.signal?.aborted) throw new Error('aborted')
      return normalizeAssistantMessage(streamAccToMessage(acc))
    }

    let reqBody: Record<string, unknown> = body
    try {
      return await runStream(reqBody)
    } catch (e1) {
      const err1 = String(e1)
      if (wantRequired && (err1.includes('400') || err1.includes('422'))) {
        callbacks.onStatus('Retrying stream without required tools…')
        reqBody = { ...body, tool_choice: 'auto' as const }
        callbacks.onAssistantStreamBegin?.()
        callbacks.onStatus(`Streaming… (step ${streamStep})`)
        try {
          return await runStream(reqBody)
        } catch (e2) {
          callbacks.onStatus(`Stream failed: ${String(e2)}. Using non-streaming…`)
        }
      } else {
        callbacks.onStatus(`Stream failed: ${err1}. Using non-streaming…`)
      }
      callbacks.onAssistantStreamBegin?.()
      callbacks.onStatus(`Streaming… (step ${streamStep}) · non-streaming`)
      const msg = await fetchChatCompletionJson(
        endpoint,
        { ...reqBody, stream: false },
        headers,
        callbacks.signal
      )
      const t = msg.content ?? ''
      if (t) callbacks.onAssistantDelta?.(t)
      callbacks.onAssistantStreamEnd?.()
      return normalizeAssistantMessage(msg)
    }
  }

  let forceRequiredNextRound = false

  for (let round = 0; round < maxRounds; round++) {
    // Rebuild tools every round — picks up any tools registered via create_tool
    const tools = getToolSchemas()
    if (callbacks.signal?.aborted) break

    const body: Record<string, unknown> = {
      model,
      // o-series models reject temperature; use reasoning_effort instead
      ...(useReasoning
        ? { reasoning_effort: reasoningEffort }
        : { temperature: 0.7 }),
      tools,
      messages: history,
    }
    const wantRequired = round === 0 || forceRequiredNextRound
    forceRequiredNextRound = false
    body.tool_choice = wantRequired ? 'required' : 'auto'

    let message: ApiMessage | null
    try {
      message = await completeOneRound(body, wantRequired, round + 1)
    } catch (e) {
      callbacks.onStatus(`API error: ${String(e)}`)
      break
    }
    if (callbacks.signal?.aborted || !message) break

    history.push(stripAssistantForHistory(message))

    const toolCalls = message.tool_calls as ToolCallDef[] | undefined
    const hasTools = Array.isArray(toolCalls) && toolCalls.length > 0

    if (!hasTools) {
      if (round === 0) {
        callbacks.onStatus(
          'No tool calls in response. Ensure the endpoint supports OpenAI-style tools + tool_calls over streaming.'
        )
      } else {
        callbacks.onStatus('Done.')
      }
      break
    }

    let createdToolThisRound = false
    const newlyCreatedToolNames: string[] = []

    for (const tc of toolCalls!) {
      const name = tc.function.name
      let args: Record<string, unknown> = {}
      try { args = JSON.parse(tc.function.arguments) as Record<string, unknown> } catch { /* ignore */ }

      callbacks.onStatus(`Calling ${name}…`)
      const result = dispatchTool(name, args, ctx)
      callbacks.onToolCall(name, args, result)

      if (name === 'create_tool' && result.ok) {
        createdToolThisRound = true
        newlyCreatedToolNames.push(String(args.tool_name ?? ''))
      }

      history.push({
        role: 'tool',
        tool_call_id: tc.id,
        name,
        content: JSON.stringify(result),
      })
    }

    // After a create_tool round, inject a nudge and force tool_choice: required
    // so the model immediately uses the newly registered tool instead of just narrating.
    if (createdToolThisRound && newlyCreatedToolNames.length > 0) {
      history.push({
        role: 'user',
        content: `The following tools are now registered and available: ${newlyCreatedToolNames.join(', ')}. Call them now to build the scene — do not describe what you will do, just call the tools.`,
      })
      forceRequiredNextRound = true
    }
  }

  callbacks.onDone()
}
