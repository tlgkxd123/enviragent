import { getToolSchemas, dispatchTool } from './scene-tools'
import type { ToolContext, ToolResult } from './scene-tools'

/**
 * Agentic scene-building loop using OpenAI function-calling.
 *
 * Uses streaming (SSE) for chat completions: assistant text streams live;
 * tool_calls are accumulated from deltas, then executed.
 */

export interface AgentOptions {
  apiKey: string
  model?: string
  baseUrl?: string
  maxRounds?: number
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
After calling tools, briefly describe what you created in plain text. Be creative and cinematic.
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

interface StreamAcc {
  content: string
  toolCalls: Map<number, { id: string; name: string; arguments: string }>
}

function createStreamAcc(): StreamAcc {
  return { content: '', toolCalls: new Map() }
}

function applyStreamDelta(acc: StreamAcc, delta: Record<string, unknown>): void {
  if (typeof delta.content === 'string' && delta.content.length > 0) {
    acc.content += delta.content
  }
  if (!Array.isArray(delta.tool_calls)) return
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

async function readSseJsonLines(
  res: Response,
  onData: (obj: Record<string, unknown>) => void,
  signal?: AbortSignal
): Promise<void> {
  const reader = res.body?.getReader()
  if (!reader) throw new Error('Response has no readable body')
  const dec = new TextDecoder()
  let buf = ''
  try {
    while (true) {
      if (signal?.aborted) {
        await reader.cancel()
        return
      }
      const { done, value } = await reader.read()
      if (done) break
      buf += dec.decode(value, { stream: true })
      let nl = buf.indexOf('\n')
      while (nl >= 0) {
        const line = buf.slice(0, nl).replace(/\r$/, '').trimEnd()
        buf = buf.slice(nl + 1)
        nl = buf.indexOf('\n')
        if (!line.startsWith('data:')) continue
        const data = line.slice(5).trim()
        if (data === '[DONE]') continue
        try {
          onData(JSON.parse(data) as Record<string, unknown>)
        } catch {
          /* ignore */
        }
      }
    }
    buf += dec.decode()
    const tail = buf.replace(/\r$/, '').trimEnd()
    if (tail.startsWith('data:')) {
      const data = tail.slice(5).trim()
      if (data && data !== '[DONE]') {
        try {
          onData(JSON.parse(data) as Record<string, unknown>)
        } catch {
          /* ignore */
        }
      }
    }
  } finally {
    /* empty */
  }
}

async function fetchChatCompletionStreaming(
  endpoint: string,
  body: Record<string, unknown>,
  headers: HeadersInit,
  signal: AbortSignal | undefined,
  onDelta: (delta: Record<string, unknown>) => void
): Promise<void> {
  const res = await fetch(endpoint, {
    method: 'POST',
    signal,
    headers,
    body: JSON.stringify({ ...body, stream: true }),
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`${res.status}: ${text}`)
  }
  await readSseJsonLines(
    res,
    (obj) => {
      const choices = obj.choices as Array<Record<string, unknown>> | undefined
      const choice = choices?.[0]
      if (!choice) return
      const delta = choice.delta as Record<string, unknown> | undefined
      if (delta && Object.keys(delta).length > 0) onDelta(delta)
    },
    signal
  )
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
    onAssistantDelta?: (delta: string) => void
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
  } = opts

  const endpoint = `${normalizeBaseUrl(baseUrl)}/chat/completions`
  const tools = getToolSchemas()
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
    wantRequired: boolean
  ): Promise<ApiMessage | null> {
    callbacks.onAssistantStreamBegin?.()

    const runStream = async (b: Record<string, unknown>): Promise<ApiMessage> => {
      const acc = createStreamAcc()
      await fetchChatCompletionStreaming(endpoint, b, headers, callbacks.signal, (delta) => {
        if (typeof delta.content === 'string' && delta.content.length > 0) {
          callbacks.onAssistantDelta?.(delta.content)
        }
        applyStreamDelta(acc, delta)
      })
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
        try {
          return await runStream(reqBody)
        } catch (e2) {
          callbacks.onStatus(`Stream failed: ${String(e2)}. Using non-streaming…`)
        }
      } else {
        callbacks.onStatus(`Stream failed: ${err1}. Using non-streaming…`)
      }
      callbacks.onAssistantStreamBegin?.()
      const msg = await fetchChatCompletionJson(
        endpoint,
        { ...reqBody, stream: false },
        headers,
        callbacks.signal
      )
      const t = msg.content ?? ''
      if (t) callbacks.onAssistantDelta?.(t)
      return normalizeAssistantMessage(msg)
    }
  }

  for (let round = 0; round < maxRounds; round++) {
    if (callbacks.signal?.aborted) break

    callbacks.onStatus(`Streaming… (step ${round + 1})`)

    const body: Record<string, unknown> = {
      model,
      temperature: 0.7,
      tools,
      messages: history,
    }
    const wantRequired = round === 0
    body.tool_choice = wantRequired ? 'required' : 'auto'

    let message: ApiMessage | null
    try {
      message = await completeOneRound(body, wantRequired)
    } catch (e) {
      callbacks.onStatus(`API error: ${String(e)}`)
      break
    }
    if (callbacks.signal?.aborted || !message) break

    history.push(message)

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

    for (const tc of toolCalls!) {
      const name = tc.function.name
      let args: Record<string, unknown> = {}
      try { args = JSON.parse(tc.function.arguments) as Record<string, unknown> } catch { /* ignore */ }

      callbacks.onStatus(`Calling ${name}…`)
      const result = dispatchTool(name, args, ctx)
      callbacks.onToolCall(name, args, result)

      history.push({
        role: 'tool',
        tool_call_id: tc.id,
        name,
        content: JSON.stringify(result),
      })
    }
  }

  callbacks.onDone()
}
