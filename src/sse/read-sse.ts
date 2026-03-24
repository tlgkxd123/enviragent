/**
 * Browser-side SSE (Server-Sent Events) reader for POST streaming APIs
 * (e.g. OpenAI chat.completions with stream: true).
 *
 * Parses `event:` + `data:` frames (named events for CoT / reasoning streams),
 * comment lines (`:` …), heartbeats, and `[DONE]`.
 */

export interface SseJsonMeta {
  /** Last `event:` name before this `data:` line (null if none). */
  event: string | null
}

export interface ReadSseJsonOptions {
  signal?: AbortSignal
  idleTimeoutMs: number
  /**
   * Each parsed `data:` JSON object.
   * `meta.event` is set when the server sent `event: name` before this `data:` line.
   */
  onJson: (obj: Record<string, unknown>, meta: SseJsonMeta) => void
}

function idleError(idleMs: number): Error {
  const s = Math.max(1, Math.round(idleMs / 1000))
  return new Error(`SSE stream timed out (no data for ${s}s)`)
}

type ParsedLine =
  | { kind: 'event'; name: string }
  | { kind: 'data'; payload: string }
  | { kind: 'skip' }

function parseSseLine(line: string): ParsedLine {
  const t = line.replace(/\r$/, '').trimEnd()
  if (t === '' || t.startsWith(':')) return { kind: 'skip' }
  const ev = /^event:\s*(.*)$/i.exec(t)
  if (ev) return { kind: 'event', name: ev[1].trim() }
  if (t.startsWith('data:')) {
    return { kind: 'data', payload: t.slice(5).trimStart().trim() }
  }
  return { kind: 'skip' }
}

function dispatchDataLine(
  payload: string,
  lastEvent: string | null,
  onJson: (obj: Record<string, unknown>, meta: SseJsonMeta) => void
): void {
  if (payload === '[DONE]') return
  try {
    const obj = JSON.parse(payload) as Record<string, unknown>
    onJson(obj, { event: lastEvent })
  } catch {
    /* ignore malformed chunk */
  }
}

/**
 * Consume a fetch `Response` whose body is `text/event-stream` (newline-delimited SSE).
 */
export async function readSseJsonLines(
  res: Response,
  options: ReadSseJsonOptions
): Promise<void> {
  const { signal, idleTimeoutMs, onJson } = options
  const reader = res.body?.getReader()
  if (!reader) throw new Error('Response has no readable body')

  const dec = new TextDecoder()
  let buf = ''
  let lastEvent: string | null = null

  try {
    while (true) {
      if (signal?.aborted) {
        await reader.cancel()
        return
      }
      let idleTimer: ReturnType<typeof setTimeout> | undefined
      const chunkPromise = reader.read()
      const timeoutPromise = new Promise<never>((_, reject) => {
        idleTimer = setTimeout(() => reject(idleError(idleTimeoutMs)), idleTimeoutMs)
      })
      let done: boolean
      let value: Uint8Array | undefined
      try {
        const result = await Promise.race([chunkPromise, timeoutPromise])
        done = result.done
        value = result.value
      } finally {
        if (idleTimer !== undefined) clearTimeout(idleTimer)
      }
      if (done) break
      buf += dec.decode(value, { stream: true })
      let nl = buf.indexOf('\n')
      while (nl >= 0) {
        const line = buf.slice(0, nl)
        buf = buf.slice(nl + 1)
        nl = buf.indexOf('\n')
        const parsed = parseSseLine(line)
        if (parsed.kind === 'event') {
          lastEvent = parsed.name
          continue
        }
        if (parsed.kind === 'data') {
          dispatchDataLine(parsed.payload, lastEvent, onJson)
        }
      }
    }
    buf += dec.decode()
    const tail = buf.replace(/\r$/, '').trimEnd()
    if (tail.length > 0) {
      const parsed = parseSseLine(tail)
      if (parsed.kind === 'data') {
        dispatchDataLine(parsed.payload, lastEvent, onJson)
      }
    }
  } finally {
    reader.cancel().catch(() => { /* ignore */ })
  }
}
