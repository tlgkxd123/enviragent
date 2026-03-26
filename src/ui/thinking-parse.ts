/** Markers for chain-of-thought (case-insensitive; optional spaces before `>`). */
// NOTE: Do NOT use module-level regex with lastIndex — without the `g` flag lastIndex is ignored
// and exec() always starts from 0, causing an infinite loop. Create fresh instances per call.
function makeOpenRe()  { return /<thinking\s*>/i }
function makeCloseRe() { return /<\/thinking\s*>/i }

function stripPartialOpenSuffix(s: string): string {
  const openLower = '<thinking>'
  for (let k = Math.min(s.length, openLower.length); k >= 1; k--) {
    const tail = s.slice(-k)
    if (tail[0] === '<' && openLower.startsWith(tail.toLowerCase())) {
      return s.slice(0, -k)
    }
  }
  return s
}

/** While streaming, strip incomplete `</thinking>` from the end of inner text. */
function stripPartialCloseSuffix(s: string): string {
  const closeLower = '</thinking>'
  for (let k = Math.min(s.length, closeLower.length); k >= 1; k--) {
    const tail = s.slice(-k)
    if (tail[0] === '<' && closeLower.startsWith(tail.toLowerCase())) {
      return s.slice(0, -k)
    }
  }
  return s
}

export interface ThinkingSplit {
  /** Text outside `<thinking>` blocks, safe to show as the main reply. */
  visible: string
  /** Concatenated inner text of all complete blocks + streaming tail inside an open block. */
  thinking: string
  /** True when `<thinking>` is open but `</thinking>` has not arrived yet. */
  inThinking: boolean
}

/**
 * Split streamed assistant text into visible vs thinking.
 * - Case-insensitive tags, optional spaces in `<thinking >`.
 * - As soon as the opening tag is complete, inner text streams into `thinking` until `</thinking>`.
 * - Strips chunk-split partial closing tags so CoT does not flash `</think`.
 */
export function parseThinkingDisplay(raw: string): ThinkingSplit {
  let visible = ''
  let thinking = ''
  let i = 0

  while (i < raw.length) {
    // Create a fresh regex each iteration — module-level regex without `g` ignores lastIndex
    const om = makeOpenRe().exec(raw.slice(i))
    if (!om) {
      visible += stripPartialOpenSuffix(raw.slice(i))
      break
    }
    visible += raw.slice(i, i + om.index)
    const innerStart = i + om.index + om[0].length

    const cm = makeCloseRe().exec(raw.slice(innerStart))

    if (!cm) {
      let inner = raw.slice(innerStart)
      inner = stripPartialCloseSuffix(inner)
      thinking += inner
      return { visible, thinking, inThinking: true }
    }

    const block = raw.slice(innerStart, innerStart + cm.index)
    thinking += (thinking ? '\n\n' : '') + block
    i = innerStart + cm.index + cm[0].length
  }

  return { visible, thinking, inThinking: false }
}

/** Remove all `<thinking>...</thinking>` blocks for chat/completions history (case-insensitive). */
export function stripThinkingTags(s: string): string {
  return s
    .replace(/<thinking\s*>[\s\S]*?<\/thinking\s*>/gi, '')
    .replace(/\n{3,}/g, '\n\n')
    .trim()
}
