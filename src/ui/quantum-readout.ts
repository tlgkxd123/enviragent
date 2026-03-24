import {
  deltaR,
  degeneracy,
  expectL2,
  expectLz,
  expectR,
  indicativeUncertaintyProduct,
  selectionRule,
  type TransitionType,
} from '../physics/expectation'
import type { OrbitalParams } from '../physics/state'

const shellLetter = ['s', 'p', 'd', 'f', 'g', 'h']

function fmt(x: number, dp = 3): string {
  return x.toFixed(dp)
}

function transitionLabel(t: TransitionType): string {
  if (t === 'same') return ''
  if (t === 'allowed') return '<span class="qr-allowed">E1 allowed</span>'
  return '<span class="qr-forbidden">E1 forbidden (\u0394l\u22601)</span>'
}

function setText(id: string, text: string) {
  const el = document.getElementById(id)
  if (el) el.textContent = text
}

function setHtml(id: string, html: string) {
  const el = document.getElementById(id)
  if (el) el.innerHTML = html
}

export function updateQuantumReadout(
  primary: OrbitalParams,
  secondary: OrbitalParams | null,
  superpositionActive: boolean
): void {
  const { n, l, m } = primary

  // Quantum number labels
  const letter = shellLetter[l] ?? `l${l}`
  setText('qr-state', `|${n}, ${l}, ${m}\u27E9  (${n}${letter})`)
  setText('qr-qnums', `n=${n}  \u2113=${l}  m\u2097=${m}`)

  // Energy
  const E = -1 / (2 * n * n)
  setText('qr-energy', `E${n} = \u22121/(2n\u00B2) = ${fmt(E, 5)} a.u.`)

  // Angular momentum
  const L2 = expectL2(l)
  const Lz = expectLz(m)
  setText('qr-L2', `\u27E8L\u00B2\u27E9 = \u2113(\u2113+1)\u210F\u00B2 = ${fmt(L2, 3)} a.u.`)
  setText('qr-Lz', `\u27E8Lz\u27E9 = m\u210F = ${Lz >= 0 ? '+' : ''}${fmt(Lz, 0)} a.u.`)

  // Position expectation & uncertainty
  const rExp = expectR(n, l)
  const dr = deltaR(n, l)
  setText('qr-r', `\u27E8r\u27E9 = ${fmt(rExp, 3)} a\u2080`)
  setText('qr-dr', `\u0394r = ${fmt(dr, 3)} a\u2080`)

  // Heisenberg
  const hup = indicativeUncertaintyProduct(n, l)
  const hupStr = `\u0394r\u00B7|p| \u2248 ${fmt(hup, 3)} \u210F  (min = 0.5\u210F)`
  setText('qr-hup', hupStr)

  // Degeneracy
  const degen = degeneracy(n, false)
  const degenSpin = degeneracy(n, true)
  setText('qr-degen', `g = n\u00B2 = ${degen}  (with spin: ${degenSpin})`)

  // Selection rules (only if superposition is active)
  const transEl = document.getElementById('qr-transition-row')
  if (transEl) {
    if (superpositionActive && secondary) {
      const t = selectionRule(l, m, secondary.l, secondary.m)
      const secLetter = shellLetter[secondary.l] ?? `l${secondary.l}`
      const label = `|${secondary.n}, ${secondary.l}, ${secondary.m}\u27E9 (${secondary.n}${secLetter})`
      setHtml('qr-transition', `\u2192 ${label}: ${transitionLabel(t)}`)
      transEl.style.display = ''
    } else {
      transEl.style.display = 'none'
    }
  }
}
