/**
 * Analytic expectation values for hydrogen eigenstates in atomic units
 * (ℏ = m_e = e = a₀ = 1).
 *
 * All formulae are standard results from Griffiths / Bethe–Salpeter.
 */

/** ⟨r⟩ = a₀ n² [ 3/2 - l(l+1)/(2n²) ]  (a₀ = 1) */
export function expectR(n: number, l: number): number {
  return (n * n * (3 / 2)) - (l * (l + 1)) / 2
}

/** ⟨r²⟩ = n⁴ a₀² [ 5n²+1 - 3l(l+1) ] / 2  (a₀ = 1) */
export function expectR2(n: number, l: number): number {
  return (n * n * n * n * (5 * n * n + 1 - 3 * l * (l + 1))) / 2
}

/** Δr = sqrt(⟨r²⟩ − ⟨r⟩²) */
export function deltaR(n: number, l: number): number {
  const r1 = expectR(n, l)
  const r2 = expectR2(n, l)
  return Math.sqrt(Math.max(0, r2 - r1 * r1))
}

/** ⟨L²⟩ = l(l+1) ℏ²  (ℏ = 1) */
export function expectL2(l: number): number {
  return l * (l + 1)
}

/** ⟨Lz⟩ = m ℏ  (ℏ = 1) */
export function expectLz(m: number): number {
  return m
}

/**
 * Number of degenerate states for principal quantum number n
 * (ignoring spin: n²; including spin: 2n²).
 */
export function degeneracy(n: number, includeSpin = true): number {
  return includeSpin ? 2 * n * n : n * n
}

/**
 * Electric-dipole selection rules for a transition (n,l,m) → (n2,l2,m2).
 * Returns 'allowed' | 'forbidden' | 'same'.
 */
export type TransitionType = 'same' | 'allowed' | 'forbidden'

export function selectionRule(
  l1: number, m1: number,
  l2: number, m2: number
): TransitionType {
  if (l1 === l2 && m1 === m2) return 'same'
  const dl = Math.abs(l2 - l1)
  const dm = Math.abs(m2 - m1)
  if (dl === 1 && dm <= 1) return 'allowed'
  return 'forbidden'
}

/**
 * Heisenberg radial uncertainty product ΔrΔp_r (lower bound ≥ ℏ/2 = 0.5).
 * Uses ⟨p_r²⟩ = (1/n²)[1 - l(l+1)/n²] / 1  (from virial / kinetic energy).
 * ⟨p_r⟩ = 0 for stationary states.
 * This is an approximation via the virial theorem: ⟨T⟩ = -E_n = 1/(2n²),
 * and ⟨p²⟩ = 2⟨T⟩ = 1/n².
 * We use Δp ≈ sqrt(⟨p²⟩) as an indicative quantity (not the exact radial spread).
 */
export function indicativeUncertaintyProduct(n: number, l: number): number {
  const dr = deltaR(n, l)
  const p2 = 1 / (n * n) // ⟨p²⟩ from virial
  const dp = Math.sqrt(p2)
  return dr * dp
}
