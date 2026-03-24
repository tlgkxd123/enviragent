/**
 * Analytic expectation values, physical constants, and derived observables
 * for hydrogen eigenstates.
 *
 * Primary basis: atomic units (ℏ = m_e = e = a₀ = 1).
 * SI conversions use CODATA 2022 recommended values.
 *
 * References:
 *   Griffiths, "Introduction to Quantum Mechanics" (3rd ed.)
 *   Bethe & Salpeter, "Quantum Mechanics of One- and Two-Electron Atoms"
 *   NIST Atomic Spectra Database
 *   Drake (ed.), "Springer Handbook of Atomic, Molecular and Optical Physics"
 */

// ── Physical constants (SI) ───────────────────────────────────────────────────

export const CONST = {
  /** Speed of light, m/s */
  c:      2.99792458e8,
  /** Planck constant, J·s */
  h:      6.62607015e-34,
  /** Reduced Planck constant, J·s */
  hbar:   1.054571817e-34,
  /** Elementary charge, C */
  e:      1.602176634e-19,
  /** Electron mass, kg */
  me:     9.1093837015e-31,
  /** Bohr radius, m */
  a0:     5.29177210903e-11,
  /** Hartree energy, J (= 1 a.u. of energy) */
  Eh:     4.3597447222071e-18,
  /** Hartree energy, eV */
  Eh_eV:  27.211386245988,
  /** Fine-structure constant α */
  alpha:  7.2973525693e-3,
  /** Bohr magneton, J/T */
  muB:    9.2740100783e-24,
  /** Nuclear magneton, J/T */
  muN:    5.0507837461e-27,
  /** Proton g-factor */
  gp:     5.5856946893,
  /** Rydberg constant, m⁻¹ */
  Rinf:   1.0973731568160e7,
  /** Rydberg energy, eV */
  Ry_eV:  13.605693122994,
  /** Boltzmann constant, J/K */
  kB:     1.380649e-23,
  /** Avogadro constant */
  NA:     6.02214076e23,
  /** Vacuum permittivity, F/m */
  eps0:   8.8541878128e-12,
} as const

// ── Atomic unit conversions ───────────────────────────────────────────────────

/** Convert atomic unit energy to eV */
export function auToEv(au: number): number { return au * CONST.Eh_eV }
/** Convert atomic unit length (a₀) to Ångström */
export function auToAngstrom(au: number): number { return au * 0.529177210903 }
/** Convert atomic unit length to nm */
export function auToNm(au: number): number { return au * 0.0529177210903 }
/** Convert energy difference in a.u. to photon wavelength in nm */
export function deltaEToWavelengthNm(deltaE_au: number): number {
  if (Math.abs(deltaE_au) < 1e-12) return Infinity
  // λ = h·c / ΔE  in SI, then convert to nm
  const deltaE_J = Math.abs(deltaE_au) * CONST.Eh
  return (CONST.h * CONST.c) / deltaE_J * 1e9
}
/** Convert energy difference in a.u. to frequency in Hz */
export function deltaEToFreqHz(deltaE_au: number): number {
  return Math.abs(deltaE_au) * CONST.Eh / CONST.hbar
}
/** Convert wavelength in nm to approximate RGB hex color (CIE approximation) */
export function wavelengthToHex(nm: number): string {
  if (nm < 380 || nm > 780) return '#888888'
  let r = 0, g = 0, b = 0
  if (nm >= 380 && nm < 440) { r = -(nm - 440) / 60; g = 0; b = 1 }
  else if (nm >= 440 && nm < 490) { r = 0; g = (nm - 440) / 50; b = 1 }
  else if (nm >= 490 && nm < 510) { r = 0; g = 1; b = -(nm - 510) / 20 }
  else if (nm >= 510 && nm < 580) { r = (nm - 510) / 70; g = 1; b = 0 }
  else if (nm >= 580 && nm < 645) { r = 1; g = -(nm - 645) / 65; b = 0 }
  else { r = 1; g = 0; b = 0 }
  // Intensity falloff at edges
  let factor = 1.0
  if (nm >= 380 && nm < 420) factor = 0.3 + 0.7 * (nm - 380) / 40
  else if (nm >= 700 && nm <= 780) factor = 0.3 + 0.7 * (780 - nm) / 80
  const R = Math.round(255 * Math.pow(Math.max(0, r) * factor, 0.8))
  const G = Math.round(255 * Math.pow(Math.max(0, g) * factor, 0.8))
  const B = Math.round(255 * Math.pow(Math.max(0, b) * factor, 0.8))
  return `#${R.toString(16).padStart(2,'0')}${G.toString(16).padStart(2,'0')}${B.toString(16).padStart(2,'0')}`
}

// ── Energy levels ─────────────────────────────────────────────────────────────

/** Bohr energy in atomic units: E_n = -1/(2n²) */
export function energyBohr(n: number): number {
  return -1 / (2 * n * n)
}

/**
 * Fine-structure energy correction (Dirac formula to order α²).
 * ΔE_fs = (α²/2n⁴) · [ n/(j+½) - 3/4 ]  in a.u.,  j = l ± ½
 * For the readout we use j = l + ½ (upper fine-structure level).
 */
export function energyFineStructure(n: number, _l: number, j: number): number {
  const alpha = CONST.alpha
  return (alpha * alpha / (2 * n * n * n * n)) * (n / (j + 0.5) - 0.75)
}

/**
 * Full energy including fine structure (leading relativistic + spin-orbit).
 * Uses Dirac result to O(α⁴): E_nj = -1/(2n²) · [1 + α²/n² · (n/(j+½) - 3/4)]
 */
export function energyDirac(n: number, _l: number, j: number): number {
  const alpha = CONST.alpha
  const En = -1 / (2 * n * n)
  const correction = 1 + (alpha * alpha / (n * n)) * (n / (j + 0.5) - 0.75)
  return En * correction
}

/**
 * Lamb shift (leading QED correction) in a.u.
 * Dominant term for s-states: ΔE_Lamb ≈ (α³/πn³) · [ ln(1/α²) + C_l ]
 * For l > 0 states the Lamb shift is much smaller (included approximately).
 * Bethe log C_s ≈ 2.984 for s-states.
 */
export function energyLambShift(n: number, l: number): number {
  const alpha = CONST.alpha
  if (l === 0) {
    // Bethe logarithm approximation
    const betheLog = Math.log(1 / (alpha * alpha)) + 2.984
    return (alpha * alpha * alpha) / (Math.PI * n * n * n) * betheLog
  }
  // l>0: smaller correction, approximate
  return (alpha * alpha * alpha) / (Math.PI * n * n * n * (l + 0.5) * (l + 1)) * 0.1
}

/**
 * Hyperfine splitting — hydrogen ground-state HFS frequency ≈ 1420.405 MHz
 * For state (n,l): ΔE_hfs ∝ g_p · α² · (m_e/m_p) / (n³ · l(l+½)(l+1))  [l>0]
 * For l=0: ΔE_hfs = (g_p · α² · (m_e/m_p)) / (3 n³)   (Fermi contact term)
 * Returns splitting in Hz (more intuitive for HFS).
 */
export function hyperfineFreqHz(n: number, l: number): number {
  // Ground state reference: 1420.405751767 MHz
  const f0 = 1420.405751767e6 // Hz
  if (l === 0) {
    // Scales as 1/n³
    return f0 / (n * n * n)
  }
  // Non-s states: much smaller, scales as 1/[n³ l(l+½)(l+1)]
  return f0 / (n * n * n * l * (l + 0.5) * (l + 1)) * 0.33
}

// ── Expectation values ────────────────────────────────────────────────────────

/** ⟨r⟩ = a₀ n² [ 3/2 - l(l+1)/(2n²) ]  (a₀ = 1) */
export function expectR(n: number, l: number): number {
  return (n * n * (3 / 2)) - (l * (l + 1)) / 2
}

/** ⟨r²⟩ = n⁴ a₀² [ 5n²+1 - 3l(l+1) ] / 2  (a₀ = 1) */
export function expectR2(n: number, l: number): number {
  return (n * n * n * n * (5 * n * n + 1 - 3 * l * (l + 1))) / 2
}

/** ⟨1/r⟩ = 1/n²  (a₀ = 1) */
export function expectInvR(n: number): number {
  return 1 / (n * n)
}

/** ⟨1/r²⟩ = 1/[n³(l+½)]  (a₀ = 1) */
export function expectInvR2(n: number, l: number): number {
  return 1 / (n * n * n * (l + 0.5))
}

/** ⟨1/r³⟩ = 1/[n³ l(l+½)(l+1)]  for l > 0  (a₀ = 1) */
export function expectInvR3(n: number, l: number): number {
  if (l === 0) return Infinity // diverges for s-states
  return 1 / (n * n * n * l * (l + 0.5) * (l + 1))
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

/** ⟨T⟩ = 1/(2n²) (virial theorem: ⟨T⟩ = -E_n) in a.u. */
export function expectKineticEnergy(n: number): number {
  return 1 / (2 * n * n)
}

/** ⟨V⟩ = -1/n² (virial theorem: ⟨V⟩ = 2E_n) in a.u. */
export function expectPotentialEnergy(n: number): number {
  return -1 / (n * n)
}

/** ⟨p²⟩ = 2⟨T⟩ = 1/n² in a.u. */
export function expectP2(n: number): number {
  return 1 / (n * n)
}

/** RMS momentum √⟨p²⟩ in a.u. */
export function rmsP(n: number): number {
  return 1 / n
}

/** Electron speed as fraction of c: v/c = α/n */
export function electronSpeedOverC(n: number): number {
  return CONST.alpha / n
}

/**
 * Relativistic kinetic energy correction ⟨p⁴⟩/(8m³c²) in a.u.
 * ΔE_rel = -(α²/8n⁴)(4n/(l+½) - 3)  (Darwin + mass-velocity combined)
 */
export function relativisticCorrection(n: number, l: number): number {
  const alpha = CONST.alpha
  return -(alpha * alpha / (8 * n * n * n * n)) * (4 * n / (l + 0.5) - 3)
}

/**
 * Spin-orbit coupling energy ⟨S·L⟩ contribution in a.u.
 * ΔE_SO = (α²/4n³) · [j(j+1) - l(l+1) - s(s+1)] / [l(l+½)(l+1)]
 * For l=0: zero (no spin-orbit for s-states, Darwin term handles it)
 */
export function spinOrbitEnergy(n: number, l: number, j: number): number {
  if (l === 0) return 0
  const alpha = CONST.alpha
  const s = 0.5
  const slDot = 0.5 * (j * (j + 1) - l * (l + 1) - s * (s + 1))
  return (alpha * alpha / (4 * n * n * n)) * slDot / (l * (l + 0.5) * (l + 1))
}

// ── Quantum numbers & degeneracy ──────────────────────────────────────────────

/**
 * Number of degenerate states for principal quantum number n
 * (ignoring spin: n²; including spin: 2n²).
 */
export function degeneracy(n: number, includeSpin = true): number {
  return includeSpin ? 2 * n * n : n * n
}

/** Number of radial nodes = n - l - 1 */
export function radialNodes(n: number, l: number): number {
  return n - l - 1
}

/** Number of angular nodes = l */
export function angularNodes(l: number): number {
  return l
}

/** Total nodes = n - 1 */
export function totalNodes(n: number): number {
  return n - 1
}

// ── Transition properties ─────────────────────────────────────────────────────

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
 * Photon wavelength for transition n1→n2 in nm (Rydberg formula).
 * 1/λ = R∞ (1/n2² - 1/n1²)
 */
export function transitionWavelengthNm(n1: number, n2: number): number {
  if (n1 === n2) return Infinity
  const deltaE = Math.abs(1 / (2 * n2 * n2) - 1 / (2 * n1 * n1))
  return deltaEToWavelengthNm(deltaE)
}

/**
 * Transition series name.
 */
export function seriesName(n_lower: number): string {
  switch (n_lower) {
    case 1: return 'Lyman'
    case 2: return 'Balmer'
    case 3: return 'Paschen'
    case 4: return 'Brackett'
    case 5: return 'Pfund'
    default: return `n=${n_lower}`
  }
}

/**
 * Einstein A coefficient (spontaneous emission rate) for hydrogen n,l → n',l'
 * A ≈ (4α³ ω³ |⟨r⟩|²) / (3c²)  in a.u. → s⁻¹
 * We use the approximate hydrogenic oscillator strength formula.
 * Reference: Bethe & Salpeter, table of hydrogenic transition rates.
 */
export function einsteinA_s(n1: number, l1: number, n2: number, l2: number): number {
  if (n1 <= n2) return 0 // emission: n1 > n2
  const rule = selectionRule(l1, 0, l2, 0)
  if (rule === 'forbidden') return 0
  const deltaE_au = Math.abs(1 / (2 * n2 * n2) - 1 / (2 * n1 * n1))
  const omega_au = deltaE_au // ω = ΔE/ℏ, ℏ=1 in a.u.
  // Approximate |⟨r⟩|² from hydrogenic matrix element (simplified)
  const rMatSq = Math.pow((6 * n1 * n2) / ((n1 * n1 - n2 * n2) * (n1 + n2)), 2) * n1 * n2
  const alpha = CONST.alpha
  const A_au = (4.0 / 3.0) * alpha * alpha * alpha * omega_au * omega_au * omega_au * rMatSq
  // Convert from a.u. time (ℏ/Eh) to seconds
  const tau_au = CONST.hbar / CONST.Eh
  return A_au / tau_au
}

/**
 * Natural linewidth (FWHM) from Einstein A coefficient, in Hz.
 * Γ = A / (2π)
 */
export function naturalLinewidthHz(A: number): number {
  return A / (2 * Math.PI)
}

// ── Zeeman effect ─────────────────────────────────────────────────────────────

/**
 * Normal Zeeman splitting energy in a.u. for field B (in Tesla).
 * ΔE = μ_B · B · m_l / E_h
 * Returns energy shift per unit m_l.
 */
export function zeemanShiftPerM(B_tesla: number): number {
  return CONST.muB * B_tesla / CONST.Eh
}

/**
 * Anomalous Zeeman: total shift including spin (m_j = m_l + m_s, g_j Landé factor).
 * g_J = 1 + [j(j+1) + s(s+1) - l(l+1)] / [2j(j+1)]
 */
export function landeG(l: number, j: number, s = 0.5): number {
  if (j === 0) return 0
  return 1 + (j * (j + 1) + s * (s + 1) - l * (l + 1)) / (2 * j * (j + 1))
}

// ── Heisenberg uncertainty ────────────────────────────────────────────────────

/**
 * Heisenberg radial uncertainty product ΔrΔp_r (lower bound ≥ ℏ/2 = 0.5).
 */
export function indicativeUncertaintyProduct(n: number, l: number): number {
  const dr = deltaR(n, l)
  const p2 = 1 / (n * n)
  const dp = Math.sqrt(p2)
  return dr * dp
}

/**
 * Position-momentum uncertainty ratio to the Heisenberg minimum (ℏ/2).
 * Values > 1 mean the state is more spread than the minimum uncertainty state.
 */
export function uncertaintyRatio(n: number, l: number): number {
  return indicativeUncertaintyProduct(n, l) / 0.5
}

// ── Lifetime & decay ─────────────────────────────────────────────────────────

/**
 * Spontaneous emission lifetime τ of state (n,l) in seconds.
 * τ = 1 / Σ_lower A(n,l → n',l')
 * Sums over all lower n' with Δl=±1.
 */
export function statLifetimeSec(n: number, l: number): number {
  let totalA = 0
  for (let np = 1; np < n; np++) {
    if (l > 0) totalA += einsteinA_s(n, l, np, l - 1)
    if (l < n - 1) totalA += einsteinA_s(n, l, np, l + 1)
  }
  if (totalA <= 0) return Infinity // ground state or forbidden
  return 1 / totalA
}

// ── Orbital geometry ─────────────────────────────────────────────────────────

/**
 * Most probable radius r_mp for hydrogen state (n,l).
 * For l = n-1 (circular orbits): r_mp = n²  (in a₀)
 * For general states: solve d(r²|R_nl|²)/dr = 0  — approximate here as
 * the outermost maximum of the radial probability density.
 * Exact analytic result: n² [1 + √(1 - l(l+1)/n²)] / 2  (outer lobe)
 */
export function mostProbableRadius(n: number, l: number): number {
  const disc = 1 - (l * (l + 1)) / (n * n)
  if (disc < 0) return n * n
  return (n * n * (1 + Math.sqrt(Math.max(0, disc)))) / 2
}

/**
 * Ionization energy from state n in eV.
 * E_ion = -E_n = 1/(2n²) in a.u. → eV
 */
export function ionizationEnergyEv(n: number): number {
  return auToEv(1 / (2 * n * n))
}

/**
 * Bohr orbital velocity for circular orbit n (v = α/n · c).
 * Returns v/c.
 */
export function bohrVelocityOverC(n: number): number {
  return CONST.alpha / n
}