import {
  CONST,
  auToEv,
  auToAngstrom,
  deltaEToFreqHz,
  wavelengthToHex,
  energyBohr,
  energyFineStructure,
  energyLambShift,
  hyperfineFreqHz,
  expectR,
  expectR2,
  expectInvR,
  expectInvR2,
  expectKineticEnergy,
  expectPotentialEnergy,
  expectL2,
  expectLz,
  deltaR,
  indicativeUncertaintyProduct,
  uncertaintyRatio,
  degeneracy,
  radialNodes,
  angularNodes,
  selectionRule,
  transitionWavelengthNm,
  seriesName,
  einsteinA_s,
  naturalLinewidthHz,
  statLifetimeSec,
  mostProbableRadius,
  ionizationEnergyEv,
  bohrVelocityOverC,
  landeG,
  zeemanShiftPerM,
  relativisticCorrection,
  spinOrbitEnergy,
  type TransitionType,
} from '../physics/expectation'
import type { OrbitalParams } from '../physics/state'

const shellLetter = ['s', 'p', 'd', 'f', 'g', 'h']

function fmt(x: number, dp = 3): string {
  if (!isFinite(x)) return '∞'
  if (Math.abs(x) < 1e-15) return '0'
  return x.toFixed(dp)
}

function fmtSci(x: number, dp = 3): string {
  if (!isFinite(x)) return '∞'
  if (Math.abs(x) < 1e-15) return '0'
  return x.toExponential(dp)
}

function fmtFreq(hz: number): string {
  if (!isFinite(hz)) return '∞'
  if (hz >= 1e12) return `${(hz / 1e12).toFixed(3)} THz`
  if (hz >= 1e9)  return `${(hz / 1e9).toFixed(3)} GHz`
  if (hz >= 1e6)  return `${(hz / 1e6).toFixed(3)} MHz`
  if (hz >= 1e3)  return `${(hz / 1e3).toFixed(3)} kHz`
  return `${hz.toFixed(1)} Hz`
}

function fmtTime(sec: number): string {
  if (!isFinite(sec)) return '∞ (stable)'
  if (sec >= 1)       return `${sec.toExponential(2)} s`
  if (sec >= 1e-3)    return `${(sec * 1e3).toFixed(3)} ms`
  if (sec >= 1e-6)    return `${(sec * 1e6).toFixed(3)} μs`
  if (sec >= 1e-9)    return `${(sec * 1e9).toFixed(3)} ns`
  if (sec >= 1e-12)   return `${(sec * 1e12).toFixed(3)} ps`
  return `${(sec * 1e15).toFixed(3)} fs`
}

function transitionLabel(t: TransitionType): string {
  if (t === 'same') return ''
  if (t === 'allowed') return '<span class="qr-allowed">E1 allowed</span>'
  return '<span class="qr-forbidden">E1 forbidden (Δℓ≠±1)</span>'
}

function setText(id: string, text: string) {
  const el = document.getElementById(id)
  if (el) el.textContent = text
}

function setHtml(id: string, html: string) {
  const el = document.getElementById(id)
  if (el) el.innerHTML = html
}

function setStyle(id: string, prop: string, val: string) {
  const el = document.getElementById(id) as HTMLElement | null
  if (el) (el.style as unknown as Record<string, string>)[prop] = val
}

export function updateQuantumReadout(
  primary: OrbitalParams,
  secondary: OrbitalParams | null,
  superpositionActive: boolean
): void {
  const { n, l, m } = primary
  const j_upper = l + 0.5  // j = l + s, upper fine-structure level
  // ── State identity ──────────────────────────────────────────────────────────
  const letter = shellLetter[l] ?? `l${l}`
  setText('qr-state', `|n=${n}, ℓ=${l}, m=${m}⟩  (${n}${letter})`)
  setText('qr-qnums', `n=${n}  ℓ=${l}  mℓ=${m}  ms=±½  j=${fmt(j_upper,1)}`)

  // ── Nodes ───────────────────────────────────────────────────────────────────
  setText('qr-nodes', `radial=${radialNodes(n,l)}  angular=${angularNodes(l)}  total=${n-1}`)

  // ── Energy (multi-level) ─────────────────────────────────────────────────────
  const E_bohr_au = energyBohr(n)
  const E_bohr_eV = auToEv(E_bohr_au)
  const E_fs_au   = energyFineStructure(n, l, j_upper)
  const E_lamb_au = energyLambShift(n, l)
  const E_total_eV = auToEv(E_bohr_au + E_fs_au + E_lamb_au)
  setText('qr-energy',    `E_n = ${fmt(E_bohr_eV, 6)} eV  (${fmt(E_bohr_au, 6)} a.u.)`)
  setText('qr-energy-fs', `+FS: ${fmtSci(auToEv(E_fs_au),3)} eV  +Lamb: ${fmtSci(auToEv(E_lamb_au),3)} eV`)
  setText('qr-energy-total', `E_total = ${fmt(E_total_eV, 6)} eV`)
  setText('qr-ionization', `E_ion = ${fmt(ionizationEnergyEv(n), 4)} eV`)
  const relCorr_eV = auToEv(relativisticCorrection(n, l))
  const soCorr_eV  = auToEv(spinOrbitEnergy(n, l, j_upper))
  setText('qr-rel', `Δrel = ${fmtSci(relCorr_eV,3)} eV   ΔSO = ${fmtSci(soCorr_eV,3)} eV`)

  // ── Velocity & relativity ───────────────────────────────────────────────────
  const vOverC = bohrVelocityOverC(n)
  const gamma  = 1 / Math.sqrt(1 - vOverC * vOverC)
  setText('qr-velocity', `v/c = α/n = ${fmtSci(vOverC,4)}   γ = ${fmt(gamma,8)}`)

  // ── Angular momentum ────────────────────────────────────────────────────────
  const L2 = expectL2(l)
  const Lz = expectLz(m)
  const gJ  = landeG(l, j_upper)
  setText('qr-L2', `⟨L²⟩ = ℓ(ℓ+1)ℏ² = ${fmt(L2,3)} ℏ²   |L| = ${fmt(Math.sqrt(L2),3)} ℏ`)
  setText('qr-Lz', `⟨Lz⟩ = mℏ = ${Lz>=0?'+':''}${fmt(Lz,0)} ℏ   gJ = ${fmt(gJ,4)}`)

  // ── Position & momentum ─────────────────────────────────────────────────────
  const rExp    = expectR(n, l)
  const r2Exp   = expectR2(n, l)
  const invR    = expectInvR(n)
  const invR2   = expectInvR2(n, l)
  const dr      = deltaR(n, l)
  const rMp     = mostProbableRadius(n, l)
  const rExp_A  = auToAngstrom(rExp)
  const rMp_A   = auToAngstrom(rMp)
  const T_au    = expectKineticEnergy(n)
  const V_au    = expectPotentialEnergy(n)
  setText('qr-r',    `⟨r⟩ = ${fmt(rExp,4)} a₀ = ${fmt(rExp_A,4)} Å`)
  setText('qr-r2',   `⟨r²⟩ = ${fmt(r2Exp,3)} a₀²   rmp = ${fmt(rMp,4)} a₀ = ${fmt(rMp_A,4)} Å`)
  setText('qr-invr', `⟨1/r⟩ = ${fmt(invR,5)} a₀⁻¹   ⟨1/r²⟩ = ${fmt(invR2,5)} a₀⁻²`)
  setText('qr-dr',   `Δr = ${fmt(dr,4)} a₀   (${fmt(auToAngstrom(dr),4)} Å)`)
  setText('qr-TV',   `⟨T⟩ = ${fmt(auToEv(T_au),4)} eV   ⟨V⟩ = ${fmt(auToEv(V_au),4)} eV   (virial: T=-E ✓)`)

  // ── Heisenberg ──────────────────────────────────────────────────────────────
  const hup   = indicativeUncertaintyProduct(n, l)
  const hRatio = uncertaintyRatio(n, l)
  setText('qr-hup', `Δr·|p| ≈ ${fmt(hup,4)} ℏ   (${fmt(hRatio,2)}× min)`)

  // ── Hyperfine ───────────────────────────────────────────────────────────────
  const hfsHz = hyperfineFreqHz(n, l)
  setText('qr-hfs', `HFS Δν ≈ ${fmtFreq(hfsHz)}`)

  // ── Transition to ground (if n>1) ────────────────────────────────────────────
  if (n > 1) {
    const wl_nm   = transitionWavelengthNm(n, 1)
    const freq_hz = deltaEToFreqHz(Math.abs(E_bohr_au - energyBohr(1)))
    const hexCol  = wavelengthToHex(wl_nm)
    const series  = seriesName(1)
    setText('qr-photon', `n→1: ${fmt(wl_nm,2)} nm  ${fmtFreq(freq_hz)}  (${series})`)
    setStyle('qr-photon-swatch', 'background', hexCol)
    setStyle('qr-photon-swatch', 'display', 'inline-block')
  } else {
    setText('qr-photon', 'Ground state — no photon emission')
    setStyle('qr-photon-swatch', 'display', 'none')
  }

  // ── Lifetime ─────────────────────────────────────────────────────────────────
  const tau = statLifetimeSec(n, l)
  const A_total = tau === Infinity ? 0 : 1 / tau
  const lw_hz   = naturalLinewidthHz(A_total)
  setText('qr-lifetime', `τ = ${fmtTime(tau)}`)
  setText('qr-linewidth', `Δν_nat = ${fmtFreq(lw_hz)}  (ΔE = ${fmtSci(lw_hz * CONST.h / CONST.e * 1e9, 3)} neV)`)

  // ── Zeeman splitting ─────────────────────────────────────────────────────────
  const B_ref = 1.0 // 1 Tesla reference field
  const zShift_eV = auToEv(zeemanShiftPerM(B_ref))
  setText('qr-zeeman', `Zeeman: ΔE/m·B = ${fmtSci(zShift_eV,4)} eV/T   (${l===0?'no splitting for ℓ=0':((2*l+1)+' levels')})`)

  // ── Degeneracy ───────────────────────────────────────────────────────────────
  const degen = degeneracy(n, false)
  const degenSpin = degeneracy(n, true)
  setText('qr-degen', `g = n² = ${degen}  (with spin: ${degenSpin})  [n levels: ${Array.from({length:n},(_,i)=>i+1).join(',')}]`)

  // ── Transition (superposition) ────────────────────────────────────────────────
  const transEl = document.getElementById('qr-transition-row')
  if (transEl) {
    if (superpositionActive && secondary) {
      const t = selectionRule(l, m, secondary.l, secondary.m)
      const secLetter = shellLetter[secondary.l] ?? `l${secondary.l}`
      const label = `|${secondary.n},${secondary.l},${secondary.m}⟩ (${secondary.n}${secLetter})`
      const wl2 = transitionWavelengthNm(Math.max(n, secondary.n), Math.min(n, secondary.n))
      const hex2 = wavelengthToHex(wl2)
      const A_emit = einsteinA_s(Math.max(n,secondary.n), l, Math.min(n,secondary.n), secondary.l)
      setHtml('qr-transition',
        `→ ${label}: ${transitionLabel(t)}<br>` +
        `λ = ${isFinite(wl2)?fmt(wl2,2)+' nm':'—'} ` +
        `<span id="qr-trans-swatch" style="display:inline-block;width:10px;height:10px;border-radius:2px;background:${hex2};vertical-align:middle"></span><br>` +
        `A = ${A_emit>0?fmtSci(A_emit,2)+' s⁻¹':'—'}`
      )
      transEl.style.display = ''
    } else {
      transEl.style.display = 'none'
    }
  }
}
