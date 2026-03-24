import { energyHydrogen, isValidNLM, orbitalLabel } from '../physics/hydrogen'
import type { OrbitalParams } from '../physics/state'

const N_MAX = 6

export interface UiState {
  primary: OrbitalParams
  secondary: OrbitalParams | null
  mix: number
  evolve: boolean
  time: number
}

function fillOptions(select: HTMLSelectElement, values: number[]) {
  select.innerHTML = ''
  for (const v of values) {
    const opt = document.createElement('option')
    opt.value = String(v)
    opt.textContent = String(v)
    select.appendChild(opt)
  }
}

function allowedL(n: number): number[] {
  const out: number[] = []
  for (let l = 0; l < n; l++) out.push(l)
  return out
}

function allowedM(l: number): number[] {
  const out: number[] = []
  for (let m = -l; m <= l; m++) out.push(m)
  return out
}

function clampStateToValid(n: number, l: number, m: number): OrbitalParams {
  const nn = Math.max(1, Math.min(N_MAX, Math.floor(n)))
  let ll = Math.max(0, Math.min(nn - 1, Math.floor(l)))
  let mm = Math.max(-ll, Math.min(ll, Math.floor(m)))
  if (!isValidNLM(nn, ll, mm)) {
    ll = 0
    mm = 0
  }
  return { n: nn, l: ll, m: mm }
}

export function bindUi(onChange: (state: UiState) => void): () => void {
  const n = document.getElementById('n') as HTMLSelectElement
  const l = document.getElementById('l') as HTMLSelectElement
  const m = document.getElementById('m') as HTMLSelectElement
  const n2 = document.getElementById('n2') as HTMLSelectElement
  const l2 = document.getElementById('l2') as HTMLSelectElement
  const m2 = document.getElementById('m2') as HTMLSelectElement
  const mix = document.getElementById('mix') as HTMLInputElement
  const mixVal = document.getElementById('mixVal') as HTMLSpanElement
  const time = document.getElementById('time') as HTMLInputElement
  const timeVal = document.getElementById('timeVal') as HTMLSpanElement
  const evolve = document.getElementById('evolve') as HTMLInputElement
  const superposition = document.getElementById('superposition') as HTMLInputElement
  const labelPrimary = document.getElementById('labelPrimary') as HTMLSpanElement
  const labelSecondary = document.getElementById('labelSecondary') as HTMLSpanElement
  const energyReadout = document.getElementById('energyReadout') as HTMLSpanElement

  fillOptions(
    n,
    Array.from({ length: N_MAX }, (_, i) => i + 1)
  )

  function readPrimary(): OrbitalParams {
    return clampStateToValid(Number(n.value), Number(l.value), Number(m.value))
  }

  function readSecondary(): OrbitalParams {
    return clampStateToValid(Number(n2.value), Number(l2.value), Number(m2.value))
  }

  function syncPrimarySelectors(p: OrbitalParams) {
    fillOptions(l, allowedL(p.n))
    fillOptions(m, allowedM(p.l))
    l.value = String(p.l)
    m.value = String(p.m)
  }

  function syncSecondarySelectors(p: OrbitalParams) {
    fillOptions(l2, allowedL(p.n))
    fillOptions(m2, allowedM(p.l))
    l2.value = String(p.l)
    m2.value = String(p.m)
  }

  function emit() {
    const primary = readPrimary()
    const useSup = superposition.checked
    const secondary = useSup ? readSecondary() : null
    const mixValNum = Number(mix.value) / 100
    const t = (Number(time.value) / 1000) * 24 * Math.PI
    const st: UiState = {
      primary,
      secondary,
      mix: mixValNum,
      evolve: evolve.checked,
      time: t,
    }
    labelPrimary.textContent = orbitalLabel(primary.n, primary.l, primary.m)
    if (secondary) {
      labelSecondary.textContent = orbitalLabel(secondary.n, secondary.l, secondary.m)
    } else {
      labelSecondary.textContent = '—'
    }
    const e1 = energyHydrogen(primary.n)
    if (secondary && useSup) {
      const e2 = energyHydrogen(secondary.n)
      energyReadout.textContent = `E₁ = ${e1.toFixed(4)}  E₂ = ${e2.toFixed(4)} a.u.`
    } else {
      energyReadout.textContent = `E = ${e1.toFixed(4)} a.u.`
    }
    onChange(st)
  }

  function setSecondaryEnabled(en: boolean) {
    n2.disabled = !en
    l2.disabled = !en
    m2.disabled = !en
    mix.disabled = !en
  }

  const onAny = () => {
    const p = readPrimary()
    syncPrimarySelectors(p)
    if (superposition.checked) {
      const s = readSecondary()
      syncSecondarySelectors(s)
    }
    setSecondaryEnabled(superposition.checked)
    time.disabled = !evolve.checked
    mixVal.textContent = (Number(mix.value) / 100).toFixed(2)
    timeVal.textContent = ((Number(time.value) / 1000) * 24 * Math.PI).toFixed(2)
    emit()
  }

  function onPrimaryMetaChange() {
    const p = clampStateToValid(Number(n.value), Number(l.value), Number(m.value))
    syncPrimarySelectors(p)
    onAny()
  }

  n.addEventListener('change', onPrimaryMetaChange)
  l.addEventListener('change', onPrimaryMetaChange)
  m.addEventListener('change', onAny)

  function onSecondaryMetaChange() {
    const s = clampStateToValid(Number(n2.value), Number(l2.value), Number(m2.value))
    syncSecondarySelectors(s)
    onAny()
  }

  n2.addEventListener('change', onSecondaryMetaChange)
  l2.addEventListener('change', onSecondaryMetaChange)
  m2.addEventListener('change', onAny)

  mix.addEventListener('input', onAny)
  time.addEventListener('input', onAny)
  evolve.addEventListener('change', onAny)
  superposition.addEventListener('change', onAny)

  const p0 = clampStateToValid(2, 1, 0)
  n.value = String(p0.n)
  syncPrimarySelectors(p0)
  fillOptions(n2, Array.from({ length: N_MAX }, (_, i) => i + 1))
  const s0 = clampStateToValid(2, 1, 1)
  n2.value = String(s0.n)
  syncSecondarySelectors(s0)
  mix.value = '35'
  time.value = '0'
  evolve.checked = false
  superposition.checked = false
  setSecondaryEnabled(false)
  mixVal.textContent = '0.35'
  timeVal.textContent = '0.00'
  onAny()

  return () => {
    n.removeEventListener('change', onPrimaryMetaChange)
    l.removeEventListener('change', onPrimaryMetaChange)
    m.removeEventListener('change', onAny)
    n2.removeEventListener('change', onSecondaryMetaChange)
    l2.removeEventListener('change', onSecondaryMetaChange)
    m2.removeEventListener('change', onAny)
    mix.removeEventListener('input', onAny)
    time.removeEventListener('input', onAny)
    evolve.removeEventListener('change', onAny)
    superposition.removeEventListener('change', onAny)
  }
}
