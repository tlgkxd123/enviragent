import type { Complex } from './complex'
import { cAdd, cExpImag, cMul, cScale } from './complex'
import { energyHydrogen, hydrogenPsi } from './hydrogen'

export interface OrbitalParams {
  n: number
  l: number
  m: number
}

/**
 * ψ at (r,θ,φ): optional equal-weight superposition of two orbitals with
 * independent time evolution exp(−i E_n t) when `evolve` is true.
 */
export function psiSuperposition(
  primary: OrbitalParams,
  secondary: OrbitalParams | null,
  mix: number,
  r: number,
  theta: number,
  phi: number,
  time: number,
  evolve: boolean
): Complex {
  const psi1 = hydrogenPsi(primary.n, primary.l, primary.m, r, theta, phi)
  if (!secondary || mix <= 1e-6) {
    if (!evolve) return psi1
    const E1 = energyHydrogen(primary.n)
    return cMul(psi1, cExpImag(-E1 * time))
  }
  const psi2 = hydrogenPsi(secondary.n, secondary.l, secondary.m, r, theta, phi)
  const w2 = Math.sqrt(Math.min(1, Math.max(0, mix)))
  const w1 = Math.sqrt(1 - w2 * w2)
  let t1 = cScale(psi1, w1)
  let t2 = cScale(psi2, w2)
  if (evolve) {
    const E1 = energyHydrogen(primary.n)
    const E2 = energyHydrogen(secondary.n)
    t1 = cMul(t1, cExpImag(-E1 * time))
    t2 = cMul(t2, cExpImag(-E2 * time))
  }
  return cAdd(t1, t2)
}
