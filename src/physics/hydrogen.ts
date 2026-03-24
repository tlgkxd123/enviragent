import type { Complex } from './complex'
import { cExpImag, cScale } from './complex'

/** Atomic units: ℏ = m_e = e = a_0 = 1 */

function factorial(n: number): number {
  if (n < 0) throw new Error('factorial: n < 0')
  let r = 1
  for (let i = 2; i <= n; i++) r *= i
  return r
}

/** Generalized Laguerre L_n^{(α)}(x) = Σ (-1)^j C(n+α, n-j) x^j / j! */
function generalizedLaguerre(n: number, alpha: number, x: number): number {
  let s = 0
  for (let j = 0; j <= n; j++) {
    const bin =
      factorial(n + alpha) /
      (factorial(n - j) * factorial(alpha + j) * factorial(j))
    s += (j % 2 === 0 ? 1 : -1) * bin * Math.pow(x, j)
  }
  return s
}

/** Associated Legendre P_l^m(x), m ≥ 0, x ∈ [-1, 1]. Numerical Recipes style. */
function assocLegendre(l: number, m: number, x: number): number {
  const mAbs = Math.abs(m)
  if (mAbs > l) return 0
  const xm = Math.sqrt(Math.max(0, (1 - x) * (1 + x)))
  let pmm = 1
  if (mAbs > 0) {
    let fact = 1
    for (let i = 1; i <= mAbs; i++) {
      pmm *= -fact * xm
      fact += 2
    }
  }
  if (l === mAbs) return pmm
  let pmmp1 = x * (2 * mAbs + 1) * pmm
  if (l === mAbs + 1) return pmmp1
  for (let ll = mAbs + 2; ll <= l; ll++) {
    const pll = (x * (2 * ll - 1) * pmmp1 - (ll + mAbs - 1) * pmm) / (ll - mAbs)
    pmm = pmmp1
    pmmp1 = pll
  }
  return pmmp1
}

/**
 * Y_l^m for m ≥ 0. θ = polar angle from +Z, φ azimuth. Condon–Shortley.
 */
function sphericalHarmonicYPos(
  l: number,
  m: number,
  theta: number,
  phi: number
): Complex {
  const x = Math.cos(theta)
  const Plm = assocLegendre(l, m, x)
  const pre =
    Math.sqrt(((2 * l + 1) / (4 * Math.PI)) * (factorial(l - m) / factorial(l + m)))
  const coef = (-1) ** m * pre * Plm
  const e = cExpImag(m * phi)
  return cScale(e, coef)
}

/**
 * Complex spherical harmonic Y_l^m(θ, φ). Uses Y_l^{-m} = (-1)^m conj(Y_l^m).
 */
export function sphericalHarmonicY(
  l: number,
  m: number,
  theta: number,
  phi: number
): Complex {
  if (m >= 0) return sphericalHarmonicYPos(l, m, theta, phi)
  const yp = sphericalHarmonicYPos(l, -m, theta, phi)
  const conj = { re: yp.re, im: -yp.im }
  return cScale(conj, (-1) ** -m)
}

/** Radial factor R_nl(r) for hydrogen (atomic units). */
export function radialHydrogen(n: number, l: number, r: number): number {
  const rho = (2 * r) / n
  const pref = Math.sqrt(
    (2 / n) ** 3 * (factorial(n - l - 1) / (2 * n * factorial(n + l)))
  )
  const L = generalizedLaguerre(n - l - 1, 2 * l + 1, rho)
  return pref * Math.exp(-rho / 2) * Math.pow(rho, l) * L
}

export function hydrogenPsi(
  n: number,
  l: number,
  m: number,
  r: number,
  theta: number,
  phi: number
): Complex {
  const R = radialHydrogen(n, l, r)
  const Y = sphericalHarmonicY(l, m, theta, phi)
  return cScale(Y, R)
}

export function energyHydrogen(n: number): number {
  return -1 / (2 * n * n)
}

export function isValidNLM(n: number, l: number, m: number): boolean {
  if (n < 1 || !Number.isInteger(n)) return false
  if (l < 0 || l >= n || !Number.isInteger(l)) return false
  if (Math.abs(m) > l || !Number.isInteger(m)) return false
  return true
}

export function allowedLForN(n: number): number[] {
  const out: number[] = []
  for (let l = 0; l < n; l++) out.push(l)
  return out
}

export function allowedMForL(l: number): number[] {
  const out: number[] = []
  for (let m = -l; m <= l; m++) out.push(m)
  return out
}

const shellLetter = ['s', 'p', 'd', 'f', 'g', 'h']

export function orbitalLabel(n: number, l: number, m: number): string {
  const letter = shellLetter[l] ?? `l${l}`
  if (l === 0) return `${n}${letter}`
  if (m === 0) return `${n}${letter}_z`
  if (m > 0) return `${n}${letter}_+${m}`
  return `${n}${letter}_${m}`
}

/** Map local unit-sphere vertex (Y-up Three.js) to physics angles (Z-up). */
export function localSphereToPhysicsAngles(
  lx: number,
  ly: number,
  lz: number
): { theta: number; phi: number } {
  const wx = lx
  const wy = -lz
  const wz = ly
  const r = Math.hypot(wx, wy, wz)
  if (r < 1e-12) return { theta: 0, phi: 0 }
  const theta = Math.acos(Math.min(1, Math.max(-1, wz / r)))
  const phi = Math.atan2(wy, wx)
  return { theta, phi }
}

export function radialShellRadius(n: number): number {
  return n * n
}
