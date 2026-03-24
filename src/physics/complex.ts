export interface Complex {
  re: number
  im: number
}

export function cAdd(a: Complex, b: Complex): Complex {
  return { re: a.re + b.re, im: a.im + b.im }
}

export function cSub(a: Complex, b: Complex): Complex {
  return { re: a.re - b.re, im: a.im - b.im }
}

export function cScale(c: Complex, s: number): Complex {
  return { re: c.re * s, im: c.im * s }
}

export function cMul(a: Complex, b: Complex): Complex {
  return { re: a.re * b.re - a.im * b.im, im: a.re * b.im + a.im * b.re }
}

export function cConj(a: Complex): Complex {
  return { re: a.re, im: -a.im }
}

export function cExpImag(theta: number): Complex {
  return { re: Math.cos(theta), im: Math.sin(theta) }
}

export function cAbsSq(a: Complex): number {
  return a.re * a.re + a.im * a.im
}

export function cArg(a: Complex): number {
  return Math.atan2(a.im, a.re)
}
