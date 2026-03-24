import type { OrbitalScene } from '../render/orbital'
import type { LiquidOptions } from '../render/liquid-sim'

interface LiquidApi extends Pick<OrbitalScene, 'setLiquidOptions'> {}

export function bindLiquidUi(orbital: LiquidApi) {
  const enabled = document.getElementById('liquidEnabled') as HTMLInputElement
  const count = document.getElementById('liquidCount') as HTMLInputElement
  const countVal = document.getElementById('liquidCountVal') as HTMLSpanElement
  const viscosity = document.getElementById('liquidViscosity') as HTMLInputElement
  const viscosityVal = document.getElementById('liquidViscosityVal') as HTMLSpanElement

  function emit() {
    const opts: Partial<LiquidOptions> = {
      enabled: enabled.checked,
      particleCount: Number(count.value),
      viscosity: Number(viscosity.value) / 100,
    }
    countVal.textContent = String(opts.particleCount)
    viscosityVal.textContent = (opts.viscosity ?? 0).toFixed(2)
    orbital.setLiquidOptions(opts)
  }

  enabled.addEventListener('change', emit)
  count.addEventListener('input', emit)
  viscosity.addEventListener('input', emit)
  emit()
}
