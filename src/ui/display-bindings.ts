import type { OrbitalScene } from '../render/orbital'

export function bindDisplayUi(orbital: Pick<OrbitalScene, 'setAtomHalftone'>) {
  const enabled = document.getElementById('atomHalftoneEnabled') as HTMLInputElement
  const cellSize = document.getElementById('atomCellSize') as HTMLInputElement
  const cellSizeVal = document.getElementById('atomCellSizeVal') as HTMLSpanElement
  const strength = document.getElementById('atomStrength') as HTMLInputElement
  const strengthVal = document.getElementById('atomStrengthVal') as HTMLSpanElement

  function emit() {
    cellSizeVal.textContent = cellSize.value
    const s = Number(strength.value) / 100
    strengthVal.textContent = s.toFixed(2)
    orbital.setAtomHalftone({
      enabled: enabled.checked,
      cellSize: Number(cellSize.value),
      strength: s,
    })
  }

  enabled.addEventListener('change', emit)
  cellSize.addEventListener('input', emit)
  strength.addEventListener('input', emit)
  emit()
}
