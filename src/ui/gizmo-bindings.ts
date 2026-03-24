import type {
  TransformGizmoMode,
  TransformGizmoTarget,
} from '../render/orbital'

interface OrbitalGizmoApi {
  setTransformGizmoVisible: (visible: boolean) => void
  setTransformGizmoTarget: (target: TransformGizmoTarget) => void
  setTransformGizmoMode: (mode: TransformGizmoMode) => void
  hasImportedModel: () => boolean
}

export function bindGizmoUi(orbital: OrbitalGizmoApi) {
  const enabled = document.getElementById('gizmoEnabled') as HTMLInputElement
  const mode = document.getElementById('gizmoMode') as HTMLSelectElement
  const target = document.getElementById('gizmoTarget') as HTMLSelectElement
  const importOpt = target.querySelector('option[value="import"]') as HTMLOptionElement

  function syncImportOption() {
    importOpt.disabled = !orbital.hasImportedModel()
    if (importOpt.disabled && target.value === 'import') {
      target.value = 'orbital'
      orbital.setTransformGizmoTarget('orbital')
    }
  }

  function emit() {
    orbital.setTransformGizmoVisible(enabled.checked)
    orbital.setTransformGizmoMode(mode.value as TransformGizmoMode)
    orbital.setTransformGizmoTarget(target.value as TransformGizmoTarget)
  }

  enabled.addEventListener('change', emit)
  mode.addEventListener('change', emit)
  target.addEventListener('change', emit)

  syncImportOption()
  emit()

  return { syncImportOption }
}
