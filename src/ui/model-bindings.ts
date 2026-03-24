export type VisualizationMode = 'orbital' | 'model'

export interface ModelUiState {
  viewMode: VisualizationMode
  showImportedMesh: boolean
  particlesEnabled: boolean
  particleCount: number
}

export function bindModelUi(onChange: (state: ModelUiState) => void): () => void {
  const viewMode = document.getElementById('viewMode') as HTMLSelectElement
  const showMesh = document.getElementById('showImportedMesh') as HTMLInputElement
  const particles = document.getElementById('particlesEnabled') as HTMLInputElement
  const particleCount = document.getElementById('particleCount') as HTMLInputElement
  const particleVal = document.getElementById('particleCountVal') as HTMLSpanElement

  function read(): ModelUiState {
    const vm = viewMode.value === 'model' ? 'model' : 'orbital'
    return {
      viewMode: vm,
      showImportedMesh: showMesh.checked,
      particlesEnabled: particles.checked,
      particleCount: Number(particleCount.value),
    }
  }

  function emit() {
    particleVal.textContent = String(Number(particleCount.value))
    onChange(read())
  }

  viewMode.addEventListener('change', emit)
  showMesh.addEventListener('change', emit)
  particles.addEventListener('change', emit)
  particleCount.addEventListener('input', emit)

  emit()

  return () => {
    viewMode.removeEventListener('change', emit)
    showMesh.removeEventListener('change', emit)
    particles.removeEventListener('change', emit)
    particleCount.removeEventListener('input', emit)
  }
}
