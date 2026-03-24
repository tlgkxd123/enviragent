import {
  BufferAttribute,
  BufferGeometry,
  Color,
  Points,
  PointsMaterial,
  Vector3,
} from 'three'

export interface LiquidOptions {
  enabled: boolean
  particleCount: number
  viscosity: number
}

export interface LiquidSystem {
  points: Points
  setEnabled: (enabled: boolean) => void
  setParticleCount: (count: number) => void
  setViscosity: (value: number) => void
  step: (dt: number) => void
  dispose: () => void
}

const tmpA = new Vector3()
const tmpB = new Vector3()

export function createLiquidSystem(
  initialCount = 480,
  radius = 1.28
): LiquidSystem {
  let particleCount = Math.min(1200, Math.max(120, Math.floor(initialCount)))
  let viscosity = 0.22
  let enabled = false

  const geometry = new BufferGeometry()
  const positions = new Float32Array(1200 * 3)
  const velocities = new Float32Array(1200 * 3)
  const seeds = new Float32Array(1200)

  const scatter = () => {
    for (let i = 0; i < particleCount; i++) {
      const t = i / Math.max(1, particleCount - 1)
      const a = t * Math.PI * 9.2
      const r = 0.16 + 0.84 * Math.sqrt(t)
      const x = Math.cos(a) * r * 0.52
      const y = -0.25 + (i % 17) * 0.03
      const z = Math.sin(a) * r * 0.52
      positions[i * 3] = x
      positions[i * 3 + 1] = y
      positions[i * 3 + 2] = z
      velocities[i * 3] = 0
      velocities[i * 3 + 1] = 0
      velocities[i * 3 + 2] = 0
      seeds[i] = ((i * 16807) % 2147483647) / 2147483647
    }
    for (let i = particleCount; i < 1200; i++) {
      positions[i * 3] = 99
      positions[i * 3 + 1] = 99
      positions[i * 3 + 2] = 99
      velocities[i * 3] = 0
      velocities[i * 3 + 1] = 0
      velocities[i * 3 + 2] = 0
    }
  }

  scatter()
  geometry.setAttribute('position', new BufferAttribute(positions, 3))
  geometry.setDrawRange(0, particleCount)

  const material = new PointsMaterial({
    size: 0.03,
    color: new Color(0x7ec7ff),
    transparent: true,
    opacity: 0.82,
    depthWrite: false,
    sizeAttenuation: true,
  })
  const points = new Points(geometry, material)
  points.position.set(0, 0.12, 0)
  points.visible = enabled

  function clampToContainer(i: number) {
    const px = positions[i * 3]
    const py = positions[i * 3 + 1]
    const pz = positions[i * 3 + 2]
    tmpA.set(px, py, pz)
    const floorY = -1.06
    const ceilingY = 1.22
    if (tmpA.y < floorY) {
      tmpA.y = floorY
      velocities[i * 3 + 1] *= -0.35
      velocities[i * 3] *= 0.93
      velocities[i * 3 + 2] *= 0.93
    } else if (tmpA.y > ceilingY) {
      tmpA.y = ceilingY
      velocities[i * 3 + 1] *= -0.15
    }
    const r = Math.hypot(tmpA.x, tmpA.z)
    if (r > radius) {
      const inv = 1 / Math.max(r, 1e-6)
      tmpA.x *= radius * inv
      tmpA.z *= radius * inv
      velocities[i * 3] *= -0.2
      velocities[i * 3 + 2] *= -0.2
    }
    positions[i * 3] = tmpA.x
    positions[i * 3 + 1] = tmpA.y
    positions[i * 3 + 2] = tmpA.z
  }

  function setParticleCount(count: number) {
    particleCount = Math.min(1200, Math.max(120, Math.floor(count)))
    geometry.setDrawRange(0, particleCount)
    scatter()
    geometry.attributes.position.needsUpdate = true
  }

  function step(dtRaw: number) {
    if (!enabled) return
    const dt = Math.min(0.033, Math.max(0.001, dtRaw))
    const gravity = -4.7
    const interactionR = 0.16
    const interactionR2 = interactionR * interactionR
    const pressureK = 0.18
    const swirl = 0.35

    for (let i = 0; i < particleCount; i++) {
      velocities[i * 3 + 1] += gravity * dt
      const px = positions[i * 3]
      const pz = positions[i * 3 + 2]
      const tangX = -pz
      const tangZ = px
      velocities[i * 3] += tangX * swirl * dt * 0.15
      velocities[i * 3 + 2] += tangZ * swirl * dt * 0.15
    }

    for (let i = 0; i < particleCount; i++) {
      tmpA.set(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2])
      for (let j = i + 1; j < particleCount; j++) {
        tmpB.set(positions[j * 3], positions[j * 3 + 1], positions[j * 3 + 2])
        const dx = tmpA.x - tmpB.x
        const dy = tmpA.y - tmpB.y
        const dz = tmpA.z - tmpB.z
        const d2 = dx * dx + dy * dy + dz * dz
        if (d2 > interactionR2 || d2 < 1e-9) continue
        const d = Math.sqrt(d2)
        const nX = dx / d
        const nY = dy / d
        const nZ = dz / d
        const strength = (interactionR - d) * pressureK
        velocities[i * 3] += nX * strength
        velocities[i * 3 + 1] += nY * strength
        velocities[i * 3 + 2] += nZ * strength
        velocities[j * 3] -= nX * strength
        velocities[j * 3 + 1] -= nY * strength
        velocities[j * 3 + 2] -= nZ * strength
      }
    }

    const damping = Math.max(0.82, 0.97 - viscosity * 0.2)
    for (let i = 0; i < particleCount; i++) {
      velocities[i * 3] *= damping
      velocities[i * 3 + 1] *= damping
      velocities[i * 3 + 2] *= damping
      positions[i * 3] += velocities[i * 3] * dt
      positions[i * 3 + 1] += velocities[i * 3 + 1] * dt
      positions[i * 3 + 2] += velocities[i * 3 + 2] * dt

      const s = seeds[i]
      positions[i * 3] += Math.sin((positions[i * 3 + 1] + s) * 12) * 0.0008
      positions[i * 3 + 2] += Math.cos((positions[i * 3] + s) * 11) * 0.0008
      clampToContainer(i)
    }

    geometry.attributes.position.needsUpdate = true
  }

  return {
    points,
    setEnabled(value) {
      enabled = value
      points.visible = value
    },
    setParticleCount,
    setViscosity(value) {
      viscosity = Math.min(1, Math.max(0, value))
    },
    step,
    dispose() {
      geometry.dispose()
      material.dispose()
    },
  }
}
