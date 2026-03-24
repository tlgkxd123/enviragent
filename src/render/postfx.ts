import { Vector2 } from 'three'
import type { PerspectiveCamera, Scene, WebGLRenderer } from 'three'
import { createAtomHalftonePass, type AtomHalftoneUniforms } from './atom-halftone'
import { createSSRPass } from './ssr-pass'
import { createLensFlarePass, type LensFlarePass } from './lens-flare-pass'
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js'
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js'
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js'
import { SMAAPass } from 'three/addons/postprocessing/SMAAPass.js'
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js'
import type { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js'

/**
 * Post stack:
 *   scene → SSR → bloom (threshold-gated, highlight-only) →
 *   output → lens flare → atom halftone → SMAA
 *
 * Bloom threshold raised to 0.78 so only true specular highlights bloom;
 * the dark/shadowed hemisphere no longer glows.
 */
export function createOrbitalComposer(
  renderer: WebGLRenderer,
  scene: Scene,
  camera: PerspectiveCamera
): {
  composer: EffectComposer
  atomHalftonePass: ShaderPass
  atomHalftoneUniforms: AtomHalftoneUniforms
  ssrSetSize: (w: number, h: number) => void
  lensFlare: LensFlarePass
} {
  const composer = new EffectComposer(renderer)
  composer.addPass(new RenderPass(scene, camera))

  // Screen-space reflections (after geometry, before bloom)
  const ssr = createSSRPass()
  composer.addPass(ssr.pass)

  const size = new Vector2()
  renderer.getSize(size)
  ssr.setSize(size.x, size.y)

  // Selective bloom — high threshold keeps glow only on bright specular peaks,
  // eliminating the ambient glow that was washing out the shadowed side.
  const bloom = new UnrealBloomPass(
    size,
    0.28,   // strength  (was 0.32)
    0.38,   // radius    (was 0.42)
    0.78    // threshold (was 0.84 — now higher = only brightest pixels bloom)
  )
  composer.addPass(bloom)

  composer.addPass(new OutputPass())

  // Lens flare — additive, after tone-mapping output
  const lensFlare = createLensFlarePass()
  lensFlare.setSize(size.x, size.y)
  composer.addPass(lensFlare.pass)

  const { pass: atomHalftonePass, uniforms: atomHalftoneUniforms } = createAtomHalftonePass()
  composer.addPass(atomHalftonePass)

  composer.addPass(new SMAAPass())

  return { composer, atomHalftonePass, atomHalftoneUniforms, ssrSetSize: ssr.setSize, lensFlare }
}
