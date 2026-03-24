import { ShaderMaterial, UniformsUtils, Vector2, Vector3, type IUniform } from 'three'
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js'

/** Default matches scene.background 0x040506 */
const DEFAULT_BG = new Vector3(4 / 255, 5 / 255, 6 / 255)

const vertexShader = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`

const fragmentShader = /* glsl */ `
precision highp float;

uniform sampler2D tDiffuse;
uniform vec2 resolution;
uniform float cellSize;
uniform float mixAmount;
uniform vec3 bgColor;

varying vec2 vUv;

void main() {
  vec2 px = vUv * resolution;
  float cs = max(cellSize, 2.0);
  vec2 cellId = floor(px / cs);
  vec2 center = (cellId + 0.5) * cs;
  vec2 centerUv = center / resolution;
  vec4 sampleC = texture2D(tDiffuse, centerUv);
  float L = dot(sampleC.rgb, vec3(0.299, 0.587, 0.114));
  float maxR = cs * 0.48;
  float r = maxR * clamp(sqrt(L), 0.0, 1.0);
  float d = distance(px, center);
  vec3 halftone = (d <= r) ? sampleC.rgb : bgColor;
  vec3 orig = texture2D(tDiffuse, vUv).rgb;
  gl_FragColor = vec4(mix(orig, halftone, mixAmount), 1.0);
}
`

const baseUniforms = {
  tDiffuse: { value: null },
  resolution: { value: new Vector2(800, 600) },
  cellSize: { value: 8 },
  mixAmount: { value: 0 },
  bgColor: { value: DEFAULT_BG.clone() },
}

export interface AtomHalftoneUniforms {
  tDiffuse: { value: unknown }
  resolution: { value: Vector2 }
  cellSize: { value: number }
  mixAmount: { value: number }
  bgColor: { value: Vector3 }
}

export function createAtomHalftonePass(): {
  pass: ShaderPass
  uniforms: AtomHalftoneUniforms
} {
  const uniforms = UniformsUtils.clone(baseUniforms) as unknown as AtomHalftoneUniforms
  const material = new ShaderMaterial({
    uniforms: uniforms as unknown as { [key: string]: IUniform },
    vertexShader,
    fragmentShader,
    depthTest: false,
    depthWrite: false,
  })
  const pass = new ShaderPass(material)
  return { pass, uniforms }
}
