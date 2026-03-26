import * as THREE from 'three'

// ═══════════════════════════════════════════════════════════════════════════════
// Procedural Terrain Generator
//
// Multi-octave gradient noise heightmap → vertex displacement on PlaneGeometry.
// Custom ShaderMaterial blends 4 tileable textures (grass/rock/dirt/snow) based
// on height band + surface steepness, with built-in directional + ambient light.
// ═══════════════════════════════════════════════════════════════════════════════

// ── 2D gradient noise (CPU) ─────────────────────────────────────────────────

function hash2(ix: number, iy: number): [number, number] {
  let h = ix * 374761393 + iy * 668265263 + 1376312589
  h = (h ^ (h >>> 13)) * 1274126177
  h = h ^ (h >>> 16)
  const a = ((h & 0xffff) / 32768.0) - 1.0
  const b = (((h >>> 16) & 0xffff) / 32768.0) - 1.0
  return [a, b]
}

function quintic(t: number) { return t * t * t * (t * (t * 6 - 15) + 10) }

function noise2D(x: number, y: number): number {
  const ix = Math.floor(x), iy = Math.floor(y)
  const fx = x - ix, fy = y - iy
  const ux = quintic(fx), uy = quintic(fy)

  const [gax, gay] = hash2(ix, iy)
  const [gbx, gby] = hash2(ix + 1, iy)
  const [gcx, gcy] = hash2(ix, iy + 1)
  const [gdx, gdy] = hash2(ix + 1, iy + 1)

  const va = gax * fx + gay * fy
  const vb = gbx * (fx - 1) + gby * fy
  const vc = gcx * fx + gcy * (fy - 1)
  const vd = gdx * (fx - 1) + gdy * (fy - 1)

  return va + ux * (vb - va) + uy * (vc - va) + ux * uy * (va - vb - vc + vd)
}

function fbm(x: number, y: number, octaves: number, persistence: number, lacunarity: number): number {
  let sum = 0, amp = 1, freq = 1, maxAmp = 0
  for (let i = 0; i < octaves; i++) {
    sum += noise2D(x * freq, y * freq) * amp
    maxAmp += amp
    amp *= persistence
    freq *= lacunarity
  }
  return sum / maxAmp
}

function ridgedFbm(x: number, y: number, octaves: number, persistence: number, lacunarity: number): number {
  let sum = 0, amp = 1, freq = 1, maxAmp = 0, prev = 1
  for (let i = 0; i < octaves; i++) {
    let v = noise2D(x * freq, y * freq)
    v = 1.0 - Math.abs(v)
    v = v * v * prev
    prev = v
    sum += v * amp
    maxAmp += amp
    amp *= persistence
    freq *= lacunarity
  }
  return sum / maxAmp
}

// ── Biome presets ───────────────────────────────────────────────────────────

export interface BiomeParams {
  octaves: number
  persistence: number
  lacunarity: number
  ridgeFraction: number   // 0 = pure FBM, 1 = pure ridged
  heightScale: number
  noiseScale: number      // world-space frequency
  seed: number
  snowLine: number        // 0-1 normalized height where snow starts
  treeLine: number        // 0-1 where grass→rock transition happens
}

const BIOME_PRESETS: Record<string, Partial<BiomeParams>> = {
  mountains:     { octaves: 8, persistence: 0.48, lacunarity: 2.1, ridgeFraction: 0.6, heightScale: 8,  noiseScale: 0.04, snowLine: 0.72, treeLine: 0.45 },
  rolling_hills: { octaves: 6, persistence: 0.45, lacunarity: 2.0, ridgeFraction: 0.0, heightScale: 3,  noiseScale: 0.06, snowLine: 0.95, treeLine: 0.7 },
  desert:        { octaves: 5, persistence: 0.35, lacunarity: 2.3, ridgeFraction: 0.2, heightScale: 4,  noiseScale: 0.05, snowLine: 1.0,  treeLine: 0.3 },
  arctic:        { octaves: 7, persistence: 0.5,  lacunarity: 2.0, ridgeFraction: 0.4, heightScale: 5,  noiseScale: 0.045, snowLine: 0.25, treeLine: 0.15 },
  volcanic:      { octaves: 7, persistence: 0.55, lacunarity: 1.9, ridgeFraction: 0.7, heightScale: 10, noiseScale: 0.035, snowLine: 0.85, treeLine: 0.35 },
  canyon:        { octaves: 6, persistence: 0.52, lacunarity: 2.2, ridgeFraction: 0.5, heightScale: 7,  noiseScale: 0.04, snowLine: 0.9,  treeLine: 0.4 },
  plateau:       { octaves: 5, persistence: 0.4,  lacunarity: 2.0, ridgeFraction: 0.3, heightScale: 4,  noiseScale: 0.05, snowLine: 0.85, treeLine: 0.5 },
}

function getBiome(name: string, overrides: Partial<BiomeParams> = {}): BiomeParams {
  const base: BiomeParams = {
    octaves: 7, persistence: 0.48, lacunarity: 2.1, ridgeFraction: 0.4,
    heightScale: 6, noiseScale: 0.04, seed: 42, snowLine: 0.75, treeLine: 0.45,
  }
  const preset = BIOME_PRESETS[name] ?? {}
  return { ...base, ...preset, ...overrides }
}

// ── GLSL shaders ────────────────────────────────────────────────────────────

const VERT = /* glsl */ `
out vec3 vWorldPos;
out vec3 vWorldNormal;
out vec2 vUv;

void main() {
  vec4 wp = modelMatrix * vec4(position, 1.0);
  vWorldPos = wp.xyz;
  vWorldNormal = normalize(mat3(modelMatrix) * normal);
  vUv = uv;
  gl_Position = projectionMatrix * viewMatrix * wp;
}
`

const FRAG = /* glsl */ `
precision highp float;

uniform sampler2D uGrass;
uniform sampler2D uRock;
uniform sampler2D uDirt;
uniform sampler2D uSnow;

uniform vec3 uSunDir;
uniform vec3 uSunColor;
uniform float uHeightMin;
uniform float uHeightMax;
uniform float uSnowLine;
uniform float uTreeLine;
uniform float uTexScale;

in vec3 vWorldPos;
in vec3 vWorldNormal;
in vec2 vUv;
out vec4 fragColor;

void main() {
  vec3 N = normalize(vWorldNormal);
  float hRange = uHeightMax - uHeightMin + 0.001;
  float h = clamp((vWorldPos.y - uHeightMin) / hRange, 0.0, 1.0);
  float steepness = 1.0 - abs(N.y);

  vec2 tUv = vWorldPos.xz * uTexScale;
  vec3 tGrass = texture(uGrass, tUv).rgb;
  vec3 tRock  = texture(uRock,  tUv * 0.7).rgb;
  vec3 tDirt  = texture(uDirt,  tUv * 0.6).rgb;
  vec3 tSnow  = texture(uSnow,  tUv * 0.4).rgb;

  // Height-band weights
  float wDirt  = 1.0 - smoothstep(0.0, uTreeLine * 0.5, h);
  float wGrass = smoothstep(0.0, uTreeLine * 0.4, h) * (1.0 - smoothstep(uTreeLine, uTreeLine + 0.15, h));
  float wRock  = smoothstep(uTreeLine - 0.1, uTreeLine + 0.1, h) * (1.0 - smoothstep(uSnowLine, uSnowLine + 0.1, h));
  float wSnow  = smoothstep(uSnowLine - 0.05, uSnowLine + 0.1, h);

  // Steep slopes → rock
  float slopeRock = smoothstep(0.3, 0.65, steepness);
  wRock  = max(wRock, slopeRock);
  wGrass *= 1.0 - smoothstep(0.25, 0.55, steepness);
  wSnow  *= 1.0 - smoothstep(0.4, 0.7, steepness);
  wDirt  *= 1.0 - smoothstep(0.5, 0.8, steepness);

  float tw = wGrass + wRock + wDirt + wSnow + 0.001;
  vec3 albedo = (tGrass * wGrass + tRock * wRock + tDirt * wDirt + tSnow * wSnow) / tw;

  // Lighting
  vec3 L = normalize(uSunDir);
  float NdL = max(dot(N, L), 0.0);
  vec3 ambient = vec3(0.18, 0.2, 0.26);
  vec3 diffuse = uSunColor * NdL * 0.82;

  // Hemisphere sky fill
  float skyFill = N.y * 0.5 + 0.5;
  vec3 sky = mix(vec3(0.08, 0.06, 0.04), vec3(0.1, 0.14, 0.22), skyFill) * 0.3;

  vec3 lit = albedo * (ambient + diffuse + sky);
  fragColor = vec4(lit, 1.0);
}
`

// ── Public API ──────────────────────────────────────────────────────────────

export interface LandscapeOptions {
  size?: [number, number]
  resolution?: number
  biome?: string
  height_scale?: number
  seed?: number
  octaves?: number
  persistence?: number
  lacunarity?: number
  ridge_fraction?: number
  snow_line?: number
  tree_line?: number
  tex_scale?: number
}

export interface LandscapeHandle {
  mesh: THREE.Mesh
  heightMin: number
  heightMax: number
  dispose: () => void
}

const texLoader = new THREE.TextureLoader()

function loadTileTexture(path: string): THREE.Texture {
  const tex = texLoader.load(path)
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping
  tex.magFilter = THREE.LinearFilter
  tex.minFilter = THREE.LinearMipmapLinearFilter
  tex.colorSpace = THREE.SRGBColorSpace
  return tex
}

export function createLandscape(opts: LandscapeOptions = {}): LandscapeHandle {
  const [Lx, Lz] = opts.size ?? [40, 40]
  const res = Math.min(512, Math.max(16, opts.resolution ?? 128))
  const biome = getBiome(opts.biome ?? 'mountains', {
    heightScale: opts.height_scale,
    seed: opts.seed,
    octaves: opts.octaves,
    persistence: opts.persistence,
    lacunarity: opts.lacunarity,
    ridgeFraction: opts.ridge_fraction,
    snowLine: opts.snow_line,
    treeLine: opts.tree_line,
  })
  const texScale = opts.tex_scale ?? 0.25

  const Nx = res, Nz = res
  const geo = new THREE.PlaneGeometry(Lx, Lz, Nx, Nz)
  geo.rotateX(-Math.PI / 2)

  const pos = geo.attributes.position as THREE.BufferAttribute
  const pArr = pos.array as Float32Array
  const count = pos.count

  const seedOffX = biome.seed * 13.37
  const seedOffZ = biome.seed * 7.91

  let hMin = Infinity, hMax = -Infinity

  for (let i = 0; i < count; i++) {
    const x = pArr[i * 3] + seedOffX
    const z = pArr[i * 3 + 2] + seedOffZ

    const nx = x * biome.noiseScale
    const nz = z * biome.noiseScale

    const f = fbm(nx, nz, biome.octaves, biome.persistence, biome.lacunarity)
    const r = ridgedFbm(nx * 1.1, nz * 1.1, biome.octaves, biome.persistence, biome.lacunarity)
    const h = (f * (1 - biome.ridgeFraction) + r * biome.ridgeFraction) * biome.heightScale

    pArr[i * 3 + 1] = h
    if (h < hMin) hMin = h
    if (h > hMax) hMax = h
  }

  geo.computeVertexNormals()

  // Textures
  const grassTex = loadTileTexture('/textures/terrain/grass.png')
  const rockTex  = loadTileTexture('/textures/terrain/rock.png')
  const dirtTex  = loadTileTexture('/textures/terrain/dirt.png')
  const snowTex  = loadTileTexture('/textures/terrain/snow.png')

  const mat = new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3,
    vertexShader: VERT,
    fragmentShader: FRAG,
    uniforms: {
      uGrass:    { value: grassTex },
      uRock:     { value: rockTex },
      uDirt:     { value: dirtTex },
      uSnow:     { value: snowTex },
      uSunDir:   { value: new THREE.Vector3(0.5, 0.8, 0.3).normalize() },
      uSunColor: { value: new THREE.Vector3(1.0, 0.95, 0.85) },
      uHeightMin: { value: hMin },
      uHeightMax: { value: hMax },
      uSnowLine:  { value: biome.snowLine },
      uTreeLine:  { value: biome.treeLine },
      uTexScale:  { value: texScale },
    },
    side: THREE.FrontSide,
  })

  const mesh = new THREE.Mesh(geo, mat)
  mesh.receiveShadow = true
  mesh.castShadow = true

  return {
    mesh,
    heightMin: hMin,
    heightMax: hMax,
    dispose() {
      geo.dispose()
      mat.dispose()
      grassTex.dispose()
      rockTex.dispose()
      dirtTex.dispose()
      snowTex.dispose()
    },
  }
}
