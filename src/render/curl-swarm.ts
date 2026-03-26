import * as THREE from 'three'

// ═══════════════════════════════════════════════════════════════════════════════
// Advanced GPU Curl Noise Particle Swarm
//
// Algorithms:
//   1. 3D Simplex/Perlin Noise in GLSL
//   2. Analytical Curl Noise (cross product of noise gradients)
//   3. GPU Instancing for millions of particles
//   4. Time-based advection integrated in the vertex shader
// ═══════════════════════════════════════════════════════════════════════════════

const curlSwarmVertexShader = `
uniform float uTime;
uniform float uSpeed;
uniform float uScale;
uniform float uSwirl;

attribute vec3 offset;
attribute vec3 customColor;
attribute float size;

varying vec3 vColor;

// --- Simplex Noise 3D ---
vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 permute(vec4 x) { return mod289(((x*34.0)+1.0)*x); }
vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

float snoise(vec3 v) {
  const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
  const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);

  vec3 i  = floor(v + dot(v, C.yyy) );
  vec3 x0 = v - i + dot(i, C.xxx) ;

  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min( g.xyz, l.zxy );
  vec3 i2 = max( g.xyz, l.zxy );

  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy;
  vec3 x3 = x0 - D.yyy;

  i = mod289(i);
  vec4 p = permute( permute( permute(
             i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
           + i.y + vec4(0.0, i1.y, i2.y, 1.0 ))
           + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

  float n_ = 0.142857142857;
  vec3  ns = n_ * D.wyz - D.xzx;

  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);

  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_ );

  vec4 x = x_ *ns.x + ns.yyyy;
  vec4 y = y_ *ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);

  vec4 b0 = vec4( x.xy, y.xy );
  vec4 b1 = vec4( x.zw, y.zw );

  vec4 s0 = floor(b0)*2.0 + 1.0;
  vec4 s1 = floor(b1)*2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));

  vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
  vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

  vec3 p0 = vec3(a0.xy,h.x);
  vec3 p1 = vec3(a0.zw,h.y);
  vec3 p2 = vec3(a1.xy,h.z);
  vec3 p3 = vec3(a1.zw,h.w);

  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
  p0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;

  vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  m = m * m;
  return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3) ) );
}

vec3 snoiseVec3( vec3 x ){
  float s  = snoise(vec3( x ));
  float s1 = snoise(vec3( x.y - 19.1 , x.z + 33.4 , x.x + 47.2 ));
  float s2 = snoise(vec3( x.z + 74.2 , x.x - 124.5 , x.y + 99.4 ));
  vec3 c = vec3( s , s1 , s2 );
  return c;
}

vec3 curlNoise( vec3 p ){
  const float e = .1;
  vec3 dx = vec3( e   , 0.0 , 0.0 );
  vec3 dy = vec3( 0.0 , e   , 0.0 );
  vec3 dz = vec3( 0.0 , 0.0 , e   );

  vec3 p_x0 = snoiseVec3( p - dx );
  vec3 p_x1 = snoiseVec3( p + dx );
  vec3 p_y0 = snoiseVec3( p - dy );
  vec3 p_y1 = snoiseVec3( p + dy );
  vec3 p_z0 = snoiseVec3( p - dz );
  vec3 p_z1 = snoiseVec3( p + dz );

  float x = p_y1.z - p_y0.z - p_z1.y + p_z0.y;
  float y = p_z1.x - p_z0.x - p_x1.z + p_x0.z;
  float z = p_x1.y - p_x0.y - p_y1.x + p_y0.x;

  const float divisor = 1.0 / ( 2.0 * e );
  return normalize( vec3( x , y , z ) * divisor );
}

void main() {
  vColor = customColor;
  
  // Calculate curl noise displacement
  vec3 pos = offset;
  vec3 noisePos = pos * uScale + vec3(0.0, uTime * uSpeed, 0.0);
  vec3 curl = curlNoise(noisePos);
  
  // Advect particle
  vec3 advectedPos = pos + curl * uSwirl;
  
  // Instance transform
  vec4 mvPosition = modelViewMatrix * vec4(advectedPos, 1.0);
  
  // Add local geometry (for the instanced mesh)
  mvPosition.xyz += position * size;
  
  gl_Position = projectionMatrix * mvPosition;
}
`;

const curlSwarmFragmentShader = `
varying vec3 vColor;

void main() {
  // Simple soft circle
  vec2 coord = gl_PointCoord - vec2(0.5);
  float dist = length(coord);
  if(dist > 0.5) discard;
  
  float alpha = smoothstep(0.5, 0.1, dist);
  
  // For instanced mesh, we don't have gl_PointCoord, so we just use a flat color
  // or we can compute lighting. Here we just output the color.
  gl_FragColor = vec4(vColor, 1.0);
}
`;

const curlSwarmInstancedFragmentShader = `
varying vec3 vColor;

void main() {
  // Basic shading for the instanced geometry
  // We assume a simple unlit material for glowing particles
  gl_FragColor = vec4(vColor, 1.0);
}
`;

export interface CurlSwarmOptions {
  count?: number;
  radius?: number;
  color1?: number;
  color2?: number;
  sizeMin?: number;
  sizeMax?: number;
  speed?: number;
  scale?: number;
  swirl?: number;
}

export function createCurlSwarm(opts: CurlSwarmOptions = {}): THREE.Group {
  const count = opts.count ?? 10000;
  const radius = opts.radius ?? 5.0;
  const color1 = new THREE.Color(opts.color1 ?? 0x00ffff);
  const color2 = new THREE.Color(opts.color2 ?? 0xff00ff);
  
  const sizeMin = opts.sizeMin ?? 0.02;
  const sizeMax = opts.sizeMax ?? 0.08;

  const geometry = new THREE.TetrahedronGeometry(1, 0);
  const instancedGeo = new THREE.InstancedBufferGeometry();
  instancedGeo.index = geometry.index;
  instancedGeo.attributes.position = geometry.attributes.position;
  instancedGeo.attributes.normal = geometry.attributes.normal;

  const offsets = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  const sizes = new Float32Array(count);

  for (let i = 0; i < count; i++) {
    // Random point in sphere
    const u = Math.random();
    const v = Math.random();
    const theta = u * 2.0 * Math.PI;
    const phi = Math.acos(2.0 * v - 1.0);
    const r = Math.cbrt(Math.random()) * radius;

    const sinPhi = Math.sin(phi);
    const x = r * sinPhi * Math.cos(theta);
    const y = r * sinPhi * Math.sin(theta);
    const z = r * Math.cos(phi);

    offsets[i * 3 + 0] = x;
    offsets[i * 3 + 1] = y;
    offsets[i * 3 + 2] = z;

    const mixedColor = color1.clone().lerp(color2, Math.random());
    colors[i * 3 + 0] = mixedColor.r;
    colors[i * 3 + 1] = mixedColor.g;
    colors[i * 3 + 2] = mixedColor.b;

    sizes[i] = sizeMin + Math.random() * (sizeMax - sizeMin);
  }

  instancedGeo.setAttribute('offset', new THREE.InstancedBufferAttribute(offsets, 3));
  instancedGeo.setAttribute('customColor', new THREE.InstancedBufferAttribute(colors, 3));
  instancedGeo.setAttribute('size', new THREE.InstancedBufferAttribute(sizes, 1));

  const material = new THREE.ShaderMaterial({
    vertexShader: curlSwarmVertexShader,
    fragmentShader: curlSwarmInstancedFragmentShader,
    uniforms: {
      uTime: { value: 0 },
      uSpeed: { value: opts.speed ?? 0.2 },
      uScale: { value: opts.scale ?? 0.5 },
      uSwirl: { value: opts.swirl ?? 1.5 },
    },
    transparent: true,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });

  const mesh = new THREE.Mesh(instancedGeo, material);
  mesh.frustumCulled = false;

  const group = new THREE.Group();
  group.add(mesh);

  // Attach update function to user data so it can be called each frame
  group.userData.update = (dt: number, t: number) => {
    material.uniforms.uTime.value = t;
  };

  return group;
}
