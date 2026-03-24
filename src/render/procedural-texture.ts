import * as THREE from 'three'

/**
 * GPU-side procedural texture generator — WebGL2 / GLSL ES 3.0.
 *
 * All textures rendered into WebGLRenderTarget via a full-screen quad.
 * Reaction-diffusion (Gray-Scott) uses ping-pong double-buffer iteration.
 *
 * Domains:
 *   atomic   — quantum wavefunction density / phase / interference
 *   cellular — voronoi membranes, reaction-diffusion, cytoskeleton, mitochondria
 *   material — crystal lattice, thin-film, grain boundary, dislocation field
 */

export type TextureDomain = 'atomic' | 'cellular' | 'material'

export interface ProceduralTextureOpts {
  domain: TextureDomain
  preset: string
  resolution?: number
  params?: Record<string, number>
  animate?: boolean
  /** Bake once at full resolution. updateTime becomes a no-op for static presets.
   *  Enables mipmaps for crisp results at any display size. Default false. */
  bake?: boolean
}

export interface ProceduralTextureHandle {
  texture: THREE.Texture
  dispose: () => void
  updateTime: (t: number) => void
}

// ── Shared quad infrastructure ────────────────────────────────────────────────

const QUAD_GEO   = new THREE.PlaneGeometry(2, 2)
const QUAD_CAM   = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1)
const QUAD_SCENE = new THREE.Scene()

const VERT_QUAD = /* glsl */`
uniform float uTime;
out vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`

// ── Shared GLSL library chunks ────────────────────────────────────────────────

const GLSL_UTIL = /* glsl */`
const float PI = 3.14159265358979;
float hash(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}
vec2  hash2(vec2 p){return fract(sin(vec2(dot(p,vec2(127.1,311.7)),dot(p,vec2(269.5,183.3))))*43758.5453);}
float noise(vec2 p){
  vec2 i=floor(p),f=fract(p),u=f*f*(3.0-2.0*f);
  return mix(mix(hash(i),hash(i+vec2(1,0)),u.x),mix(hash(i+vec2(0,1)),hash(i+vec2(1,1)),u.x),u.y);
}
float fbm(vec2 p,int oct){float v=0.,a=0.5;for(int i=0;i<8;i++){if(i>=oct)break;v+=a*noise(p);p*=2.1;a*=0.5;}return v;}
float worley(vec2 p,float jit){
  vec2 i=floor(p);float md=9e9;
  for(int y=-1;y<=1;y++)for(int x=-1;x<=1;x++){vec2 n=i+vec2(x,y);vec2 f=n+jit*hash2(n);md=min(md,length(p-f));}
  return md;
}
vec3 hsv2rgb(vec3 c){
  vec3 p=abs(fract(c.xxx+vec3(0,2./3.,1./3.))*6.-3.);
  return c.z*mix(vec3(1),clamp(p-1.,0.,1.),c.y);
}
vec3 wavelengthToRGB(float nm){
  float r=0.,g=0.,b=0.;
  if(nm>=380.&&nm<440.){r=-(nm-440.)/60.;b=1.;}
  else if(nm<490.){g=(nm-440.)/50.;b=1.;}
  else if(nm<510.){g=1.;b=-(nm-510.)/20.;}
  else if(nm<580.){r=(nm-510.)/70.;g=1.;}
  else if(nm<645.){r=1.;g=-(nm-645.)/65.;}
  else if(nm<=780.){r=1.;}
  float f=1.;
  if(nm<420.) f=0.3+0.7*(nm-380.)/40.;
  else if(nm>700.) f=0.3+0.7*(780.-nm)/80.;
  return pow(vec3(r,g,b)*f,vec3(0.8));
}
`

const GLSL_HYDROGEN = /* glsl */`
float fact(int n){float r=1.;for(int i=2;i<=12;i++){if(i>n)break;r*=float(i);}return r;}
float laguerre(int n,float alpha,float x){
  float s=0.;
  for(int j=0;j<=8;j++){
    if(j>n)break;
    float bin=fact(n+int(alpha))/(fact(n-j)*fact(int(alpha)+j)*fact(j));
    s+=(mod(float(j),2.)==0.?1.:-1.)*bin*pow(x,float(j));
  }
  return s;
}
float R_nl(int n,int l,float r){
  float rho=2.*r/float(n);
  float norm=sqrt(pow(2./float(n),3.)*fact(n-l-1)/(2.*float(n)*pow(fact(n+l),3.)));
  return norm*exp(-rho*.5)*pow(rho,float(l))*laguerre(n-l-1,float(2*l+1),rho);
}
float P_lm(int l,int m,float x){
  int ma=abs(m);float pmm=1.;
  if(ma>0){float xm=sqrt(max(0.,(1.-x)*(1.+x))),f=1.;for(int i=1;i<=6;i++){if(i>ma)break;pmm*=-f*xm;f+=2.;}}
  if(l==ma)return pmm;
  float pmmp1=x*float(2*ma+1)*pmm;
  if(l==ma+1)return pmmp1;
  float pll=0.;
  for(int ll=ma+2;ll<=6;ll++){if(ll>l)break;pll=(x*float(2*ll-1)*pmmp1-float(ll+ma-1)*pmm)/float(ll-ma);pmm=pmmp1;pmmp1=pll;}
  return pmmp1;
}
float Y_real(int l,int m,float theta,float phi){
  int ma=abs(m);
  float norm=sqrt((float(2*l+1)/PI*4.)*fact(l-ma)/fact(l+ma));
  float Plm=P_lm(l,m,cos(theta));
  if(m==0)return norm*Plm;
  if(m>0)return sqrt(2.)*norm*Plm*cos(float(ma)*phi);
  return sqrt(2.)*norm*Plm*sin(float(ma)*phi);
}
float psiDensity(int n,int l,int m,float r,float theta,float phi){
  float R=R_nl(n,l,r),Y=Y_real(l,m,theta,phi);
  return R*R*Y*Y;
}
`

// ══ ATOMIC ════════════════════════════════════════════════════════════════════

const FRAG_ORBITAL_DENSITY = /* glsl */`precision highp float;
${GLSL_UTIL}${GLSL_HYDROGEN}
uniform float uTime,uN,uL,uM,uSliceZ;
in vec2 vUv; out vec4 fragColor;
void main(){
  vec2 p=(vUv-.5)*24.; float z=uSliceZ*12.;
  float r=length(vec3(p,z)); if(r<.001)r=.001;
  float theta=acos(clamp(z/r,-1.,1.)),phi=atan(p.y,p.x);
  float d=psiDensity(int(uN),int(uL),int(uM),r,theta,phi);
  d=log(1.+d*900.)/log(900.);
  float hue=fract(uL*.25+float(int(uM)+4)*.07);
  fragColor=vec4(hsv2rgb(vec3(hue,.75,d)),1.);
}
`

const FRAG_ORBITAL_PHASE = /* glsl */`precision highp float;
${GLSL_UTIL}${GLSL_HYDROGEN}
uniform float uTime,uN,uL,uM,uSliceZ;
in vec2 vUv; out vec4 fragColor;
void main(){
  vec2 p=(vUv-.5)*24.; float z=uSliceZ*12.;
  float r=length(vec3(p,z)); if(r<.001)r=.001;
  float theta=acos(clamp(z/r,-1.,1.)),phi=atan(p.y,p.x)+uTime*.25;
  float d=psiDensity(int(uN),int(uL),int(uM),r,theta,phi);
  float phase=phi*uM/(2.*PI)+uTime*.12;
  d=log(1.+d*700.)/log(700.);
  fragColor=vec4(hsv2rgb(vec3(fract(phase),.9,d)),1.);
}
`

const FRAG_INTERFERENCE = /* glsl */`precision highp float;
${GLSL_UTIL}${GLSL_HYDROGEN}
uniform float uTime,uN1,uL1,uM1,uN2,uL2,uM2,uMix,uSliceZ;
in vec2 vUv; out vec4 fragColor;
void main(){
  vec2 p=(vUv-.5)*24.; float z=uSliceZ*12.;
  float r=length(vec3(p,z)); if(r<.001)r=.001;
  float theta=acos(clamp(z/r,-1.,1.)),phi=atan(p.y,p.x);
  float E1=-1./(2.*uN1*uN1),E2=-1./(2.*uN2*uN2);
  float A1=R_nl(int(uN1),int(uL1),r)*Y_real(int(uL1),int(uM1),theta,phi);
  float A2=R_nl(int(uN2),int(uL2),r)*Y_real(int(uL2),int(uM2),theta,phi);
  float w1=sqrt(1.-uMix),w2=sqrt(uMix);
  float sr=w1*A1*cos(-E1*uTime)+w2*A2*cos(-E2*uTime);
  float si=w1*A1*sin(-E1*uTime)+w2*A2*sin(-E2*uTime);
  float d=sr*sr+si*si;
  d=log(1.+d*700.)/log(700.);
  float phase=atan(si,sr);
  fragColor=vec4(hsv2rgb(vec3(fract(phase/(2.*PI)),.88,d)),1.);
}
`

const FRAG_RADIAL_PROB = /* glsl */`precision highp float;
${GLSL_UTIL}${GLSL_HYDROGEN}
uniform float uTime,uN,uL;
in vec2 vUv; out vec4 fragColor;
void main(){
  float r=vUv.x*22.;
  float Rnl=R_nl(int(uN),int(uL),r);
  float prob=4.*PI*r*r*Rnl*Rnl;
  float val=log(1.+prob*80.)/log(80.);
  float y=vUv.y;
  float line=smoothstep(.014,.0,abs(y-val));
  float bohr=0.;
  for(int ni=1;ni<=6;ni++){bohr+=smoothstep(.005,.0,abs(vUv.x-float(ni*ni)/22.))*.55;}
  float hue=.55+vUv.x*.28+uTime*.015;
  vec3 base=hsv2rgb(vec3(fract(hue),.6,val*.35));
  fragColor=vec4(base+vec3(.9,.95,1.)*line+vec3(1.,.6,.2)*bohr,1.);
}
`

const FRAG_ELECTRON_CLOUD = /* glsl */`precision highp float;
${GLSL_UTIL}${GLSL_HYDROGEN}
uniform float uTime,uN,uL,uM;
in vec2 vUv; out vec4 fragColor;
void main(){
  vec2 p=(vUv-.5)*24.;
  float accum=0.;
  for(int iz=-3;iz<=3;iz++){
    float z=float(iz)*2.;
    float r=length(vec3(p,z)); if(r<.001)r=.001;
    float theta=acos(clamp(z/r,-1.,1.)),phi=atan(p.y,p.x)+uTime*.08*float(iz+4);
    float d=psiDensity(int(uN),int(uL),int(uM),r,theta,phi);
    accum+=d/7.;
  }
  float th=.02+noise(p*.7+uTime*.1)*.015;
  accum=log(1.+accum*1200.)/log(1200.);
  float hue=fract(.6+accum*.4+uTime*.04);
  vec3 col=hsv2rgb(vec3(hue,.8,accum));
  fragColor=vec4(col,1.);
}
`

// ══ CELLULAR ══════════════════════════════════════════════════════════════════

const FRAG_VORONOI_MEMBRANE = /* glsl */`precision highp float;
${GLSL_UTIL}
uniform float uTime,uCellScale,uMembraneWidth,uJitter;
in vec2 vUv; out vec4 fragColor;
void main(){
  vec2 p=vUv*uCellScale+vec2(uTime*.03);
  float d=worley(p,uJitter);
  float membrane=1.-smoothstep(0.,uMembraneWidth,d);
  // cell interior: slight noise variation
  float interior=fbm(p*3.,4.);
  vec2 ci=floor(p); float cellHue=hash(ci);
  // lipid bilayer: two bright rings
  float bilayer=smoothstep(uMembraneWidth*.3,.0,abs(d-uMembraneWidth*.5));
  vec3 cellCol=hsv2rgb(vec3(cellHue,.25,.12+interior*.1));
  vec3 memCol=hsv2rgb(vec3(.13+cellHue*.2,.5,.95))*membrane;
  vec3 bilayerCol=vec3(.9,.95,1.)*bilayer*1.5;
  fragColor=vec4(cellCol+memCol+bilayerCol,1.);
}
`

const FRAG_RD_INIT = /* glsl */`precision highp float;
${GLSL_UTIL}
uniform float uTime;
in vec2 vUv; out vec4 fragColor;
void main(){
  float n=noise(vUv*18.+3.7)*noise(vUv*7.+1.2);
  float a=1.-smoothstep(.3,.6,n);
  float b=smoothstep(.35,.55,n);
  fragColor=vec4(a,b,0.,1.);
}
`

const FRAG_RD_STEP = /* glsl */`precision highp float;
${GLSL_UTIL}
uniform sampler2D uState;
uniform float uFeed,uKill,uDa,uDb,uDt;
in vec2 vUv; out vec4 fragColor;
void main(){
  vec2 texel=1./vec2(textureSize(uState,0));
  vec4 c=texture(uState,vUv);
  vec4 lap=
    texture(uState,vUv+texel*vec2(-1,-1))*.05+
    texture(uState,vUv+texel*vec2( 0,-1))*.2+
    texture(uState,vUv+texel*vec2( 1,-1))*.05+
    texture(uState,vUv+texel*vec2(-1, 0))*.2+
    c*(-1.)+
    texture(uState,vUv+texel*vec2( 1, 0))*.2+
    texture(uState,vUv+texel*vec2(-1, 1))*.05+
    texture(uState,vUv+texel*vec2( 0, 1))*.2+
    texture(uState,vUv+texel*vec2( 1, 1))*.05;
  float a=c.r,b=c.g;
  float reaction=a*b*b;
  float na=a+uDt*(uDa*lap.r-reaction+uFeed*(1.-a));
  float nb=b+uDt*(uDb*lap.g+reaction-(uKill+uFeed)*b);
  fragColor=vec4(clamp(na,0.,1.),clamp(nb,0.,1.),0.,1.);
}
`

const FRAG_RD_DISPLAY = /* glsl */`precision highp float;
${GLSL_UTIL}
uniform sampler2D uState;
uniform float uTime;
in vec2 vUv; out vec4 fragColor;
void main(){
  vec2 cv=texture(uState,vUv).rg;
  float v=cv.r-cv.g;
  float hue=.6+v*.5+uTime*.01;
  vec3 col=hsv2rgb(vec3(fract(hue),.85,smoothstep(-.1,.65,v)));
  fragColor=vec4(col,1.);
}
`

const FRAG_CYTOSKELETON = /* glsl */`precision highp float;
${GLSL_UTIL}
uniform float uTime,uFiberDensity,uThickness;
in vec2 vUv; out vec4 fragColor;
void main(){
  vec2 p=vUv*uFiberDensity;
  vec2 warp=vec2(fbm(p+uTime*.04,5),fbm(p+vec2(5.2,1.3)+uTime*.04,5));
  vec2 wp=p+1.2*warp;
  const float PI2=3.14159265;
  float f1=abs(sin(wp.x*PI2+fbm(wp*.5,4)*3.));
  float f2=abs(sin(wp.y*PI2+fbm(wp*.5+3.7,4)*3.));
  float fiber=smoothstep(uThickness+.02,uThickness,min(f1,f2));
  float glow=smoothstep(.25,.0,min(f1,f2))*.3;
  vec3 actin=hsv2rgb(vec3(.45+fbm(wp*.3,3)*.1,.7,.9+glow));
  vec3 micro=hsv2rgb(vec3(.08,.8,smoothstep(.06,.0,min(f1,f2))*.8));
  fragColor=vec4(actin*fiber+micro+vec3(0.,.04,.06)*fbm(p,4),1.);
}
`

const FRAG_MITOCHONDRIA = /* glsl */`precision highp float;
${GLSL_UTIL}
uniform float uTime;
in vec2 vUv; out vec4 fragColor;
void main(){
  vec2 p=vUv*6.+vec2(uTime*.015);
  float outer=worley(p,0.75);
  vec2 warp=vec2(fbm(p*2.,4),fbm(p*2.+3.7,4))*.4;
  float inner=worley(p*4.+warp,0.65);
  float outerMem=smoothstep(.08,.0,outer-.04);
  float innerMem=smoothstep(.06,.0,inner-.03);
  float matrix=smoothstep(.3,.8,outer)*(1.-innerMem);
  vec3 matrixCol=vec3(.18,.06,.02)*matrix;
  vec3 cristaeCol=hsv2rgb(vec3(.07,.9,.95))*innerMem;
  vec3 outerCol=vec3(.8,.5,.1)*outerMem;
  fragColor=vec4(matrixCol+cristaeCol+outerCol,1.);
}
`

const FRAG_CRYSTAL_LATTICE = /* glsl */`precision highp float;
${GLSL_UTIL}
uniform float uTime,uA,uB,uAngle,uHklH,uHklK;
in vec2 vUv; out vec4 fragColor;
void main(){
  const float PI3=3.14159265;
  vec2 p=(vUv-.5)*12.;
  float ca=cos(uAngle),sa=sin(uAngle);
  vec2 b1=vec2(ca,sa)/uA,b2=vec2(-sa,ca)/uB;
  float h1=cos(2.*PI3*(dot(p,b1)*uHklH));
  float h2=cos(2.*PI3*(dot(p,b2)*uHklK));
  float lattice=(h1+h2)*.5;
  float spots=0.;
  for(int hh=-3;hh<=3;hh++)for(int kk=-3;kk<=3;kk++){
    vec2 G=b1*float(hh)+b2*float(kk);
    float dist=length(p-G*8.);
    spots+=exp(-dist*dist*.8)*float((hh*hh+kk*kk)>0?1:0);
  }
  float hue=.6+lattice*.2+uTime*.005;
  fragColor=vec4(hsv2rgb(vec3(fract(hue),.7,clamp(lattice*.5+.5+spots*.4,0.,1.))),1.);
}
`

const FRAG_THIN_FILM = /* glsl */`precision highp float;
${GLSL_UTIL}
uniform float uTime,uThicknessNm,uNFilm,uNSubstrate;
in vec2 vUv; out vec4 fragColor;
void main(){
  const float PI4=3.14159265;
  float t=uThicknessNm*(1.+.4*sin(vUv.x*PI4*3.+uTime*.4))*(1.+.3*cos(vUv.y*PI4*2.+uTime*.3));
  float cosTheta=cos((vUv.x-.5)*PI4*.7);
  vec3 col=vec3(0.);
  for(int i=0;i<16;i++){
    float nm=380.+float(i)*(700.-380.)/15.;
    float sinT2=pow((1./uNFilm),2.)*(1.-cosTheta*cosTheta);
    float cosThT=sqrt(max(0.,1.-sinT2));
    float opd=2.*uNFilm*t*cosThT;
    float phi=2.*PI4*opd/nm;
    float r12=(1.-uNFilm)/(1.+uNFilm),r12s=r12*r12;
    float r23=(uNFilm-uNSubstrate)/(uNFilm+uNSubstrate),r23s=r23*r23;
    float R=(r12s+r23s+2.*r12*r23*cos(phi))/(1.+r12s*r23s+2.*r12*r23*cos(phi));
    col+=wavelengthToRGB(nm)*R;
  }
  fragColor=vec4(clamp(col/8.,0.,1.),1.);
}
`

const FRAG_GRAIN_BOUNDARY = /* glsl */`precision highp float;
${GLSL_UTIL}
uniform float uTime,uGrainCount,uBoundarySharpness;
in vec2 vUv; out vec4 fragColor;
void main(){
  vec2 p=vUv*uGrainCount;
  vec2 ip=floor(p);float d1=9e9,d2=9e9;
  vec2 site1=vec2(0.);
  for(int y=-2;y<=2;y++)for(int x=-2;x<=2;x++){
    vec2 n=ip+vec2(x,y),site=n+hash2(n);
    float d=length(p-site);
    if(d<d1){d2=d1;d1=d;site1=site;}else if(d<d2){d2=d;}
  }
  float boundary=smoothstep(uBoundarySharpness*.02,.0,d2-d1);
  float grainAngle=hash(floor(site1))*3.14159;
  float aniso=abs(cos(2.*(atan(p.y-site1.y,p.x-site1.x)-grainAngle)));
  float hue=hash(floor(site1));
  vec3 grainCol=hsv2rgb(vec3(hue,.5,.3+aniso*.5));
  fragColor=vec4(mix(grainCol,vec3(.04,.04,.07),boundary),1.);
}
`

const FRAG_DISLOCATION = /* glsl */`precision highp float;
${GLSL_UTIL}
uniform float uTime,uDislocationCount,uBurgers;
in vec2 vUv; out vec4 fragColor;
void main(){
  const float PI5=3.14159265;
  vec2 p=(vUv-.5)*10.;
  float stress=0.;
  for(int i=0;i<8;i++){
    if(float(i)>=uDislocationCount)break;
    float a=float(i)/uDislocationCount*PI5*2.;
    vec2 core=vec2(cos(a+uTime*.05),sin(a+uTime*.05))*3.;
    vec2 r=p-core;
    float rsq=dot(r,r)+.01;
    float nu=0.3;
    float pre=uBurgers/(2.*PI5*(1.-nu));
    float sxy=pre*r.x*(r.x*r.x-r.y*r.y)/(rsq*rsq);
    stress+=sxy;
  }
  float val=clamp(stress*.4+.5,0.,1.);
  float hue=val*.7;
  // slip plane lines
  float slip=smoothstep(.02,.0,abs(mod(p.y+uTime*.02,.8)-.4))*.4;
  fragColor=vec4(hsv2rgb(vec3(hue,.9,val))+vec3(slip),1.);
}
`

// ── TypeScript runtime ────────────────────────────────────────────────────────

type UniformMap = Record<string, THREE.IUniform>

function makeRT(res: number, bake = false): THREE.WebGLRenderTarget {
  const rt = new THREE.WebGLRenderTarget(res, res, {
    type: THREE.HalfFloatType,
    minFilter: bake ? THREE.LinearMipmapLinearFilter : THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
    wrapS: THREE.RepeatWrapping,
    wrapT: THREE.RepeatWrapping,
    generateMipmaps: bake,
  })
  return rt
}

function makeMat(frag: string, uniforms: UniformMap): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    vertexShader: VERT_QUAD,
    fragmentShader: frag,
    uniforms,
    glslVersion: THREE.GLSL3,
    depthTest: false,
    depthWrite: false,
  })
}

function renderQuad(renderer: THREE.WebGLRenderer, mat: THREE.ShaderMaterial, target: THREE.WebGLRenderTarget): void {
  const mesh = new THREE.Mesh(QUAD_GEO, mat)
  QUAD_SCENE.clear()
  QUAD_SCENE.add(mesh)
  const prevTarget = renderer.getRenderTarget()
  renderer.setRenderTarget(target)
  renderer.render(QUAD_SCENE, QUAD_CAM)
  renderer.setRenderTarget(prevTarget)
  QUAD_SCENE.clear()
}

function paramsGet(p: Record<string, number> | undefined, key: string, def: number): number {
  return p && key in p ? p[key] : def
}

// ── Preset builders ───────────────────────────────────────────────────────────

function buildAtomic(
  renderer: THREE.WebGLRenderer,
  preset: string,
  res: number,
  params: Record<string, number> | undefined,
  bake = false
): ProceduralTextureHandle {
  const rt = makeRT(res, bake)
  let frag: string
  let uniforms: UniformMap

  if (preset === 'orbital_phase') {
    frag = FRAG_ORBITAL_PHASE
    uniforms = {
      uTime:   { value: 0 },
      uN:      { value: paramsGet(params, 'n', 2) },
      uL:      { value: paramsGet(params, 'l', 1) },
      uM:      { value: paramsGet(params, 'm', 0) },
      uSliceZ: { value: paramsGet(params, 'slice_z', 0) },
    }
  } else if (preset === 'interference') {
    frag = FRAG_INTERFERENCE
    uniforms = {
      uTime:   { value: 0 },
      uN1:     { value: paramsGet(params, 'n1', 2) },
      uL1:     { value: paramsGet(params, 'l1', 1) },
      uM1:     { value: paramsGet(params, 'm1', 0) },
      uN2:     { value: paramsGet(params, 'n2', 3) },
      uL2:     { value: paramsGet(params, 'l2', 2) },
      uM2:     { value: paramsGet(params, 'm2', 1) },
      uMix:    { value: paramsGet(params, 'mix', 0.5) },
      uSliceZ: { value: paramsGet(params, 'slice_z', 0) },
    }
  } else if (preset === 'radial_probability') {
    frag = FRAG_RADIAL_PROB
    uniforms = {
      uTime: { value: 0 },
      uN:    { value: paramsGet(params, 'n', 2) },
      uL:    { value: paramsGet(params, 'l', 1) },
    }
  } else if (preset === 'electron_cloud') {
    frag = FRAG_ELECTRON_CLOUD
    uniforms = {
      uTime: { value: 0 },
      uN:    { value: paramsGet(params, 'n', 3) },
      uL:    { value: paramsGet(params, 'l', 2) },
      uM:    { value: paramsGet(params, 'm', 0) },
    }
  } else {
    // default: orbital_density
    frag = FRAG_ORBITAL_DENSITY
    uniforms = {
      uTime:   { value: 0 },
      uN:      { value: paramsGet(params, 'n', 2) },
      uL:      { value: paramsGet(params, 'l', 1) },
      uM:      { value: paramsGet(params, 'm', 0) },
      uSliceZ: { value: paramsGet(params, 'slice_z', 0) },
    }
  }

  const mat = makeMat(frag, uniforms)
  renderQuad(renderer, mat, rt)
  // For baked static presets, generate mipmaps immediately
  if (bake) renderer.initTexture(rt.texture)

  return {
    texture: rt.texture,
    dispose() { rt.dispose(); mat.dispose() },
    updateTime(t) {
      if (bake) return // baked — never re-render
      uniforms.uTime.value = t
      renderQuad(renderer, mat, rt)
    },
  }
}

function buildCellular(
  renderer: THREE.WebGLRenderer,
  preset: string,
  res: number,
  params: Record<string, number> | undefined,
  bake = false
): ProceduralTextureHandle {
  // Reaction-diffusion: ping-pong
  // Sim buffer capped at 512 regardless of display resolution — keeps GPU cost constant
  if (preset === 'reaction_diffusion') {
    const simRes = Math.min(res, 512)
    const rtA = makeRT(simRes)
    const rtB = makeRT(simRes)

    // Init
    const initUniforms: UniformMap = { uTime: { value: 0 } }
    const initMat = makeMat(FRAG_RD_INIT, initUniforms)
    renderQuad(renderer, initMat, rtA)
    initMat.dispose()

    const stepUniforms: UniformMap = {
      uState: { value: rtA.texture },
      uFeed:  { value: paramsGet(params, 'feed', 0.037) },
      uKill:  { value: paramsGet(params, 'kill', 0.06) },
      uDa:    { value: paramsGet(params, 'diffusion_a', 1.0) },
      uDb:    { value: paramsGet(params, 'diffusion_b', 0.5) },
      uDt:    { value: paramsGet(params, 'dt', 1.0) },
    }
    const stepMat = makeMat(FRAG_RD_STEP, stepUniforms)

    const dispUniforms: UniformMap = {
      uState: { value: rtA.texture },
      uTime:  { value: 0 },
    }
    const dispMat = makeMat(FRAG_RD_DISPLAY, dispUniforms)
    // Display RT at full requested resolution — sim stays at simRes
    const displayRT = makeRT(res, bake)
    renderQuad(renderer, dispMat, displayRT)

    let ping = rtA, pong = rtB

    return {
      texture: displayRT.texture,
      dispose() {
        rtA.dispose(); rtB.dispose(); displayRT.dispose()
        stepMat.dispose(); dispMat.dispose()
      },
      updateTime(t) {
        // 8 Gray-Scott steps on the capped sim buffer (cheap regardless of display res)
        for (let i = 0; i < 8; i++) {
          stepUniforms.uState.value = ping.texture
          renderQuad(renderer, stepMat, pong)
          const tmp = ping; ping = pong; pong = tmp
        }
        // Blit sim result to full-res display target
        dispUniforms.uState.value = ping.texture
        dispUniforms.uTime.value = t
        renderQuad(renderer, dispMat, displayRT)
      },
    }
  }

  // Single-pass cellular presets
  let frag: string
  let uniforms: UniformMap

  if (preset === 'cytoskeleton') {
    frag = FRAG_CYTOSKELETON
    uniforms = {
      uTime:        { value: 0 },
      uFiberDensity:{ value: paramsGet(params, 'fiber_density', 8) },
      uThickness:   { value: paramsGet(params, 'thickness', 0.04) },
    }
  } else if (preset === 'mitochondria') {
    frag = FRAG_MITOCHONDRIA
    uniforms = { uTime: { value: 0 } }
  } else {
    // default: voronoi_membrane
    frag = FRAG_VORONOI_MEMBRANE
    uniforms = {
      uTime:          { value: 0 },
      uCellScale:     { value: paramsGet(params, 'cell_scale', 7) },
      uMembraneWidth: { value: paramsGet(params, 'membrane_width', 0.08) },
      uJitter:        { value: paramsGet(params, 'jitter', 0.85) },
    }
  }

  const rt = makeRT(res, bake)
  const mat = makeMat(frag, uniforms)
  renderQuad(renderer, mat, rt)
  if (bake) renderer.initTexture(rt.texture)

  return {
    texture: rt.texture,
    dispose() { rt.dispose(); mat.dispose() },
    updateTime(t) {
      if (bake) return
      uniforms.uTime.value = t
      renderQuad(renderer, mat, rt)
    },
  }
}

function buildMaterial(
  renderer: THREE.WebGLRenderer,
  preset: string,
  res: number,
  params: Record<string, number> | undefined,
  bake = false
): ProceduralTextureHandle {
  let frag: string
  let uniforms: UniformMap

  if (preset === 'thin_film') {
    frag = FRAG_THIN_FILM
    uniforms = {
      uTime:          { value: 0 },
      uThicknessNm:   { value: paramsGet(params, 'thickness_nm', 400) },
      uNFilm:         { value: paramsGet(params, 'n_film', 1.5) },
      uNSubstrate:    { value: paramsGet(params, 'n_substrate', 1.0) },
    }
  } else if (preset === 'grain_boundary') {
    frag = FRAG_GRAIN_BOUNDARY
    uniforms = {
      uTime:              { value: 0 },
      uGrainCount:        { value: paramsGet(params, 'grain_count', 12) },
      uBoundarySharpness: { value: paramsGet(params, 'boundary_sharpness', 1.0) },
    }
  } else if (preset === 'dislocation_field') {
    frag = FRAG_DISLOCATION
    uniforms = {
      uTime:              { value: 0 },
      uDislocationCount:  { value: paramsGet(params, 'dislocation_count', 4) },
      uBurgers:           { value: paramsGet(params, 'burgers', 1.0) },
    }
  } else {
    // default: crystal_lattice
    frag = FRAG_CRYSTAL_LATTICE
    uniforms = {
      uTime:   { value: 0 },
      uA:      { value: paramsGet(params, 'a', 1.0) },
      uB:      { value: paramsGet(params, 'b', 1.2) },
      uAngle:  { value: paramsGet(params, 'angle', 0) },
      uHklH:   { value: paramsGet(params, 'hkl_h', 1) },
      uHklK:   { value: paramsGet(params, 'hkl_k', 1) },
    }
  }

  const rt = makeRT(res, bake)
  const mat = makeMat(frag, uniforms)
  renderQuad(renderer, mat, rt)
  if (bake) renderer.initTexture(rt.texture)

  return {
    texture: rt.texture,
    dispose() { rt.dispose(); mat.dispose() },
    updateTime(t) {
      if (bake) return
      uniforms.uTime.value = t
      renderQuad(renderer, mat, rt)
    },
  }
}

// ── Public API ────────────────────────────────────────────────────────────────

export function buildProceduralTexture(
  renderer: THREE.WebGLRenderer,
  opts: ProceduralTextureOpts
): ProceduralTextureHandle {
  const res  = Math.min(4096, Math.max(64, opts.resolution ?? 512))
  const bake = opts.bake ?? false
  const p    = opts.params

  switch (opts.domain) {
    case 'atomic':   return buildAtomic(renderer, opts.preset, res, p, bake)
    case 'cellular': return buildCellular(renderer, opts.preset, res, p, bake)
    case 'material': return buildMaterial(renderer, opts.preset, res, p, bake)
    default: throw new Error(`Unknown texture domain: ${String(opts.domain)}`)
  }
}
