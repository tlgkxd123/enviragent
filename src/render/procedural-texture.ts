import * as THREE from 'three'

/**
 * GPU-side procedural texture generator — WebGL2 / GLSL ES 3.0.
 *
 * All textures rendered into WebGLRenderTarget via a full-screen quad.
 * Reaction-diffusion (Gray-Scott) uses ping-pong double-buffer iteration.
 *
 * Domains:
 *   cellular — voronoi membranes, reaction-diffusion, cytoskeleton, mitochondria
 *   material — crystal lattice, thin-film, grain boundary, dislocation field
 *   surface  — PBR map stacks (albedo + normal + ORM) from presets
 */

export type TextureDomain = 'cellular' | 'material' | 'surface'

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
  normalMap?: THREE.Texture
  ormMap?: THREE.Texture
  emissiveMap?: THREE.Texture
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

// ══ SURFACE — PBR hyperreal materials ═════════════════════════════════════════
//
// Each preset outputs albedo / tangent-space normal / ORM (AO, Roughness,
// Metalness packed R/G/B per glTF convention) via a `uOutputMode` int uniform.
// Normals are computed analytically from gradient-noise derivatives (no finite
// differences) for perfect smooth results at any resolution.

const GLSL_PBR_NOISE = /* glsl */`
const float PI=3.14159265;

// ── robust hashes (Dave Hoskins) ──────────────────────────────────────────
vec2 hash22(vec2 p){
  vec3 p3=fract(vec3(p.xyx)*vec3(.1031,.1030,.0973));
  p3+=dot(p3,p3.yzx+33.33);
  return fract((p3.xx+p3.yz)*p3.zy)*2.-1.;
}
float hash21(vec2 p){
  vec3 p3=fract(vec3(p.xyx)*.1031);
  p3+=dot(p3,p3.yzx+33.33);
  return fract((p3.x+p3.y)*p3.z);
}

// ── 2D gradient noise with analytical derivatives → (value, dv/dx, dv/dy) ─
vec3 noised2(vec2 x){
  vec2 i=floor(x),f=fract(x);
  vec2 u=f*f*f*(f*(f*6.-15.)+10.);
  vec2 du=30.*f*f*(f*(f-2.)+1.);
  vec2 ga=hash22(i),gb=hash22(i+vec2(1,0)),
       gc=hash22(i+vec2(0,1)),gd=hash22(i+vec2(1,1));
  float va=dot(ga,f),vb=dot(gb,f-vec2(1,0)),
        vc=dot(gc,f-vec2(0,1)),vd=dot(gd,f-vec2(1,1));
  float v=va+u.x*(vb-va)+u.y*(vc-va)+u.x*u.y*(va-vb-vc+vd);
  vec2 d=ga+u.x*(gb-ga)+u.y*(gc-ga)+u.x*u.y*(ga-gb-gc+gd)
    +du*(vec2(vb-va,vc-va)+u.yx*vec2(va-vb-vc+vd));
  return vec3(v,d);
}

// ── FBM with chain-rule derivative accumulation ───────────────────────────
vec3 fbmD(vec2 p,int oct,float lac,float gain){
  float v=0.,a=1.,at=0.,freq=1.;vec2 d=vec2(0.);
  for(int i=0;i<8;i++){if(i>=oct)break;
    vec3 n=noised2(p);v+=a*n.x;d+=a*freq*n.yz;at+=a;
    p*=lac;a*=gain;freq*=lac;}
  return vec3(v/at,d/at);
}

// ── absolute-value turbulence ─────────────────────────────────────────────
vec3 turbD(vec2 p,int oct,float lac,float gain){
  float v=0.,a=1.,at=0.,freq=1.;vec2 d=vec2(0.);
  for(int i=0;i<8;i++){if(i>=oct)break;
    vec3 n=noised2(p);v+=a*abs(n.x);d+=a*freq*sign(n.x)*n.yz;at+=a;
    p*=lac;a*=gain;freq*=lac;}
  return vec3(v/at,d/at);
}

// ── ridge noise (sharp peaks / veins) ─────────────────────────────────────
vec3 ridgeD(vec2 p,int oct,float lac,float gain){
  float v=0.,a=1.,at=0.,w=1.,freq=1.;vec2 d=vec2(0.);
  for(int i=0;i<8;i++){if(i>=oct)break;
    vec3 n=noised2(p);float r=1.-abs(n.x);r*=r;
    v+=a*w*r;d+=a*w*freq*2.*(1.-abs(n.x))*(-sign(n.x))*n.yz;
    w=clamp(r*2.,0.,1.);at+=a;p*=lac;a*=gain;freq*=lac;}
  return vec3(v/at,d/at);
}

// ── 2D Voronoi → (F1, F2, cellHash) ──────────────────────────────────────
vec3 voronoi2(vec2 p){
  vec2 i=floor(p),f=fract(p);float d1=9.,d2=9.;vec2 s1=vec2(0.);
  for(int y=-1;y<=1;y++)for(int x=-1;x<=1;x++){
    vec2 n=vec2(x,y),c=n+hash22(i+n)*.5+.5-f;float dd=dot(c,c);
    if(dd<d1){d2=d1;d1=dd;s1=i+n;}else if(dd<d2)d2=dd;}
  return vec3(sqrt(d1),sqrt(d2),hash21(s1));
}

// ── domain warp ───────────────────────────────────────────────────────────
vec2 warp2(vec2 p,float str){
  return p+str*vec2(noised2(p+vec2(0.,1.7)).x,noised2(p+vec2(5.2,3.1)).x);
}

// ── analytical derivatives → tangent-space normal ─────────────────────────
vec3 heightNormal(vec2 d,float str){return normalize(vec3(-d*str,1.));}

float smin(float a,float b,float k){
  float h=clamp(.5+.5*(b-a)/k,0.,1.);return mix(b,a,h)-k*h*(1.-h);
}
vec3 hsv2rgb(vec3 c){
  vec3 p=abs(fract(c.xxx+vec3(0.,2./3.,1./3.))*6.-3.);
  return c.z*mix(vec3(1.),clamp(p-1.,0.,1.),c.y);
}
`

// ── Surface presets ───────────────────────────────────────────────────────────

const FRAG_SURFACE_WEATHERED_METAL = /* glsl */`precision highp float;
${GLSL_PBR_NOISE}
uniform float uTime;uniform int uOutputMode;
uniform float uScale,uScratchDensity,uGrime;
in vec2 vUv;out vec4 fragColor;
void main(){
  vec2 p=vUv*uScale;
  vec3 brushN=fbmD(vec2(p.x,p.y*12.),6,2.2,.45);
  float brushH=brushN.x*.12;
  float scrA=noised2(p*2.3+7.).x*PI;
  vec2 scrD=vec2(cos(scrA),sin(scrA));
  float scrL=fract(dot(vUv*uScratchDensity,scrD)*50.);
  float scratch=pow(smoothstep(.94,1.,scrL)*smoothstep(0.,.15,noised2(p*5.).x*.5+.5),.7);
  vec3 vor=voronoi2(p*18.);
  float pit=smoothstep(.08,.0,vor.x);
  vec3 grimeN=fbmD(p*3.+11.,5,2.,.5);
  float grime=smoothstep(.3,.8,grimeN.x*.5+.5)*uGrime;
  grime*=smoothstep(.05,-.06,brushH-pit*.1);
  float height=brushH-scratch*.06-pit*.12;
  vec2 deriv=brushN.yz*.12+grimeN.yz*grime*.05;
  vec3 metalCol=mix(vec3(.72,.73,.74),vec3(.64,.65,.67),brushN.x*.5+.5);
  metalCol=mix(metalCol,vec3(.5,.52,.55),scratch*.5);
  metalCol=mix(metalCol,vec3(.1,.08,.06),grime);
  metalCol*=1.-pit*.3;
  float rough=.25+brushN.x*.06;
  rough=mix(rough,.7,scratch);rough=mix(rough,.6,pit);rough=mix(rough,.8,grime);
  float metal=1.-grime*.9;
  float ao=1.-pit*.5-grime*.3-scratch*.1;
  vec3 norm=heightNormal(deriv,3.);
  if(uOutputMode==1)fragColor=vec4(norm*.5+.5,1.);
  else if(uOutputMode==2)fragColor=vec4(clamp(ao,0.,1.),clamp(rough,0.,1.),clamp(metal,0.,1.),1.);
  else fragColor=vec4(metalCol,1.);
}
`

const FRAG_SURFACE_MARBLE = /* glsl */`precision highp float;
${GLSL_PBR_NOISE}
uniform float uTime;uniform int uOutputMode;
uniform float uScale,uVeinIntensity,uVeinFreq,uColorTemp;
in vec2 vUv;out vec4 fragColor;
void main(){
  vec2 p=vUv*uScale;
  vec2 wp=warp2(p,1.5);
  vec3 vein1=turbD(wp*uVeinFreq,6,2.3,.5);
  vec2 wp2=warp2(p+vec2(7.3,3.1),.8);
  vec3 vein2=turbD(wp2*uVeinFreq*2.5+5.,5,2.1,.45);
  vec3 zone=fbmD(p*.5+3.,3,2.,.5);
  float v1=smoothstep(.4,.7,vein1.x)*uVeinIntensity;
  float v2=smoothstep(.45,.65,vein2.x)*uVeinIntensity*.4;
  float veinMask=max(v1,v2);
  float height=-veinMask*.03;
  vec2 deriv=-vein1.yz*v1*.03-vein2.yz*v2*.02;
  vec3 baseCol=mix(vec3(.95,.93,.90),vec3(.92,.94,.96),zone.x*.5+.5);
  baseCol=mix(baseCol,vec3(.97,.96,.92),uColorTemp);
  vec3 veinCol=mix(vec3(.2,.18,.16),vec3(.55,.45,.3),vein2.x);
  vec3 albedo=mix(baseCol,veinCol,veinMask);
  float rough=.05+veinMask*.08+fbmD(p*20.,3,2.,.5).x*.015;
  float metal=0.;
  float ao=1.-veinMask*.15;
  vec3 norm=heightNormal(deriv,2.);
  if(uOutputMode==1)fragColor=vec4(norm*.5+.5,1.);
  else if(uOutputMode==2)fragColor=vec4(clamp(ao,0.,1.),clamp(rough,0.,1.),metal,1.);
  else fragColor=vec4(albedo,1.);
}
`

const FRAG_SURFACE_ROUGH_STONE = /* glsl */`precision highp float;
${GLSL_PBR_NOISE}
uniform float uTime;uniform int uOutputMode;
uniform float uScale,uCrackDepth,uWeathering,uMineralVar;
in vec2 vUv;out vec4 fragColor;
void main(){
  vec2 p=vUv*uScale;
  vec3 macro=fbmD(p,4,2.,.5);
  vec3 detail=fbmD(p*4.+3.,5,2.2,.45);
  vec3 grain=fbmD(p*25.+7.,3,2.,.4);
  vec3 vor=voronoi2(p*3.);
  float crackDist=vor.y-vor.x;
  float crack=smoothstep(.06,.0,crackDist)*uCrackDepth;
  vec2 wp=warp2(p*.7,2.);
  float layer=sin(wp.y*8.+fbmD(wp,3,2.,.5).x*3.)*.5+.5;
  float height=macro.x*.15+detail.x*.05+grain.x*.01-crack*.2;
  vec2 deriv=macro.yz*.15+detail.yz*.05+grain.yz*.01;
  vec3 baseGray=mix(vec3(.45,.43,.40),vec3(.55,.52,.48),macro.x*.5+.5);
  vec3 albedo=mix(baseGray,mix(vec3(.52,.42,.35),vec3(.40,.45,.50),layer),uMineralVar);
  albedo=mix(albedo,albedo*.3,crack);
  float weather=fbmD(p*1.5+20.,4,2.,.5).x*.5+.5;
  albedo=mix(albedo,albedo*vec3(.85,.9,.8),smoothstep(.4,.7,weather)*uWeathering);
  float rough=.65+detail.x*.1+grain.x*.05;
  rough=mix(rough,.5,smoothstep(.04,.0,crackDist));
  float ao=1.-crack*.7-smoothstep(0.,-.1,macro.x)*.2;
  vec3 norm=heightNormal(deriv,4.);
  if(uOutputMode==1)fragColor=vec4(norm*.5+.5,1.);
  else if(uOutputMode==2)fragColor=vec4(clamp(ao,0.,1.),clamp(rough,0.,1.),0.,1.);
  else fragColor=vec4(albedo,1.);
}
`

const FRAG_SURFACE_AGED_WOOD = /* glsl */`precision highp float;
${GLSL_PBR_NOISE}
uniform float uTime;uniform int uOutputMode;
uniform float uScale,uRingFreq,uGrainStrength,uAge;
in vec2 vUv;out vec4 fragColor;
void main(){
  vec2 p=vUv*uScale;
  vec2 wp=warp2(p*.8,.6);
  float dist=length(wp-vec2(uScale*.5));
  float rings=sin(dist*uRingFreq+turbD(p*2.,4,2.,.5).x*4.)*.5+.5;
  float angle=atan(wp.y-uScale*.5,wp.x-uScale*.5);
  vec3 grainN=fbmD(vec2(p.x+sin(angle)*.3,p.y*15.),5,2.3,.45);
  float fiber=grainN.x*uGrainStrength;
  vec3 knotVor=voronoi2(p*.8);
  float knotMask=smoothstep(.35,.1,knotVor.x)*step(knotVor.z,.15);
  float knotRing=sin(knotVor.x*40.)*.5+.5;
  vec3 ageCrack=ridgeD(p*6.,4,2.,.5);
  float crack=smoothstep(.7,.9,ageCrack.x)*uAge;
  float height=rings*.03+fiber*.02-crack*.04+knotMask*knotRing*.03;
  vec2 deriv=grainN.yz*.02+ageCrack.yz*crack*.04;
  vec3 earlyWood=vec3(.65,.45,.25);
  vec3 lateWood=vec3(.4,.25,.12);
  vec3 woodCol=mix(earlyWood,lateWood,rings);
  woodCol+=fiber*vec3(.08,.05,.02);
  woodCol=mix(woodCol,vec3(.35,.22,.1),knotMask);
  woodCol=mix(woodCol,woodCol*.7,crack);
  woodCol*=mix(1.,.8,uAge*.5);
  float rough=.45+rings*.1;
  rough=mix(rough,.7,crack);rough+=fiber*.05;
  float ao=1.-crack*.5-knotMask*.2;
  vec3 norm=heightNormal(deriv,3.);
  if(uOutputMode==1)fragColor=vec4(norm*.5+.5,1.);
  else if(uOutputMode==2)fragColor=vec4(clamp(ao,0.,1.),clamp(rough,0.,1.),0.,1.);
  else fragColor=vec4(woodCol,1.);
}
`

const FRAG_SURFACE_RUST_IRON = /* glsl */`precision highp float;
${GLSL_PBR_NOISE}
uniform float uTime;uniform int uOutputMode;
uniform float uScale,uCorrosion,uPitting,uCleanPatches;
in vec2 vUv;out vec4 fragColor;
void main(){
  vec2 p=vUv*uScale;
  vec2 wp=warp2(p,1.2);
  vec3 corr=fbmD(wp*2.,6,2.1,.5);
  float corrLevel=corr.x*.5+.5;
  corrLevel=smoothstep(1.-uCorrosion,1.,corrLevel);
  vec3 patchN=fbmD(p*1.5+30.,4,2.,.5);
  float cleanMask=smoothstep(.5,.8,patchN.x*.5+.5)*uCleanPatches*(1.-corrLevel*.5);
  vec3 vor=voronoi2(p*uPitting);
  float pit=smoothstep(.1,.0,vor.x);
  vec3 flakeN=ridgeD(p*8.+5.,4,2.2,.45);
  float flake=smoothstep(.6,.85,flakeN.x)*corrLevel;
  float height=-corrLevel*.1-pit*.15+flake*.05+cleanMask*.02;
  vec2 deriv=corr.yz*corrLevel*.1+flakeN.yz*flake*.05;
  vec3 cleanMetal=vec3(.6,.62,.64);
  vec3 rustCol=mix(vec3(.15,.12,.1),vec3(.55,.2,.08),smoothstep(.2,.6,corrLevel));
  rustCol=mix(rustCol,vec3(.7,.35,.1),smoothstep(.6,.95,corrLevel));
  rustCol+=vec3(.1,.05,0.)*fbmD(p*12.,3,2.,.5).x;
  vec3 albedo=mix(rustCol,cleanMetal,cleanMask);
  albedo=mix(albedo,albedo*.4,pit);
  float rough=mix(.85,.3,cleanMask)+corrLevel*.1+flake*.1;
  float metal=mix(.1,.95,cleanMask);
  float ao=1.-pit*.6-flake*.2;
  vec3 norm=heightNormal(deriv,4.);
  if(uOutputMode==1)fragColor=vec4(norm*.5+.5,1.);
  else if(uOutputMode==2)fragColor=vec4(clamp(ao,0.,1.),clamp(rough,0.,1.),clamp(metal,0.,1.),1.);
  else fragColor=vec4(albedo,1.);
}
`

const FRAG_SURFACE_CRACKED_EARTH = /* glsl */`precision highp float;
${GLSL_PBR_NOISE}
uniform float uTime;uniform int uOutputMode;
uniform float uScale,uCrackWidth,uDryness,uDustColor;
in vec2 vUv;out vec4 fragColor;
void main(){
  vec2 p=vUv*uScale;
  vec3 vor=voronoi2(p*3.);
  float edgeDist=vor.y-vor.x;
  float crack=smoothstep(uCrackWidth,.0,edgeDist);
  float crackEdge=smoothstep(uCrackWidth*2.,uCrackWidth*.5,edgeDist);
  vec3 vor2=voronoi2(p*8.+1.);
  float crack2=smoothstep(uCrackWidth*.5,.0,vor2.y-vor2.x)*.5;
  vec3 surfN=fbmD(p*15.,5,2.1,.45);
  float dome=1.-smoothstep(0.,.5,vor.x);
  float height=dome*.08+surfN.x*.02-crack*.15-crack2*.05;
  float curl=crackEdge*(1.-crack)*.03;
  height+=curl;
  vec2 deriv=surfN.yz*.02;
  vec3 dryCol=mix(vec3(.6,.48,.32),vec3(.7,.55,.38),uDustColor);
  vec3 cellCol=mix(vec3(.35,.28,.18),dryCol,uDryness);
  cellCol*=.9+surfN.x*.15+vor.z*.08;
  vec3 crackCol=vec3(.2,.15,.1)*(1.-crack*.5);
  vec3 albedo=mix(cellCol,crackCol,max(crack,crack2));
  albedo=mix(albedo,dryCol*1.15,curl*8.);
  // crack wall normals via finite-difference on Voronoi edge distance
  if(crack>.01){
    float dEx=(voronoi2((p+vec2(.001,0.))*3.).y-voronoi2((p+vec2(.001,0.))*3.).x)-edgeDist;
    float dEy=(voronoi2((p+vec2(0.,.001))*3.).y-voronoi2((p+vec2(0.,.001))*3.).x)-edgeDist;
    deriv+=vec2(dEx,dEy)*crack*80.;
  }
  float rough=.75+surfN.x*.08-crack*.1;
  float ao=1.-crack*.7-crack2*.3;
  vec3 norm=heightNormal(deriv,3.5);
  if(uOutputMode==1)fragColor=vec4(norm*.5+.5,1.);
  else if(uOutputMode==2)fragColor=vec4(clamp(ao,0.,1.),clamp(rough,0.,1.),0.,1.);
  else fragColor=vec4(albedo,1.);
}
`

const FRAG_SURFACE_CONCRETE = /* glsl */`precision highp float;
${GLSL_PBR_NOISE}
uniform float uTime;uniform int uOutputMode;
uniform float uScale,uAggregate,uCrackDensity,uStaining;
in vec2 vUv;out vec4 fragColor;
void main(){
  vec2 p=vUv*uScale;
  vec3 fineN=fbmD(p*20.,5,2.2,.45);
  vec3 aggVor=voronoi2(p*uAggregate);
  float aggMask=smoothstep(.25,.15,aggVor.x);
  float aggBump=aggMask*(.5+hash21(floor(p*uAggregate+.5))*.5);
  vec3 poreVor=voronoi2(p*35.);
  float pore=smoothstep(.04,.0,poreVor.x);
  vec3 crackVor=voronoi2(p*uCrackDensity);
  float crackDist=crackVor.y-crackVor.x;
  float crack=smoothstep(.03,.0,crackDist);
  vec3 stainN=fbmD(p*.8+15.,4,2.,.5);
  float stain=smoothstep(.2,.6,stainN.x*.5+.5)*uStaining;
  float form=fbmD(vec2(p.x*.5,p.y*8.),3,2.,.5).x*.3;
  float height=fineN.x*.015+aggBump*.04-pore*.03-crack*.02;
  vec2 deriv=fineN.yz*.015;
  vec3 cementCol=vec3(.62,.60,.57)+fineN.x*vec3(.04);
  vec3 aggCol=mix(vec3(.55,.53,.50),vec3(.68,.65,.60),aggVor.z);
  vec3 albedo=mix(cementCol,aggCol,aggMask);
  albedo=mix(albedo,albedo*.5,pore);
  albedo=mix(albedo,albedo*.6,crack);
  albedo=mix(albedo,albedo*vec3(.85,.87,.82),stain);
  albedo+=form*.04;
  float rough=.7+fineN.x*.08-aggMask*.1+pore*.1;
  float ao=1.-pore*.5-crack*.4;
  vec3 norm=heightNormal(deriv,2.5);
  if(uOutputMode==1)fragColor=vec4(norm*.5+.5,1.);
  else if(uOutputMode==2)fragColor=vec4(clamp(ao,0.,1.),clamp(rough,0.,1.),0.,1.);
  else fragColor=vec4(albedo,1.);
}
`

const FRAG_SURFACE_LAVA = /* glsl */`precision highp float;
${GLSL_PBR_NOISE}
uniform float uTime;uniform int uOutputMode;
uniform float uScale,uCrackGlow,uCoolness,uFlowSpeed;
in vec2 vUv;out vec4 fragColor;
void main(){
  vec2 p=vUv*uScale;
  float t=uTime*uFlowSpeed;
  vec3 vor=voronoi2(p*3.+vec2(t*.02));
  float edgeDist=vor.y-vor.x;
  float crack=smoothstep(.08,.0,edgeDist);
  float plate=vor.x;
  vec2 flow=warp2(p+vec2(t*.05,0.),1.5);
  vec3 flowN=fbmD(flow*4.,5,2.1,.48);
  float temp=crack*uCrackGlow+(1.-plate)*.3;
  temp*=(1.-uCoolness);
  temp+=flowN.x*.15*(1.-uCoolness);
  temp=clamp(temp,0.,1.);
  vec3 surfN=fbmD(p*12.,4,2.2,.45);
  float height=-crack*.2+plate*.05+surfN.x*.02;
  vec2 deriv=surfN.yz*.02+flowN.yz*.03;
  vec3 basalt=vec3(.08,.07,.06)+surfN.x*vec3(.03);
  vec3 albedo=mix(basalt,basalt*.5,crack*(1.-temp));
  vec3 emissive=mix(vec3(1.,.15,0.),vec3(1.,.8,.2),temp)*temp*3.;
  float rough=mix(.3,.85,1.-temp);
  float ao=1.-crack*.3*(1.-temp);
  vec3 norm=heightNormal(deriv,3.);
  if(uOutputMode==1)fragColor=vec4(norm*.5+.5,1.);
  else if(uOutputMode==2)fragColor=vec4(clamp(ao,0.,1.),clamp(rough,0.,1.),0.,1.);
  else if(uOutputMode==3)fragColor=vec4(emissive,1.);
  else fragColor=vec4(albedo,1.);
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

// ── Surface builder (multi-map PBR pipeline) ─────────────────────────────────

function buildSurface(
  renderer: THREE.WebGLRenderer,
  preset: string,
  res: number,
  params: Record<string, number> | undefined,
  bake = true
): ProceduralTextureHandle {
  let frag: string
  let uniforms: UniformMap
  let hasEmissive = false

  switch (preset) {
    case 'weathered_metal':
      frag = FRAG_SURFACE_WEATHERED_METAL
      uniforms = {
        uTime: { value: 0 }, uOutputMode: { value: 0 },
        uScale: { value: paramsGet(params, 'scale', 8) },
        uScratchDensity: { value: paramsGet(params, 'scratch_density', 40) },
        uGrime: { value: paramsGet(params, 'grime', 0.4) },
      }
      break
    case 'marble':
      frag = FRAG_SURFACE_MARBLE
      uniforms = {
        uTime: { value: 0 }, uOutputMode: { value: 0 },
        uScale: { value: paramsGet(params, 'scale', 6) },
        uVeinIntensity: { value: paramsGet(params, 'vein_intensity', 0.8) },
        uVeinFreq: { value: paramsGet(params, 'vein_freq', 3) },
        uColorTemp: { value: paramsGet(params, 'color_temp', 0.3) },
      }
      break
    case 'rough_stone':
      frag = FRAG_SURFACE_ROUGH_STONE
      uniforms = {
        uTime: { value: 0 }, uOutputMode: { value: 0 },
        uScale: { value: paramsGet(params, 'scale', 6) },
        uCrackDepth: { value: paramsGet(params, 'crack_depth', 0.8) },
        uWeathering: { value: paramsGet(params, 'weathering', 0.5) },
        uMineralVar: { value: paramsGet(params, 'mineral_variation', 0.4) },
      }
      break
    case 'aged_wood':
      frag = FRAG_SURFACE_AGED_WOOD
      uniforms = {
        uTime: { value: 0 }, uOutputMode: { value: 0 },
        uScale: { value: paramsGet(params, 'scale', 5) },
        uRingFreq: { value: paramsGet(params, 'ring_freq', 20) },
        uGrainStrength: { value: paramsGet(params, 'grain_strength', 0.3) },
        uAge: { value: paramsGet(params, 'age', 0.4) },
      }
      break
    case 'rust_iron':
      frag = FRAG_SURFACE_RUST_IRON
      uniforms = {
        uTime: { value: 0 }, uOutputMode: { value: 0 },
        uScale: { value: paramsGet(params, 'scale', 6) },
        uCorrosion: { value: paramsGet(params, 'corrosion', 0.7) },
        uPitting: { value: paramsGet(params, 'pitting', 15) },
        uCleanPatches: { value: paramsGet(params, 'clean_patches', 0.3) },
      }
      break
    case 'cracked_earth':
      frag = FRAG_SURFACE_CRACKED_EARTH
      uniforms = {
        uTime: { value: 0 }, uOutputMode: { value: 0 },
        uScale: { value: paramsGet(params, 'scale', 4) },
        uCrackWidth: { value: paramsGet(params, 'crack_width', 0.06) },
        uDryness: { value: paramsGet(params, 'dryness', 0.8) },
        uDustColor: { value: paramsGet(params, 'dust_color', 0.5) },
      }
      break
    case 'concrete':
      frag = FRAG_SURFACE_CONCRETE
      uniforms = {
        uTime: { value: 0 }, uOutputMode: { value: 0 },
        uScale: { value: paramsGet(params, 'scale', 6) },
        uAggregate: { value: paramsGet(params, 'aggregate', 8) },
        uCrackDensity: { value: paramsGet(params, 'crack_density', 2) },
        uStaining: { value: paramsGet(params, 'staining', 0.3) },
      }
      break
    case 'lava':
      hasEmissive = true
      frag = FRAG_SURFACE_LAVA
      uniforms = {
        uTime: { value: 0 }, uOutputMode: { value: 0 },
        uScale: { value: paramsGet(params, 'scale', 4) },
        uCrackGlow: { value: paramsGet(params, 'crack_glow', 0.8) },
        uCoolness: { value: paramsGet(params, 'coolness', 0.4) },
        uFlowSpeed: { value: paramsGet(params, 'flow_speed', 0.1) },
      }
      break
    default:
      throw new Error(`Unknown surface preset: ${preset}`)
  }

  const mat = makeMat(frag, uniforms)

  const rtAlbedo = makeRT(res, bake)
  const rtNormal = makeRT(res, bake)
  const rtORM    = makeRT(res, bake)
  const rtEmissive = hasEmissive ? makeRT(res, bake) : undefined

  function renderAll(): void {
    uniforms.uOutputMode.value = 0
    renderQuad(renderer, mat, rtAlbedo)
    uniforms.uOutputMode.value = 1
    renderQuad(renderer, mat, rtNormal)
    uniforms.uOutputMode.value = 2
    renderQuad(renderer, mat, rtORM)
    if (rtEmissive) {
      uniforms.uOutputMode.value = 3
      renderQuad(renderer, mat, rtEmissive)
    }
  }

  renderAll()

  if (bake) {
    renderer.initTexture(rtAlbedo.texture)
    renderer.initTexture(rtNormal.texture)
    renderer.initTexture(rtORM.texture)
    if (rtEmissive) renderer.initTexture(rtEmissive.texture)
  }

  rtNormal.texture.colorSpace = THREE.LinearSRGBColorSpace
  rtORM.texture.colorSpace = THREE.LinearSRGBColorSpace

  return {
    texture: rtAlbedo.texture,
    normalMap: rtNormal.texture,
    ormMap: rtORM.texture,
    emissiveMap: rtEmissive?.texture,
    dispose() {
      rtAlbedo.dispose(); rtNormal.dispose(); rtORM.dispose()
      rtEmissive?.dispose(); mat.dispose()
    },
    updateTime(t) {
      if (bake) return
      uniforms.uTime.value = t
      renderAll()
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
    case 'cellular': return buildCellular(renderer, opts.preset, res, p, bake)
    case 'material': return buildMaterial(renderer, opts.preset, res, p, bake)
    case 'surface':  return buildSurface(renderer, opts.preset, res, p, bake)
    default: throw new Error(`Unknown texture domain: ${String(opts.domain)}`)
  }
}
