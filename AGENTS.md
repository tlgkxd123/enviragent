## Learned User Preferences

- User prefers agentic/tool-calling LLM approaches over code-generation/sandbox approaches — when asked for LLM features, default to function-calling with explicit tools, not eval/sandboxed JS output.
- User expects UI fixes to be visual and layout-complete — when panels overlap or inputs are hidden, fix the full layout, not just a single property.
- User wants all LLM-related config (API key, base URL, model) directly in the relevant UI panel, not in a separate settings area.
- User prefers vibe-driven, expansive feature requests ("make it 1000000000000000000000000x better") — interpret liberally and implement something impressive rather than asking for clarification.
- User iterates quickly and expects features to build on each other without breaking existing ones.
- User uses Windows (PowerShell) — avoid Unix-only commands like `tail`; use PowerShell equivalents (`Select-Object -Last N`). Use semicolons not `&&` to chain commands.
- User reports bugs by sending a screenshot with "fix this" — always check attached images and read layout/CSS files to diagnose before changing code.
- When the user says "use your tools", they mean use available agent tools/actions to complete the task rather than just describing what to do.
- User expects LLM config fields (API key, base URL, model) to persist across page reloads via localStorage.
- When custom implementations get complex, user prefers switching to established libraries ("nvm just use a library") — default to well-known library solutions when available.

## Learned Workspace Facts

- Project: `c:\dev\subatomicrenderer` — Three.js 3D environment AI agent, Vite + TypeScript, GLSL ES 3.0 (WebGL2), custom post-processing (bloom, SSR, lens flares).
- Build command: `npm run build` (runs `tsc && vite build`); verify with `Select-Object -Last 20` not `tail`.
- All panels use `position: fixed` and must be explicitly positioned — missing `top`/`left`/`right`/`bottom` causes overlap bugs.
- Three.js `ShaderMaterial` requires `glslVersion: GLSL3` for GLSL ES 3.0; use `out vec4 fragColor` not `gl_FragColor`, `in`/`out` not `varying`. Never redeclare Three.js built-in attributes/uniforms (position, uv, normal, modelViewMatrix, projectionMatrix, etc.) — `scene-tools.ts` has `sanitizeVertexShader()` that strips them automatically.
- Agentic scene builder: `scene-agent.ts` (OpenAI function-calling + SSE streaming), `scene-tools.ts` (tools + executors), `chatbot-bindings.ts` (UI). Chat config in `#chatApiKey`/`#chatBaseUrl`/`#chatModel`, persisted to localStorage. SSE parser handles both `choice.delta` streaming and `choice.message` fallback for local LLMs (Ollama, LM Studio, proxies).
- Dynamic tools registered via `create_tool` require a user-nudge message + `tool_choice: required` on the next agent round or the model just narrates instead of calling the new tool.
- Volumetric clouds use `SphereGeometry` (BackSide) proxy + analytic ray-ellipsoid intersection in GLSL — Perlin-Worley noise blend with cumulus height gradient, detail erosion, Beer-Powder lighting, and dual-lobe Henyey-Greenstein phase.
- Camera near/far planes update every frame in `orbital.ts step()` proportional to camera distance — enables infinite zoom without clipping.
- `gen_texture` tool generates GPU procedural textures. `domain="surface"` presets (weathered_metal, marble, rough_stone, aged_wood, rust_iron, cracked_earth, concrete, lava) auto-generate full PBR stacks: albedo + analytical normal map + ORM (AO/Roughness/Metalness) via `GLSL_PBR_NOISE` gradient noise with analytical derivatives. `bake: true` renders once at up to 4096px for zero per-frame cost.
- `add_water` tool creates a 3D Gerstner wave ocean mesh (`src/render/fft-water.ts`) with real vertex displacement every frame — sharp crests, wide troughs, Jacobian-based foam. MeshPhysicalMaterial with transmission/IOR 1.33. Auto-removes previous water. Key params: `wave_scale`, `choppiness`, `wave_count`, `resolution`, `speed`.
- `add_terrain` tool creates a procedural landscape mesh (`src/render/terrain.ts`) with multi-octave gradient + ridged noise heightmap. Custom GLSL3 ShaderMaterial blends 4 photo textures (grass/rock/dirt/snow from `public/textures/terrain/`) by height band + surface steepness. Biome presets: mountains, rolling_hills, desert, arctic, volcanic, canyon, plateau. Auto-removes previous terrain.
- Scene agent auto-captures a JPEG screenshot after each tool-call round and injects it as a vision message for self-correction. Implemented via `captureFrame()` in `chatbot-bindings.ts` (≤768px, detail=low).
- `tessellate_mesh` subdivides meshes via edge-midpoint subdivision (1-4 levels, each ×4 triangles). TransformControls helper objects must be excluded from raycasts for mesh selection to work reliably.
