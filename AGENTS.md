## Learned User Preferences

- User prefers agentic/tool-calling LLM approaches over code-generation/sandbox approaches — when asked for LLM features, default to function-calling with explicit tools, not eval/sandboxed JS output.
- User expects UI fixes to be visual and layout-complete — when panels overlap or inputs are hidden, fix the full layout, not just a single property.
- User wants all LLM-related config (API key, base URL, model) directly in the relevant UI panel, not in a separate settings area.
- User prefers vibe-driven, expansive feature requests ("make it 1000000000000000000000000x better", "add quantum physics") — interpret liberally and implement something impressive rather than asking for clarification.
- User iterates quickly and expects features to build on each other without breaking existing ones.
- User uses Windows (PowerShell) — avoid Unix-only commands like `tail`; use PowerShell equivalents (`Select-Object -Last N`).
- User reports bugs by sending a screenshot with "fix this" — always check attached images and read layout/CSS files to diagnose before changing code.
- When the user says "use your tools", they mean use available agent tools/actions to complete the task rather than just describing what to do.

## Learned Workspace Facts

- Project: `c:\dev\subatomicrenderer` — a Three.js hydrogen orbital visualizer built with Vite + TypeScript.
- Stack: Three.js, Vite, TypeScript, GLSL ES 3.0 (WebGL2), custom post-processing (bloom, SSR, lens flares).
- Build command: `npm run build` (runs `tsc && vite build`); verify with `Select-Object -Last 20` not `tail`.
- All panels use `position: fixed` and must be explicitly positioned — missing `top`/`left`/`right`/`bottom` causes overlap bugs.
- Three.js `ShaderMaterial` requires `glslVersion: GLSL3` for GLSL ES 3.0; use `out vec4 fragColor` not `gl_FragColor`, use `in`/`out` not `varying`.
- The agentic scene builder uses OpenAI function-calling via `src/render/scene-agent.ts`, tools defined in `src/render/scene-tools.ts`, UI in `src/ui/chatbot-bindings.ts`.
- Environment generator LLM config lives in `#envApiKey` / `#envModel` elements; chat agent config lives in `#chatApiKey` / `#chatBaseUrl` / `#chatModel`.
- TransformControls helper objects must be excluded from raycasts for mesh selection to work reliably.
