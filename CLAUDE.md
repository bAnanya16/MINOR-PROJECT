# CLAUDE.md

This is a browser 3D replay viewer and Python research pipeline for a warehouse multi-robot (MAPF) simulation with LLM-guided path repair, disruption injection, verified strategy cache, and deterministic validator.

It reads `/public/replays/*.json` (schema in README.md) and animates it. It never simulates in the browser.

## Rules
- Three.js only, imported from 'three/webgpu' with WebGPURenderer (it falls back to WebGL itself). Vite + vanilla TypeScript. No React, no other 3D libs.
- Grid cell = 1 metre. Cell (x, y) is at world (x + 0.5, 0, y + 0.5).
- All robots are one InstancedMesh per sub-mesh. All shelves are one InstancedMesh. Never clone a model per robot.
- Robots interpolate smoothly between steps (ease-in-out), turn before they move, never teleport.
- Lighting: one DirectionalLight with soft shadows + HDRI environment map, ACES tone mapping, no AmbientLight.
- Post-processing: ambient occlusion and a light bloom only.
- Every visual state comes from the replay file at the current step. No accumulated mutable state, so scrubbing the timeline backwards always shows the right picture.
- UI is plain HTML/CSS positioned over the canvas, dark translucent panels, 14 px minimum font.
- Keep 60 fps with 200 robots on an integrated-GPU laptop. Measure before and after each change.
- After every change, run `npm run dev` and tell me exactly what to look at in the browser.
- Never invent data fields. If the replay lacks something, say so.

## Python Pipeline Rules
- Python 3.11, numpy + networkx only for core logic. Pure functions, no global state.
- Everything is seeded: every random choice goes through one `numpy.random.Generator` passed in explicitly.
- Grid: cells are (x, y) ints; time is discrete steps; a path is a list of cells, one per step; a robot may wait (same cell twice).
- The validator is the single source of truth for "valid plan". Nothing executes without passing it.
- The LLM never sees the map. It sees only an IncidentRecord and returns a Directive from a fixed menu (Pydantic model, enforced by Instructor).
- Max one LLM retry per incident, then the fixed fallback. Never loop.
- Write a pytest for every module before marking it done. Run `pytest -q` after every change.
- Every run writes runs/<run_id>.json in the replay schema in README.md.
