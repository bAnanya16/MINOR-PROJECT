# Lifelong MAPF Simulator with LLM-Guided Path Repair & 3D Replay Viewer

## Replay JSON Schema

The browser 3D viewer reads files matching this schema (`runs/*.json` or `/public/replays/*.json`):

```json
{
  "meta": {
    "seed": 42,
    "mode": "llm", // "llm" | "baseline_pibt" | "baseline_lns" | "baseline_pbs"
    "map_name": "warehouse-10-20-10-2-1",
    "steps": 600
  },
  "map": {
    "width": 161,
    "height": 63,
    "obstacles": [[1, 2], [1, 3]]
  },
  "frames": [
    {
      "t": 0,
      "pos": [[10, 15], [20, 30]], // indexed by robot id
      "state": ["moving", "idle"]    // "moving" | "idle" | "delayed" | "failed" | "replanning"
    }
  ],
  "incidents": [
    {
      "id": "inc_017",
      "t_detected": 212,
      "disruptions": [
        {
          "type": "blocked_cell",    // "blocked_cell" | "robot_failure" | "robot_delay"
          "cells": [[52, 17]],
          "robot": 17,
          "k_steps": 5
        }
      ],
      "affected_robots": [3, 17, 41],
      "congestion": {
        "level": "medium",          // "low" | "medium" | "high"
        "region": [[48, 14], [56, 20]]
      },
      "interacts_with": ["inc_016"],
      "cache": {
        "hit": true,
        "matched": "inc_004",
        "similarity": 0.91
      },
      "dry_run": {
        "ran": true,
        "valid": false,
        "violation": "edge swap robots 3,41 t=214"
      },
      "llm": {
        "called": true,
        "latency_ms": 1430,
        "tokens": 916,
        "directive": {
          "robots": [3, 17, 41],
          "solver": "pibt",         // "pibt" | "lns" | "windowed_pbs"
          "priority": [17, 3, 41],
          "horizon": 20
        }
      },
      "validator": [
        { "attempt": 1, "valid": false, "violation": "vertex robots 3,41 cell [52,17] t=215" },
        { "attempt": 2, "valid": true }
      ],
      "fallback_used": false,
      "executed_at": 214,
      "paths": {
        "invalidated": { "3": [[50, 17], [51, 17], [52, 17]] },
        "repaired": { "3": [[50, 17], [50, 18], [51, 18], [52, 18], [53, 17]] }
      },
      "outcome": {
        "recovery_steps": 4,
        "cost_delta": 11
      }
    }
  ],
  "metrics": {
    "repair_success_rate": 0.97,
    "mean_recovery_steps": 6.2,
    "cost_delta_pct": 3.1,
    "llm_calls": 14,
    "llm_tokens": 21800,
    "cache_hits": 9
  }
}
```

## Repository Structure

```
├── CLAUDE.md
├── README.md
├── amr_repair/
│   ├── sim/             # Stage 1: Lifelong MAPF simulator (grid, agents, lifelong)
│   ├── disrupt/         # Stage 2: Disruption injector
│   ├── sentinel/        # Stage 3: Incident detection & feature extraction
│   ├── cache/           # Stages 4–5: Strategy cache, negative memory, dry-run
│   ├── llm/             # Stage 6: Pydantic Directive, prompts, strategist
│   ├── solvers/         # Stage 7: PIBT, LNS, Windowed PBS
│   ├── validate/        # Stage 8: Deterministic plan validator
│   ├── loop/            # Stage 9: Closed-loop controller & run logger
│   ├── experiments/     # Stage 10: Baselines, ablations, experiment runner
│   └── replay/          # Replay Pydantic models & JSON exporter
├── tests/               # Pytest suite for every stage
├── maps/                # MovingAI benchmark warehouse .map files
├── runs/                # Replay JSON outputs
└── viewer/              # Part A: Vite + TypeScript Three.js 3D Replay Viewer
```
