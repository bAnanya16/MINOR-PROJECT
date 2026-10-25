"""Generates a realistic valid replay JSON matching the schema with 100 robots, 300 steps,
and 6 incidents including one interacting pair, grounded in warehouse-10-20-10-2-1.map."""

import json
import random
import os
import sys

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from amr_repair.replay.models import (
    Replay, ReplayMeta, MapData, Frame, Incident, Disruption, Congestion,
    CacheResult, DryRunResult, LLMResult, LLMDirective, ValidatorAttempt,
    PathsData, IncidentOutcome, ReplayMetrics
)


def load_map(map_path: str):
    with open(map_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip() for line in f if line.strip()]
    
    height = int(lines[1].split()[1])
    width = int(lines[2].split()[1])
    grid_lines = lines[4:4 + height]
    
    obstacles = []
    free_cells = []
    for y, row in enumerate(grid_lines):
        for x, ch in enumerate(row):
            if ch in ("@", "T", "O"):
                obstacles.append((x, y))
            else:
                free_cells.append((x, y))
                
    return width, height, obstacles, free_cells


def generate_fake_replay(seed: int = 42, mode: str = "llm", num_robots: int = 100, num_steps: int = 300) -> Replay:
    rng = random.Random(seed)
    map_path = os.path.join(os.path.dirname(__file__), "..", "maps", "warehouse-10-20-10-2-1.map")
    width, height, obstacles, free_cells = load_map(map_path)
    obstacle_set = set(obstacles)
    
    # Select start positions for 100 robots
    start_positions = rng.sample(free_cells, num_robots)
    
    # Build a simple 4-connected adjacency for simulation movements
    free_set = set(free_cells)
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)]
    
    # We will simulate positions for each robot over 300 steps
    # Keep track of robot current positions
    current_pos = list(start_positions)
    robot_states = ["idle"] * num_robots
    
    frames = []
    
    # Define incidents schedule
    # 6 incidents:
    # 1: Blocked cell at t=35 (aisle blocked)
    # 2: Robot delay at t=80
    # 3: Robot failure at t=140
    # 4 & 5: Interacting pair at t=195 and t=198 (same aisle / region!)
    # 6: Cache hit incident at t=250 (repeat disruption type)
    
    incident_defs = [
        {
            "id": "inc_001",
            "t_detected": 35,
            "type": "blocked_cell",
            "robot": 12,
            "cell": (45, 14),
            "affected": [12, 18, 27],
            "congestion_level": "medium",
            "congestion_region": ((40, 10), (52, 18)),
            "interacts_with": [],
            "cache": {"hit": False, "matched": None, "similarity": 0.42},
            "dry_run": {"ran": False, "valid": False, "violation": None},
            "llm": {
                "called": True,
                "latency_ms": 1120,
                "tokens": 820,
                "directive": {
                    "robots": [12, 18, 27],
                    "solver": "lns",
                    "priority": [12, 27, 18],
                    "horizon": 20,
                    "rationale": "Moderate congestion in corridor, LNS optimizes throughput around blocked cell."
                }
            },
            "validator": [{"attempt": 1, "valid": True, "violation": None}],
            "fallback_used": False,
            "executed_at": 37,
            "recovery_steps": 5,
            "cost_delta": 8
        },
        {
            "id": "inc_002",
            "t_detected": 80,
            "type": "robot_delay",
            "robot": 45,
            "cell": (70, 20),
            "k_steps": 6,
            "affected": [45, 62],
            "congestion_level": "low",
            "congestion_region": ((66, 17), (75, 24)),
            "interacts_with": [],
            "cache": {"hit": False, "matched": None, "similarity": 0.38},
            "dry_run": {"ran": False, "valid": False, "violation": None},
            "llm": {
                "called": True,
                "latency_ms": 890,
                "tokens": 640,
                "directive": {
                    "robots": [45, 62],
                    "solver": "pibt",
                    "priority": [62, 45],
                    "horizon": 15,
                    "rationale": "Single delay with low local density; PIBT quickly yields right of way."
                }
            },
            "validator": [{"attempt": 1, "valid": True, "violation": None}],
            "fallback_used": False,
            "executed_at": 81,
            "recovery_steps": 4,
            "cost_delta": 3
        },
        {
            "id": "inc_003",
            "t_detected": 140,
            "type": "robot_failure",
            "robot": 8,
            "cell": (95, 32),
            "affected": [8, 14, 53, 71],
            "congestion_level": "high",
            "congestion_region": ((90, 28), (102, 38)),
            "interacts_with": [],
            "cache": {"hit": False, "matched": None, "similarity": 0.55},
            "dry_run": {"ran": False, "valid": False, "violation": None},
            "llm": {
                "called": True,
                "latency_ms": 1340,
                "tokens": 980,
                "directive": {
                    "robots": [8, 14, 53, 71],
                    "solver": "windowed_pbs",
                    "priority": [14, 53, 71, 8],
                    "horizon": 30,
                    "rationale": "Permanent breakdown in high density zone; Windowed PBS resolves priority tree."
                }
            },
            "validator": [
                {"attempt": 1, "valid": False, "violation": "vertex collision robots 14,53 cell [97,32] t=143"},
                {"attempt": 2, "valid": True, "violation": None}
            ],
            "fallback_used": False,
            "executed_at": 144,
            "recovery_steps": 7,
            "cost_delta": 14
        },
        # Interacting pair: inc_004 and inc_005
        {
            "id": "inc_004",
            "t_detected": 195,
            "type": "blocked_cell",
            "robot": 33,
            "cell": (120, 24),
            "affected": [33, 58, 82],
            "congestion_level": "high",
            "congestion_region": ((114, 20), (128, 30)),
            "interacts_with": ["inc_005"],
            "cache": {"hit": False, "matched": None, "similarity": 0.61},
            "dry_run": {"ran": False, "valid": False, "violation": None},
            "llm": {
                "called": True,
                "latency_ms": 1580,
                "tokens": 1150,
                "directive": {
                    "robots": [33, 58, 82, 89],
                    "solver": "windowed_pbs",
                    "priority": [82, 33, 58, 89],
                    "horizon": 25,
                    "rationale": "Interacting cascade detected with neighboring failure; joint PBS repair."
                }
            },
            "validator": [{"attempt": 1, "valid": True, "violation": None}],
            "fallback_used": False,
            "executed_at": 197,
            "recovery_steps": 6,
            "cost_delta": 16
        },
        {
            "id": "inc_005",
            "t_detected": 198,
            "type": "robot_failure",
            "robot": 89,
            "cell": (124, 25),
            "affected": [89, 58],
            "congestion_level": "high",
            "congestion_region": ((116, 21), (129, 31)),
            "interacts_with": ["inc_004"],
            "cache": {"hit": False, "matched": None, "similarity": 0.70},
            "dry_run": {"ran": False, "valid": False, "violation": None},
            "llm": {
                "called": True,
                "latency_ms": 1420,
                "tokens": 940,
                "directive": {
                    "robots": [89, 58, 33],
                    "solver": "windowed_pbs",
                    "priority": [58, 33, 89],
                    "horizon": 25,
                    "rationale": "Co-located second failure; merged into unified windowed PBS plan."
                }
            },
            "validator": [{"attempt": 1, "valid": True, "violation": None}],
            "fallback_used": False,
            "executed_at": 200,
            "recovery_steps": 5,
            "cost_delta": 12
        },
        {
            "id": "inc_006",
            "t_detected": 250,
            "type": "blocked_cell",
            "robot": 77,
            "cell": (48, 14),
            "affected": [77, 91],
            "congestion_level": "medium",
            "congestion_region": ((42, 10), (55, 18)),
            "interacts_with": [],
            "cache": {
                "hit": True,
                "matched": "inc_001",
                "similarity": 0.94
            },
            "dry_run": {"ran": True, "valid": True, "violation": None},
            "llm": {
                "called": False,
                "latency_ms": None,
                "tokens": None,
                "directive": {
                    "robots": [77, 91],
                    "solver": "lns",
                    "priority": [77, 91],
                    "horizon": 20,
                    "rationale": "Cached strategy from inc_001 applied directly after valid dry run."
                }
            },
            "validator": [{"attempt": 1, "valid": True, "violation": None}],
            "fallback_used": False,
            "executed_at": 251,
            "recovery_steps": 3,
            "cost_delta": 4
        }
    ]
    
    # If mode == baseline, alter LLM fields and cache
    if mode.startswith("baseline"):
        solver_name = mode.replace("baseline_", "")
        if solver_name not in ("pibt", "lns", "pbs"):
            solver_name = "pibt"
        mapped_solver = "windowed_pbs" if solver_name == "pbs" else solver_name
        for inc in incident_defs:
            inc["cache"] = None
            inc["dry_run"] = None
            inc["llm"] = {
                "called": False,
                "latency_ms": None,
                "tokens": None,
                "directive": {
                    "robots": inc["affected"],
                    "solver": mapped_solver,
                    "priority": inc["affected"],
                    "horizon": 20,
                    "rationale": f"Fixed baseline {mode} execution"
                }
            }
            inc["validator"] = [{"attempt": 1, "valid": True, "violation": None}]
    
    # Build incident models and sample paths
    incidents = []
    for idef in incident_defs:
        disruptions = []
        if idef["type"] == "blocked_cell":
            disruptions.append(Disruption(type="blocked_cell", cells=[idef["cell"]]))
        elif idef["type"] == "robot_failure":
            disruptions.append(Disruption(type="robot_failure", cells=[idef["cell"]], robot=idef["robot"]))
        elif idef["type"] == "robot_delay":
            disruptions.append(Disruption(type="robot_delay", cells=[idef["cell"]], robot=idef["robot"], k_steps=idef.get("k_steps", 5)))

        # Build dummy invalidated and repaired paths for affected robots
        inv_paths = {}
        rep_paths = {}
        for r_id in idef["affected"]:
            cx, cy = idef["cell"]
            # Invalidated path cuts through or near cell
            inv = [(cx - 2, cy), (cx - 1, cy), (cx, cy), (cx + 1, cy), (cx + 2, cy)]
            # Repaired path routes around cell
            rep = [(cx - 2, cy), (cx - 2, cy + 1), (cx - 1, cy + 1), (cx, cy + 1), (cx + 1, cy + 1), (cx + 2, cy + 1), (cx + 2, cy)]
            inv_paths[str(r_id)] = inv
            rep_paths[str(r_id)] = rep

        inc = Incident(
            id=idef["id"],
            t_detected=idef["t_detected"],
            disruptions=disruptions,
            affected_robots=idef["affected"],
            congestion=Congestion(level=idef["congestion_level"], region=idef["congestion_region"]),
            interacts_with=idef["interacts_with"],
            cache=CacheResult(**idef["cache"]) if idef["cache"] else None,
            dry_run=DryRunResult(**idef["dry_run"]) if idef["dry_run"] else None,
            llm=LLMResult(
                called=idef["llm"]["called"],
                latency_ms=idef["llm"]["latency_ms"],
                tokens=idef["llm"]["tokens"],
                directive=LLMDirective(**idef["llm"]["directive"]) if idef["llm"]["directive"] else None
            ),
            validator=[ValidatorAttempt(**v) for v in idef["validator"]],
            fallback_used=idef["fallback_used"],
            executed_at=idef["executed_at"],
            paths=PathsData(invalidated=inv_paths, repaired=rep_paths),
            outcome=IncidentOutcome(recovery_steps=idef["recovery_steps"], cost_delta=idef["cost_delta"])
        )
        incidents.append(inc)

    # Pre-generate robot paths for 300 steps
    # Keep failed robots stopped permanently after their failure
    failed_robots = {8: 140, 89: 198}
    
    # Store robot path history
    for t in range(num_steps):
        step_pos = []
        step_state = []
        for r in range(num_robots):
            if r in failed_robots and t >= failed_robots[r]:
                # Failed permanently
                step_pos.append(current_pos[r])
                step_state.append("failed")
                continue
                
            # Check for active delay
            if r == 45 and 80 <= t < 86:
                step_pos.append(current_pos[r])
                step_state.append("delayed")
                continue
                
            # Check for active replanning
            if r in (12, 18, 27) and 35 <= t < 37:
                step_pos.append(current_pos[r])
                step_state.append("replanning")
                continue
            if r in (33, 58, 82) and 195 <= t < 197:
                step_pos.append(current_pos[r])
                step_state.append("replanning")
                continue

            # Regular movement: choose a neighbor that is free
            cx, cy = current_pos[r]
            candidates = []
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if (nx, ny) in free_set:
                    candidates.append((nx, ny))
            
            # Prefer continuing in roughly same direction or slight wander
            if candidates:
                nxt = rng.choice(candidates)
                current_pos[r] = nxt
                state = "idle" if nxt == (cx, cy) else "moving"
            else:
                state = "idle"
                
            step_pos.append(current_pos[r])
            step_state.append(state)

        frames.append(Frame(t=t, pos=step_pos, state=step_state))

    metrics = ReplayMetrics(
        repair_success_rate=0.98 if mode == "llm" else 0.85,
        mean_recovery_steps=4.8 if mode == "llm" else 7.4,
        cost_delta_pct=3.2 if mode == "llm" else 8.9,
        llm_calls=5 if mode == "llm" else 0,
        llm_tokens=4530 if mode == "llm" else 0,
        cache_hits=1 if mode == "llm" else 0
    )

    return Replay(
        meta=ReplayMeta(seed=seed, mode=mode, map_name="warehouse-10-20-10-2-1", steps=num_steps),
        map=MapData(width=width, height=height, obstacles=obstacles),
        frames=frames,
        incidents=incidents,
        metrics=metrics
    )


def main():
    os.makedirs("runs", exist_ok=True)
    os.makedirs("viewer/public/replays", exist_ok=True)
    
    # 1. Full Demo Replay (6 incidents)
    print("Generating demo.json (mode=llm)...")
    llm_replay = generate_fake_replay(seed=42, mode="llm")
    
    with open("runs/demo.json", "w", encoding="utf-8") as f:
        f.write(llm_replay.model_dump_json(indent=2))
    with open("viewer/public/replays/demo.json", "w", encoding="utf-8") as f:
        f.write(llm_replay.model_dump_json(indent=2))
        
    print("Generating demo_baseline.json (mode=baseline_pibt)...")
    baseline_replay = generate_fake_replay(seed=42, mode="baseline_pibt")
    with open("runs/demo_baseline.json", "w", encoding="utf-8") as f:
        f.write(baseline_replay.model_dump_json(indent=2))
    with open("viewer/public/replays/demo_baseline.json", "w", encoding="utf-8") as f:
        f.write(baseline_replay.model_dump_json(indent=2))

    # 2. Scenario Isolated (inc_001, inc_002, inc_003)
    iso_replay = generate_fake_replay(seed=101, mode="llm")
    iso_replay.incidents = [i for i in iso_replay.incidents if i.id in ("inc_001", "inc_002", "inc_003")]
    with open("viewer/public/replays/scenario_isolated.json", "w", encoding="utf-8") as f:
        f.write(iso_replay.model_dump_json(indent=2))
    iso_base = generate_fake_replay(seed=101, mode="baseline_pibt")
    iso_base.incidents = [i for i in iso_base.incidents if i.id in ("inc_001", "inc_002", "inc_003")]
    with open("viewer/public/replays/scenario_isolated_baseline.json", "w", encoding="utf-8") as f:
        f.write(iso_base.model_dump_json(indent=2))

    # 3. Scenario Interacting (inc_004 & inc_005)
    inter_replay = generate_fake_replay(seed=202, mode="llm")
    inter_replay.incidents = [i for i in inter_replay.incidents if i.id in ("inc_004", "inc_005")]
    with open("viewer/public/replays/scenario_interacting.json", "w", encoding="utf-8") as f:
        f.write(inter_replay.model_dump_json(indent=2))
    inter_base = generate_fake_replay(seed=202, mode="baseline_pibt")
    inter_base.incidents = [i for i in inter_base.incidents if i.id in ("inc_004", "inc_005")]
    with open("viewer/public/replays/scenario_interacting_baseline.json", "w", encoding="utf-8") as f:
        f.write(inter_base.model_dump_json(indent=2))

    # 4. Scenario Cache (inc_001 & inc_006)
    cache_replay = generate_fake_replay(seed=303, mode="llm")
    cache_replay.incidents = [i for i in cache_replay.incidents if i.id in ("inc_001", "inc_006")]
    with open("viewer/public/replays/scenario_cache.json", "w", encoding="utf-8") as f:
        f.write(cache_replay.model_dump_json(indent=2))
    cache_base = generate_fake_replay(seed=303, mode="baseline_pibt")
    cache_base.incidents = [i for i in cache_base.incidents if i.id in ("inc_001", "inc_006")]
    with open("viewer/public/replays/scenario_cache_baseline.json", "w", encoding="utf-8") as f:
        f.write(cache_base.model_dump_json(indent=2))
        
    print("Successfully generated all demo scenario replays!")


if __name__ == "__main__":
    main()

