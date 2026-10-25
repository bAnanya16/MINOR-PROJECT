"""Closed-loop lifelong MAPF simulation controller with LLM path repair and caching."""

import os
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np

from amr_repair.sim.grid import load_map
from amr_repair.sim.lifelong import World, manhattan_distance
from amr_repair.disrupt.injector import Injector, Disruption
from amr_repair.sentinel.detect import IncidentDetector
from amr_repair.sentinel.incident import IncidentRecord
from amr_repair.cache.store import StrategyCache
from amr_repair.cache.negative import NegativeMemory
from amr_repair.llm.schema import Directive
from amr_repair.llm.strategist import Strategist
from amr_repair.validate.validator import validate, Verdict
from amr_repair.loop.logger import SimulationLogger
import amr_repair.solvers.pibt as pibt_solver
import amr_repair.solvers.lns as lns_solver
import amr_repair.solvers.windowed_pbs as pbs_solver
from amr_repair.replay.models import (
    Replay, ReplayMeta, MapData, Frame,
    Incident, Disruption as ReplayDisruption, Congestion as ReplayCongestion,
    CacheResult, DryRunResult, LLMDirective, LLMResult, ValidatorAttempt,
    PathsData, IncidentOutcome
)
from amr_repair.replay.exporter import export_replay


SOLVER_REGISTRY = {
    "pibt": pibt_solver.solve,
    "lns": lns_solver.solve,
    "windowed_pbs": pbs_solver.solve,
}


@dataclass
class SimulationConfig:
    map_path: str
    num_robots: int = 100
    steps: int = 300
    seed: int = 42
    mode: str = "llm"  # 'llm', 'baseline_pibt', 'baseline_lns', 'baseline_pbs'
    schedule: Optional[List[Disruption]] = None
    cache_enabled: bool = True
    validator_enabled: bool = True
    cache_threshold: float = 0.85
    run_id: str = "demo"
    output_dir: str = "runs"
    cache_dir: str = "runs/cache"


class SimulationController:
    """Orchestrates the lifelong MAPF simulation, disruption injection, sentinel, cache, LLM, solvers, and validator."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        # 1. Initialize Map and World
        self.grid_map = load_map(config.map_path)
        from amr_repair.sim.agents import Robot
        free_cells = self.grid_map.get_free_cells()
        n_robots = min(config.num_robots, len(free_cells))
        chosen_indices = self.rng.choice(len(free_cells), size=n_robots, replace=False)
        robots = []
        for i, idx in enumerate(chosen_indices):
            start = free_cells[idx]
            goal_idx = (idx + 15) % len(free_cells)
            goal = free_cells[goal_idx]
            robots.append(Robot(id=i, pos=start, goal=goal))
        self.world = World(self.grid_map, robots=robots, rng=self.rng)

        # 2. Initialize Injector and Detector
        self.injector = Injector(rng=self.rng, schedule=config.schedule)
        self.detector = IncidentDetector()

        # 3. Strategy Cache and Negative Memory
        neg_mem_path = os.path.join(config.cache_dir, "negative_memory.json") if config.cache_enabled else None
        cache_path = os.path.join(config.cache_dir, "strategy_cache.json") if config.cache_enabled else None
        self.negative_memory = NegativeMemory(filepath=neg_mem_path) if config.cache_enabled else NegativeMemory()
        self.strategy_cache = StrategyCache(
            threshold=config.cache_threshold,
            negative_memory=self.negative_memory,
            filepath=cache_path,
        ) if config.cache_enabled else None

        # 4. LLM Strategist & Logger
        self.strategist = Strategist()
        self.logger = SimulationLogger()

    def run(self) -> Replay:
        """Executes the closed-loop simulation over the configured number of steps."""
        for t in range(self.config.steps):
            # 1. Apply disruptions
            fired_disruptions = self.injector.apply(self.world, t)

            # 2. Detect incidents
            incidents = self.detector.detect(self.world, fired_disruptions, t)

            # 3. Process each detected incident
            for inc in incidents:
                self._handle_incident(inc, t)

            # 4. Step the world forward
            self.world.step()

        # Build full replay model
        replay = self._build_replay()

        # Export replay to disk
        out_path = os.path.join(self.config.output_dir, f"{self.config.run_id}.json")
        export_replay(replay, out_path)

        return replay

    def _handle_incident(self, incident: IncidentRecord, t: int) -> None:
        """Handle path repair for a detected incident using Cache -> LLM -> Solvers -> Validator -> Fallback."""
        affected = [r_id for r_id in incident.affected_robots if r_id < len(self.world.robots)]
        if not affected:
            return

        # Record invalidated paths
        invalidated_paths: Dict[str, List[Tuple[int, int]]] = {}
        for r_id in affected:
            r = self.world.robots[r_id]
            invalidated_paths[str(r_id)] = list(r.path[:20])

        cache_result: Optional[CacheResult] = None
        dry_run_result: Optional[DryRunResult] = None
        llm_result: Optional[LLMResult] = None
        validator_attempts: List[ValidatorAttempt] = []
        fallback_used = False
        repaired_paths: Dict[int, List[Tuple[int, int]]] = {}
        chosen_directive: Optional[Directive] = None

        # Mode: Baselines skip cache and LLM
        if self.config.mode.startswith("baseline_"):
            solver_key = self.config.mode.replace("baseline_", "")
            if solver_key == "pbs":
                solver_key = "windowed_pbs"
            chosen_solver = SOLVER_REGISTRY.get(solver_key, pibt_solver.solve)
            chosen_directive = Directive(
                robots=affected,
                solver="pibt" if solver_key == "pibt" else ("lns" if solver_key == "lns" else "windowed_pbs"),
                priority=affected,
                horizon=20,
                rationale=f"Baseline {solver_key} execution",
            )
            repaired_paths = chosen_solver(self.world, affected, affected, 20)
            verdict = validate(repaired_paths, self.world, t) if self.config.validator_enabled else Verdict(valid=True)
            validator_attempts.append(ValidatorAttempt(attempt=1, valid=verdict.valid, violation=verdict.violation))
        else:
            # LLM Mode: Cache lookup -> Dry run -> LLM choose -> Validate -> Retry -> Fallback
            cache_hit_directive: Optional[Directive] = None
            if self.config.cache_enabled and self.strategy_cache is not None:
                hit = self.strategy_cache.lookup(incident)
                if hit is not None:
                    cache_hit_directive, matched_id, sim = hit
                    cache_result = CacheResult(hit=True, matched=matched_id, similarity=round(sim, 2))

                    # Dry-run verification on current live state
                    solver_fn = SOLVER_REGISTRY.get(cache_hit_directive.solver, pibt_solver.solve)
                    dry_paths = solver_fn(
                        self.world,
                        cache_hit_directive.robots,
                        cache_hit_directive.priority,
                        cache_hit_directive.horizon,
                    )
                    dry_verdict = validate(dry_paths, self.world, t) if self.config.validator_enabled else Verdict(valid=True)
                    dry_run_result = DryRunResult(
                        ran=True,
                        valid=dry_verdict.valid,
                        violation=dry_verdict.violation,
                    )

                    if dry_verdict.valid:
                        chosen_directive = cache_hit_directive
                        repaired_paths = dry_paths
                        validator_attempts.append(ValidatorAttempt(attempt=1, valid=True))
                    else:
                        # Dry run failed: add to negative memory and fall through to LLM
                        self.negative_memory.add(incident, cache_hit_directive)
                else:
                    cache_result = CacheResult(hit=False)

            # If not resolved by cache hit
            if chosen_directive is None:
                # Call LLM Strategist (Attempt 1)
                directive1 = self.strategist.choose(incident)
                llm_result = LLMResult(
                    called=True,
                    latency_ms=int(self.strategist.last_latency_ms),
                    tokens=self.strategist.last_tokens,
                    directive=LLMDirective(
                        robots=directive1.robots,
                        solver=directive1.solver,
                        priority=directive1.priority,
                        horizon=directive1.horizon,
                        rationale=directive1.rationale,
                    ),
                )
                solver_fn = SOLVER_REGISTRY.get(directive1.solver, pibt_solver.solve)
                paths1 = solver_fn(self.world, directive1.robots, directive1.priority, directive1.horizon)
                verdict1 = validate(paths1, self.world, t) if self.config.validator_enabled else Verdict(valid=True)
                validator_attempts.append(ValidatorAttempt(attempt=1, valid=verdict1.valid, violation=verdict1.violation))

                if verdict1.valid:
                    chosen_directive = directive1
                    repaired_paths = paths1
                    if self.config.cache_enabled and self.strategy_cache is not None:
                        self.strategy_cache.store(incident, directive1)
                else:
                    # Attempt 2: One LLM retry carrying exact violation
                    self.negative_memory.add(incident, directive1)
                    directive2 = self.strategist.choose(incident, violation=verdict1.violation)
                    solver_fn = SOLVER_REGISTRY.get(directive2.solver, pibt_solver.solve)
                    paths2 = solver_fn(self.world, directive2.robots, directive2.priority, directive2.horizon)
                    verdict2 = validate(paths2, self.world, t) if self.config.validator_enabled else Verdict(valid=True)
                    validator_attempts.append(ValidatorAttempt(attempt=2, valid=verdict2.valid, violation=verdict2.violation))

                    if verdict2.valid:
                        chosen_directive = directive2
                        repaired_paths = paths2
                        if self.config.cache_enabled and self.strategy_cache is not None:
                            self.strategy_cache.store(incident, directive2)
                    else:
                        # Fixed Fallback: Windowed PBS on all affected robots, horizon 30, priority by distance
                        fallback_used = True
                        self.negative_memory.add(incident, directive2)
                        sorted_priority = sorted(
                            affected,
                            key=lambda rid: manhattan_distance(self.world.robots[rid].pos, self.world.robots[rid].goal),
                        )
                        fallback_paths = pbs_solver.solve(self.world, affected, sorted_priority, 30)
                        verdict_fb = validate(fallback_paths, self.world, t) if self.config.validator_enabled else Verdict(valid=True)
                        validator_attempts.append(ValidatorAttempt(attempt=3, valid=verdict_fb.valid, violation=verdict_fb.violation))
                        chosen_directive = Directive(
                            robots=affected,
                            solver="windowed_pbs",
                            priority=sorted_priority,
                            horizon=30,
                            rationale="Fixed deterministic fallback strategy after validation retries.",
                        )
                        repaired_paths = fallback_paths

        # Apply repaired paths to robots in the world
        for r_id, p in repaired_paths.items():
            if r_id < len(self.world.robots):
                robot = self.world.robots[r_id]
                robot.path = list(p)
                robot.state = "replanning"

        # Format repaired paths for log
        repaired_paths_dict = {str(k): [list(pos) for pos in v] for k, v in repaired_paths.items()}
        recovery_steps = max(len(p) for p in repaired_paths.values()) if repaired_paths else 4
        cost_delta = sum(len(p) for p in repaired_paths.values()) - sum(len(p) for p in invalidated_paths.values())

        # Build Incident log model
        replay_incident = Incident(
            id=incident.id,
            t_detected=incident.t_detected,
            disruptions=[
                ReplayDisruption(type=d.type, cells=d.cells, robot=d.robot, k_steps=d.k_steps)
                for d in incident.disruptions
            ],
            affected_robots=incident.affected_robots,
            congestion=ReplayCongestion(
                level=incident.congestion.level,
                region=incident.congestion.region,
                density=incident.congestion.density,
            ),
            interacts_with=incident.interacts_with,
            cache=cache_result,
            dry_run=dry_run_result,
            llm=llm_result,
            validator=validator_attempts,
            fallback_used=fallback_used,
            executed_at=t,
            paths=PathsData(
                invalidated=invalidated_paths,
                repaired=repaired_paths_dict,
            ),
            outcome=IncidentOutcome(
                recovery_steps=recovery_steps,
                cost_delta=cost_delta,
            ),
        )

        self.logger.record_incident(replay_incident)

    def _build_replay(self) -> Replay:
        """Assembles the final Replay model from simulation history."""
        # Convert frames from world history
        frames: List[Frame] = list(self.world.frames)

        metrics = self.logger.compute_metrics()

        return Replay(
            meta=ReplayMeta(
                seed=self.config.seed,
                mode=self.config.mode,  # type: ignore
                map_name=os.path.basename(self.config.map_path).replace(".map", ""),
                steps=len(frames),
            ),
            map=MapData(
                width=self.grid_map.width,
                height=self.grid_map.height,
                obstacles=self.grid_map.get_obstacle_cells(),
            ),
            frames=frames,
            incidents=self.logger.incidents,
            metrics=metrics,
        )


def run(config: SimulationConfig) -> Replay:
    """Convenience function to run a simulation with the given configuration."""
    controller = SimulationController(config)
    return controller.run()
