"""Baseline strategy definitions and runners."""

from typing import List, Optional
from amr_repair.loop.controller import SimulationConfig, run, Replay
from amr_repair.disrupt.injector import Disruption


def run_baseline(
    map_path: str,
    solver_mode: str,  # 'baseline_pibt', 'baseline_lns', 'baseline_pbs'
    num_robots: int = 100,
    steps: int = 300,
    seed: int = 42,
    schedule: Optional[List[Disruption]] = None,
    output_dir: str = "runs",
) -> Replay:
    """Runs a baseline simulation skipping LLM and Cache."""
    assert solver_mode in ("baseline_pibt", "baseline_lns", "baseline_pbs"), f"Unknown baseline mode: {solver_mode}"
    config = SimulationConfig(
        map_path=map_path,
        num_robots=num_robots,
        steps=steps,
        seed=seed,
        mode=solver_mode,
        schedule=schedule,
        cache_enabled=False,
        validator_enabled=True,
        run_id=f"baseline_{solver_mode}_s{seed}",
        output_dir=output_dir,
    )
    return run(config)
