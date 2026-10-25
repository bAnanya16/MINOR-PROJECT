"""Ablation study presets and experiment matrix generator."""

from typing import List, Dict, Any
from amr_repair.loop.controller import SimulationConfig


def get_ablation_configs(map_path: str, seed: int = 42) -> Dict[str, SimulationConfig]:
    """Returns preset simulation configs for ablation studies."""
    base = {
        "map_path": map_path,
        "num_robots": 50,
        "steps": 150,
        "seed": seed,
    }

    return {
        "full_system": SimulationConfig(
            **base,
            mode="llm",
            cache_enabled=True,
            validator_enabled=True,
            cache_threshold=0.85,
            run_id=f"ablation_full_s{seed}",
        ),
        "no_cache": SimulationConfig(
            **base,
            mode="llm",
            cache_enabled=False,
            validator_enabled=True,
            run_id=f"ablation_no_cache_s{seed}",
        ),
        "no_validator": SimulationConfig(
            **base,
            mode="llm",
            cache_enabled=True,
            validator_enabled=False,
            cache_threshold=0.85,
            run_id=f"ablation_no_validator_s{seed}",
        ),
        "threshold_070": SimulationConfig(
            **base,
            mode="llm",
            cache_enabled=True,
            validator_enabled=True,
            cache_threshold=0.70,
            run_id=f"ablation_thresh_070_s{seed}",
        ),
        "threshold_095": SimulationConfig(
            **base,
            mode="llm",
            cache_enabled=True,
            validator_enabled=True,
            cache_threshold=0.95,
            run_id=f"ablation_thresh_095_s{seed}",
        ),
    }
