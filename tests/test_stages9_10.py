"""Tests for Stages 9 & 10: Closed-loop Controller, Baselines, and Replay Exporter."""

import os
import json
from amr_repair.loop.controller import SimulationConfig, run
from amr_repair.experiments.baselines import run_baseline
from amr_repair.replay.models import Replay
from amr_repair.disrupt.injector import Disruption


MAP_PATH = "maps/warehouse-10-20-10-2-1.map"


def test_closed_loop_controller_llm_mode(tmp_path):
    out_dir = str(tmp_path / "runs")
    schedule = [
        Disruption(type="blocked_cell", t=5, cells=[(30, 10)]),
        Disruption(type="robot_delay", t=10, robot=2, k_steps=3),
    ]
    config = SimulationConfig(
        map_path=MAP_PATH,
        num_robots=10,
        steps=20,
        seed=42,
        mode="llm",
        schedule=schedule,
        cache_enabled=True,
        validator_enabled=True,
        run_id="test_run_llm",
        output_dir=out_dir,
    )
    replay = run(config)

    assert isinstance(replay, Replay)
    assert replay.meta.mode == "llm"
    assert len(replay.frames) == 21  # t=0 plus 20 steps
    assert len(replay.incidents) >= 0
    assert replay.metrics.repair_success_rate >= 0.0

    # Verify JSON file was exported and can be loaded
    out_file = os.path.join(out_dir, "test_run_llm.json")
    assert os.path.exists(out_file)
    with open(out_file, "r") as f:
        data = json.load(f)
    assert "frames" in data
    assert "incidents" in data
    assert "metrics" in data


def test_baseline_pibt_run(tmp_path):
    out_dir = str(tmp_path / "runs")
    schedule = [
        Disruption(type="blocked_cell", t=5, cells=[(30, 10)]),
    ]
    replay = run_baseline(
        map_path=MAP_PATH,
        solver_mode="baseline_pibt",
        num_robots=10,
        steps=15,
        seed=42,
        schedule=schedule,
        output_dir=out_dir,
    )
    assert replay.meta.mode == "baseline_pibt"
    assert len(replay.frames) == 16  # t=0 plus 15 steps
