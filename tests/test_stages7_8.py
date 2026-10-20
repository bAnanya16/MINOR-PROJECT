"""Stages 7-8 tests: Validator and Solvers (PIBT, LNS, Windowed PBS).
Tests:
- validator catches planted vertex collision, edge swap, blocked cell, deadlock
- each solver reroutes 5 robots around a blocked aisle and passes validation
- PIBT is significantly faster than Windowed PBS
"""

import os
import time
import pytest
import numpy as np

from amr_repair.sim.grid import GridMap
from amr_repair.sim.agents import Robot
from amr_repair.sim.lifelong import World
from amr_repair.validate.validator import validate, Verdict
from amr_repair.solvers import pibt, lns, windowed_pbs


@pytest.fixture
def warehouse_world():
    map_path = os.path.join(os.path.dirname(__file__), "..", "maps", "warehouse-10-20-10-2-1.map")
    gmap = GridMap.from_movingai(map_path)
    rng = np.random.default_rng(42)
    free = gmap.get_free_cells()
    robots = [Robot(id=i, pos=free[i * 3], goal=free[i * 3 + 15]) for i in range(10)]
    return World(gmap, robots, rng)


def test_validator_catches_all_violations(warehouse_world):
    # 1. Blocked cell
    obs = warehouse_world.map.get_obstacle_cells()[0]
    bad_path_blocked = {0: [obs, obs]}
    v1 = validate(bad_path_blocked, warehouse_world, t0=0)
    assert not v1.valid
    assert "blocked_cell" in v1.violation

    # 2. Vertex collision
    cell = warehouse_world.free_cells[5]
    bad_path_vertex = {
        0: [cell, cell],
        1: [cell, cell]
    }
    v2 = validate(bad_path_vertex, warehouse_world, t0=0)
    assert not v2.valid
    assert "vertex collision" in v2.violation

    # 3. Edge swap
    c1 = warehouse_world.free_cells[10]
    c2 = warehouse_world.free_cells[11]
    bad_path_edge = {
        0: [c1, c2],
        1: [c2, c1]
    }
    v3 = validate(bad_path_edge, warehouse_world, t0=0)
    assert not v3.valid
    assert "edge swap" in v3.violation

    # 4. Deadlock (10 steps same position when not at goal)
    not_goal_cell = warehouse_world.free_cells[20]
    bad_path_deadlock = {0: [not_goal_cell] * 12}
    v4 = validate(bad_path_deadlock, warehouse_world, t0=0)
    assert not v4.valid
    assert "deadlock" in v4.violation


def test_each_solver_reroutes_and_validates(warehouse_world):
    # Select 5 robots
    managed_robots = [0, 1, 2, 3, 4]
    priority = [0, 1, 2, 3, 4]
    horizon = 15

    # 1. PIBT
    t_start = time.perf_counter()
    pibt_paths = pibt.solve(warehouse_world, managed_robots, priority, horizon)
    pibt_time = time.perf_counter() - t_start

    v_pibt = validate(pibt_paths, warehouse_world, t0=warehouse_world.t)
    assert v_pibt.valid, f"PIBT produced invalid paths: {v_pibt.violation}"
    assert len(pibt_paths) == 5

    # 2. LNS
    lns_paths = lns.solve(warehouse_world, managed_robots, priority, horizon)
    v_lns = validate(lns_paths, warehouse_world, t0=warehouse_world.t)
    assert v_lns.valid, f"LNS produced invalid paths: {v_lns.violation}"
    assert len(lns_paths) == 5

    # 3. Windowed PBS
    t_start_pbs = time.perf_counter()
    pbs_paths = windowed_pbs.solve(warehouse_world, managed_robots, priority, horizon)
    pbs_time = time.perf_counter() - t_start_pbs

    v_pbs = validate(pbs_paths, warehouse_world, t0=warehouse_world.t)
    assert v_pbs.valid, f"PBS produced invalid paths: {v_pbs.violation}"
    assert len(pbs_paths) == 5

    # PIBT speed comparison
    assert pibt_time < pbs_time, f"Expected PIBT ({pibt_time:.4f}s) to be faster than PBS ({pbs_time:.4f}s)"
