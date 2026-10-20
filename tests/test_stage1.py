"""Stage 1 tests: Lifelong MAPF simulator.
Tests:
- robots never occupy an obstacle
- a robot at its goal gets a new goal next step
- two runs with the same seed produce identical frames
"""

import os
import pytest
import numpy as np

from amr_repair.sim.grid import GridMap
from amr_repair.sim.agents import Robot
from amr_repair.sim.lifelong import World


@pytest.fixture
def warehouse_map():
    map_path = os.path.join(os.path.dirname(__file__), "..", "maps", "warehouse-10-20-10-2-1.map")
    return GridMap.from_movingai(map_path)


def test_robots_never_occupy_obstacle(warehouse_map):
    rng = np.random.default_rng(12345)
    free_cells = warehouse_map.get_free_cells()

    # Create 50 robots
    robots = []
    start_cells = rng.choice(len(free_cells), size=50, replace=False)
    for i, idx in enumerate(start_cells):
        start = free_cells[idx]
        goal = free_cells[(idx + 10) % len(free_cells)]
        robots.append(Robot(id=i, pos=start, goal=goal))

    world = World(warehouse_map, robots, rng)

    # Run 40 steps
    for _ in range(40):
        world.step()
        for r in world.robots:
            assert warehouse_map.is_free(r.pos), f"Robot {r.id} at {r.pos} occupies an obstacle!"


def test_robot_at_goal_gets_new_goal_next_step(warehouse_map):
    rng = np.random.default_rng(999)
    free_cells = warehouse_map.get_free_cells()

    start = free_cells[0]
    # Pick adjacent goal if possible
    neighbors = list(warehouse_map.graph.neighbors(start))
    goal = neighbors[0]

    robot = Robot(id=0, pos=start, goal=goal)
    world = World(warehouse_map, [robot], rng)

    # Robot is 1 step away from goal
    assert robot.goal == goal
    old_goal = robot.goal

    # Step 1: Robot should arrive at goal and get assigned a new goal
    world.step()
    assert robot.pos == old_goal
    assert robot.goal != old_goal, "Robot at goal did not receive a new goal!"


def test_identical_seed_produces_identical_frames(warehouse_map):
    def run_simulation(seed):
        rng = np.random.default_rng(seed)
        free_cells = warehouse_map.get_free_cells()
        robots = []
        start_indices = rng.choice(len(free_cells), size=20, replace=False)
        for i, idx in enumerate(start_indices):
            start = free_cells[idx]
            goal = free_cells[(idx + 5) % len(free_cells)]
            robots.append(Robot(id=i, pos=start, goal=goal))

        world = World(warehouse_map, robots, rng)
        for _ in range(25):
            world.step()
        return world.frames

    frames1 = run_simulation(42)
    frames2 = run_simulation(42)

    assert len(frames1) == len(frames2)
    for f1, f2 in zip(frames1, frames2):
        assert f1.t == f2.t
        assert f1.pos == f2.pos
        assert f1.state == f2.state
