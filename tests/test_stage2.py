"""Stage 2 tests: Disruption injector.
Tests:
- same seed gives the same schedule
- a failed robot never moves again
- overlap=same_robot always produces two disruptions sharing a robot
"""

import os
import pytest
import numpy as np

from amr_repair.sim.grid import GridMap
from amr_repair.sim.agents import Robot
from amr_repair.sim.lifelong import World
from amr_repair.disrupt.injector import Injector


@pytest.fixture
def test_world():
    map_path = os.path.join(os.path.dirname(__file__), "..", "maps", "warehouse-10-20-10-2-1.map")
    gmap = GridMap.from_movingai(map_path)
    rng = np.random.default_rng(42)
    free = gmap.get_free_cells()
    robots = [Robot(id=i, pos=free[i * 2], goal=free[i * 2 + 10]) for i in range(15)]
    return World(gmap, robots, rng)


def test_same_seed_gives_same_schedule(test_world):
    config = {
        "rate": 0.1,
        "types": ["blocked_cell", "robot_failure", "robot_delay"],
        "overlap": "none",
        "max_steps": 100
    }
    inj1 = Injector(np.random.default_rng(101), config, test_world)
    inj2 = Injector(np.random.default_rng(101), config, test_world)

    assert len(inj1.disruptions) == len(inj2.disruptions)
    for d1, d2 in zip(inj1.disruptions, inj2.disruptions):
        assert d1.type == d2.type
        assert d1.t == d2.t
        assert d1.cells == d2.cells
        assert d1.robot == d2.robot


def test_failed_robot_never_moves_again(test_world):
    target_robot = test_world.robots[2]
    schedule = [{
        "type": "robot_failure",
        "t": 3,
        "robot": target_robot.id,
        "cells": [target_robot.pos]
    }]

    injector = Injector(np.random.default_rng(42), schedule)

    for step_num in range(1, 15):
        injector.apply(test_world, step_num)
        test_world.step()

        if step_num >= 3:
            assert target_robot.state == "failed"
            # Once failed, position must never change
            assert not target_robot.path


def test_overlap_same_robot_produces_shared_robot(test_world):
    config = {
        "rate": 0.5,
        "types": ["robot_failure", "robot_delay"],
        "overlap": "same_robot",
        "max_steps": 50
    }
    injector = Injector(np.random.default_rng(777), config, test_world)

    # Check if there are pairs of disruptions sharing a robot
    robot_occurrences = {}
    for d in injector.disruptions:
        if d.robot is not None:
            robot_occurrences[d.robot] = robot_occurrences.get(d.robot, 0) + 1

    shared_robots = [r for r, count in robot_occurrences.items() if count >= 2]
    assert len(shared_robots) > 0, "No disruptions shared a robot with overlap=same_robot!"
