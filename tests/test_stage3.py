"""Stage 3 tests: Sentinel incident detection.
Tests:
- a blocked cell not on any path yields no affected robots
- two disruptions on the same robot merge into one record
- density is in [0, 1]
- features() returns 12-element numpy array
"""

import os
import pytest
import numpy as np

from amr_repair.sim.grid import GridMap
from amr_repair.sim.agents import Robot
from amr_repair.sim.lifelong import World
from amr_repair.disrupt.injector import DisruptionItem
from amr_repair.sentinel.detect import IncidentDetector


@pytest.fixture
def warehouse_world():
    map_path = os.path.join(os.path.dirname(__file__), "..", "maps", "warehouse-10-20-10-2-1.map")
    gmap = GridMap.from_movingai(map_path)
    rng = np.random.default_rng(10)
    free = gmap.get_free_cells()
    # 5 robots located at the left side of map
    robots = [Robot(id=i, pos=free[i], goal=free[i + 50]) for i in range(5)]
    return World(gmap, robots, rng)


def test_blocked_cell_not_on_path_yields_no_affected_robots(warehouse_world):
    detector = IncidentDetector()
    # Pick a free cell far on the opposite side of the map (e.g. x=150, y=50)
    remote_cell = (150, 50)
    if not warehouse_world.map.is_free(remote_cell):
        remote_cell = warehouse_world.free_cells[-1]

    disruption = DisruptionItem(type="blocked_cell", t=5, cells=[remote_cell])
    incidents = detector.detect(warehouse_world, [disruption], t=5)

    assert len(incidents) == 1
    assert incidents[0].affected_robots == []


def test_two_disruptions_same_robot_merge(warehouse_world):
    detector = IncidentDetector()
    target_robot = warehouse_world.robots[0]

    d1 = DisruptionItem(type="robot_delay", t=5, cells=[target_robot.pos], robot=target_robot.id, k_steps=4)
    d2 = DisruptionItem(type="robot_failure", t=5, cells=[target_robot.pos], robot=target_robot.id)

    incidents = detector.detect(warehouse_world, [d1, d2], t=5)

    assert len(incidents) == 1, "Two disruptions on the same robot must merge into 1 incident record"
    assert target_robot.id in incidents[0].affected_robots
    assert len(incidents[0].disruptions) == 2
    assert len(incidents[0].interacts_with) == 2


def test_density_in_range_and_features_shape(warehouse_world):
    detector = IncidentDetector()
    cell = warehouse_world.robots[0].pos
    d = DisruptionItem(type="blocked_cell", t=1, cells=[cell])
    incidents = detector.detect(warehouse_world, [d], t=1)

    assert len(incidents) == 1
    inc = incidents[0]
    assert 0.0 <= inc.congestion.density <= 1.0

    feat = inc.features()
    assert isinstance(feat, np.ndarray)
    assert feat.shape == (12,)
    assert feat.dtype == np.float32
