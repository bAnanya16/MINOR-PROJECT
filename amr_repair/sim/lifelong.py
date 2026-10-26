"""Lifelong MAPF simulation world."""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import networkx as nx

from amr_repair.sim.grid import GridMap
from amr_repair.sim.agents import Robot
from amr_repair.replay.models import Frame, MapData, ReplayMeta, ReplayMetrics


def manhattan_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class World:
    def __init__(self, grid_map: GridMap, robots: List[Robot], rng: np.random.Generator):
        self.map: GridMap = grid_map
        self.robots: List[Robot] = robots
        self.rng: np.random.Generator = rng
        self.free_cells: List[Tuple[int, int]] = self.map.get_free_cells()
        self.t: int = 0
        self.frames: List[Frame] = []

        # Ensure every robot has a path planned towards its goal
        for robot in self.robots:
            if not robot.path:
                self._plan_path(robot)

        # Record initial frame at t=0
        self._record_frame()

    def _plan_path(self, robot: Robot):
        if robot.pos == robot.goal:
            # Pick a new random goal from free cells
            self._assign_new_goal(robot)

        try:
            path = nx.astar_path(
                self.map.graph,
                robot.pos,
                robot.goal,
                heuristic=manhattan_distance
            )
            # path includes current position as path[0]
            robot.path = path[1:]  # Steps to execute
            robot.state = "moving" if robot.path else "idle"
        except (nx.NetworkXNoPath, nx.NodeNotFound, nx.NetworkXError):
            robot.path = []
            robot.state = "idle"

    def _assign_new_goal(self, robot: Robot):
        idx = int(self.rng.integers(0, len(self.free_cells)))
        candidate = self.free_cells[idx]
        while candidate == robot.pos:
            idx = int(self.rng.integers(0, len(self.free_cells)))
            candidate = self.free_cells[idx]
        robot.goal = candidate

    def step(self):
        self.t += 1

        for robot in self.robots:
            if robot.state == "failed":
                # Permanently immobile
                continue

            if robot.state == "delayed":
                robot.delay_remaining -= 1
                if robot.delay_remaining <= 0:
                    robot.state = "moving" if robot.path else "idle"
                continue

            if robot.path:
                next_cell = robot.path.pop(0)
                # Verify that next cell is not an obstacle
                if self.map.is_free(next_cell):
                    robot.pos = next_cell
                    robot.state = "moving"
                else:
                    # Obstacle encountered, wait and re-plan
                    robot.state = "idle"
                    self._plan_path(robot)

                # Check if arrived at goal
                if robot.pos == robot.goal:
                    self._assign_new_goal(robot)
                    self._plan_path(robot)
            else:
                robot.state = "idle"
                self._assign_new_goal(robot)
                self._plan_path(robot)

        self._record_frame()

    def _record_frame(self):
        positions = [r.pos for r in self.robots]
        states = [r.state for r in self.robots]
        goals = [r.goal for r in self.robots]
        self.frames.append(Frame(t=self.t, pos=positions, state=states, goals=goals))

    def to_replay(self, seed: int = 42, mode: str = "baseline_pibt", map_name: str = "warehouse-10-20-10-2-1") -> Dict[str, Any]:
        return {
            "meta": {
                "seed": seed,
                "mode": mode,
                "map_name": map_name,
                "steps": len(self.frames)
            },
            "map": {
                "width": self.map.width,
                "height": self.map.height,
                "obstacles": self.map.get_obstacle_cells()
            },
            "frames": [f.model_dump() for f in self.frames],
            "incidents": [],
            "metrics": {
                "repair_success_rate": 1.0,
                "mean_recovery_steps": 0.0,
                "cost_delta_pct": 0.0,
                "llm_calls": 0,
                "llm_tokens": 0,
                "cache_hits": 0
            }
        }
