"""Disruption injection engine for lifelong MAPF simulation."""

from typing import List, Dict, Any, Optional, Tuple, Literal, Union
from dataclasses import dataclass, field
import numpy as np

from amr_repair.sim.lifelong import World
from amr_repair.sim.agents import Robot


@dataclass
class DisruptionItem:
    type: Literal["blocked_cell", "robot_failure", "robot_delay"]
    t: int
    cells: List[Tuple[int, int]] = field(default_factory=list)
    robot: Optional[int] = None
    k_steps: Optional[int] = None
    duration: Optional[int] = None  # For temporary blocked cells (unblocked at t + duration)


Disruption = DisruptionItem


class Injector:
    def __init__(self, rng: np.random.Generator, schedule: Optional[Union[List[Union[Dict[str, Any], DisruptionItem]], Dict[str, Any]]] = None, world: Optional[World] = None):
        self.rng: np.random.Generator = rng
        self.raw_schedule = schedule
        self.disruptions: List[DisruptionItem] = []
        self._unblock_schedule: Dict[int, List[Tuple[int, int]]] = {}

        if isinstance(schedule, list):
            # Explicit list of disruptions
            for item in schedule:
                if isinstance(item, DisruptionItem):
                    self.disruptions.append(item)
                elif isinstance(item, dict):
                    self.disruptions.append(DisruptionItem(
                        type=item["type"],
                        t=item["t"],
                        cells=[tuple(c) for c in item.get("cells", [])],
                        robot=item.get("robot"),
                        k_steps=item.get("k_steps"),
                        duration=item.get("duration")
                    ))
        elif isinstance(schedule, dict):
            # Generator config: {rate, types, size_range, overlap: none | same_aisle | same_robot, max_steps}
            if world is not None:
                self._generate_schedule(schedule, world)

        # Sort by step t
        self.disruptions.sort(key=lambda d: d.t)

    def _generate_schedule(self, config: Dict[str, Any], world: World):
        rate = config.get("rate", 0.05)
        allowed_types = config.get("types", ["blocked_cell", "robot_failure", "robot_delay"])
        overlap_mode = config.get("overlap", "none")
        max_steps = config.get("max_steps", 200)
        free_cells = world.free_cells

        t = 10
        while t < max_steps:
            if self.rng.random() < rate:
                dtype = str(self.rng.choice(allowed_types))
                d_item = self._create_disruption(dtype, t, world, free_cells)
                self.disruptions.append(d_item)

                # If overlap requested, place a second disruption within 5 cells or same robot within 10 steps
                if overlap_mode != "none":
                    t2 = t + int(self.rng.integers(1, 10))
                    if overlap_mode == "same_robot" and d_item.robot is not None:
                        # Second disruption affects the same robot
                        second_type = "robot_delay" if d_item.type != "robot_delay" else "blocked_cell"
                        r_target = d_item.robot
                        r_obj = next((r for r in world.robots if r.id == r_target), None)
                        r_cell = r_obj.pos if r_obj else free_cells[0]
                        d_item2 = DisruptionItem(
                            type=second_type,
                            t=t2,
                            cells=[r_cell],
                            robot=r_target,
                            k_steps=5
                        )
                        self.disruptions.append(d_item2)
                    elif overlap_mode in ("same_aisle", "none", "same_robot"):
                        # Within 5 cells of the first disruption
                        base_cell = d_item.cells[0] if d_item.cells else free_cells[0]
                        nearby_x = max(0, min(world.map.width - 1, base_cell[0] + int(self.rng.integers(-4, 5))))
                        nearby_y = max(0, min(world.map.height - 1, base_cell[1] + int(self.rng.integers(-2, 3))))
                        near_cell = (nearby_x, nearby_y)
                        if not world.map.is_free(near_cell):
                            near_cell = base_cell

                        d_item2 = DisruptionItem(
                            type="blocked_cell",
                            t=t2,
                            cells=[near_cell]
                        )
                        self.disruptions.append(d_item2)

                t += 15  # Spacing
            else:
                t += 1

    def _create_disruption(self, dtype: str, t: int, world: World, free_cells: List[Tuple[int, int]]) -> DisruptionItem:
        if dtype == "blocked_cell":
            cell_idx = int(self.rng.integers(0, len(free_cells)))
            cell = free_cells[cell_idx]
            duration = int(self.rng.integers(15, 45)) if self.rng.random() < 0.5 else None
            return DisruptionItem(type="blocked_cell", t=t, cells=[cell], duration=duration)

        elif dtype == "robot_failure":
            robot = self.rng.choice(world.robots)
            return DisruptionItem(type="robot_failure", t=t, cells=[robot.pos], robot=robot.id)

        else:  # robot_delay
            robot = self.rng.choice(world.robots)
            k_steps = int(self.rng.integers(3, 8))
            return DisruptionItem(type="robot_delay", t=t, cells=[robot.pos], robot=robot.id, k_steps=k_steps)

    def apply(self, world: World, t: int) -> List[DisruptionItem]:
        # Handle unblocking from previous temporary blocks
        if t in self._unblock_schedule:
            for cell in self._unblock_schedule[t]:
                cx, cy = cell
                world.map.blocked[cy, cx] = False
                world.map.graph.add_node(cell)
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nx_, ny_ = cx + dx, cy + dy
                    if world.map.is_free((nx_, ny_)):
                        world.map.graph.add_edge(cell, (nx_, ny_))
            del self._unblock_schedule[t]

        # Fired disruptions for step t
        fired: List[DisruptionItem] = []
        for d in self.disruptions:
            if d.t == t:
                fired.append(d)
                if d.type == "blocked_cell":
                    for cx, cy in d.cells:
                        world.map.blocked[cy, cx] = True
                        if world.map.graph.has_node((cx, cy)):
                            world.map.graph.remove_node((cx, cy))
                        if d.duration:
                            unblock_t = t + d.duration
                            self._unblock_schedule.setdefault(unblock_t, []).append((cx, cy))

                elif d.type == "robot_failure":
                    # Robot stops forever and its cell becomes obstacle
                    for r in world.robots:
                        if r.id == d.robot:
                            r.state = "failed"
                            r.path = []
                            cx, cy = r.pos
                            world.map.blocked[cy, cx] = True
                            if world.map.graph.has_node((cx, cy)):
                                world.map.graph.remove_node((cx, cy))

                elif d.type == "robot_delay":
                    # Robot waits k steps then resumes
                    for r in world.robots:
                        if r.id == d.robot:
                            r.state = "delayed"
                            r.delay_remaining = d.k_steps or 5

        return fired
