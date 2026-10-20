"""Priority-Based Search (PBS) over a window of horizon steps."""

import time
from typing import List, Dict, Tuple, Set, Optional
from amr_repair.sim.lifelong import World
from amr_repair.solvers.space_time_astar import plan_space_time_path
from amr_repair.validate.validator import validate


def solve(world: World, robots: List[int], priority: List[int], horizon: int) -> Dict[int, List[Tuple[int, int]]]:
    """Windowed PBS: Plans paths in priority order using Space-Time A*.
    Time-boxed at 500ms.
    """
    start_time = time.perf_counter()
    robot_map = {r.id: r for r in world.robots}
    planned_robots = [r_id for r_id in priority if r_id in robot_map]

    # Pre-populate static moving obstacle reservations from unmanaged robots
    base_vertex_res: Dict[int, Set[Tuple[int, int]]] = {}
    base_edge_res: Set[Tuple[int, Tuple[int, int], Tuple[int, int]]] = set()

    for r in world.robots:
        if r.id not in planned_robots:
            prev_cell = r.pos
            base_vertex_res.setdefault(0, set()).add(prev_cell)
            for step_idx in range(1, horizon + 1):
                cell = r.path[step_idx - 1] if step_idx - 1 < len(r.path) else r.pos
                base_vertex_res.setdefault(step_idx, set()).add(cell)
                base_edge_res.add((step_idx - 1, prev_cell, cell))
                prev_cell = cell

    # Try orderings starting with given priority
    orderings_to_try = [planned_robots]

    # Add alternate pairwise swaps if priority ordering conflicts
    if len(planned_robots) >= 2:
        alt1 = list(planned_robots)
        alt1[0], alt1[1] = alt1[1], alt1[0]
        orderings_to_try.append(alt1)

    for ordering in orderings_to_try:
        if (time.perf_counter() - start_time) > 0.48:
            break

        v_res = {k: set(v) for k, v in base_vertex_res.items()}
        e_res = set(base_edge_res)
        candidate_paths: Dict[int, List[Tuple[int, int]]] = {}
        success = True

        for r_id in ordering:
            r = robot_map[r_id]
            path = plan_space_time_path(world, r.pos, r.goal, horizon, v_res, e_res)
            if path is None:
                success = False
                break
            candidate_paths[r_id] = path

            # Reserve this robot's path for subsequent lower-priority robots
            prev_c = r.pos
            v_res.setdefault(0, set()).add(prev_c)
            for s_idx, cell in enumerate(path):
                v_res.setdefault(s_idx + 1, set()).add(cell)
                e_res.add((s_idx, prev_c, cell))
                prev_c = cell

        if success:
            verdict = validate(candidate_paths, world, world.t)
            if verdict.valid:
                return candidate_paths

    # Fallback to candidate_paths if found or dummy wait
    if candidate_paths:
        return candidate_paths
    return {r_id: [robot_map[r_id].pos] * horizon for r_id in planned_robots}
