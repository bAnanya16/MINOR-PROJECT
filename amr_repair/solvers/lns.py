"""Large Neighborhood Search (LNS) solver."""

import time
import random
from typing import List, Dict, Tuple, Set
from amr_repair.sim.lifelong import World, manhattan_distance
from amr_repair.solvers.space_time_astar import plan_space_time_path
from amr_repair.solvers.pibt import solve as pibt_solve


def solve(world: World, robots: List[int], priority: List[int], horizon: int) -> Dict[int, List[Tuple[int, int]]]:
    """LNS: starts from initial paths (via PIBT/prioritized search),
    repeatedly destroys a neighbourhood of 3-5 robots and repairs them with space-time A*.
    Accepts if sum-of-costs does not increase. Stops after 50 iterations or 200ms.
    """
    start_time = time.perf_counter()
    robot_map = {r.id: r for r in world.robots}
    planned_robots = [r_id for r_id in priority if r_id in robot_map]

    # Initial plan using PIBT as starting solution
    current_paths = pibt_solve(world, robots, priority, horizon)

    def sum_of_costs(paths: Dict[int, List[Tuple[int, int]]]) -> int:
        cost = 0
        for r_id, p in paths.items():
            goal = robot_map[r_id].goal
            # Cost is distance remaining + steps to arrive
            dist = manhattan_distance(p[-1], goal) if p else 0
            cost += len(p) + dist
        return cost

    best_cost = sum_of_costs(current_paths)
    best_paths = {k: list(v) for k, v in current_paths.items()}

    rng = random.Random(42)

    for _ in range(50):
        if (time.perf_counter() - start_time) > 0.18:
            break

        if len(planned_robots) <= 1:
            break

        # Select random neighbourhood of min(len, 3..5) robots
        k = min(len(planned_robots), rng.randint(2, min(5, len(planned_robots))))
        destroyed = rng.sample(planned_robots, k)
        undestroyed = [r_id for r_id in planned_robots if r_id not in destroyed]

        # Build reservations from undestroyed robots
        v_res: Dict[int, Set[Tuple[int, int]]] = {}
        e_res: Set[Tuple[int, Tuple[int, int], Tuple[int, int]]] = set()

        # Reservations from other unmanaged world robots
        for r in world.robots:
            if r.id not in planned_robots:
                for step_idx in range(horizon + 1):
                    cell = r.path[step_idx] if step_idx < len(r.path) else r.pos
                    v_res.setdefault(step_idx, set()).add(cell)

        # Reservations from undestroyed managed robots
        for r_id in undestroyed:
            p = best_paths[r_id]
            prev_cell = robot_map[r_id].pos
            v_res.setdefault(0, set()).add(prev_cell)
            for s_idx, cell in enumerate(p):
                v_res.setdefault(s_idx + 1, set()).add(cell)
                e_res.add((s_idx, prev_cell, cell))
                prev_cell = cell

        # Re-plan destroyed robots with space-time A*
        new_paths = {k: list(v) for k, v in best_paths.items()}
        repair_ok = True

        for r_id in destroyed:
            r = robot_map[r_id]
            rep_path = plan_space_time_path(world, r.pos, r.goal, horizon, v_res, e_res)
            if rep_path is None:
                repair_ok = False
                break
            new_paths[r_id] = rep_path

            # Reserve this robot's path
            prev_c = r.pos
            v_res.setdefault(0, set()).add(prev_c)
            for s_idx, cell in enumerate(rep_path):
                v_res.setdefault(s_idx + 1, set()).add(cell)
                e_res.add((s_idx, prev_c, cell))
                prev_c = cell

        if repair_ok:
            new_cost = sum_of_costs(new_paths)
            if new_cost <= best_cost:
                best_cost = new_cost
                best_paths = new_paths

    return best_paths
