"""Priority Inheritance with Backtracking (PIBT) solver."""

from typing import List, Dict, Tuple, Set, Optional
import networkx as nx
from amr_repair.sim.lifelong import World, manhattan_distance


def solve(world: World, robots: List[int], priority: List[int], horizon: int) -> Dict[int, List[Tuple[int, int]]]:
    """PIBT: per step, each robot in priority order picks best neighbour toward goal.
    On conflict, the higher-priority robot inherits and pushes the lower one.
    Treats unmanaged robots as moving obstacles according to their current paths.
    """
    robot_map = {r.id: r for r in world.robots}
    planned_robots = [r_id for r_id in priority if r_id in robot_map]

    # Pre-populate static and dynamic reservations of unmanaged robots
    other_reservations: Dict[int, Set[Tuple[int, int]]] = {}
    for r in world.robots:
        if r.id not in planned_robots:
            for step_idx in range(horizon + 1):
                if step_idx < len(r.path):
                    cell = r.path[step_idx]
                else:
                    cell = r.pos
                other_reservations.setdefault(step_idx, set()).add(cell)

    # Initialize paths with start positions
    current_pos: Dict[int, Tuple[int, int]] = {r_id: robot_map[r_id].pos for r_id in planned_robots}
    paths: Dict[int, List[Tuple[int, int]]] = {r_id: [current_pos[r_id]] for r_id in planned_robots}

    for step in range(1, horizon + 1):
        reserved_now = other_reservations.get(step, set()).copy()
        next_pos: Dict[int, Tuple[int, int]] = {}

        def pibt_step(r_id: int) -> bool:
            curr = current_pos[r_id]
            goal = robot_map[r_id].goal

            # Candidate neighbors (including wait)
            if world.map.graph.has_node(curr):
                neighbors = list(world.map.graph.neighbors(curr)) + [curr]
            else:
                neighbors = [
                    (curr[0] + dx, curr[1] + dy)
                    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]
                    if world.map.is_free((curr[0] + dx, curr[1] + dy))
                ] + [curr]

            # Sort by heuristic distance to goal
            neighbors.sort(key=lambda c: manhattan_distance(c, goal))

            for cand in neighbors:
                if cand in reserved_now:
                    continue

                # Check edge collision with already decided robots
                edge_conflict = False
                for other_id, other_next in next_pos.items():
                    if other_next == curr and current_pos[other_id] == cand:
                        edge_conflict = True
                        break
                if edge_conflict:
                    continue

                # Candidate is free, tentatively reserve
                reserved_now.add(cand)
                next_pos[r_id] = cand

                # If another robot currently occupies this cell, try pushing it
                conflict_robot = None
                for other_id, pos in current_pos.items():
                    if other_id != r_id and other_id not in next_pos and pos == cand:
                        conflict_robot = other_id
                        break

                if conflict_robot is not None:
                    # Recursive push
                    if pibt_step(conflict_robot):
                        return True
                    else:
                        # Cannot push, backtrack
                        reserved_now.remove(cand)
                        del next_pos[r_id]
                        continue
                else:
                    return True

            # If no neighbor worked, attempt to stay still
            if curr not in reserved_now:
                reserved_now.add(curr)
                next_pos[r_id] = curr
                return True

            return False

        for r_id in planned_robots:
            if r_id not in next_pos:
                pibt_step(r_id)

        # Record chosen positions
        for r_id in planned_robots:
            pos = next_pos.get(r_id, current_pos[r_id])
            current_pos[r_id] = pos
            paths[r_id].append(pos)

    # Return paths excluding the starting t0 position
    return {r_id: p[1:] for r_id, p in paths.items()}
