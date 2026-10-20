"""Space-Time A* search for prioritized multi-agent path planning."""

from typing import List, Tuple, Set, Dict, Optional
import heapq
from amr_repair.sim.lifelong import World, manhattan_distance


def plan_space_time_path(
    world: World,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    horizon: int,
    vertex_reservations: Dict[int, Set[Tuple[int, int]]],
    edge_reservations: Set[Tuple[int, Tuple[int, int], Tuple[int, int]]]
) -> Optional[List[Tuple[int, int]]]:
    """Plans a collision-free path for one agent over 'horizon' steps using Space-Time A*.
    Returns list of cells of length horizon.
    """
    # Priority queue holds (f_score, g_score, (x, y), t, path)
    h0 = manhattan_distance(start, goal)
    pq = [(h0, 0, start, [start])]
    visited = set()

    best_path = None
    min_dist_to_goal = float("inf")

    while pq:
        f, g, current, path = heapq.heappop(pq)
        state_key = (current, g)
        if state_key in visited:
            continue
        visited.add(state_key)

        if g == horizon:
            return path[1:]  # Found full horizon path

        dist = manhattan_distance(current, goal)
        if dist < min_dist_to_goal:
            min_dist_to_goal = dist
            best_path = path

        next_t = g + 1
        # Possible actions: 4-connected movements + wait
        candidates = list(world.map.graph.neighbors(current)) + [current]

        for cand in candidates:
            # Vertex conflict check at next_t
            if cand in vertex_reservations.get(next_t, set()):
                continue

            # Edge conflict check: (next_t - 1, cand, current)
            if (g, cand, current) in edge_reservations:
                continue

            h = manhattan_distance(cand, goal)
            heapq.heappush(pq, (next_t + h, next_t, cand, path + [cand]))

    if best_path:
        # Pad with last position if stopped early
        res = best_path[1:]
        while len(res) < horizon:
            res.append(res[-1] if res else start)
        return res

    return [start] * horizon
