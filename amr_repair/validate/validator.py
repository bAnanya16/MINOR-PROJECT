"""Deterministic validator for multi-agent paths."""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from amr_repair.sim.lifelong import World


@dataclass
class Verdict:
    valid: bool
    violation: Optional[str] = None


def validate(paths: Dict[int, List[Tuple[int, int]]], world: World, t0: int) -> Verdict:
    """Validates multi-agent paths starting at step t0 against static/moving obstacles and conflicts.
    
    Checks in exact order:
    1. A cell is blocked (world static obstacle or temporary blocked cell): 'blocked_cell robot <r> cell [<x>,<y>] t=<t>'
    2. A cell is a failed robot's cell: 'failed_robot robot <r> cell [<x>,<y>] t=<t>'
    3. Vertex collision (two robots occupy the same cell at the same step): 'vertex collision robots <r1>,<r2> cell [<x>,<y>] t=<t>'
    4. Edge collision (two robots swap cells between t and t+1): 'edge swap robots <r1>,<r2> cells [<x1>,<y1>]->[<x2>,<y2>] t=<t>'
    5. Deadlock (no robot in the set moves for 10 consecutive steps while not at goal): 'deadlock robots <r_list> steps <t_start>..<t_end>'
    """
    if not paths:
        return Verdict(valid=True, violation=None)

    robot_ids = sorted(list(paths.keys()))
    max_len = max(len(p) for p in paths.values())

    failed_cells = set()
    for r in world.robots:
        if r.state == "failed":
            failed_cells.add(r.pos)

    # 1 & 2: Check static obstacles and failed robot cells
    for r_id in robot_ids:
        path = paths[r_id]
        for step_idx, cell in enumerate(path):
            current_t = t0 + step_idx
            if not world.map.is_free(cell):
                return Verdict(
                    valid=False,
                    violation=f"blocked_cell robot {r_id} cell [{cell[0]},{cell[1]}] t={current_t}"
                )
            if cell in failed_cells:
                return Verdict(
                    valid=False,
                    violation=f"failed_robot robot {r_id} cell [{cell[0]},{cell[1]}] t={current_t}"
                )

    # 3 & 4: Check vertex and edge collisions across steps
    for step_idx in range(max_len):
        current_t = t0 + step_idx
        occupied: Dict[Tuple[int, int], int] = {}

        # Vertex collision check at step_idx
        for r_id in robot_ids:
            path = paths[r_id]
            cell = path[step_idx] if step_idx < len(path) else path[-1]
            if cell in occupied:
                other_r = occupied[cell]
                r_pair = sorted([r_id, other_r])
                return Verdict(
                    valid=False,
                    violation=f"vertex collision robots {r_pair[0]},{r_pair[1]} cell [{cell[0]},{cell[1]}] t={current_t}"
                )
            occupied[cell] = r_id

        # Edge swap check between step_idx and step_idx + 1
        if step_idx < max_len - 1:
            for i in range(len(robot_ids)):
                r1 = robot_ids[i]
                p1 = paths[r1]
                c1_now = p1[step_idx] if step_idx < len(p1) else p1[-1]
                c1_next = p1[step_idx + 1] if step_idx + 1 < len(p1) else p1[-1]

                for j in range(i + 1, len(robot_ids)):
                    r2 = robot_ids[j]
                    p2 = paths[r2]
                    c2_now = p2[step_idx] if step_idx < len(p2) else p2[-1]
                    c2_next = p2[step_idx + 1] if step_idx + 1 < len(p2) else p2[-1]

                    if c1_now == c2_next and c2_now == c1_next and c1_now != c2_now:
                        r_pair = sorted([r1, r2])
                        return Verdict(
                            valid=False,
                            violation=f"edge swap robots {r_pair[0]},{r_pair[1]} cells [{c1_now[0]},{c1_now[1]}]->[{c1_next[0]},{c1_next[1]}] t={current_t}"
                        )

    # 5: Deadlock check (no robot in the set moves for 10 consecutive steps while not at goal)
    if max_len >= 10:
        robot_goals = {r.id: r.goal for r in world.robots if r.id in paths}
        for start_step in range(max_len - 9):
            all_stayed = True
            for r_id in robot_ids:
                p = paths[r_id]
                window = [p[s] if s < len(p) else p[-1] for s in range(start_step, start_step + 10)]
                # If robot moved or is already at its goal, not deadlocked
                goal = robot_goals.get(r_id)
                if len(set(window)) > 1 or (goal and window[-1] == goal):
                    all_stayed = False
                    break

            if all_stayed:
                return Verdict(
                    valid=False,
                    violation=f"deadlock robots {robot_ids} steps {t0 + start_step}..{t0 + start_step + 9}"
                )

    return Verdict(valid=True, violation=None)
