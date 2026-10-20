"""Sentinel detection engine: converts fired disruptions into IncidentRecords."""

from typing import List, Tuple, Set, Optional
import numpy as np

from amr_repair.sim.lifelong import World
from amr_repair.disrupt.injector import DisruptionItem
from amr_repair.sentinel.incident import (
    IncidentRecord, DisruptionData, CongestionData, SizeBucket, CongestionLevel
)


class IncidentDetector:
    def __init__(self):
        self._incident_counter = 0

    def detect(self, world: World, fired_disruptions: List[DisruptionItem], t: int) -> List[IncidentRecord]:
        if not fired_disruptions:
            return []

        raw_incidents: List[IncidentRecord] = []

        for d in fired_disruptions:
            self._incident_counter += 1
            inc_id = f"inc_{self._incident_counter:03d}"

            # 1. Blocked/disruption cells
            cells = list(d.cells)
            if not cells and d.robot is not None:
                robot_obj = next((r for r in world.robots if r.id == d.robot), None)
                if robot_obj:
                    cells = [robot_obj.pos]

            # 2. Affected robots:
            # - whose remaining planned path crosses any disruption cell
            # - or who are themselves failed / delayed
            affected: Set[int] = set()
            cell_set = set(cells)

            for r in world.robots:
                if d.robot is not None and r.id == d.robot:
                    affected.add(r.id)
                elif any(step_cell in cell_set for step_cell in r.path):
                    affected.add(r.id)

            affected_list = sorted(list(affected))

            # 3. Congestion region: bounding box padded by 3 cells
            if cells:
                min_x = min(c[0] for c in cells) - 3
                max_x = max(c[0] for c in cells) + 3
                min_y = min(c[1] for c in cells) - 3
                max_y = max(c[1] for c in cells) + 3
            else:
                min_x, max_x, min_y, max_y = 0, 6, 0, 6

            x0 = max(0, min_x)
            x1 = min(world.map.width - 1, max_x)
            y0 = max(0, min_y)
            y1 = min(world.map.height - 1, max_y)

            # Count free cells and robots inside region
            free_count = 0
            robots_in_region = 0

            for y in range(y0, y1 + 1):
                for x in range(x0, x1 + 1):
                    if world.map.is_free((x, y)):
                        free_count += 1

            for r in world.robots:
                rx, ry = r.pos
                if x0 <= rx <= x1 and y0 <= ry <= y1:
                    robots_in_region += 1

            density = float(robots_in_region) / float(max(1, free_count))
            density = float(np.clip(density, 0.0, 1.0))

            if density < 0.15:
                level: CongestionLevel = "low"
            elif density <= 0.35:
                level = "medium"
            else:
                level = "high"

            # Size bucket: small <= 2, medium <= 6, large
            count_metric = max(len(cells), len(affected_list))
            if count_metric <= 2:
                size_bucket: SizeBucket = "small"
            elif count_metric <= 6:
                size_bucket = "medium"
            else:
                size_bucket = "large"

            inc_rec = IncidentRecord(
                id=inc_id,
                t_detected=t,
                disruptions=[DisruptionData(
                    type=d.type,
                    cells=cells,
                    robot=d.robot,
                    k_steps=d.k_steps
                )],
                affected_robots=affected_list,
                size_bucket=size_bucket,
                congestion=CongestionData(
                    level=level,
                    region=((x0, y0), (x1, y1)),
                    density=density
                ),
                interacts_with=[]
            )
            raw_incidents.append(inc_rec)

        # Merge interacting incidents
        return self._merge_interacting_incidents(raw_incidents)

    def _merge_interacting_incidents(self, incidents: List[IncidentRecord]) -> List[IncidentRecord]:
        if len(incidents) <= 1:
            return incidents

        merged: List[IncidentRecord] = []
        visited = set()

        for i in range(len(incidents)):
            if i in visited:
                continue

            current = incidents[i]
            interacting_ids = []

            for j in range(i + 1, len(incidents)):
                if j in visited:
                    continue
                other = incidents[j]

                # Two incidents interact if affected robots overlap or regions intersect
                robots_overlap = bool(set(current.affected_robots).intersection(set(other.affected_robots)))
                regions_overlap = self._regions_intersect(current.congestion.region, other.congestion.region)

                if robots_overlap or regions_overlap:
                    visited.add(j)
                    interacting_ids.append(other.id)

                    # Merge into current
                    merged_disruptions = current.disruptions + other.disruptions
                    merged_affected = sorted(list(set(current.affected_robots).union(set(other.affected_robots))))

                    # Union of bounding boxes
                    rx0 = min(current.congestion.region[0][0], other.congestion.region[0][0])
                    ry0 = min(current.congestion.region[0][1], other.congestion.region[0][1])
                    rx1 = max(current.congestion.region[1][0], other.congestion.region[1][0])
                    ry1 = max(current.congestion.region[1][1], other.congestion.region[1][1])

                    avg_density = (current.congestion.density + other.congestion.density) / 2.0
                    avg_level: CongestionLevel = "high" if avg_density > 0.35 else ("medium" if avg_density >= 0.15 else "low")

                    count_metric = max(len(merged_disruptions), len(merged_affected))
                    new_size: SizeBucket = "small" if count_metric <= 2 else ("medium" if count_metric <= 6 else "large")

                    current = IncidentRecord(
                        id=current.id,
                        t_detected=current.t_detected,
                        disruptions=merged_disruptions,
                        affected_robots=merged_affected,
                        size_bucket=new_size,
                        congestion=CongestionData(
                            level=avg_level,
                            region=((rx0, ry0), (rx1, ry1)),
                            density=avg_density
                        ),
                        interacts_with=sorted(list(set(current.interacts_with + [current.id, other.id])))
                    )

            if interacting_ids and not current.interacts_with:
                current.interacts_with = sorted(list(set([current.id] + interacting_ids)))

            merged.append(current)
            visited.add(i)

        return merged

    def _regions_intersect(self, reg1: Tuple[Tuple[int, int], Tuple[int, int]], reg2: Tuple[Tuple[int, int], Tuple[int, int]]) -> bool:
        (x1_0, y1_0), (x1_1, y1_1) = reg1
        (x2_0, y2_0), (x2_1, y2_1) = reg2
        return not (x1_1 < x2_0 or x1_0 > x2_1 or y1_1 < y2_0 or y1_0 > y2_1)
