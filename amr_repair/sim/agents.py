"""Robot agent representation for lifelong MAPF simulation."""

from dataclasses import dataclass, field
from typing import List, Tuple, Literal

RobotState = Literal["moving", "idle", "delayed", "failed", "replanning"]


@dataclass
class Robot:
    id: int
    pos: Tuple[int, int]
    goal: Tuple[int, int]
    path: List[Tuple[int, int]] = field(default_factory=list)
    state: RobotState = "idle"
    delay_remaining: int = 0
