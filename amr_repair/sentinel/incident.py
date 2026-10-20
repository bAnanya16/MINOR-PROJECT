"""Incident representation and feature vector extraction for the strategy cache and LLM."""

from typing import List, Tuple, Literal, Optional
from pydantic import BaseModel, Field
import numpy as np


SizeBucket = Literal["small", "medium", "large"]
CongestionLevel = Literal["low", "medium", "high"]
DisruptionType = Literal["blocked_cell", "robot_failure", "robot_delay"]


class DisruptionData(BaseModel):
    type: DisruptionType
    cells: List[Tuple[int, int]] = Field(default_factory=list)
    robot: Optional[int] = None
    k_steps: Optional[int] = None


class CongestionData(BaseModel):
    level: CongestionLevel
    region: Tuple[Tuple[int, int], Tuple[int, int]]  # ((x0, y0), (x1, y1))
    density: float


class IncidentRecord(BaseModel):
    id: str
    t_detected: int
    disruptions: List[DisruptionData]
    affected_robots: List[int]
    size_bucket: SizeBucket
    congestion: CongestionData
    interacts_with: List[str] = Field(default_factory=list)

    def features(self) -> np.ndarray:
        """Returns a fixed-length numpy feature vector:
        - one-hot type x3: [is_blocked_cell, is_robot_failure, is_robot_delay]
        - size bucket x3: [is_small, is_medium, is_large]
        - congestion level x3: [is_low, is_medium, is_high]
        - affected count (normalized by e.g. 10)
        - interaction flag: 1.0 if interacts_with else 0.0
        - density: float in [0, 1]
        Total length: 12 floats
        """
        # Types present
        type_set = {d.type for d in self.disruptions}
        type_vec = [
            1.0 if "blocked_cell" in type_set else 0.0,
            1.0 if "robot_failure" in type_set else 0.0,
            1.0 if "robot_delay" in type_set else 0.0
        ]

        # Size bucket
        size_vec = [
            1.0 if self.size_bucket == "small" else 0.0,
            1.0 if self.size_bucket == "medium" else 0.0,
            1.0 if self.size_bucket == "large" else 0.0
        ]

        # Congestion level
        cong_vec = [
            1.0 if self.congestion.level == "low" else 0.0,
            1.0 if self.congestion.level == "medium" else 0.0,
            1.0 if self.congestion.level == "high" else 0.0
        ]

        affected_count_norm = float(len(self.affected_robots)) / 10.0
        has_interaction = 1.0 if len(self.interacts_with) > 0 else 0.0
        density = float(np.clip(self.congestion.density, 0.0, 1.0))

        vec = type_vec + size_vec + cong_vec + [affected_count_norm, has_interaction, density]
        return np.array(vec, dtype=np.float32)
