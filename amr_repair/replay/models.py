"""Pydantic schemas and serialization models for the warehouse simulation replay."""

from typing import List, Dict, Optional, Literal, Tuple
from pydantic import BaseModel, Field


class ReplayMeta(BaseModel):
    seed: int
    mode: Literal["llm", "baseline_pibt", "baseline_lns", "baseline_pbs"]
    map_name: str
    steps: int


class MapData(BaseModel):
    width: int
    height: int
    obstacles: List[Tuple[int, int]]


RobotState = Literal["moving", "idle", "delayed", "failed", "replanning"]


class Frame(BaseModel):
    t: int
    pos: List[Tuple[int, int]]  # Indexed by robot id
    state: List[RobotState]     # Indexed by robot id


DisruptionType = Literal["blocked_cell", "robot_failure", "robot_delay"]


class Disruption(BaseModel):
    type: DisruptionType
    cells: List[Tuple[int, int]] = Field(default_factory=list)
    robot: Optional[int] = None
    k_steps: Optional[int] = None


CongestionLevel = Literal["low", "medium", "high"]


class Congestion(BaseModel):
    level: CongestionLevel
    region: Tuple[Tuple[int, int], Tuple[int, int]]  # ((x0, y0), (x1, y1))
    density: Optional[float] = None


class CacheResult(BaseModel):
    hit: bool
    matched: Optional[str] = None
    similarity: Optional[float] = None


class DryRunResult(BaseModel):
    ran: bool
    valid: bool
    violation: Optional[str] = None


SolverType = Literal["pibt", "lns", "windowed_pbs"]


class LLMDirective(BaseModel):
    robots: List[int]
    solver: SolverType
    priority: List[int]
    horizon: int
    rationale: Optional[str] = None


class LLMResult(BaseModel):
    called: bool
    latency_ms: Optional[int] = None
    tokens: Optional[int] = None
    directive: Optional[LLMDirective] = None


class ValidatorAttempt(BaseModel):
    attempt: int
    valid: bool
    violation: Optional[str] = None


class PathsData(BaseModel):
    invalidated: Dict[str, List[Tuple[int, int]]] = Field(default_factory=dict)
    repaired: Dict[str, List[Tuple[int, int]]] = Field(default_factory=dict)


class IncidentOutcome(BaseModel):
    recovery_steps: int
    cost_delta: int


class Incident(BaseModel):
    id: str
    t_detected: int
    disruptions: List[Disruption]
    affected_robots: List[int]
    congestion: Congestion
    interacts_with: List[str] = Field(default_factory=list)
    cache: Optional[CacheResult] = None
    dry_run: Optional[DryRunResult] = None
    llm: Optional[LLMResult] = None
    validator: List[ValidatorAttempt] = Field(default_factory=list)
    fallback_used: bool = False
    executed_at: int
    paths: PathsData
    outcome: IncidentOutcome


class ReplayMetrics(BaseModel):
    repair_success_rate: float
    mean_recovery_steps: float
    cost_delta_pct: float
    llm_calls: int
    llm_tokens: int
    cache_hits: int


class Replay(BaseModel):
    meta: ReplayMeta
    map: MapData
    frames: List[Frame]
    incidents: List[Incident]
    metrics: ReplayMetrics
