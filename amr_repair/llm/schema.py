"""Directive schema for LLM-guided multi-robot path repair."""

from typing import List, Literal
from pydantic import BaseModel, Field, field_validator, model_validator


class Directive(BaseModel):
    """Path repair directive produced by the LLM or retrieved from strategy cache."""

    robots: List[int] = Field(
        ...,
        description="List of robot IDs to replan. Must be non-empty.",
    )
    solver: Literal["pibt", "lns", "windowed_pbs"] = Field(
        ...,
        description="The multi-agent solver to execute: 'pibt', 'lns', or 'windowed_pbs'.",
    )
    priority: List[int] = Field(
        ...,
        description="Planning priority order. Must be a valid permutation of robots.",
    )
    horizon: int = Field(
        ...,
        ge=10,
        le=40,
        description="Planning horizon window in discrete time steps (10 to 40).",
    )
    rationale: str = Field(
        default="",
        max_length=200,
        description="Brief justification under 200 characters explaining the choice.",
    )

    @field_validator("robots")
    @classmethod
    def validate_robots_non_empty(cls, v: List[int]) -> List[int]:
        if not v:
            raise ValueError("robots list cannot be empty.")
        if len(v) != len(set(v)):
            raise ValueError("robots list contains duplicate robot IDs.")
        return v

    @model_validator(mode="after")
    def validate_priority_is_permutation(self) -> "Directive":
        if set(self.priority) != set(self.robots) or len(self.priority) != len(self.robots):
            raise ValueError(
                f"priority ({self.priority}) must be an exact permutation of robots ({self.robots})."
            )
        return self
