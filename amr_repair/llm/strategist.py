"""LLM Strategist using Instructor with OpenAI-compatible backend."""

import os
import time
import logging
from typing import Optional, Tuple
from pydantic import BaseModel

from amr_repair.sentinel.incident import IncidentRecord
from amr_repair.llm.schema import Directive
from amr_repair.llm.prompts import SYSTEM_PROMPT, format_user_prompt

logger = logging.getLogger(__name__)


class Strategist:
    """Strategist that queries an LLM (or deterministic heuristic fallback) to produce repair directives."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        client = None,
        force_heuristic: bool = False,
    ):
        self.model = model
        self.temperature = temperature
        self.force_heuristic = force_heuristic
        self.client = client
        self.last_latency_ms: float = 0.0
        self.last_tokens: int = 0
        self.total_calls: int = 0
        self.total_tokens: int = 0

        if not self.force_heuristic and self.client is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            base_url = os.environ.get("OPENAI_BASE_URL")
            if api_key:
                try:
                    import instructor
                    from openai import OpenAI
                    base_client = OpenAI(api_key=api_key, base_url=base_url)
                    self.client = instructor.from_openai(base_client)
                except Exception as e:
                    logger.warning(f"Could not initialize Instructor OpenAI client: {e}. Falling back to heuristic.")
                    self.client = None

    def choose(self, incident: IncidentRecord, violation: Optional[str] = None) -> Directive:
        """Choose a path repair directive for the given incident and optional previous violation."""
        start_time = time.perf_counter()
        self.total_calls += 1

        if self.client is not None and not self.force_heuristic:
            try:
                user_msg = format_user_prompt(incident, violation)
                response = self.client.chat.completions.create(
                    model=self.model,
                    response_model=Directive,
                    max_retries=1,
                    temperature=self.temperature,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                )
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                self.last_latency_ms = round(elapsed_ms, 2)
                # Estimated token count
                self.last_tokens = 450 + len(incident.affected_robots) * 30
                self.total_tokens += self.last_tokens
                return response
            except Exception as e:
                logger.warning(f"LLM call failed: {e}. Falling back to deterministic heuristic.")

        # Deterministic heuristic fallback (always valid and predictable)
        directive = self._heuristic_choice(incident, violation)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        self.last_latency_ms = round(elapsed_ms, 2)
        self.last_tokens = 0
        return directive

    def _heuristic_choice(self, incident: IncidentRecord, violation: Optional[str] = None) -> Directive:
        robots = list(incident.affected_robots)
        if not robots:
            robots = [0]

        # Prioritize robots: reverse on retry or if violation occurred
        if violation:
            # On validation failure, upgrade to windowed_pbs with larger horizon and reversed priority
            priority = list(reversed(robots))
            return Directive(
                robots=robots,
                solver="windowed_pbs",
                priority=priority,
                horizon=30,
                rationale=f"Validation retry resolving '{violation[:40]}...': PBS with extended horizon 30.",
            )

        # Interacting disruptions -> Windowed PBS
        if len(incident.interacts_with) > 0 or len(robots) >= 4:
            priority = list(robots)
            return Directive(
                robots=robots,
                solver="windowed_pbs",
                priority=priority,
                horizon=25,
                rationale="Interacting / multi-robot disruption; chosen windowed_pbs for joint conflict resolution.",
            )

        # High congestion -> LNS
        if incident.congestion.level == "high" or len(robots) >= 2:
            priority = list(robots)
            return Directive(
                robots=robots,
                solver="lns",
                priority=priority,
                horizon=20,
                rationale="Congested area; chosen LNS for neighborhood trajectory repair.",
            )

        # Default / isolated disruption -> PIBT
        priority = list(robots)
        return Directive(
            robots=robots,
            solver="pibt",
            priority=priority,
            horizon=15,
            rationale="Single isolated disruption; chosen PIBT for minimal latency.",
        )
