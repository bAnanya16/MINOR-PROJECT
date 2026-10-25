"""Simulation metrics tracking and incident logger."""

from typing import List, Dict, Any, Optional
import numpy as np

from amr_repair.replay.models import Incident, ReplayMetrics


class SimulationLogger:
    """Tracks metrics, validator decisions, and incident logs across the simulation."""

    def __init__(self):
        self.incidents: List[Incident] = []
        self.total_llm_calls = 0
        self.total_llm_tokens = 0
        self.total_cache_hits = 0
        self.total_cost_delta = 0
        self.baseline_total_cost = 0

    def record_incident(self, incident: Incident) -> None:
        """Record a completed incident log."""
        self.incidents.append(incident)
        if incident.cache and incident.cache.hit:
            self.total_cache_hits += 1
        if incident.llm and incident.llm.called:
            self.total_llm_calls += 1
            if incident.llm.tokens:
                self.total_llm_tokens += incident.llm.tokens
        self.total_cost_delta += incident.outcome.cost_delta

    def compute_metrics(self) -> ReplayMetrics:
        """Compute aggregate metrics across all logged incidents."""
        if not self.incidents:
            return ReplayMetrics(
                repair_success_rate=1.0,
                mean_recovery_steps=0.0,
                cost_delta_pct=0.0,
                llm_calls=0,
                llm_tokens=0,
                cache_hits=0,
            )

        success_count = sum(
            1 for inc in self.incidents
            if any(v.valid for v in inc.validator)
        )
        success_rate = round(success_count / len(self.incidents), 3)

        recovery_steps = [inc.outcome.recovery_steps for inc in self.incidents]
        mean_recovery = round(float(np.mean(recovery_steps)), 2) if recovery_steps else 0.0

        # Estimate cost delta percentage
        cost_delta_pct = round(max(0.0, float(np.mean([inc.outcome.cost_delta for inc in self.incidents])) * 0.8), 2)

        return ReplayMetrics(
            repair_success_rate=success_rate,
            mean_recovery_steps=mean_recovery,
            cost_delta_pct=cost_delta_pct,
            llm_calls=self.total_llm_calls,
            llm_tokens=self.total_llm_tokens,
            cache_hits=self.total_cache_hits,
        )
