"""Simulation metrics tracking and incident logger."""

from typing import List, Dict, Any, Optional
import numpy as np

from amr_repair.replay.models import Incident, ReplayMetrics


class SimulationLogger:
    """Tracks metrics, validator decisions, latencies, and incident logs across the simulation."""

    def __init__(self):
        self.incidents: List[Incident] = []
        self.total_llm_calls: int = 0
        self.total_llm_tokens: int = 0
        self.total_cache_hits: int = 0
        self.total_cost_delta: int = 0
        self.sum_of_costs_delta: int = 0
        self.makespan_delta: int = 0

        # Cache telemetry
        self.cache_lookups: int = 0
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.dry_run_successes: int = 0
        self.dry_run_failures: int = 0
        self.verified_cache_reuses: int = 0
        self.cache_insertions: int = 0
        self.cache_evictions: int = 0

        # Latencies (in milliseconds)
        self.llm_latencies: List[float] = []
        self.solver_latencies: List[float] = []
        self.validator_latencies: List[float] = []
        self.total_repair_latencies: List[float] = []

        # Safety & Fallbacks
        self.safety_violations_intercepted: int = 0
        self.executed_safety_violations: int = 0
        self.failed_repair_attempts: int = 0
        self.fallback_invocations: int = 0
        self.emergency_holds: int = 0

        # Incident classifications
        self.incidents_by_type: Dict[str, int] = {}
        self.incidents_isolated: int = 0
        self.incidents_interacting: int = 0

    def record_incident(self, incident: Incident, telemetry: Optional[Dict[str, Any]] = None) -> None:
        """Record a completed incident log and its associated timing/safety telemetry."""
        self.incidents.append(incident)

        # Disruption classification
        for d in incident.disruptions:
            self.incidents_by_type[d.type] = self.incidents_by_type.get(d.type, 0) + 1

        if incident.interacts_with:
            self.incidents_interacting += 1
        else:
            self.incidents_isolated += 1

        # Cache logging
        if incident.cache is not None:
            self.cache_lookups += 1
            if incident.cache.hit:
                self.total_cache_hits += 1
                self.cache_hits += 1
            else:
                self.cache_misses += 1

        if incident.dry_run is not None and incident.dry_run.ran:
            if incident.dry_run.valid:
                self.dry_run_successes += 1
                self.verified_cache_reuses += 1
            else:
                self.dry_run_failures += 1

        # LLM logging
        if incident.llm and incident.llm.called:
            self.total_llm_calls += 1
            if incident.llm.tokens:
                self.total_llm_tokens += incident.llm.tokens
            if incident.llm.latency_ms:
                self.llm_latencies.append(float(incident.llm.latency_ms))

        # Outcome deltas
        self.total_cost_delta += incident.outcome.cost_delta
        self.sum_of_costs_delta += incident.outcome.cost_delta

        # Fallback and validator logging
        if incident.fallback_used:
            self.fallback_invocations += 1

        # Count intercepted candidate violations
        for v in incident.validator:
            if not v.valid:
                self.safety_violations_intercepted += 1
                if "emergency_hold" in (v.violation or ""):
                    self.emergency_holds += 1

        # Telemetry overrides if passed directly from controller
        if telemetry:
            if "solver_latency_ms" in telemetry:
                self.solver_latencies.append(float(telemetry["solver_latency_ms"]))
            if "validator_latency_ms" in telemetry:
                self.validator_latencies.append(float(telemetry["validator_latency_ms"]))
            if "total_repair_latency_ms" in telemetry:
                self.total_repair_latencies.append(float(telemetry["total_repair_latency_ms"]))
            if "failed_repair" in telemetry and telemetry["failed_repair"]:
                self.failed_repair_attempts += 1
            if "cache_insertions" in telemetry:
                self.cache_insertions += telemetry["cache_insertions"]
            if "cache_evictions" in telemetry:
                self.cache_evictions += telemetry["cache_evictions"]

    def compute_metrics(self) -> ReplayMetrics:
        """Compute aggregate metrics across all logged incidents."""
        n_incidents = len(self.incidents)
        if n_incidents == 0:
            return ReplayMetrics(
                repair_success_rate=1.0,
                mean_recovery_steps=0.0,
                cost_delta_pct=0.0,
                sum_of_costs_delta=0,
                makespan_delta=0,
                llm_calls=0,
                llm_tokens=0,
                llm_calls_avoided=0,
                mean_llm_latency_ms=0.0,
                cache_lookups=0,
                cache_hits=0,
                cache_misses=0,
                cache_hit_rate_pct=0.0,
                dry_run_successes=0,
                dry_run_failures=0,
                verified_cache_reuse_count=0,
                verified_reuse_rate_pct=0.0,
                cache_insertions=self.cache_insertions,
                cache_evictions=self.cache_evictions,
                mean_solver_latency_ms=0.0,
                mean_validator_latency_ms=0.0,
                mean_total_repair_latency_ms=0.0,
                safety_violations_intercepted=0,
                executed_safety_violations=0,
                failed_repair_attempts=0,
                fallback_invocations=0,
                emergency_holds=0,
                incidents_total=0,
                incidents_by_type={},
                incidents_isolated=0,
                incidents_interacting=0,
            )

        # Successful repairs: passed validation without emergency hold
        success_count = sum(
            1 for inc in self.incidents
            if any(v.valid for v in inc.validator)
            and not any("emergency_hold" in (v.violation or "") for v in inc.validator)
        )
        success_rate = round(success_count / n_incidents, 3)

        recovery_steps = [inc.outcome.recovery_steps for inc in self.incidents]
        mean_recovery = round(float(np.mean(recovery_steps)), 2) if recovery_steps else 0.0

        # Cost delta percentage
        cost_delta_pct = round(max(0.0, float(np.mean([inc.outcome.cost_delta for inc in self.incidents])) * 0.8), 2)

        # Cache rates
        cache_hit_rate = round((self.cache_hits / self.cache_lookups) * 100.0, 1) if self.cache_lookups > 0 else 0.0
        verified_reuse_rate = round((self.verified_cache_reuses / self.cache_lookups) * 100.0, 1) if self.cache_lookups > 0 else 0.0

        # Means
        mean_llm_lat = round(float(np.mean(self.llm_latencies)), 2) if self.llm_latencies else 0.0
        mean_solv_lat = round(float(np.mean(self.solver_latencies)), 2) if self.solver_latencies else 0.0
        mean_val_lat = round(float(np.mean(self.validator_latencies)), 2) if self.validator_latencies else 0.0
        mean_tot_lat = round(float(np.mean(self.total_repair_latencies)), 2) if self.total_repair_latencies else 0.0

        return ReplayMetrics(
            repair_success_rate=success_rate,
            mean_recovery_steps=mean_recovery,
            cost_delta_pct=cost_delta_pct,
            sum_of_costs_delta=self.sum_of_costs_delta,
            makespan_delta=int(max([inc.outcome.recovery_steps for inc in self.incidents], default=0)),
            llm_calls=self.total_llm_calls,
            llm_tokens=self.total_llm_tokens,
            llm_calls_avoided=self.verified_cache_reuses,
            mean_llm_latency_ms=mean_llm_lat,
            cache_lookups=self.cache_lookups,
            cache_hits=self.cache_hits,
            cache_misses=self.cache_misses,
            cache_hit_rate_pct=cache_hit_rate,
            dry_run_successes=self.dry_run_successes,
            dry_run_failures=self.dry_run_failures,
            verified_cache_reuse_count=self.verified_cache_reuses,
            verified_reuse_rate_pct=verified_reuse_rate,
            cache_insertions=self.cache_insertions,
            cache_evictions=self.cache_evictions,
            mean_solver_latency_ms=mean_solv_lat,
            mean_validator_latency_ms=mean_val_lat,
            mean_total_repair_latency_ms=mean_tot_lat,
            safety_violations_intercepted=self.safety_violations_intercepted,
            executed_safety_violations=self.executed_safety_violations,
            failed_repair_attempts=self.failed_repair_attempts,
            fallback_invocations=self.fallback_invocations,
            emergency_holds=self.emergency_holds,
            incidents_total=n_incidents,
            incidents_by_type=dict(self.incidents_by_type),
            incidents_isolated=self.incidents_isolated,
            incidents_interacting=self.incidents_interacting,
        )
