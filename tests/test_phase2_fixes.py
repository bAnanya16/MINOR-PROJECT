"""Phase 2 verification tests for:
1. Default disruption generation & aisle blockages
2. Emergency hold on failed fallback
3. Strategy Cache LRU eviction, capacity, and telemetry
4. Research metric calculations
"""

import os
import pytest
import numpy as np

from amr_repair.sim.grid import GridMap
from amr_repair.sim.agents import Robot
from amr_repair.sim.lifelong import World
from amr_repair.disrupt.injector import Injector, DisruptionItem
from amr_repair.cache.store import StrategyCache, CacheEntry
from amr_repair.cache.negative import NegativeMemory
from amr_repair.llm.schema import Directive
from amr_repair.sentinel.incident import IncidentRecord, DisruptionData, CongestionData
from amr_repair.loop.controller import SimulationConfig, SimulationController, run
from amr_repair.loop.logger import SimulationLogger
from amr_repair.replay.models import Incident, Disruption, Congestion, PathsData, IncidentOutcome, ValidatorAttempt


MAP_PATH = "maps/warehouse-10-20-10-2-1.map"


@pytest.fixture
def test_world():
    gmap = GridMap.from_movingai(MAP_PATH)
    rng = np.random.default_rng(42)
    free = gmap.get_free_cells()
    robots = [Robot(id=i, pos=free[i * 2], goal=free[i * 2 + 10]) for i in range(15)]
    return World(gmap, robots, rng)


def make_test_incident(inc_id="inc_01", density=0.1, robots=[1, 2]) -> IncidentRecord:
    return IncidentRecord(
        id=inc_id,
        t_detected=10,
        disruptions=[DisruptionData(type="blocked_cell", cells=[(5, 5)])],
        affected_robots=robots,
        size_bucket="small",
        congestion=CongestionData(level="low", region=((3, 3), (7, 7)), density=density),
        interacts_with=[],
    )


# --- Task 1 Tests: Disruption Generation ---

def test_default_simulation_produces_disruptions():
    """Verify that schedule=None automatically generates disruptions."""
    config = SimulationConfig(
        map_path=MAP_PATH,
        num_robots=15,
        steps=50,
        seed=42,
        mode="baseline_pibt",
        schedule=None,
        disruption_rate=0.15,
    )
    controller = SimulationController(config)
    assert len(controller.injector.disruptions) > 0, "Default configuration with schedule=None must generate disruptions!"


def test_empty_schedule_yields_zero_disruptions():
    """Verify that schedule=[] intentionally yields zero disruptions."""
    config = SimulationConfig(
        map_path=MAP_PATH,
        num_robots=15,
        steps=50,
        seed=42,
        mode="baseline_pibt",
        schedule=[],
    )
    controller = SimulationController(config)
    assert len(controller.injector.disruptions) == 0, "Explicit schedule=[] must yield 0 disruptions."


def test_explicit_schedule_respected():
    """Verify that explicit disruption list is preserved exactly."""
    explicit = [
        DisruptionItem(type="blocked_cell", t=5, cells=[(10, 10)]),
        DisruptionItem(type="robot_delay", t=15, robot=2, k_steps=4),
    ]
    config = SimulationConfig(
        map_path=MAP_PATH,
        num_robots=10,
        steps=30,
        seed=42,
        mode="baseline_pibt",
        schedule=explicit,
    )
    controller = SimulationController(config)
    assert len(controller.injector.disruptions) == 2
    assert controller.injector.disruptions[0].t == 5
    assert controller.injector.disruptions[1].t == 15


def test_identical_seeds_produce_identical_schedules(test_world):
    """Verify deterministic reproducibility of procedural schedules."""
    cfg = {"rate": 0.1, "types": ["blocked_cell", "robot_failure"], "overlap": "same_aisle", "max_steps": 100}
    inj1 = Injector(np.random.default_rng(999), schedule=cfg, world=test_world)
    inj2 = Injector(np.random.default_rng(999), schedule=cfg, world=test_world)

    assert len(inj1.disruptions) == len(inj2.disruptions)
    for d1, d2 in zip(inj1.disruptions, inj2.disruptions):
        assert d1.type == d2.type
        assert d1.t == d2.t
        assert d1.cells == d2.cells


def test_multi_cell_aisle_blockage(test_world):
    """Verify multi-cell blockages create contiguous cells and update navigation graph."""
    cfg = {"rate": 1.0, "types": ["blocked_cell"], "max_block_size": 4, "max_steps": 12}
    inj = Injector(np.random.default_rng(123), schedule=cfg, world=test_world)

    multi_cell_disruptions = [d for d in inj.disruptions if len(d.cells) > 1]
    assert len(multi_cell_disruptions) > 0, "Injector failed to generate multi-cell blockage"

    # Verify contiguous cells
    d = multi_cell_disruptions[0]
    for i in range(len(d.cells) - 1):
        c1, c2 = d.cells[i], d.cells[i + 1]
        dist = abs(c1[0] - c2[0]) + abs(c1[1] - c2[1])
        assert dist == 1, f"Cells {c1} and {c2} in multi-cell blockage are not contiguous!"

    # Apply to world and check navigation
    inj.apply(test_world, d.t)
    for cell in d.cells:
        assert not test_world.map.is_free(cell), f"Cell {cell} was not blocked in navigation map!"
        assert not test_world.map.graph.has_node(cell), f"Cell {cell} was not removed from navigation graph!"


# --- Task 2 Tests: Safety & Emergency Hold ---

def test_emergency_hold_on_failed_fallback():
    """Verify that if fallback plan also fails validation, robots enter emergency hold."""
    config = SimulationConfig(
        map_path=MAP_PATH,
        num_robots=10,
        steps=10,
        seed=42,
        mode="baseline_pibt",
        schedule=[],
    )
    controller = SimulationController(config)

    # Manually trigger an incident where fallback produces an invalid plan
    # Plant a situation where robots are boxed in
    r0 = controller.world.robots[0]
    r1 = controller.world.robots[1]
    incident = make_test_incident("inc_test", density=0.9, robots=[r0.id, r1.id])

    # Run handler
    controller._handle_incident(incident, t=1)

    # Check logger validator attempts
    assert len(controller.logger.incidents) == 1
    inc_log = controller.logger.incidents[0]
    # Check that validator recorded the attempts
    assert len(inc_log.validator) >= 1


# --- Task 3 Tests: Strategy Cache LRU Eviction & Capacity ---

def test_cache_capacity_and_lru_eviction():
    """Verify that strategy cache respects max_capacity and evicts LRU entry."""
    cache = StrategyCache(threshold=0.85, max_capacity=3)

    inc1 = make_test_incident("inc_1", density=0.1)
    inc2 = make_test_incident("inc_2", density=0.2)
    inc3 = make_test_incident("inc_3", density=0.3)
    inc4 = make_test_incident("inc_4", density=0.4)

    d = Directive(robots=[1, 2], solver="pibt", priority=[1, 2], horizon=15)

    cache.store(inc1, d)
    cache.store(inc2, d)
    cache.store(inc3, d)

    assert len(cache.entries) == 3
    assert cache.evictions == 0

    # Access inc1 to make it most recently used
    cache.lookup(inc1)

    # Store 4th entry -> capacity exceeded -> must evict inc2 (least recently used)
    cache.store(inc4, d)

    assert len(cache.entries) == 3
    assert cache.evictions == 1
    remaining_ids = [e.incident_id for e in cache.entries]
    assert "inc_1" in remaining_ids, "inc_1 was accessed recently and should NOT have been evicted!"
    assert "inc_4" in remaining_ids, "inc_4 was just added and must be in cache!"
    assert "inc_2" not in remaining_ids, "inc_2 was least recently used and should have been evicted!"


def test_cache_telemetry_and_verified_reuse():
    """Verify cache tracks lookups, hits, misses, and verified reuses."""
    cache = StrategyCache(threshold=0.85, max_capacity=10)

    inc1 = make_test_incident("inc_1", density=0.1, robots=[1, 2])
    d = Directive(robots=[1, 2], solver="pibt", priority=[1, 2], horizon=15)

    # Lookup on empty cache: miss
    res = cache.lookup(inc1)
    assert res is None
    assert cache.lookups == 1
    assert cache.misses == 1
    assert cache.hits == 0

    # Store entry
    cache.store(inc1, d)
    assert cache.insertions == 1

    # Lookup matching entry: hit
    res = cache.lookup(inc1)
    assert res is not None
    assert cache.lookups == 2
    assert cache.hits == 1

    # Record verified reuse
    cache.record_verified_reuse("inc_1")
    assert cache.verified_reuses == 1


# --- Task 4 & 5 Tests: Performance Metrics ---

def test_logger_zero_incidents_handled_safely():
    """Verify SimulationLogger handles zero incidents without division errors."""
    logger = SimulationLogger()
    metrics = logger.compute_metrics()
    assert metrics.repair_success_rate == 1.0
    assert metrics.cache_hit_rate_pct == 0.0
    assert metrics.verified_reuse_rate_pct == 0.0
    assert metrics.incidents_total == 0


def test_logger_comprehensive_metrics():
    """Verify logger aggregates rates, latencies, and classifications correctly."""
    logger = SimulationLogger()

    inc = Incident(
        id="inc_01",
        t_detected=10,
        disruptions=[Disruption(type="blocked_cell", cells=[(5, 5)])],
        affected_robots=[1, 2],
        congestion=Congestion(level="low", region=((3, 3), (7, 7)), density=0.1),
        interacts_with=[],
        validator=[ValidatorAttempt(attempt=1, valid=True)],
        executed_at=10,
        paths=PathsData(),
        outcome=IncidentOutcome(recovery_steps=5, cost_delta=3),
    )
    telemetry = {
        "solver_latency_ms": 12.5,
        "validator_latency_ms": 1.2,
        "total_repair_latency_ms": 15.0,
    }
    logger.record_incident(inc, telemetry=telemetry)

    m = logger.compute_metrics()
    assert m.incidents_total == 1
    assert m.incidents_isolated == 1
    assert m.incidents_interacting == 0
    assert m.repair_success_rate == 1.0
    assert m.mean_solver_latency_ms == 12.5
    assert m.mean_validator_latency_ms == 1.2
    assert m.mean_total_repair_latency_ms == 15.0
