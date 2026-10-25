"""Tests for Stages 4 & 5: Strategy Cache and Negative Memory."""

import numpy as np
from amr_repair.cache.store import StrategyCache
from amr_repair.cache.negative import NegativeMemory
from amr_repair.llm.schema import Directive
from amr_repair.sentinel.incident import IncidentRecord, DisruptionData, CongestionData


def make_sample_incident(inc_id="inc_01", density=0.1, robots=[1, 2]) -> IncidentRecord:
    return IncidentRecord(
        id=inc_id,
        t_detected=100,
        disruptions=[DisruptionData(type="blocked_cell", cells=[(5, 5)])],
        affected_robots=robots,
        size_bucket="small",
        congestion=CongestionData(level="low", region=((3, 3), (7, 7)), density=density),
        interacts_with=[],
    )


def test_strategy_cache_hit_and_miss():
    cache = StrategyCache(threshold=0.85)
    inc1 = make_sample_incident("inc_01", density=0.1, robots=[1, 2])
    directive1 = Directive(robots=[1, 2], solver="pibt", priority=[1, 2], horizon=15)

    # Empty cache should return None
    assert cache.lookup(inc1) is None

    # Store entry
    cache.store(inc1, directive1)

    # Similar incident should HIT
    inc2 = make_sample_incident("inc_02", density=0.12, robots=[3, 4])
    result = cache.lookup(inc2)
    assert result is not None
    hit_directive, matched_id, sim = result
    assert matched_id == "inc_01"
    assert sim >= 0.85
    assert hit_directive.solver == "pibt"
    assert hit_directive.robots == [3, 4]


def test_negative_memory_filters_entry():
    neg = NegativeMemory()
    cache = StrategyCache(threshold=0.85, negative_memory=neg)

    inc1 = make_sample_incident("inc_01", robots=[1, 2])
    directive = Directive(robots=[1, 2], solver="pibt", priority=[1, 2], horizon=15)
    cache.store(inc1, directive)

    # Add to negative memory
    neg.add(inc1, directive)

    # Now lookup must skip this entry
    result = cache.lookup(inc1)
    assert result is None


def test_threshold_one_disables_cache():
    cache = StrategyCache(threshold=1.0)
    inc = make_sample_incident()
    directive = Directive(robots=[1, 2], solver="pibt", priority=[1, 2], horizon=15)
    cache.store(inc, directive)

    assert cache.lookup(inc) is None
