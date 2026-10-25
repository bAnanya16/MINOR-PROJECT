"""Tests for Stage 6: LLM Strategist, Directive schema, and prompt formatting."""

import pytest
from pydantic import ValidationError
from amr_repair.llm.schema import Directive
from amr_repair.llm.prompts import format_user_prompt
from amr_repair.llm.strategist import Strategist
from amr_repair.sentinel.incident import IncidentRecord, DisruptionData, CongestionData


def make_test_incident(interacts=False) -> IncidentRecord:
    return IncidentRecord(
        id="inc_test_01",
        t_detected=50,
        disruptions=[DisruptionData(type="blocked_cell", cells=[(10, 10)])],
        affected_robots=[1, 2, 3],
        size_bucket="small",
        congestion=CongestionData(level="low", region=((8, 8), (12, 12)), density=0.1),
        interacts_with=["inc_test_00"] if interacts else [],
    )


def test_directive_valid():
    d = Directive(
        robots=[1, 2, 3],
        solver="pibt",
        priority=[3, 1, 2],
        horizon=20,
        rationale="Valid test directive",
    )
    assert d.robots == [1, 2, 3]
    assert d.solver == "pibt"
    assert d.priority == [3, 1, 2]
    assert d.horizon == 20


def test_directive_invalid_priority_not_permutation():
    with pytest.raises(ValidationError):
        Directive(
            robots=[1, 2, 3],
            solver="pibt",
            priority=[1, 2, 4],  # 4 is not in robots!
            horizon=20,
        )


def test_directive_invalid_horizon_bounds():
    with pytest.raises(ValidationError):
        Directive(
            robots=[1, 2],
            solver="lns",
            priority=[1, 2],
            horizon=5,  # Below 10
        )
    with pytest.raises(ValidationError):
        Directive(
            robots=[1, 2],
            solver="lns",
            priority=[1, 2],
            horizon=50,  # Above 40
        )


def test_prompt_formatter_includes_violation():
    inc = make_test_incident()
    prompt_normal = format_user_prompt(inc)
    assert "Previous directive failed validation" not in prompt_normal

    prompt_retry = format_user_prompt(inc, violation="vertex collision robots 1,2 cell [10,10] t=52")
    assert "Previous directive failed validation: vertex collision" in prompt_retry


def test_strategist_heuristic_choice():
    strat = Strategist(force_heuristic=True)
    inc = make_test_incident()
    d1 = strat.choose(inc)
    assert d1.solver in ["pibt", "lns", "windowed_pbs"]
    assert set(d1.priority) == set(inc.affected_robots)

    # Retry with violation should upgrade to windowed_pbs
    d_retry = strat.choose(inc, violation="vertex collision")
    assert d_retry.solver == "windowed_pbs"
    assert d_retry.horizon == 30
