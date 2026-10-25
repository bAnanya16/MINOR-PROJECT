"""System prompts and prompt formatters for LLM Strategist."""

from typing import Optional
from amr_repair.sentinel.incident import IncidentRecord

SYSTEM_PROMPT = """You are an expert multi-robot path repair strategist for lifelong warehouse MAPF.
Given an IncidentRecord describing disruptions, congestion, and affected robots, you choose the scope, priority, horizon, and solver.

Solver choices:
- pibt: Priority Inheritance with Backtracking. Fastest execution, single-step greedy resolution, but weakest in dense congestion.
- lns: Large Neighborhood Search with Space-Time A*. Best path quality and cost delta, but takes more computation time.
- windowed_pbs: Windowed Priority-Based Search. Most reliable for interacting disruptions and tight bottlenecks, but slowest.

Rules:
1. robots: list of affected robot IDs that need path repair. Must be non-empty.
2. solver: exactly one of 'pibt', 'lns', or 'windowed_pbs'.
3. priority: a strict permutation of robots indicating planning precedence.
4. horizon: integer planning window between 10 and 40 steps.
5. rationale: brief explanation under 200 characters.
6. Return JSON only conforming to the schema.
"""


def format_user_prompt(incident: IncidentRecord, violation: Optional[str] = None) -> str:
    """Format user prompt with incident data and optional previous validation violation for retries."""
    incident_json = incident.model_dump_json(indent=2)
    prompt = f"Incident detected:\n{incident_json}\n"
    if violation:
        prompt += (
            f"\nCRITICAL: Previous directive failed validation: {violation}.\n"
            "Return a corrected directive that resolves this violation (e.g. adjust priority order, increase horizon, or use windowed_pbs).\n"
        )
    else:
        prompt += "\nSelect the optimal repair directive."
    return prompt
