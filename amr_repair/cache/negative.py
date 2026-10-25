"""Negative memory store tracking directives that failed validation."""

import os
import json
from typing import Dict, List, Optional
from amr_repair.sentinel.incident import IncidentRecord
from amr_repair.llm.schema import Directive


def get_hard_filter_key(incident: IncidentRecord) -> str:
    """Generate string key from disruption types, size bucket, and congestion level."""
    type_set = sorted(list({d.type for d in incident.disruptions}))
    return f"{','.join(type_set)}|{incident.size_bucket}|{incident.congestion.level}"


class NegativeMemory:
    """Stores rejected directives per hard-filter key to prevent re-trying known invalid configurations."""

    def __init__(self, filepath: Optional[str] = None):
        self.filepath = filepath
        # key -> list of directive dicts
        self.entries: Dict[str, List[dict]] = {}
        if self.filepath and os.path.exists(self.filepath):
            self.load(self.filepath)

    def add(self, incident: IncidentRecord, directive: Directive) -> None:
        """Add a rejected directive to negative memory."""
        key = get_hard_filter_key(incident)
        if key not in self.entries:
            self.entries[key] = []
        d_dict = directive.model_dump()
        if d_dict not in self.entries[key]:
            self.entries[key].append(d_dict)

        if self.filepath:
            self.save(self.filepath)

    def contains(self, incident: IncidentRecord, directive: Directive) -> bool:
        """Check if this directive has failed previously for this incident category."""
        key = get_hard_filter_key(incident)
        if key not in self.entries:
            return False
        d_dict = directive.model_dump()
        return any(
            e["solver"] == d_dict["solver"]
            and e["horizon"] == d_dict["horizon"]
            and e["priority"] == d_dict["priority"]
            for e in self.entries[key]
        )

    def save(self, filepath: str) -> None:
        """Persist negative memory to JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.entries, f, indent=2)

    def load(self, filepath: str) -> None:
        """Load negative memory from JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            self.entries = json.load(f)
