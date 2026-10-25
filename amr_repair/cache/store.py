"""Strategy Cache for verified path repair directives."""

import os
import json
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from amr_repair.sentinel.incident import IncidentRecord
from amr_repair.llm.schema import Directive
from amr_repair.cache.negative import NegativeMemory, get_hard_filter_key


class CacheEntry:
    def __init__(
        self,
        incident_id: str,
        hard_filter_key: str,
        features: np.ndarray,
        incident_summary: dict,
        directive: Directive,
        outcome: dict,
    ):
        self.incident_id = incident_id
        self.hard_filter_key = hard_filter_key
        self.features = features
        self.incident_summary = incident_summary
        self.directive = directive
        self.outcome = outcome

    def to_dict(self) -> dict:
        return {
            "incident_id": self.incident_id,
            "hard_filter_key": self.hard_filter_key,
            "features": self.features.tolist(),
            "incident_summary": self.incident_summary,
            "directive": self.directive.model_dump(),
            "outcome": self.outcome,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CacheEntry":
        return cls(
            incident_id=data["incident_id"],
            hard_filter_key=data["hard_filter_key"],
            features=np.array(data["features"], dtype=np.float32),
            incident_summary=data.get("incident_summary", {}),
            directive=Directive.model_validate(data["directive"]),
            outcome=data.get("outcome", {}),
        )


class StrategyCache:
    """Stores verified repair strategies and retrieves them via hard filtering + cosine similarity."""

    def __init__(
        self,
        threshold: float = 0.85,
        negative_memory: Optional[NegativeMemory] = None,
        filepath: Optional[str] = None,
    ):
        self.threshold = threshold
        self.negative_memory = negative_memory or NegativeMemory()
        self.filepath = filepath
        self.entries: List[CacheEntry] = []

        if self.filepath and os.path.exists(self.filepath):
            self.load(self.filepath)

    def lookup(
        self, incident: IncidentRecord
    ) -> Optional[Tuple[Directive, str, float]]:
        """Find best matching directive above threshold that is not in negative memory.
        Returns: (Directive, matched_incident_id, similarity) or None.
        """
        if self.threshold >= 1.0 or not self.entries:
            return None

        target_key = get_hard_filter_key(incident)
        # 1. Hard filter on disruption types, size bucket, and congestion level
        candidates = [e for e in self.entries if e.hard_filter_key == target_key]
        if not candidates:
            return None

        # 2. Cosine similarity on features
        inc_feat = incident.features().reshape(1, -1)
        cand_feats = np.vstack([e.features for e in candidates])
        sims = cosine_similarity(inc_feat, cand_feats)[0]

        # 3. Sort by similarity descending
        sorted_indices = np.argsort(sims)[::-1]

        for idx in sorted_indices:
            sim = float(sims[idx])
            if sim < self.threshold:
                break
            entry = candidates[idx]

            # 4. Adapt directive robot IDs to current incident's affected robots if necessary
            curr_robots = list(incident.affected_robots)
            adapted_directive = Directive(
                robots=curr_robots,
                solver=entry.directive.solver,
                priority=curr_robots,  # preserves solver and horizon
                horizon=entry.directive.horizon,
                rationale=f"Cache hit from {entry.incident_id} (sim={sim:.2f})",
            )

            # 5. Check against negative memory
            if self.negative_memory.contains(incident, adapted_directive):
                continue

            return adapted_directive, entry.incident_id, sim

        return None

    def store(
        self,
        incident: IncidentRecord,
        directive: Directive,
        outcome: Optional[dict] = None,
    ) -> None:
        """Store a verified directive into the cache."""
        key = get_hard_filter_key(incident)
        entry = CacheEntry(
            incident_id=incident.id,
            hard_filter_key=key,
            features=incident.features(),
            incident_summary={
                "affected_count": len(incident.affected_robots),
                "congestion_level": incident.congestion.level,
            },
            directive=directive,
            outcome=outcome or {},
        )
        self.entries.append(entry)

        if self.filepath:
            self.save(self.filepath)

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = [e.to_dict() for e in self.entries]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str) -> None:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.entries = [CacheEntry.from_dict(d) for d in data]
