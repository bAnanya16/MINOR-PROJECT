"""Replay file exporter conforming to the benchmark schema."""

import os
import gzip
import json
from amr_repair.replay.models import Replay


def export_replay(replay: Replay, filepath: str) -> None:
    """Exports a Replay object to JSON or gzip-compressed JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    json_str = replay.model_dump_json(indent=2)

    if filepath.endswith(".gz"):
        with gzip.open(filepath, "wt", encoding="utf-8") as f:
            f.write(json_str)
    else:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(json_str)
