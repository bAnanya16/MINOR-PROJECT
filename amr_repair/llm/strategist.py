"""LLM Strategist with multi-tier provider cascade: Groq primary, Ollama fallback, and deterministic heuristic."""

import os
import time
import json
import logging
from typing import Optional, Tuple
from pydantic import BaseModel

from amr_repair.sentinel.incident import IncidentRecord
from amr_repair.llm.schema import Directive
from amr_repair.llm.prompts import SYSTEM_PROMPT, format_user_prompt

logger = logging.getLogger(__name__)


def _load_env_file() -> None:
    """Auto-load variables from .env file if present."""
    env_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.env"))
    if os.path.exists(env_file):
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip().strip("'\"")
                        if k and k not in os.environ:
                            os.environ[k] = v
        except Exception as e:
            logger.debug(f"Could not load .env file: {e}")


class Strategist:
    """Strategist that queries an LLM to produce repair directives.

    Provider Cascade:
    1. Primary: Groq API (free-tier, fast inference with structured JSON output)
    2. Secondary Fallback: Local Ollama (when Groq is offline or unavailable)
    3. Tertiary Fallback: Deterministic Heuristic (guaranteed valid & predictable)
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.2,
        client = None,
        force_heuristic: bool = False,
    ):
        self.temperature = temperature
        self.force_heuristic = force_heuristic
        self.client = client
        self.ollama_client = None

        self.last_provider: str = "none"
        self.last_latency_ms: float = 0.0
        self.last_tokens: int = 0
        self.total_calls: int = 0
        self.total_tokens: int = 0

        # Auto-load .env
        _load_env_file()

        # Primary configuration (Groq by default, or OpenAI fallback)
        groq_key = os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")
        groq_base_url = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
        self.model = model or os.environ.get("GROQ_MODEL") or os.environ.get("OPENAI_MODEL") or "qwen/qwen3.8-27b"

        # Secondary configuration (Local Ollama)
        self.ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        self.ollama_model = os.environ.get("OLLAMA_MODEL", "llama3.2")

        if not self.force_heuristic:
            # Initialize primary client if not explicitly passed
            if self.client is None and groq_key:
                try:
                    from openai import OpenAI
                    self.client = OpenAI(
                        api_key=groq_key,
                        base_url=groq_base_url,
                        timeout=12.0,
                    )
                except Exception as e:
                    logger.warning(f"Could not initialize primary Groq client: {e}")
                    self.client = None

            # Initialize secondary Ollama client
            try:
                from openai import OpenAI
                self.ollama_client = OpenAI(
                    api_key="ollama",
                    base_url=self.ollama_base_url,
                    timeout=3.0,
                )
            except Exception as e:
                logger.debug(f"Could not initialize Ollama client: {e}")
                self.ollama_client = None

    def _call_llm_json(self, client, model_name: str, user_msg: str) -> Tuple[Directive, int]:
        """Queries an OpenAI-compatible endpoint with JSON format enforcement and Pydantic validation."""
        schema_json = json.dumps(Directive.model_json_schema())
        system_content = (
            f"{SYSTEM_PROMPT}\n"
            f"You MUST output valid JSON strictly adhering to this JSON schema:\n{schema_json}"
        )

        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
            temperature=self.temperature,
            max_tokens=500,
        )

        content = completion.choices[0].message.content
        directive = Directive.model_validate_json(content)

        tokens = 0
        if hasattr(completion, "usage") and completion.usage and hasattr(completion.usage, "total_tokens"):
            tokens = int(completion.usage.total_tokens)
        else:
            tokens = 300 + len(directive.robots) * 30

        return directive, tokens

    def choose(self, incident: IncidentRecord, violation: Optional[str] = None) -> Directive:
        """Choose a path repair directive using Tier 1 (Groq) -> Tier 2 (Ollama) -> Tier 3 (Heuristic)."""
        start_time = time.perf_counter()
        self.total_calls += 1

        if not self.force_heuristic:
            user_msg = format_user_prompt(incident, violation)

            # --- Tier 1: Groq API ---
            if self.client is not None:
                try:
                    directive, tokens = self._call_llm_json(self.client, self.model, user_msg)
                    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                    self.last_latency_ms = round(elapsed_ms, 2)
                    self.last_tokens = tokens
                    self.total_tokens += tokens
                    self.last_provider = "groq"
                    return directive
                except Exception as e:
                    logger.warning(f"Groq LLM call failed: {e}. Attempting Ollama fallback.")

            # --- Tier 2: Ollama Local Fallback ---
            if self.ollama_client is not None:
                try:
                    directive, tokens = self._call_llm_json(self.ollama_client, self.ollama_model, user_msg)
                    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                    self.last_latency_ms = round(elapsed_ms, 2)
                    self.last_tokens = tokens
                    self.total_tokens += tokens
                    self.last_provider = "ollama"
                    return directive
                except Exception as e:
                    logger.info(f"Ollama fallback unavailable or failed: {e}. Falling back to deterministic heuristic.")

        # --- Tier 3: Deterministic Heuristic Fallback ---
        directive = self._heuristic_choice(incident, violation)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        self.last_latency_ms = round(elapsed_ms, 2)
        self.last_tokens = 0
        self.last_provider = "heuristic"
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
