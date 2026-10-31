"""Telemetry and SLA cost/latency tracking module for FinSight."""

import time
from contextlib import asynccontextmanager
from typing import Dict, Optional
from pydantic import BaseModel, Field

# Approximate OpenAI model pricing per 1M tokens (USD)
PRICING_TABLE = {
    "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
    "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
    "text-embedding-3-small": {"input": 0.02 / 1_000_000, "output": 0.0},
}


class ExecutionTelemetry(BaseModel):
    """Execution telemetry captured for each user query."""
    trace_id: str
    retrieval_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    model_used: str = "gpt-4o-mini"
    sla_latency_passed: bool = True
    sla_cost_passed: bool = True


def calculate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    embedding_tokens: int = 0
) -> float:
    """Calculates estimated API cost based on standard model pricing."""
    rates = PRICING_TABLE.get(model, PRICING_TABLE["gpt-4o-mini"])
    emb_rate = PRICING_TABLE["text-embedding-3-small"]["input"]

    llm_cost = (prompt_tokens * rates["input"]) + (completion_tokens * rates["output"])
    emb_cost = embedding_tokens * emb_rate
    return round(llm_cost + emb_cost, 6)


class LatencyTracker:
    """Helper to record sub-stage latencies."""

    def __init__(self):
        self.start_times: Dict[str, float] = {}
        self.durations_ms: Dict[str, float] = {}

    def start(self, stage: str):
        self.start_times[stage] = time.perf_counter()

    def stop(self, stage: str) -> float:
        if stage in self.start_times:
            elapsed = (time.perf_counter() - self.start_times[stage]) * 1000.0
            self.durations_ms[stage] = round(elapsed, 2)
            return self.durations_ms[stage]
        return 0.0

    def get(self, stage: str, default: float = 0.0) -> float:
        return self.durations_ms.get(stage, default)

