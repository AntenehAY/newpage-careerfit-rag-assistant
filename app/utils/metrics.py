"""Observability metrics for Career Intelligence Assistant."""

from __future__ import annotations

import csv
import json
import threading
from pathlib import Path
from typing import Any

# Cost per 1M tokens (USD)
# Claude Sonnet 4: https://www.anthropic.com/pricing
CLAUDE_INPUT_COST_PER_1M = 3.0
CLAUDE_OUTPUT_COST_PER_1M = 15.0
# OpenAI text-embedding-3-small
OPENAI_EMBEDDING_COST_PER_1M = 0.02


class MetricsCollector:
    """Collects and aggregates observability metrics. Singleton pattern."""

    _instance: MetricsCollector | None = None
    _lock = threading.Lock()

    def __new__(cls) -> MetricsCollector:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize metrics storage. Skip if already initialized (singleton)."""
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._lock = threading.Lock()
        # Counts
        self._documents_ingested: int = 0
        self._queries_processed: int = 0
        self._llm_calls: int = 0
        self._api_requests: int = 0
        self._errors: int = 0
        # Durations (cumulative for averaging)
        self._retrieval_durations: list[float] = []
        self._llm_durations: list[float] = []
        self._ingestion_durations: list[float] = []
        self._api_durations: list[float] = []
        # Tokens and cost
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._total_cost: float = 0.0
        # Errors by type
        self._errors_by_type: dict[str, int] = {}
        self._initialized = True

    def record_ingestion(
        self,
        doc_id: str,
        duration: float,
        chunk_count: int,
    ) -> None:
        """Record document ingestion metrics."""
        with self._lock:
            self._documents_ingested += 1
            self._ingestion_durations.append(duration)

    def record_retrieval(
        self,
        query: str,
        duration: float,
        chunks_returned: int,
    ) -> None:
        """Record retrieval metrics."""
        with self._lock:
            self._queries_processed += 1
            self._retrieval_durations.append(duration)

    def record_llm_call(
        self,
        duration: float,
        tokens_used: int,
        cost: float,
    ) -> None:
        """Record LLM call. tokens_used can be total or input+output separately."""
        with self._lock:
            self._llm_calls += 1
            self._llm_durations.append(duration)
            self._total_cost += cost
            # If tokens_used is combined, we don't split - track as input for simplicity
            self._total_input_tokens += tokens_used

    def record_llm_call_detailed(
        self,
        duration: float,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Record LLM call with separate input/output token counts for cost calculation."""
        cost = (
            (input_tokens / 1_000_000) * CLAUDE_INPUT_COST_PER_1M
            + (output_tokens / 1_000_000) * CLAUDE_OUTPUT_COST_PER_1M
        )
        with self._lock:
            self._llm_calls += 1
            self._llm_durations.append(duration)
            self._total_input_tokens += input_tokens
            self._total_output_tokens += output_tokens
            self._total_cost += cost

    def record_api_request(
        self,
        endpoint: str,
        status_code: int,
        duration: float,
    ) -> None:
        """Record API request metrics."""
        with self._lock:
            self._api_requests += 1
            self._api_durations.append(duration)
            if status_code >= 400:
                self._errors += 1

    def record_error(self, error_type: str, context: dict[str, Any] | None = None) -> None:
        """Record an error with optional context."""
        with self._lock:
            self._errors += 1
            self._errors_by_type[error_type] = self._errors_by_type.get(error_type, 0) + 1

    def get_metrics_summary(self) -> dict[str, Any]:
        """Return aggregated metrics summary."""
        with self._lock:
            total_requests = self._api_requests + self._queries_processed
            error_rate = self._errors / total_requests if total_requests > 0 else 0.0
            avg_retrieval = (
                sum(self._retrieval_durations) / len(self._retrieval_durations)
                if self._retrieval_durations
                else 0.0
            )
            avg_llm = (
                sum(self._llm_durations) / len(self._llm_durations)
                if self._llm_durations
                else 0.0
            )
            total_tokens = self._total_input_tokens + self._total_output_tokens

            return {
                "total_documents_ingested": self._documents_ingested,
                "total_queries_processed": self._queries_processed,
                "average_retrieval_time_seconds": round(avg_retrieval, 4),
                "average_llm_time_seconds": round(avg_llm, 4),
                "total_tokens_used": total_tokens,
                "total_input_tokens": self._total_input_tokens,
                "total_output_tokens": self._total_output_tokens,
                "total_cost_estimate_usd": round(self._total_cost, 6),
                "error_rate": round(error_rate, 4),
                "total_errors": self._errors,
                "total_llm_calls": self._llm_calls,
                "total_api_requests": self._api_requests,
                "errors_by_type": dict(self._errors_by_type),
            }

    def export_to_file(self, filepath: str) -> None:
        """Save metrics summary to JSON or CSV file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        summary = self.get_metrics_summary()

        if path.suffix.lower() == ".csv":
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                for k, v in summary.items():
                    if isinstance(v, dict):
                        writer.writerow([k, json.dumps(v)])
                    else:
                        writer.writerow([k, v])
        else:
            with open(path, "w") as f:
                json.dump(summary, f, indent=2)


def get_metrics() -> MetricsCollector:
    """Return the global singleton MetricsCollector."""
    return MetricsCollector()


def estimate_llm_cost(input_tokens: int, output_tokens: int) -> float:
    """Estimate cost for Claude Sonnet 4 based on token counts."""
    return (
        (input_tokens / 1_000_000) * CLAUDE_INPUT_COST_PER_1M
        + (output_tokens / 1_000_000) * CLAUDE_OUTPUT_COST_PER_1M
    )


def estimate_embedding_cost(token_count: int) -> float:
    """Estimate cost for OpenAI embeddings ($0.02/1M tokens)."""
    return (token_count / 1_000_000) * OPENAI_EMBEDDING_COST_PER_1M
