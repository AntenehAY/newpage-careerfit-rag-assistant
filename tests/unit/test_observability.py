"""Tests for observability: logging and metrics."""

import json
from pathlib import Path

import pytest

from app.utils.logging import (
    clear_request_context,
    get_logger,
    set_request_context,
    setup_logging,
)
from app.utils.metrics import (
    MetricsCollector,
    estimate_embedding_cost,
    estimate_llm_cost,
    get_metrics,
)


@pytest.mark.unit
def test_setup_logging_creates_log_dir(tmp_path):
    """setup_logging creates logs directory and log file."""
    log_dir = tmp_path / "test_logs"
    setup_logging(log_level="INFO", log_dir=str(log_dir))
    assert log_dir.exists()
    assert (log_dir / "app.log").exists()


@pytest.mark.unit
def test_get_logger_returns_logger_with_module():
    """get_logger returns a logger bound with module name."""
    logger = get_logger("test.module")
    assert logger is not None
    assert hasattr(logger, "info")
    assert hasattr(logger, "debug")
    assert hasattr(logger, "warning")
    assert hasattr(logger, "error")


@pytest.mark.unit
def test_request_context():
    """set_request_context and clear_request_context work."""
    set_request_context(
        request_id="req-123", user_id="user-1", operation="upload"
    )
    clear_request_context()
    set_request_context(request_id="req-456")


@pytest.mark.unit
def test_metrics_collector_singleton():
    """MetricsCollector uses singleton pattern."""
    m1 = MetricsCollector()
    m2 = MetricsCollector()
    assert m1 is m2
    assert get_metrics() is m1


@pytest.mark.unit
def test_record_ingestion():
    """record_ingestion updates metrics."""
    metrics = get_metrics()
    initial = metrics.get_metrics_summary()["total_documents_ingested"]
    metrics.record_ingestion(doc_id="doc1", duration=1.5, chunk_count=10)
    summary = metrics.get_metrics_summary()
    assert summary["total_documents_ingested"] == initial + 1


@pytest.mark.unit
def test_record_retrieval():
    """record_retrieval updates metrics."""
    metrics = get_metrics()
    initial = metrics.get_metrics_summary()["total_queries_processed"]
    metrics.record_retrieval(
        query="test query", duration=0.1, chunks_returned=5
    )
    summary = metrics.get_metrics_summary()
    assert summary["total_queries_processed"] == initial + 1


@pytest.mark.unit
def test_record_llm_call():
    """record_llm_call updates tokens and cost."""
    metrics = get_metrics()
    metrics.record_llm_call(duration=2.0, tokens_used=1000, cost=0.01)
    summary = metrics.get_metrics_summary()
    assert summary["total_llm_calls"] >= 1
    assert summary["total_cost_estimate_usd"] >= 0.0


@pytest.mark.unit
def test_record_llm_call_detailed():
    """record_llm_call_detailed calculates cost from input/output tokens."""
    metrics = get_metrics()
    metrics.record_llm_call_detailed(
        duration=1.5,
        input_tokens=500,
        output_tokens=200,
    )
    summary = metrics.get_metrics_summary()
    assert summary["total_input_tokens"] >= 500
    assert summary["total_output_tokens"] >= 200


@pytest.mark.unit
def test_record_api_request():
    """record_api_request updates counts and durations."""
    metrics = get_metrics()
    metrics.record_api_request("/api/query", 200, 0.15)
    summary = metrics.get_metrics_summary()
    assert summary["total_api_requests"] >= 1


@pytest.mark.unit
def test_record_error():
    """record_error increments error count."""
    metrics = get_metrics()
    metrics.record_error("test_error", {"details": "foo"})
    summary = metrics.get_metrics_summary()
    assert summary["total_errors"] >= 1
    assert "test_error" in summary["errors_by_type"]


@pytest.mark.unit
def test_get_metrics_summary():
    """get_metrics_summary returns expected structure."""
    metrics = get_metrics()
    summary = metrics.get_metrics_summary()
    assert "total_documents_ingested" in summary
    assert "total_queries_processed" in summary
    assert "average_retrieval_time_seconds" in summary
    assert "average_llm_time_seconds" in summary
    assert "total_tokens_used" in summary
    assert "total_cost_estimate_usd" in summary
    assert "error_rate" in summary
    assert "errors_by_type" in summary


@pytest.mark.unit
def test_export_to_file_json(tmp_path):
    """export_to_file saves metrics as JSON."""
    path = tmp_path / "metrics.json"
    metrics = get_metrics()
    metrics.export_to_file(str(path))
    assert path.exists()
    with open(path) as f:
        data = json.load(f)
    assert "total_documents_ingested" in data


@pytest.mark.unit
def test_export_to_file_csv(tmp_path):
    """export_to_file saves metrics as CSV."""
    path = tmp_path / "metrics.csv"
    metrics = get_metrics()
    metrics.export_to_file(str(path))
    assert path.exists()
    content = path.read_text()
    assert "total_documents_ingested" in content


@pytest.mark.unit
def test_estimate_llm_cost():
    """estimate_llm_cost calculates Claude pricing."""
    cost = estimate_llm_cost(input_tokens=1_000_000, output_tokens=0)
    assert abs(cost - 3.0) < 0.001
    cost = estimate_llm_cost(input_tokens=0, output_tokens=1_000_000)
    assert abs(cost - 15.0) < 0.001
    cost = estimate_llm_cost(input_tokens=1000, output_tokens=500)
    assert 0 < cost < 0.02


@pytest.mark.unit
def test_estimate_embedding_cost():
    """estimate_embedding_cost calculates OpenAI embedding pricing."""
    cost = estimate_embedding_cost(1_000_000)
    assert abs(cost - 0.02) < 0.001
    cost = estimate_embedding_cost(1000)
    assert 0 < cost < 0.001


@pytest.mark.unit
def test_metrics_endpoint_returns_summary():
    """GET /metrics returns metrics summary."""
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "total_documents_ingested" in data
        assert "total_cost_estimate_usd" in data
