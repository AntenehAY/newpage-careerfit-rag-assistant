# Architecture Decision Records

Design decisions log for the Career Intelligence Assistant project.

---

## Step 11 - Testing Approach (2025-02-13)

### Context
Step 11 introduced a comprehensive testing suite to ensure quality and reliability of the Career Intelligence Assistant.

### Decision
- **Structure**: Tests are organized into `tests/unit/`, `tests/integration/`, `tests/e2e/`, `tests/performance/`, and `tests/fixtures/`.
- **Unit tests**: Fast, isolated tests for config, models, parsers, chunker, pipeline, embeddings, vector store, retriever, RAG chain, guardrails, observability, and API routes. Mock external APIs (Anthropic, OpenAI).
- **Integration tests**: Multi-component tests using real ChromaDB (EphemeralClient) with temp directories. Mock only LLM and embedding API calls. Cover: ingestion→vector store, RAG pipeline, API with real services.
- **E2E tests**: Full workflow from empty state through upload, multiple query types, delete, cleanup. Error scenarios: invalid upload, query without docs, rate limit, 404.
- **Performance tests**: Optional benchmarks for ingestion time, retrieval latency. Marked `slow` to exclude from default runs.
- **External APIs**: Anthropic and OpenAI are mocked in all tests to avoid API costs and ensure CI reproducibility.
- **ChromaDB**: Use real ChromaDB with EphemeralClient for integration/e2e; no persistence across runs. Clean up test data after each run.
- **Coverage**: Target >70% (configurable via `--cov-fail-under`). Use `pytest-cov` for HTML and term reports.
- **CI**: GitHub Actions workflow runs on push/PR, matrix Python 3.11 and 3.12, installs deps, runs unit/integration/e2e, uploads coverage artifact.

### Commands
```bash
# Unit tests only (fast)
pytest tests/unit -v

# All tests (exclude slow)
pytest -m "not slow" -v

# With coverage
pytest tests/unit tests/integration tests/e2e --cov=app --cov-report=html
```
