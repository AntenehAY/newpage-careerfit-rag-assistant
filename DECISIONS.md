# Architecture Decision Records

Design decisions log for the Career Intelligence Assistant project.

---

## Step 12 - Docker Multi-Stage Build (2025-02-13)

### Context
Production deployment required a secure, minimal Docker image for the API and UI services.

### Decision
- **Multi-stage build**: Stage 1 (builder) installs Python deps in a venv; Stage 2 (runtime) copies only the venv and app code.
- **Rationale**: Smaller final image (~400MB vs 1.5GB+), faster layer caching, reduced attack surface.
- **Non-root user**: `appuser` (uid 1000) runs the application; avoids running as root.
- **Runtime deps**: Only `libmagic1` for document type detection; no build tools in final image.

### Consequences
- Smaller images for faster pulls and deploys.
- Security best practice for containers.

---

## Step 12 - Health Check Implementation (2025-02-13)

### Context
Docker Compose and orchestration need to know when the API is ready and healthy.

### Decision
- **Endpoint**: `GET /health` (root-level) performs component checks: vector DB connection, embedding service (lightweight cached embed), disk space.
- **Response**: `{ "status": "ok" | "degraded", "components": { ... } }`.
- **Docker HEALTHCHECK**: Uses Python `urllib.request.urlopen` to call `/health`; interval 30s, timeout 10s, start_period 40s, retries 3.
- **UI depends_on**: `condition: service_healthy` so UI only starts after API is ready.

### Consequences
- Orchestration can detect unhealthy containers.
- UI avoids connecting to a not-yet-ready API.

---

## Step 12 - Volume Mount Strategy (2025-02-13)

### Context
Vector DB, uploaded files, document registry, and logs must persist across container restarts.

### Decision
- **Named volumes**: `careerfit_data` (vectordb, uploads, registry) and `careerfit_logs` (app logs).
- **Paths**: `/app/data` and `/app/logs` inside container.
- **Alternative**: Bind mounts (`./data:/app/data`) for host-accessible data during development.
- **Backup**: Documented `tar` commands for volume backup/restore in DEPLOYMENT.md.

### Consequences
- Data survives `docker-compose down`; only `docker-compose down -v` removes volumes.
- Backup/restore procedures available for production.

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
