# Career Intelligence Assistant - Submission Checklist

**Final verification before submitting to Newpage Solutions.**

---

## Pre-Submission Checklist

Complete each item before submission. Mark with `[x]` when done.

### Code & Runtime

- [ ] All code runs without errors (API + UI locally)
- [ ] `uvicorn app.main:app` starts successfully
- [ ] `streamlit run ui/streamlit_app.py` starts successfully
- [ ] Docker builds successfully: `docker-compose build`
- [ ] Docker Compose runs: `docker-compose up -d`
- [ ] Health check passes: `curl http://localhost:8000/health`

### Testing

- [ ] All tests pass: `pytest -m "not slow" -v`
- [ ] Coverage ≥ 80%: `pytest -m "not slow" --cov=app --cov-fail-under=80`
- [ ] No critical lint/type errors (optional: `ruff check app`, `mypy app`)

### Documentation

- [ ] README.md is complete and reflects your implementation
- [ ] DECISIONS.md has all Architecture Decision Records
- [ ] docs/architecture.md exists and is detailed
- [ ] docs/API_GUIDE.md exists and documents all endpoints
- [ ] docs/DEPLOYMENT.md exists (Docker, env vars)
- [ ] All documentation reflects YOUR thinking (not generic boilerplate)

### Configuration

- [ ] .env.example provided (no secrets)
- [ ] .env.docker.example provided for Docker
- [ ] Required env vars documented: ANTHROPIC_API_KEY, OPENAI_API_KEY

### End-to-End Verification

- [ ] Sample documents work E2E: upload resume + JD, query, get cited answer
- [ ] Delete document works and removes from vector store
- [ ] Rate limit returns 429 when exceeded
- [ ] Invalid upload (wrong type, empty file) returns 400

### Repository

- [ ] GitHub repo is clean and organized
- [ ] .gitignore excludes .env, __pycache__, .venv, htmlcov
- [ ] No committed secrets (API keys, passwords)
- [ ] Clear commit history (or squashed for submission)

---

## Quick Verification Commands

```bash
# Run all tests (exclude slow)
pytest -m "not slow" -v

# Coverage
pytest -m "not slow" --cov=app --cov-report=term

# Docker build
docker-compose build

# Start services
docker-compose up -d

# Health check
curl http://localhost:8000/health

# Sample query (after uploading docs)
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What skills does this resume highlight?"}'
```

---

## Document Inventory

| Document | Purpose |
|----------|---------|
| README.md | Project overview, quick start, architecture, decisions |
| DECISIONS.md | Architecture Decision Records |
| docs/architecture.md | Detailed architecture, sequence diagrams |
| docs/API_GUIDE.md | Full API documentation |
| docs/DEPLOYMENT.md | Docker, env vars, production notes |
| SUBMISSION_CHECKLIST.md | This file |

---

## Notes for Evaluator

- **API Keys**: Create `.env` from `.env.example` with valid Anthropic and OpenAI keys.
- **Tests**: Some integration/e2e tests may skip if ChromaDB has compatibility issues (e.g., Python 3.14); unit tests should all pass.
- **Sample docs**: Use any PDF or DOCX resume and job description for E2E testing.
