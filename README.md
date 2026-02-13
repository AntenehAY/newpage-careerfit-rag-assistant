# Career Intelligence Assistant

**RAG-based conversational AI** that analyzes resumes against job descriptions to help candidates understand skill gaps, experience alignment, strengths, and interview preparation insights.

## Project Overview

The Career Intelligence Assistant is a production-ready Retrieval-Augmented Generation (RAG) system built for the Newpage Solutions assignment. It ingests resumes and job descriptions (PDF/DOCX), embeds them in a vector store, and answers career-related questions with grounded, cited responses.

### Key Features

- **Document Ingestion**: Upload PDF and DOCX resumes and job descriptions; automatic parsing, semantic chunking, and vector storage
- **Smart Retrieval**: Vector similarity search with MMR (Maximal Marginal Relevance) for diverse, relevant context
- **Query Type Detection**: Automatically routes skill-gap, experience-alignment, interview-prep, and general career questions to specialized prompts
- **Cited Answers**: Responses include explicit `[Source N]` citations tied to document chunks
- **Guardrails**: Input validation, injection detection, profanity checks, rate limiting (60 req/min per user)
- **Observability**: Structured logging, request tracing (X-Request-ID), metrics (latency, tokens, cost estimates)
- **Streamlit UI**: Simple interface for upload, query, and document management
- **Docker**: Multi-stage build, health checks, volume persistence

### Tech Stack Summary

| Layer | Technology | Rationale |
|-------|------------|-----------|
| API | FastAPI | Async, type-safe, auto OpenAPI docs |
| LLM | Claude Sonnet 4 | Strong reasoning, 200k context, cost-effective |
| Embeddings | OpenAI text-embedding-3-small | Quality/cost balance |
| Vector DB | ChromaDB | Local-first, zero-ops for MVP |
| Orchestration | LangChain | RAG patterns, prompt templates |
| Parsing | pypdf, python-docx | Mature, reliable extraction |

---

## Quick Start

### Prerequisites

- **Python 3.11+**
- **API Keys**: [Anthropic](https://console.anthropic.com/) (Claude) and [OpenAI](https://platform.openai.com/) (embeddings)

### Local Setup

```bash
# Clone repo
git clone <repo-url>
cd newpage-careerfit-rag-assistant_ayimer

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env: set ANTHROPIC_API_KEY and OPENAI_API_KEY
```

### Running the Application

**API:**
```bash
uvicorn app.main:app --reload
```

**UI (separate terminal):**
```bash
pip install -r requirements-ui.txt
streamlit run ui/streamlit_app.py
```

- **API:** http://localhost:8000  
- **Docs:** http://localhost:8000/docs  
- **UI:** http://localhost:8501  

### Docker Setup

```bash
cp .env.docker.example .env
# Edit .env with your API keys

docker-compose up -d --build
```

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for full deployment details.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Career Intelligence Assistant                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌─────────────────┐    ┌──────────────────────────┐  │
│  │   Streamlit  │    │  FastAPI (API)  │    │  Document Ingestion      │  │
│  │   UI         │───▶│  /api/upload    │───▶│  Parse → Chunk → Embed    │  │
│  │   :8501      │    │  /api/query     │    │  (pypdf, docx, LangChain) │  │
│  └──────────────┘    └────────┬────────┘    └─────────────┬──────────────┘  │
│                               │                           │                 │
│                               │            ┌──────────────▼──────────────┐    │
│                               │            │  ChromaDB Vector Store    │    │
│                               │            │  (careerfit_chunks)       │    │
│                               │            └──────────────┬────────────┘    │
│                               │                           │                 │
│                               │            ┌──────────────▼──────────────┐    │
│                               ▼            │  Retriever (MMR, filters) │    │
│  ┌──────────────────────────────────────┐  │  → Context for LLM       │    │
│  │  RAG Chain                           │◀─┘  (~4000 token limit)      │    │
│  │  Guardrails → Retrieve → Prompt →    │                              │    │
│  │  Claude Sonnet 4 → Parse → Response │                              │    │
│  └──────────────────────────────────────┘                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component Descriptions

| Component | Description |
|-----------|-------------|
| **Document Ingestion Pipeline** | Parses PDF/DOCX via pypdf and python-docx; chunks with RecursiveCharacterTextSplitter (512 chars, 50 overlap); embeds via OpenAI and stores in ChromaDB |
| **Vector Storage & Retrieval** | ChromaDB with cosine similarity; metadata filtering (doc_type, doc_id); MMR reranking for diversity |
| **RAG Chain with LLM** | Query validation → retrieval → prompt selection by type → Claude Sonnet 4 invocation → citation parsing → QueryResponse |
| **API Layer** | FastAPI routes: upload, query, list/delete documents, health, stats, metrics |
| **UI** | Streamlit: file upload, chat-style query, document list, delete |

### Data Flow

1. **Upload**: File → validate type/size → save to `data/uploads/` → parse → chunk → embed → add to ChromaDB → registry update  
2. **Query**: Request → validate input → rate limit check → embed query → vector search with filters → MMR rerank → format context → prompt with query type → LLM → validate output → parse citations → return QueryResponse  

---

## RAG/LLM Approach & Key Decisions

### LLM Selection

- **Model**: Claude Sonnet 4 (`claude-sonnet-4-20250514`)
- **Rationale**: Strong reasoning for career advice; 200k context window; cost-effective vs GPT-4; native support in LangChain
- **Alternatives considered**: GPT-4 (higher cost), Llama 3 (local but weaker for nuanced analysis)

### Embedding Model

- **Model**: OpenAI `text-embedding-3-small`
- **Why**: Good quality/cost balance; consistent with OpenAI ecosystem; low latency
- **Alternative**: `all-MiniLM-L6-v2` (local, no API cost; slightly lower quality)

### Vector Database

- **Choice**: ChromaDB
- **Rationale**: Local-first, no external service; minimal ops; easy migration path
- **Production path**: Pinecone, Weaviate, or pgvector for scalability

### Orchestration Framework

- **LangChain**: RAG abstractions, ChatPromptTemplate, ChatAnthropic integration
- **Benefits**: Prompt templates as code, observability hooks (LangSmith), consistent patterns

### Chunking Strategy

- **Approach**: Semantic chunking with `RecursiveCharacterTextSplitter`
- **Size**: 512 characters, overlap 50
- **Rationale**: Balance between context richness and retrieval precision
- **Special handling**: Section detection (Experience, Skills, Requirements, etc.) for resumes and JDs; chunks tagged with `section` metadata

### Retrieval Approach

- **Vector similarity**: Top-k (default 5), cosine distance
- **MMR**: λ=0.5 for diversity (relevance vs novelty)
- **Metadata filtering**: doc_type, doc_id
- **Future**: Hybrid retrieval (vector + BM25), cross-encoder reranking

### Prompt Engineering

- **System prompt**: Expert career advisor; factual-only; cite with `[Source N]`
- **Grounding**: "Use only provided context"
- **Output structure**: Bullet points, clear sections, actionable advice
- **Query-type prompts**: skill_gap, experience_alignment, interview_prep, general
- **Context limit**: ~4000 tokens; chunk prioritization by relevance

### Context Management

- Token budget enforcement in `get_context_for_llm`
- Chunk truncation when exceeding limit
- Source attribution in formatted context: `Document: {doc_id} (Type: {doc_type})\n{text}\n---`

### Guardrails

- **Input**: Length (3–500 chars), injection patterns (SQL, script, template), profanity
- **Output**: Citation validation, content safety
- **Rate limiting**: 60 requests/min per user (sliding window)
- **Fallback responses**: User-friendly messages for validation/rate-limit/empty-context

### Quality Controls

- **Testing**: 179+ tests (unit, integration, e2e), 80% coverage
- **Type safety**: Pydantic models throughout
- **Error handling**: Graceful degradation at all layers
- **Validation**: API request validation, pipeline error recovery

### Observability

- **Logging**: loguru, structured JSON, request_id in context
- **Metrics**: Latency, token counts, cost estimates, error rates
- **Tracing**: `X-Request-ID` header on responses
- **Endpoints**: `/health`, `/metrics`

---

## Key Technical Decisions

| Decision | Rationale |
|----------|------------|
| **FastAPI** | Async, type hints, auto OpenAPI docs, production-ready |
| **ChromaDB** | Local MVP; easy migration to Pinecone/pgvector for scale |
| **Multi-stage Docker** | Smaller image, security (non-root), layer caching |
| **Testing pyramid** | Unit → integration → e2e; mocks for external APIs |
| **Pydantic v2** | Validation, serialization, settings management |

---

## Engineering Standards Followed

- **Code structure**: Modular (`app/ingestion`, `app/rag`, `app/retrieval`, `app/api`); separation of concerns
- **Type hints**: All public functions and classes
- **Docstrings**: Every module, class, and function
- **Testing**: 80% coverage; unit, integration, e2e; no external API calls in CI
- **Error handling**: Try/except with logging; fallback responses for user-facing paths
- **Logging**: Structured (loguru); JSON for production
- **Containerization**: Multi-stage Dockerfile, health checks, non-root user

---

## Standards Skipped (with Rationale)

| Skipped | Rationale |
|---------|------------|
| Full CI/CD | Would add GitHub Actions → deploy; repo has tests workflow |
| Kubernetes manifests | Overkill for MVP; Docker Compose sufficient |
| Advanced reranking | Cross-encoder would improve precision; time constraint |
| Fine-tuned models | Out of scope; zero-shot sufficient for assignment |

---

## AI Tooling Usage

### Tools Used

- **Claude/Cursor**: Project scaffolding, boilerplate, documentation
- **GitHub Copilot**: Test cases, repetitive patterns

### Quality Assurance

- Manual code review for all AI-generated code
- Type checking (mypy optional)
- Linting (ruff optional)
- All tests written or reviewed manually

### Best Practices

- **DO**: Use AI for boilerplate, tests, docs
- **DON'T**: Blindly accept complex business logic
- **DO**: Refactor AI code for clarity
- **DON'T**: Skip manual review of critical paths

### Maintainability

- Clear naming conventions
- Comments on complex logic
- [DECISIONS.md](DECISIONS.md) for ADRs
- Prompts in `app/rag/prompts.py`

---

## Productionization for AWS/GCP/Azure

### Compute

- **AWS**: ECS Fargate, Lambda, EC2 for ChromaDB
- **GCP**: Cloud Run, GKE
- **Azure**: Container Instances, AKS

### Vector DB Migration

- **Pinecone**: Managed, scalable
- **Weaviate Cloud**: Self-hosted or managed
- **pgvector**: RDS, Cloud SQL, Azure Database

### Storage

- S3 / GCS / Blob Storage for documents
- CloudFront / CDN for static assets

### Secrets Management

- AWS Secrets Manager, GCP Secret Manager, Azure Key Vault

### Scaling Strategy

- Horizontal: Load balancer + auto-scaling
- Async: SQS/Pub-Sub + worker pools for ingestion
- Caching: Redis for embeddings, responses
- CDN: Static content

### Monitoring

- CloudWatch / Stackdriver / Azure Monitor
- Prometheus metrics export
- OpenTelemetry tracing
- ELK / CloudWatch Insights for logs

### CI/CD

- GitHub Actions → Build → Test → Deploy
- ECR / GCR / ACR for images
- Blue-green or canary deployments

### Cost Optimization

- Claude Haiku for simple queries
- Embedding cache (Redis)
- Batch ingestion
- Reserved instances for predictable load

### Security

- API auth (OAuth2/JWT)
- Rate limiting per user/API key
- Input sanitization
- Secrets rotation
- VPC/VNet isolation
- HTTPS/TLS

### Estimated Infrastructure Cost

- **MVP**: ~$100–200/month (small scale)
- **Production**: $500–2000/month (usage-dependent)

---

## Future Improvements

- Hybrid retrieval (vector + BM25)
- Cross-encoder reranking
- Multi-query retrieval for complex questions
- Fine-tune embedding on career corpus
- Conversation history for multi-turn dialogue
- Structured extraction (skills, requirements)
- A/B testing for prompts
- User feedback loop
- Advanced UI (highlighting, side-by-side comparison)
- LinkedIn API integration
- Resume scoring/ranking
- Job recommendation engine

---

## Project Structure

```
newpage-careerfit-rag-assistant_ayimer/
├── app/
│   ├── api/
│   │   ├── dependencies.py      # DI for embedding, vector store, RAG chain
│   │   ├── document_registry.py  # In-memory document metadata
│   │   └── routes.py             # FastAPI endpoints
│   ├── ingestion/
│   │   ├── chunker.py            # RecursiveCharacterTextSplitter, section detection
│   │   ├── parsers.py            # PDF (pypdf) and DOCX (python-docx)
│   │   └── pipeline.py          # End-to-end ingest_document
│   ├── rag/
│   │   ├── chain.py             # RAGChain: retrieve, prompt, LLM, parse
│   │   ├── guardrails.py        # Input/output validation, rate limiting
│   │   └── prompts.py           # Query-type prompts
│   ├── retrieval/
│   │   ├── embeddings.py        # OpenAI / sentence-transformers
│   │   ├── retriever.py         # MMR, context formatting
│   │   └── vector_store.py     # ChromaDB wrapper
│   ├── utils/
│   │   ├── logging.py           # loguru setup, request context
│   │   └── metrics.py          # Observability metrics
│   ├── config.py
│   ├── main.py
│   └── models.py
├── ui/
│   └── streamlit_app.py
├── tests/
│   ├── unit/                    # Fast, mocked
│   ├── integration/             # Real ChromaDB (ephemeral)
│   ├── e2e/                     # Full workflow
│   └── performance/             # Marked slow
├── docs/
│   ├── architecture.md
│   ├── API_GUIDE.md
│   ├── DEPLOYMENT.md
│   └── UI_TESTING.md
├── data/
│   ├── uploads/                 # Uploaded files
│   └── vectordb/                # ChromaDB persistence
├── .env.example
├── .env.docker.example
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── requirements-ui.txt
├── pytest.ini
├── DECISIONS.md
└── README.md
```

---

## API Documentation

**Interactive docs**: http://localhost:8000/docs  

**Key endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/upload` | Upload resume or job description |
| POST | `/api/query` | Query with RAG |
| GET | `/api/documents` | List uploaded documents |
| DELETE | `/api/documents/{doc_id}` | Delete document |
| GET | `/api/health` | Health check |
| GET | `/api/stats` | System statistics |
| GET | `/health` | Full health (vector DB, embedding, disk) |
| GET | `/metrics` | Observability metrics |

See [docs/API_GUIDE.md](docs/API_GUIDE.md) for full API documentation.  
See [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md) for final verification steps.

---

## Running Tests

```bash
# All tests (exclude slow)
pytest -m "not slow" -v

# With coverage
pytest -m "not slow" --cov=app --cov-report=html --cov-report=term

# Unit tests only (fast)
pytest tests/unit -v

# Coverage report
pytest --cov=app --cov-report=html
# Open htmlcov/index.html
```

---

## License

MIT (or as appropriate for your submission)

---

## Contact

- **Name**: [Your Name]
- **Email**: [Your Email]
- **GitHub**: [Your Profile]
