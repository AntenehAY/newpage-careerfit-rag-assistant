# Career Intelligence Assistant - API Guide

Comprehensive API documentation for the Career Intelligence Assistant.

**Base URL (local):** `http://localhost:8000`  
**Interactive docs:** http://localhost:8000/docs  
**ReDoc:** http://localhost:8000/redoc  

---

## 1. Overview

The API provides REST endpoints for:

- **Document management**: Upload resumes and job descriptions, list, delete
- **Query**: Ask career-related questions; RAG returns cited answers
- **Health & observability**: Health checks, stats, metrics

All endpoints return JSON unless otherwise noted.

---

## 2. Endpoints

### 2.1 Upload Document

**`POST /api/upload`**

Upload a resume or job description. Accepts PDF and DOCX (max 10 MB).

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| file | File | Yes | PDF or DOCX file |
| doc_type | string | Yes | `"resume"` or `"job_description"` |

**Response:** `DocumentUpload`

```json
{
  "file_id": "550e8400-e29b-41d4-a716-446655440000",
  "file_name": "my_resume.pdf",
  "file_type": "resume",
  "file_size": 45678,
  "uploaded_at": "2025-02-13T14:30:00Z",
  "status": "completed"
}
```

**Status values:** `pending`, `processing`, `completed`, `failed`

**Example curl:**

```bash
curl -X POST "http://localhost:8000/api/upload" \
  -F "file=@resume.pdf" \
  -F "doc_type=resume"
```

**Error responses:**

| Code | Condition |
|------|-----------|
| 400 | Invalid doc_type (not resume/job_description) |
| 400 | Invalid file type (not .pdf or .docx) |
| 400 | File too large (>10 MB) |
| 400 | File is empty |
| 500 | Parse, chunk, or vector store error |

---

### 2.2 Query

**`POST /api/query`**

Submit a career-related question. Returns an answer with source citations.

**Request body:** `application/json`

```json
{
  "query": "What skills am I missing for this role?",
  "filter_doc_type": "resume",
  "filter_doc_id": null,
  "max_results": 5
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| query | string | Yes | - | User question (3–500 chars) |
| filter_doc_type | string | No | null | `"resume"` or `"job_description"` |
| filter_doc_id | string | No | null | Restrict to specific document |
| max_results | int | No | 5 | Chunks to retrieve (1–50) |

**Response:** `QueryResponse`

```json
{
  "answer": "Based on the provided documents [Source 1, Source 2]:\n\n**Skills the job requires:**\n- Python, SQL [Source 1]\n\n**Skills you demonstrate:**\n- Python [Source 2]\n\n**Gaps:**\n- SQL not mentioned in resume\n\n**Recommendations:** Add SQL projects or certifications.",
  "sources": [
    {
      "doc_id": "550e8400-e29b-41d4-a716-446655440000",
      "doc_name": "550e8400-e29b-41d4-a716-446655440000",
      "doc_type": "job_description",
      "chunk_text": "Requirements: Python, SQL, 3+ years...",
      "relevance_score": 0.92
    }
  ],
  "confidence": 0.85,
  "generated_at": "2025-02-13T14:35:00Z"
}
```

**Example curl:**

```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What skills am I missing for this role?"}'
```

**Error responses:**

| Code | Condition |
|------|-----------|
| 400 | Validation error (query too short/long, invalid JSON) |
| 429 | Rate limit exceeded (60 req/min per user) |
| 500 | LLM failure, retrieval error |

---

### 2.3 List Documents

**`GET /api/documents`**

List all uploaded documents.

**Response:** Array of `DocumentUpload`

```json
[
  {
    "file_id": "550e8400-e29b-41d4-a716-446655440000",
    "file_name": "resume.pdf",
    "file_type": "resume",
    "file_size": 45678,
    "uploaded_at": "2025-02-13T14:30:00Z",
    "status": "completed"
  }
]
```

**Example curl:**

```bash
curl "http://localhost:8000/api/documents"
```

---

### 2.4 Delete Document

**`DELETE /api/documents/{doc_id}`**

Delete a document from the vector store and remove the file.

**Path parameter:** `doc_id` — UUID returned from upload

**Response:**

```json
{
  "success": true,
  "message": "Document 550e8400-e29b-41d4-a716-446655440000 deleted"
}
```

**Error responses:**

| Code | Condition |
|------|-----------|
| 404 | Document not found |

**Example curl:**

```bash
curl -X DELETE "http://localhost:8000/api/documents/550e8400-e29b-41d4-a716-446655440000"
```

---

### 2.5 API Health Check

**`GET /api/health`**

Lightweight health check with vector DB status.

**Response:**

```json
{
  "status": "ok",
  "components": {
    "vector_db": "ok",
    "collection": "careerfit_chunks"
  }
}
```

---

### 2.6 System Stats

**`GET /api/stats`**

System statistics: document count, chunk count, vector DB info.

**Response:**

```json
{
  "total_documents": 2,
  "total_chunks": 15,
  "vector_db": {
    "collection_name": "careerfit_chunks",
    "persist_directory": "./data/vectordb"
  },
  "documents_by_type": {
    "resume": 1,
    "job_description": 1
  }
}
```

---

### 2.7 Root Health Check

**`GET /health`**

Full health check used by Docker HEALTHCHECK. Checks vector DB, embedding service, disk space.

**Response:**

```json
{
  "status": "ok",
  "components": {
    "vector_db": {
      "status": "ok",
      "collection": "careerfit_chunks",
      "chunk_count": 15
    },
    "embedding_service": { "status": "ok" },
    "disk": {
      "status": "ok",
      "free_gb": 45.2,
      "total_gb": 256.0
    }
  }
}
```

`status` may be `ok` or `degraded` if any component fails.

---

### 2.8 Metrics

**`GET /metrics`**

Observability metrics summary (latency, tokens, errors, cost estimates).

**Response:** (structure varies; example)

```json
{
  "api_requests": { "total": 42, "by_status": { "200": 38, "400": 2 } },
  "llm_calls": { "count": 20, "total_tokens": 15000 },
  "retrievals": { "count": 20, "avg_duration_ms": 120 },
  "errors": { "validation_failure": 2 }
}
```

---

### 2.9 Root (HTML)

**`GET /`**

Returns a simple HTML status page with links to health, docs, metrics.

---

## 3. Request/Response Schemas

### DocumentUpload

| Field | Type | Description |
|-------|------|-------------|
| file_id | string | UUID |
| file_name | string | Original filename |
| file_type | "resume" \| "job_description" | Document type |
| file_size | int | Bytes |
| uploaded_at | datetime (ISO 8601) | Upload timestamp |
| status | "pending" \| "processing" \| "completed" \| "failed" | Processing status |

### QueryRequest

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| query | string | Yes | 3–500 chars |
| filter_doc_type | string \| null | No | "resume" or "job_description" |
| filter_doc_id | string \| null | No | Document UUID |
| max_results | int | No | 1–50, default 5 |

### QueryResponse

| Field | Type | Description |
|-------|------|-------------|
| answer | string | Generated answer |
| sources | SourceReference[] | Cited chunks |
| confidence | float \| null | 0–1 |
| generated_at | datetime | Response timestamp |

### SourceReference

| Field | Type | Description |
|-------|------|-------------|
| doc_id | string | Source document ID |
| doc_name | string | Display name |
| doc_type | string | "resume" or "job_description" |
| chunk_text | string | Excerpt used |
| relevance_score | float | 0–1 |

---

## 4. Error Codes and Handling

| HTTP Code | Meaning | Typical Causes |
|-----------|---------|----------------|
| 400 | Bad Request | Invalid input, validation error |
| 404 | Not Found | Document ID not in registry |
| 429 | Too Many Requests | Rate limit (60/min per user) |
| 500 | Internal Server Error | LLM, embedding, or vector DB failure |

**Error response format:**

```json
{
  "detail": "Human-readable error message"
}
```

For validation errors (400), `errors` may contain detailed field-level messages.

---

## 5. Rate Limiting

- **Limit**: 60 requests per minute per user (by client IP)
- **Scope**: `/api/query` only
- **Response**: 429 with message "You've made too many requests. Please wait a moment and try again."
- **Header**: `X-Request-ID` on all responses for tracing

---

## 6. Authentication (Future)

Currently no authentication. For production:

- **Planned**: API key or OAuth2/JWT
- **Header**: `Authorization: Bearer <token>`
- **Rate limits**: Per authenticated user

---

## 7. Example Workflows

### Full workflow (curl)

```bash
# 1. Upload resume
RESP=$(curl -s -X POST "http://localhost:8000/api/upload" \
  -F "file=@resume.pdf" \
  -F "doc_type=resume")
RESUME_ID=$(echo $RESP | jq -r '.file_id')

# 2. Upload job description
RESP=$(curl -s -X POST "http://localhost:8000/api/upload" \
  -F "file=@jd.pdf" \
  -F "doc_type=job_description")
JD_ID=$(echo $RESP | jq -r '.file_id')

# 3. Query
curl -s -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What skills am I missing?"}' | jq .

# 4. List documents
curl -s "http://localhost:8000/api/documents" | jq .

# 5. Delete
curl -X DELETE "http://localhost:8000/api/documents/$RESUME_ID"
```

### Python (requests)

```python
import requests

BASE = "http://localhost:8000"

# Upload
with open("resume.pdf", "rb") as f:
    r = requests.post(f"{BASE}/api/upload", files={"file": f}, data={"doc_type": "resume"})
    file_id = r.json()["file_id"]

# Query
r = requests.post(f"{BASE}/api/query", json={"query": "What are my strengths?"})
print(r.json()["answer"])
```
