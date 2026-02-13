# Career Intelligence Assistant - Architecture

Detailed architecture documentation for the RAG-based career analysis system.

---

## 1. System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        Career Intelligence Assistant - High-Level View            │
└─────────────────────────────────────────────────────────────────────────────────┘

                                    ┌─────────────┐
                                    │   User      │
                                    └──────┬──────┘
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    │                      │                      │
                    ▼                      ▼                      ▼
            ┌───────────────┐      ┌───────────────┐      ┌───────────────┐
            │  Streamlit UI │      │  Browser /    │      │  curl / API   │
            │  (port 8501)  │      │  Postman      │      │  Client       │
            └───────┬───────┘      └───────┬───────┘      └───────┬───────┘
                    │                      │                      │
                    └──────────────────────┼──────────────────────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                            FastAPI Application (port 8000)                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ RequestId   │  │ CORS        │  │ Exception   │  │ Routes      │              │
│  │ Middleware │──│ Middleware  │──│ Handlers    │──│ /api/*      │              │
│  └─────────────┘  └─────────────┘  └─────────────┘  └──────┬──────┘              │
└─────────────────────────────────────────────────────────────┼─────────────────────┘
                                                              │
         ┌────────────────────────────────────────────────────┼─────────────────────┐
         │                    Dependency Injection (app/api/dependencies.py)          │
         │  EmbeddingService → VectorStore → Retriever → RAGChain                    │
         └────────────────────────────────────────────────────┼─────────────────────┘
                                                              │
    ┌─────────────────────────────────────────────────────────┼─────────────────────┐
    │                                                         │                     │
    ▼                                                         ▼                     ▼
┌───────────────┐                                    ┌─────────────────┐   ┌───────────────┐
│  Ingestion    │                                    │  RAG Chain      │   │  Document     │
│  Pipeline     │                                    │  (chain.py)     │   │  Registry     │
│  parsers.py   │                                    │  - Guardrails   │   │  (in-memory)  │
│  chunker.py   │                                    │  - Retriever    │   └───────────────┘
│  pipeline.py  │                                    │  - Prompts     │
└───────┬───────┘                                    │  - Claude LLM  │
        │                                             └────────┬────────┘
        │                                                      │
        │                                                      │
        ▼                                                      ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         ChromaDB Vector Store (careerfit_chunks)                   │
│                         Cosine similarity, metadata filtering                     │
└──────────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│  External Services: OpenAI (embeddings), Anthropic (Claude LLM)                      │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Interaction Diagrams

### 2.1 Document Upload Flow

```
Client                API Routes           DocumentRegistry    Ingestion Pipeline      VectorStore      ChromaDB
  |                       |                       |                    |                    |              |
  |--POST /api/upload---->|                       |                    |                    |              |
  |  (file, doc_type)     |                       |                    |                    |              |
  |                       |--validate file--------|                    |                    |              |
  |                       |--save to data/uploads--|                    |                    |              |
  |                       |--registry.add(processing)|                  |                    |              |
  |                       |--ingest_document()---->|                    |                    |              |
  |                       |                       |                    |--parse (pypdf/docx)|              |
  |                       |                       |                    |--chunk (512/50)----|              |
  |                       |                       |                    |--add_documents()-->|--store-------|
  |                       |                       |                    |<-------------------|<-------------|
  |                       |--registry.add(completed)|                  |                    |              |
  |<--DocumentUpload------|                       |                    |                    |              |
```

### 2.2 Query Flow (RAG Chain)

```
Client                API Routes           RAGChain            Guardrails        Retriever        VectorStore      Claude API
  |                       |                    |                    |                 |                 |              |
  |--POST /api/query----->|                    |                    |                 |                 |              |
  |  {query, filters}     |--get_rag_chain()-->|                    |                 |                 |              |
  |                       |                    |--validate_query()->|                 |                 |              |
  |                       |                    |<--valid------------|                 |                 |              |
  |                       |                    |--rate_limit.check()->                 |                 |              |
  |                       |                    |--retriever.get_context_for_llm()----->|                 |              |
  |                       |                    |                    |                 |--search()------->|              |
  |                       |                    |                    |                 |--embed query-----|--OpenAI------|
  |                       |                    |                    |                 |--MMR rerank------|              |
  |                       |                    |                    |                 |<--chunks---------|              |
  |                       |                    |<--{context,sources}--|                 |                 |              |
  |                       |                    |--prompt.format()---|                 |                 |              |
  |                       |                    |--llm.invoke()------------------------------|--------->|              |
  |                       |                    |<--response----------------------------------|---------|              |
  |                       |                    |--validate_response()|                 |                 |              |
  |                       |                    |--_parse_response()--|                 |                 |              |
  |<--QueryResponse-------|<-------------------|                    |                 |                 |              |
```

---

## 3. Data Flow Diagrams

### 3.1 Ingestion Data Flow

```
PDF/DOCX File
      │
      ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Parse     │────>│   Chunk     │────>│   Embed     │
│  (extract   │     │  (512 chars,│     │  (OpenAI    │
│   text)     │     │   50 overlap)│     │   batch)    │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
      Metadata: doc_id, doc_type, chunk_index, section
                                               │
                                               ▼
                                      ┌─────────────┐
                                      │  ChromaDB   │
                                      │  add(ids,   │
                                      │  embeddings,│
                                      │  metadatas) │
                                      └─────────────┘
```

### 3.2 Retrieval Data Flow

```
User Query
      │
      ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Embed      │────>│  Search     │────>│  MMR        │
│  (OpenAI)   │     │  (cosine,   │     │  Rerank     │
│             │     │  top_k*3)   │     │  (λ=0.5)    │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                                               ▼
                                      ┌─────────────┐
                                      │  Format     │
                                      │  Context    │
                                      │  (~4000 tok)│
                                      └──────┬──────┘
                                             │
                                             ▼
                                      ┌─────────────┐
                                      │  LLM        │
                                      │  (Claude)   │
                                      └─────────────┘
```

---

## 4. Sequence Diagrams

### 4.1 Document Upload and Ingestion

```
User          API           Registry      Pipeline      Parsers    Chunker    VectorStore   EmbeddingService
  |             |               |             |            |          |            |               |
  |--upload---->|               |             |            |          |            |               |
  |             |--validate---->|             |            |          |            |               |
  |             |--save file---->|             |            |          |            |               |
  |             |--add(processing)|            |            |          |            |               |
  |             |--ingest_document()---------->|            |          |            |               |
  |             |               |             |--parse_document()----->|            |               |
  |             |               |             |<--text------------------|            |               |
  |             |               |             |--chunk_document()------------------>|            |               |
  |             |               |             |<--chunks----------------------------|            |               |
  |             |               |             |--add_documents(chunks)------------->|            |               |
  |             |               |             |               |            |         |--embed_batch()->|
  |             |               |             |               |            |         |<--vectors------|
  |             |               |             |               |            |         |--collection.add()|
  |             |               |             |<--chunks-----|            |         |               |
  |             |--add(completed)|             |            |          |            |               |
  |<--DocumentUpload-----------|             |            |          |            |               |
```

### 4.2 Query Processing

```
User          API           RAGChain      Guardrails    Retriever    VectorStore   Claude
  |             |               |               |            |            |          |
  |--query----->|               |               |            |            |          |
  |             |--answer_query()-------------->|            |            |          |
  |             |               |--validate_query()--------->|            |          |
  |             |               |<--valid---------------------|            |          |
  |             |               |--check_rate_limit()------->|            |          |
  |             |               |<--allowed-------------------|            |          |
  |             |               |--get_context_for_llm()----------------->|          |
  |             |               |               |            |--search()->|          |
  |             |               |               |            |<--results--|          |
  |             |               |               |            |--_apply_mmr()         |
  |             |               |<--{context,sources}---------|            |          |
  |             |               |--format_messages()          |            |          |
  |             |               |--llm.invoke()--------------------------------->|
  |             |               |<--response-----------------------------------|
  |             |               |--validate_response()       |            |          |
  |             |               |--_parse_response()         |            |          |
  |<--QueryResponse------------|               |            |            |          |
```

### 4.3 RAG Chain Execution (Detail)

```
RAGChain                    Retriever                 VectorStore              LLM
   |                            |                         |                    |
   |--get_context_for_llm()---->|                         |                    |
   |                            |--retrieve_for_query()--->|                    |
   |                            |   (fetch_k=15)           |--query()           |
   |                            |<--raw results-----------|                    |
   |                            |--_apply_mmr()           |                    |
   |                            |   (embed query, chunks) |                    |
   |                            |   (rerank by MMR)       |                    |
   |<--{context, sources}-------|                         |                    |
   |--_detect_query_type()      |                         |                    |
   |--get_prompt_for_type()     |                         |                    |
   |--format_messages()         |                         |                    |
   |--llm.invoke()----------------------------------------------->|            |
   |<--response---------------------------------------------------|            |
   |--validate_response()       |                         |                    |
   |--_parse_response()         |                         |                    |
   |   (extract [Source N])     |                         |                    |
   |   (map to SourceReference) |                         |                    |
   |--return QueryResponse      |                         |                    |
```

---

## 5. Technology Stack Details

| Layer | Component | Version / Model | Purpose |
|-------|------------|------------------|---------|
| Runtime | Python | 3.11+ | Application runtime |
| API | FastAPI | 0.109+ | REST API, async |
| ASGI | Uvicorn | 0.27+ | WSGI server |
| LLM | Claude Sonnet 4 | claude-sonnet-4-20250514 | Generation |
| Embeddings | OpenAI | text-embedding-3-small | Vector embeddings |
| Vector DB | ChromaDB | 0.4.22+ | Vector storage |
| Orchestration | LangChain | 0.1+ | RAG patterns |
| Parsing | pypdf | 4.0+ | PDF extraction |
| Parsing | python-docx | 1.1+ | DOCX extraction |
| Validation | Pydantic | 2.5+ | Models, settings |
| Logging | loguru | 0.7+ | Structured logs |
| UI | Streamlit | 1.30+ | Web interface |
| Container | Docker | 20.10+ | Deployment |

---

## 6. Scalability Considerations

### Current Limitations

- **Single process**: Uvicorn single worker; no horizontal scaling of API without shared vector DB
- **In-memory registry**: DocumentRegistry is process-local; lost on restart (files and vector DB persist)
- **ChromaDB**: Local file storage; not designed for high concurrency or distributed deployment
- **Synchronous ingestion**: Upload blocks until parse/chunk/embed complete

### Scaling Paths

1. **Horizontal API scaling**: Run multiple API replicas; migrate ChromaDB to Pinecone/Weaviate/pgvector (shared)
2. **Async ingestion**: Queue uploads (SQS/RabbitMQ); worker pool ingests; notify on completion
3. **Embedding cache**: Redis for frequent query embeddings; reduce OpenAI calls
4. **Read replicas**: For vector DB; separate read/write paths
5. **CDN**: Static UI assets; reduce API load

### Performance Characteristics

- **Ingestion**: ~2–5 s per document (parse + chunk + embed + store)
- **Query**: ~1–3 s (embed + search + MMR + LLM)
- **Rate limit**: 60 req/min per user (configurable)
