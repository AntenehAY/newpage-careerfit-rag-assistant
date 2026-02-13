"""FastAPI application entry point for Career Intelligence Assistant."""

import shutil
import time
import uuid
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.api.dependencies import get_embedding_service, get_vector_store
from app.api.routes import router
from app.config import settings
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.vector_store import VectorStore
from app.utils.logging import clear_request_context, get_logger, set_request_context, setup_logging
from app.utils.metrics import get_metrics

logger = get_logger(__name__)


def _create_dirs() -> None:
    """Ensure data directories exist."""
    from pathlib import Path

    for d in ["./data", "./data/uploads", "./data/vectordb"]:
        Path(d).mkdir(parents=True, exist_ok=True)
    logger.info("Data directories initialized")


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Middleware to add request ID, set logging context, and record API metrics."""

    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        user_id = request.client.host if request.client else "unknown"
        set_request_context(
            request_id=request_id,
            user_id=user_id,
            operation=request.url.path,
        )
        start = time.perf_counter()
        try:
            response = await call_next(request)
            metrics = get_metrics()
            metrics.record_api_request(request.url.path, response.status_code, time.perf_counter() - start)
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            clear_request_context()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    setup_logging(log_level=settings.log_level)
    _create_dirs()
    logger.info("Application startup complete")
    yield
    # Shutdown: Cleanup resources
    logger.info("Application shutdown")


app = FastAPI(
    title="Career Intelligence Assistant",
    description="RAG-based resume and job description analyzer",
    version="0.1.0",
    lifespan=lifespan,
)

# Request ID and observability middleware (innermost = runs last before route)
app.add_middleware(RequestIdMiddleware)

# CORS middleware - allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(router)


# Exception handlers: 400 validation, 429 rate limit, 500 server errors
@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle validation errors (400)."""
    errors = exc.errors() if hasattr(exc, "errors") else []
    detail = errors[0].get("msg", "Validation error") if errors else "Validation error"
    return JSONResponse(status_code=400, content={"detail": detail, "errors": errors})


@app.exception_handler(Exception)
async def server_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle uncaught server errors (500). HTTPException passed through by FastAPI."""
    from fastapi import HTTPException

    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )
    logger.exception("Server error: {}", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred"},
    )


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    """Root page - visible status for browser visitors."""
    return """
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"><title>Career Intelligence Assistant</title></head>
    <body style="font-family: system-ui; max-width: 600px; margin: 3rem auto; padding: 2rem;">
        <h1>Career Intelligence Assistant</h1>
        <p style="font-size: 1.25rem; padding: 1rem; background: #e8f5e9; border-radius: 8px;">
            <strong>Status:</strong> <span style="color: #2e7d32;">ok</span>
        </p>
        <p>API is running. Links:</p>
        <ul>
            <li><a href="/health">/health</a> — JSON health check</li>
            <li><a href="/api/health">/api/health</a> — API health with components</li>
            <li><a href="/api/stats">/api/stats</a> — System statistics</li>
            <li><a href="/docs">/docs</a> — Interactive API documentation</li>
            <li><a href="/metrics">/metrics</a> — Observability metrics</li>
        </ul>
    </body>
    </html>
    """


@app.get("/health")
def health_check(
    vector_store: VectorStore = Depends(get_vector_store),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
) -> dict:
    """Health check with vector DB, embedding service, and disk space status.
    Used by Docker HEALTHCHECK and orchestration.
    Returns 200 with status 'ok' or 'degraded' and component details.
    """
    components: dict = {}
    status = "ok"

    # 1. Vector DB connection
    try:
        stats = vector_store.get_collection_stats()
        components["vector_db"] = {
            "status": "ok",
            "collection": stats.get("collection_name", "careerfit_chunks"),
            "chunk_count": stats.get("count", 0),
        }
    except Exception as e:
        logger.warning("Health check: vector_db error: {}", e)
        components["vector_db"] = {"status": "error", "error": str(e)}
        status = "degraded"

    # 2. Embedding service (lightweight: single cached embed)
    try:
        embedding_service.embed_text("health", use_cache=True)
        components["embedding_service"] = {"status": "ok"}
    except Exception as e:
        logger.warning("Health check: embedding_service error: {}", e)
        components["embedding_service"] = {"status": "error", "error": str(e)}
        status = "degraded"

    # 3. Disk space (data directory)
    try:
        data_path = Path(settings.vector_db_path).resolve().parent
        if not data_path.exists():
            data_path = Path("/app/data").resolve() if Path("/app/data").exists() else Path(".").resolve()
        usage = shutil.disk_usage(data_path)
        free_gb = usage.free / (1024**3)
        components["disk"] = {
            "status": "ok" if free_gb > 0.5 else "low",
            "free_gb": round(free_gb, 2),
            "total_gb": round(usage.total / (1024**3), 2),
        }
        if free_gb < 0.5:
            status = "degraded"
    except Exception as e:
        logger.warning("Health check: disk error: {}", e)
        components["disk"] = {"status": "error", "error": str(e)}
        status = "degraded"

    return {
        "status": status,
        "components": components,
    }


@app.get("/metrics")
def metrics_endpoint() -> dict:
    """Return aggregated observability metrics summary."""
    return get_metrics().get_metrics_summary()
