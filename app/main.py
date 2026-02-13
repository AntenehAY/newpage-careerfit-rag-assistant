"""FastAPI application entry point for Career Intelligence Assistant."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from app.api.routes import router

logger = logging.getLogger(__name__)


def _create_dirs() -> None:
    """Ensure data directories exist."""
    from pathlib import Path

    for d in ["./data", "./data/uploads", "./data/vectordb"]:
        Path(d).mkdir(parents=True, exist_ok=True)
    logger.info("Data directories initialized")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup: Initialize vector store, create data dirs
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
    logger.exception("Server error: %s", exc)
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
        </ul>
    </body>
    </html>
    """


@app.get("/health")
def health_check() -> dict[str, str]:
    """Simple health check endpoint (legacy)."""
    return {"status": "ok"}
