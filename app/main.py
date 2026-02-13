"""FastAPI application entry point. Full implementation in Step 8."""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI(
    title="Career Intelligence Assistant",
    description="RAG-based resume and job description analyzer",
    version="0.1.0",
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
            <li><a href="/docs">/docs</a> — Interactive API documentation</li>
        </ul>
    </body>
    </html>
    """


@app.get("/health")
def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}
