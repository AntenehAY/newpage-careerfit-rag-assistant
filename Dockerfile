# Career Intelligence Assistant - Production Multi-Stage Dockerfile
# Optimized for minimal image size with layer caching

# -----------------------------------------------------------------------------
# Stage 1: Builder
# -----------------------------------------------------------------------------
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy requirements first for optimal layer caching
COPY requirements.txt .

# Create virtual env and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: Runtime
# -----------------------------------------------------------------------------
FROM python:3.11-slim as runtime

# Install runtime-only deps (minimal for document parsing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd --gid 1000 appgroup \
    && useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy only necessary artifacts from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY app/ ./app/
COPY ui/ ./ui/

# Ensure data directories exist and are writable by appuser
RUN mkdir -p /app/data/uploads /app/data/vectordb /app/logs \
    && chown -R appuser:appgroup /app/data /app/logs

USER appuser

EXPOSE 8000

# Health check - uses enhanced /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Production: single worker, bind to 0.0.0.0 for container access
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
