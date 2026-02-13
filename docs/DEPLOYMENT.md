# Career Intelligence Assistant - Deployment Guide

Production deployment using Docker and Docker Compose.

## Prerequisites

- **Docker** 20.10+ ([Install Docker](https://docs.docker.com/get-docker/))
- **Docker Compose** v2+ (included with Docker Desktop)
- **API Keys**: Anthropic (Claude) and OpenAI (embeddings)

## Quick Start

```bash
# 1. Clone and enter project
git clone <repo-url>
cd newpage-careerfit-rag-assistant_ayimer

# 2. Configure environment
cp .env.docker.example .env
# Edit .env with your ANTHROPIC_API_KEY and OPENAI_API_KEY

# 3. Build and run
docker-compose up -d --build

# 4. Access
# API:  http://localhost:8000
# UI:   http://localhost:8501
```

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | Claude API key for LLM |
| `OPENAI_API_KEY` | Yes | OpenAI API key for embeddings |
| `VECTOR_DB_PATH` | No | Default: `/app/data/vectordb` |
| `ANTHROPIC_MODEL` | No | Override LLM model |
| `EMBEDDING_MODEL` | No | Override embedding model |
| `LOG_LEVEL` | No | INFO, DEBUG, WARNING |

### Using .env

Create `.env` from the template:

```bash
cp .env.docker.example .env
```

Never commit `.env`; it contains secrets.

## Persistence

### Volumes

Docker Compose uses named volumes:

- **careerfit_data**: Vector DB, uploaded documents, document registry
- **careerfit_logs**: Application logs

Data persists across container restarts. To backup:

```bash
# Backup data volume
docker run --rm -v careerfit_data:/data -v $(pwd):/backup alpine tar czf /backup/careerfit-data-backup.tar.gz -C /data .

# Restore
docker run --rm -v careerfit_data:/data -v $(pwd):/backup alpine tar xzf /backup/careerfit-data-backup.tar.gz -C /data
```

### Bind Mounts (Alternative)

For host-accessible data, replace volumes in `docker-compose.yml`:

```yaml
volumes:
  - ./data:/app/data
  - ./logs:/app/logs
```

## Monitoring

### Health Checks

- **Endpoint**: `GET /health`
- **Returns**: `{ "status": "ok"|"degraded", "components": {...} }`
- **Checks**: Vector DB, embedding service, disk space

```bash
curl http://localhost:8000/health
```

### Logs

```bash
# Follow API logs
docker-compose logs -f api

# Follow UI logs
docker-compose logs -f ui

# All services
docker-compose logs -f
```

### Container Status

```bash
docker-compose ps
```

## Troubleshooting

### Container fails to start

- **Check logs**: `docker-compose logs api`
- **Verify .env**: Ensure `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` are set
- **Health check timeout**: Increase `start_period` in healthcheck if startup is slow

### API returns 500

- **Embedding/LLM errors**: Verify API keys and quotas
- **Vector DB**: Ensure data volume has write permissions

### UI cannot reach API

- **Docker network**: UI uses `http://api:8000`; ensure both services are on `careerfit-network`
- **Wait for API**: UI has `depends_on: api` with `condition: service_healthy`

### Out of disk space

- Health endpoint reports `disk.status: "low"` when &lt; 0.5 GB free
- Clean old uploads or expand volume

## Helper Scripts

| Script | Description |
|--------|-------------|
| `build.ps1` / `build.sh` | Build images (use `--with-cache` for incremental) |
| `run.ps1` / `run.sh` | Start services in background |
| `stop.ps1` / `stop.sh` | Stop and remove containers |
| `logs.ps1` / `logs.sh` | Follow logs (`logs.ps1 ui` for UI) |

## Production Considerations

1. **Secrets**: Use Docker secrets or a secrets manager instead of `.env` in production
2. **Reverse proxy**: Put nginx/traefik in front; enable TLS
3. **Resource limits**: Add `deploy.resources.limits` in docker-compose
4. **Scaling**: Run multiple API replicas behind a load balancer
5. **Log aggregation**: Ship logs to centralized logging (ELK, Loki, etc.)
6. **Backups**: Schedule regular backups of the data volume

## Scaling

For high load, scale the API service:

```bash
docker-compose up -d --scale api=3
```

Use a load balancer to distribute traffic. The UI connects to a single API URL; for multiple API instances, point it at the load balancer.
