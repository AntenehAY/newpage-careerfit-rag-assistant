#!/usr/bin/env bash
# Career Intelligence Assistant - Docker run script
# Starts all services in detached mode

set -e
cd "$(dirname "$0")"

if [[ ! -f .env ]]; then
    echo "WARNING: .env not found. Copy .env.docker.example to .env and add your API keys."
fi

echo "Starting containers..."
docker-compose up -d

echo ""
echo "Services started:"
echo "  API:  http://localhost:8000"
echo "  UI:   http://localhost:8501"
echo ""
echo "Check status: docker-compose ps"
