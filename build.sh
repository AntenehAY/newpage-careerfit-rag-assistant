#!/usr/bin/env bash
# Career Intelligence Assistant - Docker build script
# Builds images with no cache for clean production deployment

set -e
cd "$(dirname "$0")"

if [[ "${1:-}" == "--with-cache" ]]; then
    echo "Building Docker images (with cache)..."
    docker-compose build
else
    echo "Building Docker images (no cache)..."
    docker-compose build --no-cache
fi

echo "Build complete."
