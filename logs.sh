#!/usr/bin/env bash
# Career Intelligence Assistant - Docker logs script
# Follow logs for API (or specify: ./logs.sh ui)

SERVICE="${1:-api}"
cd "$(dirname "$0")"

echo "Following logs for: $SERVICE"
docker-compose logs -f "$SERVICE"
