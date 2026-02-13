#!/usr/bin/env bash
# Career Intelligence Assistant - Docker stop script
# Stops and removes containers

set -e
cd "$(dirname "$0")"

echo "Stopping containers..."
docker-compose down
echo "Containers stopped."
