#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

require_cmd docker
require_cmd uv

if [ ! -f ".env" ] && [ -f ".env.example" ]; then
  cp -n ".env.example" ".env" || true
  echo "Created .env from .env.example (edit it if needed)." >&2
fi

echo "[1/4] Starting dependency services (Docker)..."
docker compose up -d db

echo "[2/4] Waiting for Postgres to be healthy..."
for _ in $(seq 1 60); do
  status="$(docker inspect --format '{{.State.Health.Status}}' nomemory-db 2>/dev/null || true)"
  if [ "$status" = "healthy" ]; then
    break
  fi
  sleep 1
done
status="$(docker inspect --format '{{.State.Health.Status}}' nomemory-db 2>/dev/null || true)"
if [ "$status" != "healthy" ]; then
  echo "Postgres is not healthy (status=$status). Check logs: docker logs nomemory-db" >&2
  exit 1
fi

echo "[3/4] Syncing Python dependencies (uv)..."
uv sync

echo "[4/4] Migrating DB + starting API server..."
uv run alembic upgrade head

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"
exec uv run uvicorn app.main:app --reload --host "$HOST" --port "$PORT"
