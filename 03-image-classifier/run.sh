#!/usr/bin/env bash
# Build the frontend, start the backend, open the browser. Ctrl+C stops both.
set -euo pipefail

cd "$(dirname "$0")"

PORT="${IMAGE_SERVER_PORT:-5041}"
URL="http://localhost:${PORT}"

echo "→ building frontend"
(cd frontend && npm run build)

echo "→ starting backend on ${URL}"
IMAGE_SERVER_PORT="$PORT" uv run python server.py &
SERVER_PID=$!

cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "→ waiting for server"
for _ in $(seq 1 60); do
  if curl -fs -o /dev/null "${URL}/api/device"; then
    break
  fi
  sleep 0.5
done

echo "→ opening ${URL}"
open "$URL"

wait "$SERVER_PID"
