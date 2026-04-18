#!/bin/sh
set -e

MODEL="${OLLAMA_MODEL:-qwen2.5-coder:7b}"
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
export PYTHONUNBUFFERED=1
OLLAMA_VERBOSE_LOGS="${OLLAMA_VERBOSE_LOGS:-0}"
OLLAMA_LOG_FILE="${OLLAMA_LOG_FILE:-/tmp/ollama.log}"

if [ "${OLLAMA_VERBOSE_LOGS}" = "1" ]; then
  echo "Starting Ollama with verbose logs in terminal..."
  ollama serve &
else
  echo "Starting Ollama (logs -> ${OLLAMA_LOG_FILE})..."
  : > "${OLLAMA_LOG_FILE}"
  ollama serve >>"${OLLAMA_LOG_FILE}" 2>&1 &
fi
OLLAMA_PID=$!

cleanup() {
  kill "${OLLAMA_PID}" >/dev/null 2>&1 || true
  wait "${OLLAMA_PID}" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

echo "Waiting for Ollama..."
i=0
while [ "$i" -lt 120 ]; do
  if ollama list >/dev/null 2>&1; then
    break
  fi
  i=$((i + 1))
  sleep 1
done

echo "Pulling model ${MODEL} (first run may take several minutes)..."
ollama pull "${MODEL}"

if [ "$#" -eq 0 ]; then
  set -- --pretty
fi

python3 -u -m app.cli "$@"
