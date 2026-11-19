#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON="$PYTHON_BIN"
elif [[ -x "$PROJECT_DIR/.venv/bin/python" ]]; then
    PYTHON="$PROJECT_DIR/.venv/bin/python"
elif [[ -x "$PROJECT_DIR/venv/bin/python" ]]; then
    PYTHON="$PROJECT_DIR/venv/bin/python"
else
    PYTHON="$(command -v python3 || true)"
fi

if [[ -z "$PYTHON" || ! -x "$PYTHON" ]]; then
    echo "Python 3 was not found. Install Python or set PYTHON_BIN to its executable." >&2
    exit 1
fi

if ! "$PYTHON" -c 'import streamlit, uvicorn' >/dev/null 2>&1; then
    echo "This environment is missing Streamlit or Uvicorn." >&2
    echo "Install project dependencies with: $PYTHON -m pip install -r requirements.txt" >&2
    exit 1
fi

BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_HOST="${FRONTEND_HOST:-127.0.0.1}"
FRONTEND_PORT="${FRONTEND_PORT:-8501}"
backend_pid=""
frontend_pid=""

cleanup() {
    trap - EXIT INT TERM
    if [[ -n "$frontend_pid" ]]; then
        kill "$frontend_pid" 2>/dev/null || true
    fi
    if [[ -n "$backend_pid" ]]; then
        kill "$backend_pid" 2>/dev/null || true
    fi
    if [[ -n "$frontend_pid" ]]; then
        wait "$frontend_pid" 2>/dev/null || true
    fi
    if [[ -n "$backend_pid" ]]; then
        wait "$backend_pid" 2>/dev/null || true
    fi
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "Starting GreenToBuy services..."
echo "Backend:  http://${BACKEND_HOST}:${BACKEND_PORT}"
echo "Frontend: http://${FRONTEND_HOST}:${FRONTEND_PORT}"
echo "Press Ctrl+C to stop both services."

"$PYTHON" -m uvicorn server:app --reload --host "$BACKEND_HOST" --port "$BACKEND_PORT" &
backend_pid=$!

"$PYTHON" -m streamlit run app.py --server.address "$FRONTEND_HOST" --server.port "$FRONTEND_PORT" --server.headless true &
frontend_pid=$!

set +e
wait -n "$backend_pid" "$frontend_pid"
status=$?
set -e

echo "A service exited (status $status); stopping the other service."
exit "$status"
