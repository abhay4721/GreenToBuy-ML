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
FRONTEND_HOST="${FRONTEND_HOST:-127.0.0.1}"

find_free_port() {
    local host="$1" start="$2" end="$3" port use_ss=false
    if command -v ss >/dev/null 2>&1 && ss -H -ltn >/dev/null 2>&1; then
        use_ss=true
    fi
    for ((port = start; port <= end; port++)); do
        if [[ "$use_ss" == true ]]; then
            if [[ -z "$(ss -H -ltn "sport = :$port" 2>/dev/null)" ]]; then
                printf '%s' "$port"
                return 0
            fi
        elif "$PYTHON" - "$host" "$port" <<'PY'
import socket
import sys

host, port = sys.argv[1], int(sys.argv[2])
probe_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(0.2)
    raise SystemExit(1 if sock.connect_ex((probe_host, port)) == 0 else 0)
PY
        then
            printf '%s' "$port"
            return 0
        fi
    done
    echo "No free port found on $host in range $start-$end." >&2
    return 1
}

choose_port() {
    local service="$1" host="$2" requested="$3" first="$4" last="$5" port
    if [[ -n "$requested" ]]; then
        port="$(find_free_port "$host" "$requested" "$requested")" || {
            echo "$service port $requested is already in use on $host." >&2
            echo "Choose another port with the ${service^^}_PORT environment variable." >&2
            return 1
        }
    else
        port="$(find_free_port "$host" "$first" "$last")" || return 1
        if [[ "$port" != "$first" ]]; then
            echo "$service port $first is busy; using $port instead." >&2
        fi
    fi
    printf '%s' "$port"
}

BACKEND_PORT="$(choose_port Backend "$BACKEND_HOST" "${BACKEND_PORT:-}" 8000 8010)"
FRONTEND_PORT="$(choose_port Frontend "$FRONTEND_HOST" "${FRONTEND_PORT:-}" 8501 8511)"
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
