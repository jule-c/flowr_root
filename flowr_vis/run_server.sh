#!/usr/bin/env bash
# ==============================================
#  FLOWR Visualization – Frontend Server (CPU-only)
# ==============================================
# Usage:
#   ./run_server.sh                               # defaults
#   ./run_server.sh --port 8787                    # custom port
#   ./run_server.sh --worker-url http://gpu:8788   # custom worker URL
#
# This starts the lightweight CPU-only frontend server. The server code
# itself needs no PyTorch or FLOWR — just RDKit + FastAPI from the `vis`
# extra — but the root project depends on torch, so pick a wheel flavour:
#
#   uv sync --extra cpu --extra vis     # macOS / CPU
#   uv sync --extra gpu --extra vis     # Linux / CUDA
# ==============================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Defaults ──
PORT=8787
WORKER_URL="http://localhost:8788"
SERVER_URL=""  # auto-detected if not specified

# ── Parse arguments ──
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --port)        PORT="$2"; shift ;;
        --worker-url)  WORKER_URL="$2"; shift ;;
        --server-url)  SERVER_URL="$2"; shift ;;
        -h|--help)
            echo "Usage: ./run_server.sh [--port PORT] [--worker-url URL] [--server-url URL]"
            exit 0 ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$SERVER_URL" ]; then
    SERVER_URL="http://localhost:${PORT}"
fi

# ── uv must be on PATH (login shells put it in ~/.local/bin) ──
export PATH="$HOME/.local/bin:$PATH"

if [ "$(uname -s)" = "Darwin" ]; then
    SYNC_HINT="uv sync --extra cpu --extra vis"
else
    SYNC_HINT="uv sync --extra gpu --extra vis"
fi

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  FLOWR Visualization – Frontend Server  ║"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "  Project root:  ${PROJECT_ROOT}"
echo "  Port:          ${PORT}"
echo "  Worker URL:    ${WORKER_URL}"
echo "  Server URL:    ${SERVER_URL}"
echo ""

# ── Check the uv environment ──
if ! command -v uv &>/dev/null; then
    echo "ERROR: 'uv' was not found on PATH."
    echo "       Install it with:  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

if [ ! -d "${PROJECT_ROOT}/.venv" ]; then
    echo "ERROR: No virtual environment at ${PROJECT_ROOT}/.venv"
    echo "       Create it once with:"
    echo "         cd ${PROJECT_ROOT} && ${SYNC_HINT}"
    exit 1
fi

# ── Set environment ──
export FLOWR_PORT="${PORT}"

# ── OpenEye license (optional external toolkit) ──
if [ -f "${SCRIPT_DIR}/tools/oe_license.txt" ]; then
    export OE_LICENSE="${SCRIPT_DIR}/tools/oe_license.txt"
elif [ -f "${PROJECT_ROOT}/oe_license.txt" ]; then
    export OE_LICENSE="${PROJECT_ROOT}/oe_license.txt"
fi
export FLOWR_WORKER_URL="${WORKER_URL}"
export FLOWR_SERVER_URL="${SERVER_URL}"

cd "$PROJECT_ROOT"

# ── Verify critical imports ──
echo "Checking Python environment…"
echo "uv: $(uv --version)"
uv run --no-sync python -c "
import sys
print(f'Python: {sys.executable}')
print(f'Version: {sys.version}')
try:
    from rdkit import Chem; print(f'RDKit: OK')
except ImportError: print('WARNING: RDKit not installed')
try:
    import fastapi; print(f'FastAPI: {fastapi.__version__}')
except ImportError:
    print('ERROR: FastAPI not installed. Run: ${SYNC_HINT}')
    exit(1)
try:
    import torch; print(f'NOTE: PyTorch found ({torch.__version__}) but NOT needed for frontend')
except ImportError: print('PyTorch: not installed (not needed for frontend ✓)')
"
echo ""

echo "Starting frontend server on http://localhost:${PORT}"
echo ""

uv run --no-sync python "$SCRIPT_DIR/server.py"
