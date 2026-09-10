#!/usr/bin/env bash
# ==============================================
#  FLOWR Visualization – Local (Mac) Launch
# ==============================================
# Starts BOTH the frontend server and GPU worker locally.
# The worker uses MPS (Apple Silicon) or CPU as fallback.
#
# Requires a one-time environment setup at the project root:
#   uv sync --extra cpu --extra vis     # macOS / CPU
#   uv sync --extra gpu --extra vis     # Linux / CUDA
#
# Usage:
#   ./run_local.sh                          # defaults
#   ./run_local.sh --ckpt /path/to/ckpt     # custom checkpoint
#
# Then open http://localhost:8787 in your browser.
# Press Ctrl+C to stop both servers.
# ==============================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Defaults ──
SERVER_PORT=8787
WORKER_PORT=8788
CKPT_PATH="${PROJECT_ROOT}/ckpts/flowr_root.ckpt"

# ── Parse arguments ──
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --server-port) SERVER_PORT="$2"; shift ;;
        --worker-port) WORKER_PORT="$2"; shift ;;
        --ckpt)        CKPT_PATH="$2"; shift ;;
        -h|--help)
            echo "Usage: ./run_local.sh [--server-port PORT] [--worker-port PORT] [--ckpt CKPT_PATH]"
            exit 0 ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

WORKER_URL="http://localhost:${WORKER_PORT}"
SERVER_URL="http://localhost:${SERVER_PORT}"

# ── uv must be on PATH (login shells put it in ~/.local/bin) ──
export PATH="$HOME/.local/bin:$PATH"

if [ "$(uname -s)" = "Darwin" ]; then
    SYNC_HINT="uv sync --extra cpu --extra vis"
else
    SYNC_HINT="uv sync --extra gpu --extra vis"
fi

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   FLOWR Visualization – Local Launch    ║"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "  Project root:  ${PROJECT_ROOT}"
echo "  Checkpoint:    ${CKPT_PATH}"
echo "  Frontend:      ${SERVER_URL}"
echo "  Worker:        ${WORKER_URL}"
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

cd "$PROJECT_ROOT"

# ── OpenEye license (optional external toolkit) ──
if [ -f "${SCRIPT_DIR}/tools/oe_license.txt" ]; then
    export OE_LICENSE="${SCRIPT_DIR}/tools/oe_license.txt"
elif [ -f "${PROJECT_ROOT}/oe_license.txt" ]; then
    export OE_LICENSE="${PROJECT_ROOT}/oe_license.txt"
fi

# ── Quick env check ──
echo "Checking Python environment…"
echo "  uv:       $(uv --version)"
uv run --no-sync python -c "
import sys
print(f'  Python:   {sys.executable}')
try:
    import torch
    device = 'mps' if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()) else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'  PyTorch:  {torch.__version__}  (device: {device})')
except ImportError:
    print('  PyTorch:  NOT FOUND (worker will fail)')
try:
    from rdkit import Chem; print('  RDKit:    OK')
except ImportError:
    print('  RDKit:    NOT FOUND')
try:
    import fastapi; print(f'  FastAPI:  {fastapi.__version__}')
except ImportError:
    print('  FastAPI:  NOT FOUND – run: ${SYNC_HINT}')
    exit(1)
"
echo ""

# ── Check checkpoint ──
if [ ! -f "$CKPT_PATH" ]; then
    echo "WARNING: Checkpoint not found at ${CKPT_PATH}"
    echo ""
fi

# ── Trap Ctrl+C to kill both processes ──
WORKER_PID=""
SERVER_PID=""
cleanup() {
    echo ""
    echo "Shutting down…"
    [[ -n "$WORKER_PID" ]] && kill "$WORKER_PID" 2>/dev/null
    [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null
    [[ -n "$WORKER_PID" ]] && wait "$WORKER_PID" 2>/dev/null
    [[ -n "$SERVER_PID" ]] && wait "$SERVER_PID" 2>/dev/null
    echo "Done."
    exit 0
}
trap cleanup SIGINT SIGTERM

# ── Start GPU worker in background ──
echo "Starting worker on ${WORKER_URL} …"
FLOWR_CKPT_PATH="${CKPT_PATH}" \
FLOWR_WORKER_PORT="${WORKER_PORT}" \
    uv run --no-sync python "$SCRIPT_DIR/worker.py" &
WORKER_PID=$!

# Give the worker a moment to bind its port
sleep 2

# ── Start frontend server in foreground ──
echo "Starting frontend on ${SERVER_URL} …"
echo ""
echo "  ➜  Open ${SERVER_URL} in your browser"
echo ""
FLOWR_PORT="${SERVER_PORT}" \
FLOWR_WORKER_URL="${WORKER_URL}" \
FLOWR_SERVER_URL="${SERVER_URL}" \
    uv run --no-sync python "$SCRIPT_DIR/server.py" &
SERVER_PID=$!

# Wait for either to exit (compatible with macOS bash 3.2)
wait $WORKER_PID $SERVER_PID 2>/dev/null
cleanup
