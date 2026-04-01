#!/usr/bin/env bash
# ==============================================
#  FLOWR Visualization – Local (Mac) Launch
# ==============================================
# Starts BOTH the frontend server and GPU worker locally.
# The worker uses MPS (Apple Silicon) or CPU as fallback.
#
# Usage:
#   ./run_local.sh                          # defaults
#   ./run_local.sh --env flowr_root         # custom conda env
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
CONDA_ENV="flowr_root"
CKPT_PATH="${PROJECT_ROOT}/ckpts/flowr_root.ckpt"

# ── Parse arguments ──
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --server-port) SERVER_PORT="$2"; shift ;;
        --worker-port) WORKER_PORT="$2"; shift ;;
        --env)         CONDA_ENV="$2"; shift ;;
        --ckpt)        CKPT_PATH="$2"; shift ;;
        -h|--help)
            echo "Usage: ./run_local.sh [--server-port PORT] [--worker-port PORT] [--env CONDA_ENV] [--ckpt CKPT_PATH]"
            exit 0 ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

WORKER_URL="http://localhost:${WORKER_PORT}"
SERVER_URL="http://localhost:${SERVER_PORT}"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   FLOWR Visualization – Local Launch    ║"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "  Project root:  ${PROJECT_ROOT}"
echo "  Conda env:     ${CONDA_ENV}"
echo "  Checkpoint:    ${CKPT_PATH}"
echo "  Frontend:      ${SERVER_URL}"
echo "  Worker:        ${WORKER_URL}"
echo ""

# ── Activate conda environment ──
if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV" 2>/dev/null || {
        echo "WARNING: Could not activate conda env '${CONDA_ENV}'."
        echo "         Falling back to current environment."
    }
else
    echo "WARNING: conda not found. Ensure '${CONDA_ENV}' is activated."
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ── OpenEye license ──
if [ -f "${SCRIPT_DIR}/tools/oe_license.txt" ]; then
    export OE_LICENSE="${SCRIPT_DIR}/tools/oe_license.txt"
elif [ -f "${PROJECT_ROOT}/oe_license.txt" ]; then
    export OE_LICENSE="${PROJECT_ROOT}/oe_license.txt"
fi

# ── Quick env check ──
echo "Checking Python environment…"
python -c "
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
    print('  FastAPI:  NOT FOUND – run: pip install -r requirements.txt')
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
    python "$SCRIPT_DIR/worker.py" &
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
    python "$SCRIPT_DIR/server.py" &
SERVER_PID=$!

# Wait for either to exit (compatible with macOS bash 3.2)
wait $WORKER_PID $SERVER_PID 2>/dev/null
cleanup
