#!/usr/bin/env bash
# ==============================================
#  FLOWR Visualization – Frontend Server (CPU-only)
# ==============================================
# Usage:
#   ./run_server.sh                               # defaults
#   ./run_server.sh --port 8787                    # custom port
#   ./run_server.sh --worker-url http://gpu:8788   # custom worker URL
#
# This starts the lightweight CPU-only frontend server.
# It does NOT need PyTorch or FLOWR — just RDKit + FastAPI.
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

# ── Set environment ──
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export FLOWR_PORT="${PORT}"

# ── OpenEye license ──
if [ -f "${SCRIPT_DIR}/tools/oe_license.txt" ]; then
    export OE_LICENSE="${SCRIPT_DIR}/tools/oe_license.txt"
elif [ -f "${PROJECT_ROOT}/oe_license.txt" ]; then
    export OE_LICENSE="${PROJECT_ROOT}/oe_license.txt"
fi
export FLOWR_WORKER_URL="${WORKER_URL}"
export FLOWR_SERVER_URL="${SERVER_URL}"

# ── Verify critical imports ──
echo "Checking Python environment…"
python -c "
import sys
print(f'Python: {sys.executable}')
print(f'Version: {sys.version}')
try:
    from rdkit import Chem; print(f'RDKit: OK')
except ImportError: print('WARNING: RDKit not installed')
try:
    import fastapi; print(f'FastAPI: {fastapi.__version__}')
except ImportError:
    print('ERROR: FastAPI not installed. Run: pip install -r requirements.txt')
    exit(1)
try:
    import torch; print(f'NOTE: PyTorch found ({torch.__version__}) but NOT needed for frontend')
except ImportError: print('PyTorch: not installed (not needed for frontend ✓)')
"
echo ""

echo "Starting frontend server on http://localhost:${PORT}"
echo ""

cd "$SCRIPT_DIR"
python server.py
