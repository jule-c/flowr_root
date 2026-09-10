#!/usr/bin/env bash
# ==============================================
#  FLOWR Visualization – GPU Worker
# ==============================================
# Usage:
#   ./run_worker.sh                                # defaults
#   ./run_worker.sh --port 8788                    # custom port
#   ./run_worker.sh --ckpt /path/to/model.ckpt     # custom checkpoint
#
# This starts the GPU worker that handles FLOWR generation.
# Requires PyTorch with CUDA/MPS and the FLOWR package, i.e. a one-time
# environment setup at the project root:
#
#   uv sync --extra gpu --extra vis     # Linux / CUDA
#   uv sync --extra cpu --extra vis     # macOS / CPU (MPS)
# ==============================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Defaults ──
PORT=8788
CKPT_PATH="${PROJECT_ROOT}/ckpts/flowr_root.ckpt"

# ── Parse arguments ──
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --port)  PORT="$2"; shift ;;
        --ckpt)  CKPT_PATH="$2"; shift ;;
        -h|--help)
            echo "Usage: ./run_worker.sh [--port PORT] [--ckpt CKPT_PATH]"
            exit 0 ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# ── uv must be on PATH (login shells put it in ~/.local/bin) ──
export PATH="$HOME/.local/bin:$PATH"

if [ "$(uname -s)" = "Darwin" ]; then
    SYNC_HINT="uv sync --extra cpu --extra vis"
else
    SYNC_HINT="uv sync --extra gpu --extra vis"
fi

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║    FLOWR Visualization – GPU Worker     ║"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "  Project root:  ${PROJECT_ROOT}"
echo "  Checkpoint:    ${CKPT_PATH}"
echo "  Port:          ${PORT}"
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
export FLOWR_CKPT_PATH="${CKPT_PATH}"

# ── OpenEye license (optional external toolkit) ──
if [ -f "${SCRIPT_DIR}/tools/oe_license.txt" ]; then
    export OE_LICENSE="${SCRIPT_DIR}/tools/oe_license.txt"
elif [ -f "${PROJECT_ROOT}/oe_license.txt" ]; then
    export OE_LICENSE="${PROJECT_ROOT}/oe_license.txt"
fi
export FLOWR_WORKER_PORT="${PORT}"

cd "$PROJECT_ROOT"

# ── Verify critical imports ──
echo "Checking Python environment…"
echo "uv: $(uv --version)"
uv run --no-sync python -c "
import sys
print(f'Python: {sys.executable}')
print(f'Version: {sys.version}')
try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'  CUDA: {torch.cuda.is_available()} ({torch.cuda.device_count()} devices)' if torch.cuda.is_available() else f'  CUDA: False')
    print(f'  MPS: {torch.backends.mps.is_available()}' if hasattr(torch.backends, 'mps') else '  MPS: N/A')
except ImportError:
    print('ERROR: PyTorch not installed – required for GPU worker. Run: ${SYNC_HINT}')
    exit(1)
try:
    from rdkit import Chem; print(f'RDKit: OK')
except ImportError: print('WARNING: RDKit not installed')
try:
    import flowr; print(f'FLOWR: OK (from {flowr.__file__})')
except ImportError:
    print('ERROR: FLOWR package not importable. Run: ${SYNC_HINT}')
    exit(1)
try:
    import fastapi; print(f'FastAPI: {fastapi.__version__}')
except ImportError:
    print('ERROR: FastAPI not installed. Run: ${SYNC_HINT}')
    exit(1)
"
echo ""

# ── Check checkpoint ──
if [ ! -f "$CKPT_PATH" ]; then
    echo "WARNING: Checkpoint not found at ${CKPT_PATH}"
    echo "         Model loading will fail until a valid checkpoint is provided."
    echo ""
fi

echo "Starting GPU worker on http://localhost:${PORT}"
echo ""

uv run --no-sync python "$SCRIPT_DIR/worker.py"
