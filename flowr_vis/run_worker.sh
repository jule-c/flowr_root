#!/usr/bin/env bash
# ==============================================
#  FLOWR Visualization – GPU Worker
# ==============================================
# Usage:
#   ./run_worker.sh                                # defaults
#   ./run_worker.sh --port 8788                    # custom port
#   ./run_worker.sh --ckpt /path/to/model.ckpt     # custom checkpoint
#   ./run_worker.sh --env flowr_root               # custom conda env
#
# This starts the GPU worker that handles FLOWR generation.
# Requires PyTorch with CUDA/MPS and the FLOWR package.
# ==============================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Defaults ──
PORT=8788
CONDA_ENV="flowr_root"
CKPT_PATH="${PROJECT_ROOT}/ckpts/flowr_root.ckpt"

# ── Parse arguments ──
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --port)  PORT="$2"; shift ;;
        --env)   CONDA_ENV="$2"; shift ;;
        --ckpt)  CKPT_PATH="$2"; shift ;;
        -h|--help)
            echo "Usage: ./run_worker.sh [--port PORT] [--env CONDA_ENV] [--ckpt CKPT_PATH]"
            exit 0 ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║    FLOWR Visualization – GPU Worker     ║"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "  Project root:  ${PROJECT_ROOT}"
echo "  Conda env:     ${CONDA_ENV}"
echo "  Checkpoint:    ${CKPT_PATH}"
echo "  Port:          ${PORT}"
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

# ── Set environment ──
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export FLOWR_CKPT_PATH="${CKPT_PATH}"

# ── OpenEye license ──
if [ -f "${SCRIPT_DIR}/tools/oe_license.txt" ]; then
    export OE_LICENSE="${SCRIPT_DIR}/tools/oe_license.txt"
elif [ -f "${PROJECT_ROOT}/oe_license.txt" ]; then
    export OE_LICENSE="${PROJECT_ROOT}/oe_license.txt"
fi
export FLOWR_WORKER_PORT="${PORT}"

# ── Verify critical imports ──
echo "Checking Python environment…"
python -c "
import sys
print(f'Python: {sys.executable}')
print(f'Version: {sys.version}')
try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'  CUDA: {torch.cuda.is_available()} ({torch.cuda.device_count()} devices)' if torch.cuda.is_available() else f'  CUDA: False')
    print(f'  MPS: {torch.backends.mps.is_available()}' if hasattr(torch.backends, 'mps') else '  MPS: N/A')
except ImportError:
    print('ERROR: PyTorch not installed – required for GPU worker')
    exit(1)
try:
    from rdkit import Chem; print(f'RDKit: OK')
except ImportError: print('WARNING: RDKit not installed')
try:
    import flowr; print(f'FLOWR: OK (from {flowr.__file__})')
except ImportError:
    print('ERROR: FLOWR package not importable')
    exit(1)
try:
    import fastapi; print(f'FastAPI: {fastapi.__version__}')
except ImportError:
    print('ERROR: FastAPI not installed. Run: pip install -r requirements_worker.txt')
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

cd "$SCRIPT_DIR"
python worker.py
