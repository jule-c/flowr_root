#!/bin/bash
# ══════════════════════════════════════════════════════════════════════
#  FLOWR GPU Worker — Common Logic (sourced by worker_hpc.sh)
# ══════════════════════════════════════════════════════════════════════
# This file is NOT meant to be run directly. It is sourced by
# worker_hpc.sh after loading paths from hpc.env.
#
# Expected variables (set before sourcing):
#   PROJECT_ROOT   – absolute path to the project root
#   SCRIPT_DIR     – absolute path to the flowr_vis directory
#
# The Python environment is the uv-managed .venv at PROJECT_ROOT
# (one-time setup: `uv sync --extra gpu --extra vis`).
# ══════════════════════════════════════════════════════════════════════

set -e

WORKER_PORT="${FLOWR_WORKER_PORT:-8788}"
IDLE_TIMEOUT="${FLOWR_WORKER_IDLE_TIMEOUT:-120}"

# ── uv must be on PATH (SLURM jobs get a non-interactive shell) ──
export PATH="$HOME/.local/bin:$PATH"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   FLOWR GPU Worker — SLURM Job           ║"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "  Node:          $(hostname)"
echo "  GPU:           ${CUDA_VISIBLE_DEVICES:-none}"
echo "  Environment:   ${PROJECT_ROOT}/.venv (uv)"
echo "  Worker port:   ${WORKER_PORT}"
echo "  Idle timeout:  ${IDLE_TIMEOUT}s"
echo "  SLURM Job ID:  ${SLURM_JOB_ID:-n/a}"
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
    echo "         cd ${PROJECT_ROOT} && uv sync --extra gpu --extra vis"
    exit 1
fi

# ── Set environment ──
export FLOWR_WORKER_PORT="${WORKER_PORT}"
export FLOWR_WORKER_IDLE_TIMEOUT="${IDLE_TIMEOUT}"

# ── OpenEye license (optional external toolkit) ──
if [ -f "${SCRIPT_DIR}/tools/oe_license.txt" ]; then
    export OE_LICENSE="${SCRIPT_DIR}/tools/oe_license.txt"
elif [ -f "${PROJECT_ROOT}/oe_license.txt" ]; then
    export OE_LICENSE="${PROJECT_ROOT}/oe_license.txt"
fi

cd "$PROJECT_ROOT"

echo "uv:      $(uv --version)"
# Single interpreter start: importing torch three times costs ~10s on a GPU node.
uv run --no-sync python -c '
import sys
print("Python:  " + sys.version.split()[0])
try:
    import torch
    print("PyTorch: " + torch.__version__)
    avail = torch.cuda.is_available()
    print("CUDA:    " + str(avail))
    print("GPU:     " + (torch.cuda.get_device_name(0) if avail else "N/A"))
except Exception as exc:
    print("PyTorch: NOT FOUND (" + str(exc) + ")")
' || echo "PyTorch: detection failed"
echo ""
echo "Worker starting on $(hostname):${WORKER_PORT} — will auto-shutdown after ${IDLE_TIMEOUT}s idle"
echo ""

uv run --no-sync python "${SCRIPT_DIR}/worker.py"
