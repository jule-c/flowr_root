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
#   CONDA_BASE     – absolute path to conda/mamba installation
# ══════════════════════════════════════════════════════════════════════

CONDA_ENV="${CONDA_ENV:-flowr_root}"
WORKER_PORT="${FLOWR_WORKER_PORT:-8788}"
IDLE_TIMEOUT="${FLOWR_WORKER_IDLE_TIMEOUT:-120}"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   FLOWR GPU Worker — SLURM Job           ║"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "  Node:          $(hostname)"
echo "  GPU:           ${CUDA_VISIBLE_DEVICES:-none}"
echo "  Conda env:     ${CONDA_ENV}"
echo "  Worker port:   ${WORKER_PORT}"
echo "  Idle timeout:  ${IDLE_TIMEOUT}s"
echo "  SLURM Job ID:  ${SLURM_JOB_ID:-n/a}"
echo ""

# ── Load conda/mamba ──
source "${CONDA_BASE}/etc/profile.d/conda.sh"
if [ -f "${CONDA_BASE}/etc/profile.d/mamba.sh" ]; then
    source "${CONDA_BASE}/etc/profile.d/mamba.sh"
fi
conda activate "$CONDA_ENV"

# ── Set environment ──
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export FLOWR_WORKER_PORT="${WORKER_PORT}"
export FLOWR_WORKER_IDLE_TIMEOUT="${IDLE_TIMEOUT}"

# ── OpenEye license ──
if [ -f "${SCRIPT_DIR}/tools/oe_license.txt" ]; then
    export OE_LICENSE="${SCRIPT_DIR}/tools/oe_license.txt"
elif [ -f "$(dirname "$SCRIPT_DIR")/oe_license.txt" ]; then
    export OE_LICENSE="$(dirname "$SCRIPT_DIR")/oe_license.txt"
fi

echo "Python:  $(python --version 2>&1)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'NOT FOUND')"
echo "CUDA:    $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'detection failed')"
GPU_NAME=$(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")' 2>/dev/null || echo 'detection failed')
echo "GPU:     ${GPU_NAME}"
echo ""
echo "Worker starting on $(hostname):${WORKER_PORT} — will auto-shutdown after ${IDLE_TIMEOUT}s idle"
echo ""

cd "$SCRIPT_DIR"
python worker.py
