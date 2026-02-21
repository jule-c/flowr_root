#!/bin/bash
# ══════════════════════════════════════════════════════════════════════
#  FLOWR Frontend Server — HPC (CPU-only, always-on)
# ══════════════════════════════════════════════════════════════════════
# Run this on the HPC login node (or a CPU SLURM job). It serves the
# web UI and dynamically submits GPU jobs via SLURM when users click
# "Generate".
#
# Setup (one-time):
#   cd flowr_vis/hpc/
#   cp hpc.env.template hpc.env
#   # Edit hpc.env with your cluster-specific paths
#
# Usage:
#   ./hpc/run_hpc_frontend.sh                     # uses hpc/hpc.env
#   ./hpc/run_hpc_frontend.sh --config my.env     # custom config file
#
# Then SSH-tunnel from your laptop:
#   ssh -N -L 8787:<node>:8787 <user>@<hpc-login-node>
#   open http://localhost:8787
# ══════════════════════════════════════════════════════════════════════

set -e

HPC_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_FILE="${HPC_DIR}/hpc.env"

# ── Parse arguments ──
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --config) CONFIG_FILE="$2"; shift ;;
        -h|--help)
            echo "Usage: ./hpc/run_hpc_frontend.sh [--config <path-to-hpc.env>]"
            echo ""
            echo "First-time setup:"
            echo "  cp hpc/hpc.env.template hpc/hpc.env"
            echo "  # Edit hpc.env with your paths"
            exit 0 ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# ── Load user configuration ──
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Configuration file not found: ${CONFIG_FILE}"
    echo ""
    echo "Create one from the template:"
    echo "  cp ${HPC_DIR}/hpc.env.template ${HPC_DIR}/hpc.env"
    echo "  # Then edit hpc.env with your cluster-specific paths"
    exit 1
fi
# shellcheck source=hpc.env.template
source "$CONFIG_FILE"

# ── Auto-detect PROJECT_ROOT if not set ──
if [ -z "$PROJECT_ROOT" ]; then
    PROJECT_ROOT="$(cd "${HPC_DIR}/../.." && pwd)"
fi
SCRIPT_DIR="${PROJECT_ROOT}/flowr_vis"

# ── Auto-detect CONDA_BASE if not set ──
if [ -z "$CONDA_BASE" ]; then
    if command -v conda &>/dev/null; then
        CONDA_BASE="$(conda info --base 2>/dev/null)"
    fi
fi
if [ -z "$CONDA_BASE" ]; then
    echo "ERROR: CONDA_BASE is not set and could not be auto-detected."
    echo "       Set it in ${CONFIG_FILE}"
    exit 1
fi

# ── Apply defaults for anything not set in config ──
CONDA_ENV="${CONDA_ENV:-flowr_root}"
PORT="${FLOWR_PORT:-${PORT:-8787}}"
SSH_USER="${SSH_USER:-${USER}}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-300}"
WORKER_IDLE_TIMEOUT="${WORKER_IDLE_TIMEOUT:-120}"

# ── Resolve checkpoint path ──
CKPT_PATH="${CKPT_PATH:-ckpts/flowr_root.ckpt}"
if [[ "$CKPT_PATH" != /* ]]; then
    CKPT_PATH="${PROJECT_ROOT}/${CKPT_PATH}"
fi

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   FLOWR Frontend — HPC (CPU-only)        ║"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "  Config:          ${CONFIG_FILE}"
echo "  Node:            $(hostname)"
echo "  Project root:    ${PROJECT_ROOT}"
echo "  Conda base:      ${CONDA_BASE}"
echo "  Conda env:       ${CONDA_ENV}"
echo "  Port:            ${PORT}"
echo "  Checkpoint:      ${CKPT_PATH}"
echo "  Worker mode:     slurm (on-demand GPU allocation)"
echo "  Worker script:   ${HPC_DIR}/worker_hpc.sh"
echo ""

# ── Load conda/mamba ──
source "${CONDA_BASE}/etc/profile.d/conda.sh"
if [ -f "${CONDA_BASE}/etc/profile.d/mamba.sh" ]; then
    source "${CONDA_BASE}/etc/profile.d/mamba.sh"
fi
conda activate "$CONDA_ENV"

# ── Set environment ──
# Export paths so they propagate through server.py → sbatch → compute node
export CONDA_BASE
export CONDA_ENV
export PROJECT_ROOT
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export FLOWR_PORT="${PORT}"
export FLOWR_CKPT_PATH="${CKPT_PATH}"
if [ -f "${SCRIPT_DIR}/tools/oe_license.txt" ]; then
    export OE_LICENSE="${SCRIPT_DIR}/tools/oe_license.txt"
elif [ -f "${PROJECT_ROOT}/oe_license.txt" ]; then
    export OE_LICENSE="${PROJECT_ROOT}/oe_license.txt"
fi

# SLURM-based dynamic GPU allocation
export FLOWR_WORKER_MODE="slurm"
export FLOWR_SLURM_WORKER_SCRIPT="${HPC_DIR}/worker_hpc.sh"
export FLOWR_SLURM_STARTUP_TIMEOUT="${STARTUP_TIMEOUT}"
export FLOWR_WORKER_IDLE_TIMEOUT="${WORKER_IDLE_TIMEOUT}"

# Export SLURM resource settings so server.py can pass them as sbatch CLI args.
# Use FLOWR_SLURM_* prefix to avoid conflicts with SLURM-reserved env vars.
export FLOWR_SLURM_PARTITION="${FLOWR_SLURM_PARTITION:-your_partition}"
export FLOWR_SLURM_TIME="${FLOWR_SLURM_TIME:-04:00:00}"
export FLOWR_SLURM_MEM_PER_CPU="${FLOWR_SLURM_MEM_PER_CPU:-12G}"
export FLOWR_SLURM_CPUS_PER_TASK="${FLOWR_SLURM_CPUS_PER_TASK:-8}"
export FLOWR_SLURM_GRES="${FLOWR_SLURM_GRES:-gpu:1}"
export FLOWR_SLURM_OUTPUT_DIR="${FLOWR_SLURM_OUTPUT_DIR:-${HOME}/slurm_outs}"

# Pass config file path so worker_hpc.sh can source the same settings
export FLOWR_HPC_CONFIG="${CONFIG_FILE}"

# The URL where the worker can reach this frontend server (for file downloads).
export FLOWR_SERVER_URL="http://$(hostname):${PORT}"

echo "  Server URL:      ${FLOWR_SERVER_URL}"
echo "  Startup timeout: ${FLOWR_SLURM_STARTUP_TIMEOUT}s"
echo "  Idle timeout:    ${FLOWR_WORKER_IDLE_TIMEOUT}s"
echo ""

echo "Python: $(python --version)"
echo ""

# ── Print connection instructions ──
NODE=$(hostname -s)
echo "════════════════════════════════════════════"
echo "  Frontend starting on ${NODE}:${PORT}"
echo ""
echo "  To connect from your laptop:"
echo "    ssh -N -L ${PORT}:${NODE}:${PORT} ${SSH_USER}@<hpc-login-node>"
echo "    Then open: http://localhost:${PORT}"
echo ""
echo "  GPU allocation is automatic — click Generate"
echo "  in the browser and a SLURM job will be submitted."
echo "════════════════════════════════════════════"
echo ""

cd "$SCRIPT_DIR"
python server.py
