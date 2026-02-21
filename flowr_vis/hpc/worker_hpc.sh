#!/bin/bash
# ══════════════════════════════════════════════════════════════════════
#  FLOWR GPU Worker — SLURM Job Script
# ══════════════════════════════════════════════════════════════════════
# This script is submitted by the frontend server via `sbatch`.
# All user-specific paths are read from hpc.env (see hpc.env.template).
#
# IMPORTANT: #SBATCH directives MUST appear before any executable code,
# otherwise SLURM silently ignores them. Edit the defaults below, or
# pass overrides via the sbatch command line (the server does this
# automatically using values from hpc.env).
# ══════════════════════════════════════════════════════════════════════

# ── SBATCH directives (must be before any executable statements) ──
#SBATCH -J flowr_gpu
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
# ── Edit these SBATCH headers to match your cluster ──
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=12G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=your_partition

set -e

# ── Load config ──
# FLOWR_HPC_CONFIG is passed via environment from run_hpc_frontend.sh.
# NOTE: $0 in SLURM jobs may point to a spool copy, so we cannot rely on
# dirname "$0" to find hpc.env. Use FLOWR_HPC_CONFIG (env) or
# SLURM_SUBMIT_DIR (set by SLURM to the submitter's cwd) as fallbacks.
_HPC_DIR_FALLBACK="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
_CONFIG="${FLOWR_HPC_CONFIG:-${_HPC_DIR_FALLBACK}/hpc.env}"
if [ -f "$_CONFIG" ]; then
    source "$_CONFIG"
    echo "Sourced config: $_CONFIG"
else
    echo "WARNING: Config not found at $_CONFIG — relying on environment variables"
fi

# Resolve _HPC_DIR from config (reliable) or dirname fallback
if [ -n "$FLOWR_HPC_CONFIG" ]; then
    _HPC_DIR="$(dirname "$FLOWR_HPC_CONFIG")"
else
    _HPC_DIR="$_HPC_DIR_FALLBACK"
fi

set -e

# ── Resolve paths (auto-detect from script location if not in config) ──
if [ -z "$PROJECT_ROOT" ]; then
    PROJECT_ROOT="$(cd "${_HPC_DIR}/../.." && pwd)"
fi
SCRIPT_DIR="${PROJECT_ROOT}/flowr_vis"

if [ -z "$CONDA_BASE" ]; then
    if command -v conda &>/dev/null; then
        CONDA_BASE="$(conda info --base 2>/dev/null)"
    fi
fi
if [ -z "$CONDA_BASE" ]; then
    echo "ERROR: CONDA_BASE is not set. Set it in hpc.env or your environment."
    exit 1
fi

# ── Create SLURM output directory if needed ──
SLURM_OUTPUT_DIR="${SLURM_OUTPUT_DIR:-${HOME}/slurm_outs}"
mkdir -p "$SLURM_OUTPUT_DIR"

source "${_HPC_DIR}/worker_common.sh"
