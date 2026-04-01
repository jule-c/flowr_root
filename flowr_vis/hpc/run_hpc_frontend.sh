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

WORKER_SCRIPT_NAME="worker_hpc.sh"
DEFAULT_PARTITION="your_partition"

source "$(cd "$(dirname "$0")" && pwd)/run_hpc_frontend_common.sh"
