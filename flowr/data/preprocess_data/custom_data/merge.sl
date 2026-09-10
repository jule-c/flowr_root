#!/bin/bash
#SBATCH -J merge_data
#SBATCH --time=00-01:00:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=1
#SBATCH --partition=YOUR_PARTITION
# Logs land in the directory you submit from (the repo root, per the README).
# Keep these relative so a fresh clone needs no mkdir; give an absolute path if
# you want them elsewhere -- SLURM will NOT create missing directories.
#SBATCH --output=./merge_%j.out
#SBATCH --error=./merge_%j.err

# ENVIRONMENT SETUP
export PATH="$HOME/.local/bin:$PATH"
cd YOUR_CODE_PATH/flowr_root

uv run --no-sync python -m flowr.data.preprocess_data.merge_lmdbs \
    --chunks_dir ./processed \
    --output_path ./final
