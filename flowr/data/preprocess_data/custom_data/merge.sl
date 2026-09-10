#!/bin/bash
#SBATCH -J merge_data
#SBATCH --time=00-01:00:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --output=./merge_data/lmdb_%j.out
#SBATCH --error=./merge_data/lmdb_%j.err

# ENVIRONMENT SETUP
export PATH="$HOME/.local/bin:$PATH"
cd YOUR_CODE_PATH/flowr_root

uv run --no-sync python -m flowr.data.preprocess_data.merge_lmdbs \
    --chunks_dir ./processed \
    --output_path ./final \
