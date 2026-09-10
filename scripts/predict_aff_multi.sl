#!/bin/bash
#SBATCH -J AffPred
#SBATCH --time=01-00:00:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=12G
#SBATCH --cpus-per-task=12
#SBATCH --partition=YOUR_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --output=YOUR_CODE_PATH/slurm_outs/aff_pred/generate-sbdd_%j.out
#SBATCH --error=YOUR_CODE_PATH/slurm_outs/aff_pred/generate-sbdd_%j.err

# ENVIRONMENT SETUP
# One-time setup, from the repo root (pick the extra that matches the machine):
#   uv sync --extra gpu   # Linux + NVIDIA GPU (CUDA 13 wheels)
#   uv sync --extra cpu   # macOS / CPU-only
# `uv run --no-sync` then uses .venv directly without re-resolving.
export PATH="$HOME/.local/bin:$PATH"
cd YOUR_CODE_PATH/flowr_root

# COMPUTE
num_workers=12

# MAIN PATH
dataset="YOUR_PROTEIN_NAME"
data_path="YOUR_MAIN_PATH/$dataset"

# CKPT PATH
ckpt_path="YOUR_CKPT_PATH"
ckpt="$ckpt_path/flowr_root.ckpt"

# NOISE INJECTION
coord_noise_std=0.05
noise_inject="_noise-$coord_noise_std"

# BATCH SIZE
batch_cost=20

for seed in 2 42 512 1000; do
    save_dir="$main_path/flowr_logs/predict-aff${noise_inject}_seed-${seed}"
    #save_dir="$ckpt_path/predict-aff${noise_inject}_seed-${seed}"
    mkdir -p "$save_dir"

    uv run --no-sync python -m flowr.predict.predict_from_pdb \
        --pdb_file "$data_path/YOUR_PROTEIN.pdb" \
        --ligand_file "$data_path/YOUR_LIGANDS.sdf" \
        --multiple_ligands \
        --dataset $dataset \
        --gpus 1 \
        --seed $seed \
        --batch_cost $batch_cost \
        --arch pocket \
        --pocket_type holo \
        --pocket_noise fix \
        --cut_pocket \
        --pocket_cutoff 7 \
        --ckpt_path "$ckpt" \
        --save_dir "$save_dir" \
        --coord_noise_scale $coord_noise_std
done