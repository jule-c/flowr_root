#!/bin/bash
#SBATCH -J Finetune
#SBATCH --time=00-04:00:00
# Keep --ntasks-per-node and --gres in sync with num_gpus, and --cpus-per-task
# with num_workers, both set below. sbatch parses these before the shell runs,
# so they cannot reference those variables and must be literal numbers.
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=12G
#SBATCH --cpus-per-task=12
#SBATCH --partition=YOUR_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --output=./finetune_%j.out
#SBATCH --error=./finetune_%j.err

# COMPUTE
num_gpus=1  # Set the number of GPUs you want to use
num_workers=12  # Set the number of CPU workers you want to use

# ENVIRONMENT SETUP
# One-time setup, from the repo root (pick the extra that matches the machine):
#   uv sync --extra gpu   # Linux + NVIDIA GPU (CUDA 13 wheels)
#   uv sync --extra cpu   # macOS / CPU-only
# `uv run --no-sync` then uses .venv directly without re-resolving.
export PATH="$HOME/.local/bin:$PATH"
cd YOUR_CODE_PATH/flowr_root


# MLFLOW LOGGING
exp_name="finetune"
run_name="no_freeze"

# DATASET
data_name="YOUR_DATASET_NAME"

# PATHS
main_path="./$data_name"
data_path="$main_path/final"

# CKPT PATH
ckpt_path="/YOUR_CHECKPOINT_PATH"
ckpt="$ckpt_path/flowr_root_v2.2.ckpt"

# SAVE DIRECTORY
save_dir="$main_path/flowr_logs/$exp_name/$run_name"

# HYPERPARAMETERS
epochs=100
batch_cost=4
acc_batches=2
val_batch_cost=20
val_check_epochs=2
lr=1.0e-4
lr_schedule="exponential"
lr_gamma=0.995

# RUN FINETUNE
uv run --no-sync python -m flowr.finetune \
    --arch pocket \
    --pocket_noise fix \
    --seed 42 \
    --exp_name "$exp_name" \
    --run_name "$run_name" \
    --ckpt_path "$ckpt" \
    --gpus "$num_gpus" \
    --num_workers "$num_workers" \
    --batch_cost "$batch_cost" \
    --acc_batches "$acc_batches" \
    --val_batch_cost "$val_batch_cost" \
    --coord_loss_weight 3.0 \
    --type_loss_weight 1.0 \
    --bond_loss_weight 3.0 \
    --charge_loss_weight 2.0 \
    --bond_angle_loss_weight 10.0 \
    --bond_angle_huber_delta 0.5 \
    --bond_length_loss_weight 5.0 \
    --hybridization_loss_weight 1.0 \
    --distance_loss_weight_lig_pocket 10.0 \
    --coord_noise_std_dev 0.3 \
    --coord_noise_schedule "constant_decay" \
    --coord_noise_scale 0.0 \
    --pocket_coord_noise_std 0.0 \
    --time_alpha 2.0 \
    --time_beta 1.0 \
    --epochs "$epochs" \
    --val_check_epochs "$val_check_epochs" \
    --dataset "$data_name" \
    --data_path "$data_path" \
    --save_dir "$save_dir" \
    --use_ema \
    --ema_decay 0.998 \
    --lr "$lr" \
    --lr_schedule "$lr_schedule" \
    --lr_gamma "$lr_gamma"  \
    --permutation_alignment \
    --mixed_uncond_inpaint \
    --fragment_inpainting \
    --fragment_growing \
    --scaffold_hopping \
    --scaffold_elaboration \
    --predict_affinity \
    --affinity_loss_weight 3.0 \
    # --docking_loss_weight 1.0 \
    # --plddt_confidence_loss_weight 1.0 \
    # --train_confidence \
    # --confidence_loss_weight 1.0 \
    # --confidence_gen_steps 20 \
    # --predict_docking_score \
    # --use_sde_simulation \
    # --sample_schedule log \
    # --mixed_uniform_beta_time \