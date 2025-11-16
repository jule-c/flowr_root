#!/bin/bash
#SBATCH -J Finetune_LoRA
#SBATCH --time=00-04:00:00
#SBATCH --ntasks-per-node=NUM_GPUS
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=12G
#SBATCH --cpus-per-task=NUM_WORKERS
#SBATCH --partition=YOUR_PARTITION
#SBATCH --gres=gpu:NUM_GPUS
#SBATCH --output=./finetune_lora%j.out
#SBATCH --error=./finetune_lora%j.err

# COMPUTE
num_gpus=1  # Set the number of GPUs you want to use
num_workers=12  # Set the number of CPU workers you want to use

# ENVIRONMENT SETUP
cd YOUR_CODE_PATH/flowr_root
source YOUR_ENV_PATH/miniforge3/etc/profile.d/mamba.sh
source YOUR_ENV_PATH/miniforge3/etc/profile.d/conda.sh 
conda activate flowr_root

export PYTHONPATH="YOUR_CODE_PATH/flowr_root"


# MLFLOW LOGGING
exp_name="finetune"
run_name="lora"

# DATASET
data_name="YOUR_DATASET_NAME"

# PATHS
main_path="./$data_name"
data_path="$main_path/final"

# CKPT PATH
ckpt_path="/YOUR_CHECKPOINT_PATH"
ckpt="$ckpt_path/flowr_root.ckpt"

# SAVE DIRECTORY
save_dir="$main_path/flowr_logs/$exp_name/$run_name"

# HYPERPARAMETERS
epochs=300
batch_cost=4
acc_batches=2
val_batch_cost=20
val_check_epochs=2
lr=1.0e-4
lr_schedule="exponential"
lr_gamma=0.995

# LoRA
lora_rank=16
lora_alpha=32

# RUN FINETUNE
python -m flowr.finetune \
    --arch pocket \
    --pocket_noise fix \
    --seed 42 \
    --exp_name "$exp_name" \
    --run_name "$run_name" \
    --ckpt_path "$ckpt" \
    --load_pretrained_ckpt \
    --lora_finetuning \
    --lora_rank "$lora_rank" \
    --lora_alpha "$lora_alpha" \
    --gpus "$num_gpus" \
    --num_workers "$num_workers" \
    --batch_cost "$batch_cost" \
    --acc_batches "$acc_batches" \
    --val_batch_cost "$val_batch_cost" \
    --coord_loss_weight 2.0 \
    --type_loss_weight 1.0 \
    --bond_loss_weight 3.0 \
    --charge_loss_weight 2.0 \
    --hybridization_loss_weight 1.0 \
    --distance_loss_weight_lig_pocket 2.0 \
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
    --scaffold_inpainting \
    --func_group_inpainting \
    --linker_inpainting \
    --core_inpainting \
    # --interaction_inpainting \
    # --docking_loss_weight 1.0 \
    # --plddt_confidence_loss_weight 1.0 \
    # --train_confidence \
    # --confidence_loss_weight 1.0 \
    # --confidence_gen_steps 20 \
    # --predict_docking_score \
    # --use_sde_simulation \
    # --sample_schedule log \
    # --mixed_uniform_beta_time \