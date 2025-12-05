#!/bin/sh
# Modify this script to run as sbatch if needed

# Number of GPUs and workers - adjust as needed; normally we use 8 H100 GPUs
N_GPUS=1
N_WORKERS=12

# Experiment and run names
exp_name="train"
run_name="multi_cond"

# Save and data paths - Specify your dataset; here we use SPINDR
dataset="spindr"
main_path="YOUR_MAIN_PATH/$dataset"
data_path="YOUR_DATA_PATH/$dataset/final_from_smol"
save_dir="$main_path/flowr_logs/$exp_name/$run_name"

# Parameters
epochs=500
batch_cost=5 # Adjust based on GPU memory; 5 works well for 80GB GPUs
acc_batches=6
val_batch_cost=20

# Run training
python -m flowr.train \
    --exp_name "$exp_name" \
    --run_name "$run_name" \
    --arch pocket \
    --pocket_noise fix \
    --gpus "$N_GPUS" \
    --num_workers "$N_WORKERS" \
    --batch_cost "$batch_cost" \
    --acc_batches "$acc_batches" \
    --val_batch_cost "$val_batch_cost" \
    --d_model 384 \
    --d_edge 128 \
    --n_coord_sets 128 \
    --emb_size 64 \
    --n_layers 12 \
    --d_message 64 \
    --d_message_hidden 96 \
    --n_attn_heads 32 \
    --pocket_d_model 256 \
    --pocket_n_layers 4 \
    --use_distances \
    --use_crossproducts \
    --use_rbf \
    --use_lig_pocket_rbf \
    --use_fourier_time_embed \
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
    --time_alpha 1.8 \
    --time_beta 1.0 \
    --epochs "$epochs" \
    --val_check_epochs 2 \
    --self_condition \
    --dataset "$dataset" \
    --data_path "$data_path" \
    --save_dir "$save_dir" \
    --use_ema \
    --ema_decay 0.999 \
    --lr 1.0e-3 \
    --lr_schedule exponential \
    --lr_gamma 0.998 \
    --beta1 0.9 \
    --beta2 0.95 \
    --weight_decay 1.0e-4 \
    --permutation_alignment \
    --remove_hs \
    --remove_aromaticity \
    --mixed_uncond_inpaint \
    --fragment_inpainting \
    --scaffold_inpainting \
    --func_group_inpainting \
    --linker_inpainting \
    --interaction_inpainting \
    --core_inpainting \
    # --predict_affinity \
    # --affinity_loss_weight 1.0 \
    # --distance_loss_weight_lig 2.0 \
    # --use_t_loss_weights \
    # --use_bucket_sampler \
    # --bucket_cost_scale linear \
    # --add_feats \
    # --docking_loss_weight 1.0 \
    # --plddt_confidence_loss_weight 1.0 \
    # --train_confidence \
    # --confidence_loss_weight 1.0 \
    # --confidence_gen_steps 20 \
    # --predict_docking_score \
    # --use_sde_simulation \
    # --sample_schedule log \
    # --use_smol \
    # --mixed_uniform_beta_time \
    # --split_continuous_discrete_time \
    # --categorical_strategy uniform-sample \
    # --cat_sampling_noise_level 10 \
    # --corrector_iters 16 \