#!/bin/bash
#SBATCH -J SamplePDB
#SBATCH --time=01-00:00:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=12G
#SBATCH --cpus-per-task=12
#SBATCH --partition=YOUR_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --output=YOUR_CODE_PATH/slurm_outs/pdb_gen/generate-sbdd_%j.out
#SBATCH --error=YOUR_CODE_PATH/slurm_outs/pdb_gen/generate-sbdd_%j.err

cd YOUR_CODE_PATH/flowr_root
source YOUR_ENV_PATH/miniforge3/etc/profile.d/mamba.sh
source YOUR_ENV_PATH/miniforge3/etc/profile.d/conda.sh
conda activate flowr_root

export PYTHONPATH="YOUR_CODE_PATH/flowr_root"

# COMPUTE
num_gpus=1
num_workers=12

# MAIN and DATA PATH
dataset="YOUR_DATASET_NAME"
main_path="YOUR_MAIN_PATH"
data_path="$main_path/$dataset/ligand_only"

# CKPT PATH
ckpt_path="YOUR_CKPT_PATH"
ckpt="$ckpt_path/flowr_root_v2_mol.ckpt"


# SAMPLING STEPS
steps=100
sampling_steps="_$steps-steps"
#sampling_steps=""

# NOISE INJECTION
# coord_noise_std=0.0
# noise_inject=""
coord_noise_std=0.2
noise_inject="_noise-$coord_noise_std"

# SAMPLING STRATEGY
sampling_strategy="linear"
#sampling_strategy="log"
integration_strategy="ODE"
#integration_strategy="SDE"
sample_strategy="_$integration_strategy-$sampling_strategy-sampling-strategy"


# N MOLECULES PER TARGET
n_molecules_per_mol=1000
sample_mol_sizes="_sampled-mol-sizes"
#sample_mol_sizes="_fixed-mol-size"

# CONDITIONAL GENERATION
#conditional_generation=""
#conditional_generation="_interaction-cond"
#conditional_generation="_func-group-cond"
#conditional_generation="_scaffold-cond"
#conditional_generation="_linker-cond"
#conditional_generation="_core-cond"
conditional_generation="_substructure-cond"

# SAVE DIR
save_dir="$data_path/generate_hydantoin_optim_rdkit/processed$conditional_generation$sample_mol_sizes$noise_inject$sampling_steps$sample_strategy"
mkdir -p "$save_dir"

# BATCH SIZE
batch_cost=256

ligand_idx=0 # modify this index to select a specific ligand from the sdf file, or set to -1 or None to use all ligands
python -m flowr.gen.generate_from_sdf_mol \
    --sdf_path "$data_path/YOUR_SDF_FILE.sdf" \
    --ligand_idx $ligand_idx \
    --arch flowr \
    --gpus "$num_gpus" \
    --num_workers 12 \
    --batch_cost $batch_cost \
    --ckpt_path "$ckpt" \
    --save_dir "$save_dir" \
    --max_sample_iter 20 \
    --coord_noise_scale $coord_noise_std \
    --sample_n_molecules_per_mol $n_molecules_per_mol \
    --categorical_strategy uniform-sample \
    --ode_sampling_strategy "$sampling_strategy" \
    --filter_valid_unique \
    --filter_diversity \
    --diversity_threshold 0.9 \
    --add_hs_gen_mols \
    --substructure_inpainting \
    --substructure 21 23 30 31 32 33 34 35 \
    --filter_cond_substructure \
    # --calculate_strain_energies \
    # --optimize_gen_mols_rdkit \
    # --optimize_gen_mols_xtb \
    # --filter_pb_valid \
    # --scaffold_inpainting \
    # --func_group_inpainting \
    # --core_inpainting \
    # --use_sde_simulation \
    # --linker_inpainting \
    # --sample_mol_sizes \

