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

# MAIN PATH
dataset="YOUR_PROJECT_NAME"
data_path="MAIN_PATH/$dataset"

# CKPT PATH
ckpt_path="YOUR_CKPT_PATH"
ckpt="$ckpt_path/flowr_root.ckpt"


# SAMPLING STEPS
steps=100
sampling_steps="_$steps-steps"
#sampling_steps=""

# NOISE INJECTION
# coord_noise_std=0.0
# noise_inject=""
coord_noise_std=0.1
noise_inject="_noise-$coord_noise_std"

# N MOLECULES PER TARGET
n_molecules_per_target=100
#sample_mol_sizes="_sampled-mol-sizes"
sample_mol_sizes="_fixed-mol-size"

# CONDITIONAL GENERATION
conditional_generation=""
#conditional_generation="_interaction-cond"
#conditional_generation="_func-group-cond"
#conditional_generation="_scaffold-cond"
#conditional_generation="_linker-cond"
#conditional_generation="_substructure-cond"

# SAVE DIR
save_dir="$data_path/processed$conditional_generation$sample_mol_sizes$noise_inject$sampling_steps"

# BATCH SIZE
batch_cost=20

python -m flowr.gen.generate_from_pdb \
    --pdb_file "$data_path/YOUR_PROTEIN.pdb" \
    --ligand_file "$data_path/YOUR_LIGAND.sdf" \
    --arch pocket \
    --pocket_type holo \
    --cut_pocket \
    --pocket_cutoff 7 \
    --gpus "$num_gpus" \
    --batch_cost $batch_cost \
    --ckpt_path "$ckpt" \
    --save_dir "$save_dir" \
    --max_sample_iter 20 \
    --coord_noise_scale $coord_noise_std \
    --sample_n_molecules_per_target $n_molecules_per_target \
    --categorical_strategy uniform-sample \
    --filter_valid_unique \
    # --substructure_inpainting \
    # --substructure 10 11 12 33 34 \
    # --filter_pb_valid \
    # --interaction_inpainting \
    # --compute_interactions \
    # --compute_interaction_recovery \
    # --add_hs_and_optimize_gen_ligs \
    # --func_group_inpainting \
    # --scaffold_inpainting \
    # --substructure_inpainting \
    # --substructure 'c1ccccc1' \
    # --linker_inpainting \
    # --sample_mol_sizes \

