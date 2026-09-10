#!/bin/bash -l
#SBATCH -J preprocess_data
#SBATCH --time=00-01:00:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=12G
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
# The array bound MUST equal num_jobs below: one array task per chunk.
#SBATCH --array=1-10
#SBATCH --output=./preprocess_data/lmdb_%j.out
#SBATCH --error=./preprocess_data/lmdb_%j.err

# ENVIRONMENT SETUP
export PATH="$HOME/.local/bin:$PATH"
cd YOUR_CODE_PATH/flowr_root

# Number of parallel chunks. This MUST match the '#SBATCH --array=1-N' bound above:
# SLURM reads the #SBATCH directives before this variable exists, so the two are kept in
# sync by hand. Fewer array tasks than num_jobs silently leaves chunks unprocessed.
# Pick a value no larger than your number of systems (extra jobs just exit as no-ops).
num_jobs=10

uv run --no-sync python -m flowr.data.preprocess_data.preprocess \
    --data_dir ./data \
    --save_path ./processed \
    --file_type pdb \
    --add_bonds_to_protein \
    --pocket_cutoff 7.0 \
    --cut_pocket \
    --max_pocket_size 800 \
    --commit_interval 100 \
    --num_jobs $num_jobs \
    --job_index $SLURM_ARRAY_TASK_ID
