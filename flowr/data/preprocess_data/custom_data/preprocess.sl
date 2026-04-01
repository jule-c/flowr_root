#!/bin/bash -l
#SBATCH -J preprocess_data
#SBATCH --time=00-01:00:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=12G
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --array=1-10
#SBATCH --output=./preprocess_data/lmdb_%j.out
#SBATCH --error=./preprocess_data/lmdb_%j.err

cd YOUR_CODE_PATH/flowr_root
source YOUR_ENV_PATH/miniforge3/etc/profile.d/mamba.sh
source YOUR_ENV_PATH/miniforge3/etc/profile.d/conda.sh
conda activate flowr_root

export PYTHONPATH="YOUR_CODE_PATH/flowr_root"

num_jobs=40

python -m flowr.data.datasets.complex_data.preprocess \
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