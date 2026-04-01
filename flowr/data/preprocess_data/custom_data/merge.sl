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

cd YOUR_CODE_PATH/flowr_root
source YOUR_ENV_PATH/miniforge3/etc/profile.d/mamba.sh
source YOUR_ENV_PATH/miniforge3/etc/profile.d/conda.sh
conda activate flowr_root

export PYTHONPATH="YOUR_CODE_PATH/flowr_root"

python -m flowr.data.datasets.complex_data.merge_lmdbs \
    --chunks_dir ./processed \
    --output_path ./final \
