#!/bin/bash
#SBATCH -J data_statistics
#SBATCH --time=01-00:00:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=24G
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --output=./data_statistics/lmdb_%j.out
#SBATCH --error=./data_statistics/lmdb_%j.err

cd YOUR_CODE_PATH/flowr_root
source YOUR_ENV_PATH/miniforge3/etc/profile.d/mamba.sh
source YOUR_ENV_PATH/miniforge3/etc/profile.d/conda.sh
conda activate flowr_root

export PYTHONPATH="YOUR_CODE_PATH/flowr_root"

for state in train val test; do
        python -m flowr.data.preprocess_data.create_data_statistics \
                --data_path ./final \
                --remove_hs \
                --from_lmdb \
                --state $state \
                --val_size 10 \
                --test_size 100 \
                --seed 42
done