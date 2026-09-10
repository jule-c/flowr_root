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

# ENVIRONMENT SETUP
export PATH="$HOME/.local/bin:$PATH"
cd YOUR_CODE_PATH/flowr_root

# Split sizes: a value below 1 is a FRACTION of the dataset, a whole number >= 1 is an
# absolute count of systems. Fractions are used here because absolute defaults break on
# small datasets -- "--val_size 10 --test_size 100" on a 12-system set asks for 110
# systems out of 12 and aborts with a negative training split. Switch to absolute counts
# once the dataset is comfortably larger than them (e.g. --val_size 100 --test_size 225).
for state in train val test; do
        uv run --no-sync python -m flowr.data.preprocess_data.create_data_statistics \
                --data_path ./final \
                --remove_hs \
                --from_lmdb \
                --state $state \
                --val_size 0.1 \
                --test_size 0.1 \
                --seed 42
done
