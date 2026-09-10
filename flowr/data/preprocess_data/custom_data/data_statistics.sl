#!/bin/bash
#SBATCH -J data_statistics
#SBATCH --time=01-00:00:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=24G
#SBATCH --cpus-per-task=1
#SBATCH --partition=YOUR_PARTITION
# Logs land in the directory you submit from (the repo root, per the README).
# Keep these relative so a fresh clone needs no mkdir; give an absolute path if
# you want them elsewhere -- SLURM will NOT create missing directories.
#SBATCH --output=./data_statistics_%j.out
#SBATCH --error=./data_statistics_%j.err

# ENVIRONMENT SETUP
export PATH="$HOME/.local/bin:$PATH"
cd YOUR_CODE_PATH/flowr_root

# Split sizes: a value below 1 is a FRACTION of the dataset, a whole number >= 1 is an
# absolute count of systems. Fractions are used here because absolute defaults break on
# small datasets -- "--val_size 10 --test_size 100" on a 12-system set asks for 110
# systems out of 12 and aborts with a negative training split. Switch to absolute counts
# once the dataset is comfortably larger than them (e.g. --val_size 100 --test_size 225).
#
# Using your own splits.npz instead? Comment out BOTH --val_size and --test_size below.
# They are deliberately the LAST two arguments: a commented-out line ends the command, so
# anything left below it would be swallowed (or worse, run as a separate command). Keep
# any argument you want to survive ABOVE them, and always comment the pair out together.
for state in train val test; do
        uv run --no-sync python -m flowr.data.preprocess_data.create_data_statistics \
                --data_path ./final \
                --remove_hs \
                --from_lmdb \
                --state $state \
                --seed 42 \
                --val_size 0.1 \
                --test_size 0.1
done
