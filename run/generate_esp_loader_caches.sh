#!/bin/bash -l

#SBATCH --job-name=caches
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=00-04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=128G


# Each worker needs ~8G


source ~/anaconda3/etc/profile.d/conda.sh
conda activate RawByteClf
conda activate RawByteClf2


python -u src/data/generate_esp_loader_caches.py --num_workers=15 --suppress
