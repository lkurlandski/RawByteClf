#!/bin/bash -l

#SBATCH --job-name=fuzz
#SBATCH --account=admalware
#SBATCH --partition=debug
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=00-01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=32G


# Each worker needs ~8G
# 32 workers = 256G


source ~/anaconda3/etc/profile.d/conda.sh
conda activate RawByteClf2


python -u src/data/fuzz_esp_loader_endpoints.py --task=det --num_workers=1 --subset=16
