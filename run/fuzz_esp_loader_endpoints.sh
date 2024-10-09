#!/bin/bash -l

#SBATCH --job-name=fuzz
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=01-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --mem=256G


# Each worker needs ~8G


source ~/anaconda3/etc/profile.d/conda.sh
conda activate RawByteClf2


python -u src/data/fuzz_esp_loader_endpoints.py --task="beh" --num_workers=32
