#!/bin/bash -l

#SBATCH --job-name=download_sorel
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=00-02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=1G


source ~/anaconda3/etc/profile.d/conda.sh
conda activate RawByteClf


python src/data/download_sorel.py --errors=2 --shard=$1 --n_shards=200 --num=1000000 --max_length=1048576
