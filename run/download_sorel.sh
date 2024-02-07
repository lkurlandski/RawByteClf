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

python src/data/download_sorel.py \
--output_root=/home/lk3591/Documents/datasets/Sorel/binaries/ \
--num_samples=10000000 \
--num_bytes=1048576 \
--max_length=262144 \
--shard_idx=0 \
--num_shards=1000 \
--errors=2
