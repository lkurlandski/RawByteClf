#!/bin/bash -l

#SBATCH --job-name=download_sorel_labeled
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=00-01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=4G


source ~/anaconda3/etc/profile.d/conda.sh
conda activate RawByteClf
conda activate RawByteClf2


python -u src/data/download_sorel.py \
--output_root="/home/lk3591/Documents/datasets/Sorel/binaries" \
--shard_idx=$1 \
--num_shards=500 \
--packing_protocol="yes" \
--include_shas="/home/lk3591/Documents/code/RawByteClf/tmp/sorel_packed_labeled_shas.txt" \
--exclude_shas="/home/lk3591/Documents/code/RawByteClf/tmp/sorel_exclude.txt" \
--errors=2
