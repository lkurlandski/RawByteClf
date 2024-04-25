#!/bin/bash -l

#SBATCH --job-name=download
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=00-04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=4G


source ~/anaconda3/etc/profile.d/conda.sh
conda activate RawByteClf


#python -u src/data/detect_packing.py \
#    --output_root="/home/lk3591/Documents/datasets/Sorel/" \
#    --prepare \
#    --num_shards=16


for i in {0..63}; do
    python -u src/data/detect_packing.py \
	--output_root="/home/lk3591/Documents/datasets/Sorel/" \
	--shard_idx=$i \
	--num_shards=64 \
        --errors=2 \
	> logs/detect_packing_$i.out 2>&1 &
done


#python -u src/data/detect_packing.py \
#    --output_root="/home/lk3591/Documents/datasets/Sorel/" \
#    --finish \
#    --num_shards=16
