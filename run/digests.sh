#!/bin/bash -l

#SBATCH --job-name=digests
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=00-4:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --mem=128G


source ~/anaconda3/etc/profile.d/conda.sh
conda activate RawByteClf2


NUM_WORKERS=32


for dsn in ass bod sor win; do

    for ll in raw dis dec; do

        input="./data/$dsn/$ll"
        output="./data/$dsn/$ll/digests.json"
        echo "Computing digests: $input --> $output"
        echo "----------------------------------------------------------------"
        python -u src/data/digests.py --input="$input" --output="$output" --num_workers=$NUM_WORKERS
        echo "----------------------------------------------------------------"

    done

done
