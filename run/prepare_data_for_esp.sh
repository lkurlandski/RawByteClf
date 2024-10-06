#!/bin/bash -l

#SBATCH --job-name=purge-sync
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=01-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --mem=64G


source ~/anaconda3/etc/profile.d/conda.sh
conda activate RawByteClf2


num_workers="32"


# for dataset in "assemblage_pe" "bodmas_pe" "sorel_pe" "windows_pe"; do
for dataset in "assemblage_pe" "sorel_pe" "windows_pe"; do
    for lift_level in "raw" "dis" "dec"; do

        python -u src/data/prepare_data_for_esp.py --purge \
            --dataset="$dataset" --lift_level="$lift_level" --num_workers="$num_workers"

    done

    python -u src/data/prepare_data_for_esp.py --sync --dataset="$dataset" --num_workers="$num_workers"

done
