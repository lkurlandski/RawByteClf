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
conda activate RawByteClf2


lift_level_ddp="dec"
for task in "beh" "fam"; do
    for lift_level in "raw" "dis" "dec"; do
        (
            echo "Started: task='${task}', lift_level='${lift_level}', lift_level_ddp='${lift_level_ddp}'"
            python -u src/data/generate_esp_loader_caches.py \
                --task="${task}" \
                --lift_level="${lift_level}" \
                --lift_level_ddp="${lift_level_ddp}" \
                > /dev/null 2>&1
            echo "Finished: task='${task}', lift_level='${lift_level}', lift_level_ddp='${lift_level_ddp}' with exit code $?"
        ) &
    done
done

wait

exit

for task in "det" "beh" "fam" "mlm" "clm"; do
    for lift_level in "raw" "dis" "dec"; do
        for lift_level_ddp in "raw" "dis" "dec" "none"; do
            (
                python -u src/data/generate_esp_loader_caches.py \
                    --task="${task}" \
                    --lift_level="${lift_level}" \
                    --lift_level_ddp="${lift_level_ddp}" \
                    > /dev/null 2>&1
                echo "Finished: task='${task}', lift_level='${lift_level}', lift_level_ddp='${lift_level_ddp}' with exit code $?"
            ) &
        done
    done
done

wait
