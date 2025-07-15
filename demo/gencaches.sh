#!/bin/bash -l

mkdir cache/materials

for lift_level in "raw" "dis" "dec"; do
    for task in "det" "beh" "fam" "mlm" "clm"; do
        for lift_level_ddp in "dec"; do
            echo "---------------------------------------------------------------------------------------------------------"
            echo "Starting task='${task}', lift_level='${lift_level}', lift_level_ddp='${lift_level_ddp}'"
            python -u src/data/generate_esp_loader_caches.py \
                    --task="${task}" \
                    --lift_level="${lift_level}" \
                    --lift_level_ddp="${lift_level_ddp}" > /dev/null 2>&1
            echo "Finished task='${task}', lift_level='${lift_level}', lift_level_ddp='${lift_level_ddp}' with exit code $?"
            echo "---------------------------------------------------------------------------------------------------------"
        done
    done
done
