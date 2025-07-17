#!/bin/bash -l

echo "gencaches.sh: Determining the samples to use for experiments and caching the results."

mkdir cache/materials

for lift_level in "raw" "dis" "dec" "nop"; do
    for task in "det" "beh" "fam" "mlm" "clm"; do
        for lift_level_ddp in "dec"; do
            echo "Starting '${lift_level}' '${task}' '${lift_level_ddp}'"
            python src/data/generate_esp_loader_caches.py \
                --task="${task}" \
                --lift_level="${lift_level}" \
                --lift_level_ddp="${lift_level_ddp}" > /dev/null 2>&1
            echo "Finished '${lift_level}' '${task}' '${lift_level_ddp}'. Exit code $?."
        done
    done
done
