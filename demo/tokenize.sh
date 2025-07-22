#!/bin/bash -l

echo "tokenize.sh: Training tokenizers and saving the learned vocabularies."

for l in raw dis dec; do
    for a in bpe uni; do
        for v in 1024 4096 16384; do
            echo "Starting '${l}' '${a}' '${v}'"
            python src/tokenization/train.py \
                --lift_level "$l" \
                --algorithm "$a" \
                --vocab_size "$v" \
                --num_files=100 \
                --batch_size=1024 \
                --block_size=1024 \
                --max_token_length=16 > /dev/null 2>&1
            echo "Finished '${l}' '${a}' '${v}'. Exit code $?."
        done
    done
done
