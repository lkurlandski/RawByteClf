#!/bin/bash -l

#SBATCH --job-name=debug
#SBATCH --account=admalware
#SBATCH --partition=debug
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=00-01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1


#source ~/anaconda3/etc/profile.d/conda.sh
#conda init RawByteClf


for l in raw dis dec; do

    for a in bpe uni; do
        
        for v in 1024 4096 16384; do

            logfile="./logs/tok-$l-$a-$v.log"

            echo -n "Running $l-$a-$v..."

            systemd-run --scope -p MemoryLimit=96G /home/lk3591/anaconda3/envs/RawByteClf/bin/python \
                src/tokenization/train.py \
                --lift_level "$l" \
                --algorithm "$a" \
                --vocab_size "$v" \
                --num_files=4096 \
                --batch_size=1024 \
                --block_size=1024 \
                --max_token_length=16 &> "$logfile"

            echo "Done. Status: $?"

        done

    done

done
