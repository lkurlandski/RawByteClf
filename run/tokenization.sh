#!/bin/bash -l

#SBATCH --job-name=tokenization
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --mem=128G


if [ $# -eq 0 ]; then
  echo "Please provide the log2 of the vocab size as an argument."
  exit 1
fi

export vocab_size=$((2**$1))  # tokenization.py requires a log2 vocab_size

source ~/anaconda3/etc/profile.d/conda.sh
conda activate RawByteClf

python src/tokenization.py \
--algorithm="SentencePieceBPE" \
--vocab_size=$vocab_size \
--num_files=4000 \
--block_size=10000 \
--batch_size=250
