#!/bin/bash -l

#SBATCH --job-name=preprocessing
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=64G

if [ $# -eq 0 ]; then
  echo "Please provide the log2 of the vocab size as an argument."
  exit 1
fi

export vocab_size=$((2**$1 + 6))

export vocab_size=4108

source ~/anaconda3/etc/profile.d/conda.sh
conda activate RawByteClf

python src/preprocessing.py \
--max_length=10000 \
--algorithm="SentencePieceBPE" \
--vocab_size=$vocab_size \
--num_proc=1 \
--writer_batch_size=512 \
--batch_size=512 \
--shardsize=512
