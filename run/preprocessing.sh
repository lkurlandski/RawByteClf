#!/bin/bash -l

#SBATCH --job-name=preprocessing
#SBATCH --account=admalware
#SBATCH --partition=debug
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=64G


export ntasks=1
export ncpus=16
export algorithm="SentencePieceBPE"
export vocab_size=$((2**$1 + 6))

source ~/anaconda3/etc/profile.d/conda.sh
conda activate RawByteClf

python src/preprocessing.py \
--algorithm=$algorithm \
--vocab_size=$vocab_size \
--num_proc=1 \
--writer_batch_size=64 \
--batch_size=64
