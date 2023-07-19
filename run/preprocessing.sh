#!/bin/bash -l

#SBATCH --job-name=preprocessing
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./slurm/%x_%j.out
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=64G


export ntasks=1
export algorithm="SentencePieceBPE"
export vocab_size=$((2**14+6))

source ~/anaconda3/etc/profile.d/conda.sh
conda activate RawByteClf

python src/preprocessing.py \
--tokenizer_file="./output/tokenizers/$algorithm/$vocab_size/vocab.json" \
--num_files=1000 \
--num_proc=$ntasks
