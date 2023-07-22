#!/bin/bash -l

#SBATCH --job-name=train
#SBATCH --account=admalware
#SBATCH --partition=debug
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=16G


export ntasks=1
export algorithm="SentencePieceBPE"
export vocab_size=$((2**14+6))
export num_files=733

source ~/anaconda3/etc/profile.d/conda.sh
conda activate RawByteClf

python src/train.py \
--tokenizer_file="./output/tokenizers/$algorithm/$vocab_size/vocab.json" \
--dataset_path="./output/datasets/$algorithm/$vocab_size/$num_files" \
--model=longformer \
--max_length=100000 \
--scale=2 \
--output_dir="./output/models/$algorithm/$vocab_size" \
--overwrite_output_dir=true \
--load_best_model_at_end=true \
--save_strategy="epoch" \
--evaluation_strategy="epoch"
