#!/bin/bash -l

#SBATCH --job-name=train
#SBATCH --account=admalware
#SBATCH --partition=debug
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1

if [ $# -eq 0 ]; then
  echo "Please provide the log2 of the vocab size as an argument."
  exit 1
fi

export vocab_size=$((2**$1 + 6))

source ~/anaconda3/etc/profile.d/conda.sh
conda activate RawByteClf

# export CUDA_VISIBLE_DEVICES=0
# export CUDA_LAUNCH_BLOCKING=1 

python src/train.py \
--vocab_size=$vocab_size \
--num=0.1 \
--algorithm="Raw" \
--model="malconv" \
--max_length=10000 \
--scale=.25 \
--early_stopping=false \
--do_train \
--do_eval \
--output_dir=tmp \
--overwrite_output_dir=true \
--load_best_model_at_end=true \
--save_strategy="steps" \
--save_steps=10 \
--evaluation_strategy="steps" \
--eval_steps=10 \
--logging_steps=10 \
--dataloader_num_workers=-1 \
--num_train_epochs=1000 \
--per_device_train_batch_size=64 \
--per_device_eval_batch_size=64 \
--optim="adamw_torch" \
--learning_rate="5e-5" \
--group_by_length=true \
--save_total_limit=5 \
--fp16=true
# --tf32=false
