#!/bin/bash -l

#SBATCH --job-name=pretrain
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=01-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1

if [ $# -eq 0 ]; then
  echo "Please provide the log2 of the vocab size as an argument."
  exit 1
fi

export vocab_size=$((2**$1 + 6))
export steps=500
export strategy="steps"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate RawByteClf

# export CUDA_VISIBLE_DEVICES=0
# export CUDA_LAUNCH_BLOCKING=1 

python \
src/pretrain.py \
--vocab_size=$vocab_size \
--num=1 \
--algorithm="Raw" \
--model="longformer" \
--max_length=10000 \
--scale_numerator=1 \
--scale_denominator=2 \
--early_stopping=false \
--task="mlm" \
--do_train \
--do_eval \
--output_dir=tmp \
--overwrite_output_dir=true \
--load_best_model_at_end=true \
--save_strategy=$strategy \
--save_steps=$steps \
--evaluation_strategy=$strategy \
--eval_steps=$steps \
--logging_steps=$steps \
--dataloader_num_workers=14 \
--num_train_epochs=50 \
--per_device_train_batch_size=16 \
--per_device_eval_batch_size=64 \
--optim="adamw_torch" \
--learning_rate="5e-5" \
--weight_decay=0.01 \
--group_by_length=true \
--save_total_limit=5 \
--fp16=true \
--tf32=true
