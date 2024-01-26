#!/bin/bash -l

#SBATCH --job-name=top10_hrr_131072
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=02-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=5
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1

source ~/anaconda3/etc/profile.d/conda.sh
conda activate RawByteClf

# README:
# Balance gradient accumulation steps to make logical batch size 64.
# Per-device batch sizes are:
# T = 131072 -- 4
# T = 65536 -- 8
# T = 32768 -- 16
# T = 16384 -- 32
# T = 8192 -- 64

python \
src/learn/train.py \
--root="./output/" \
--tail="linear" \
--arch_config_file="./config/hrrformer.json" \
--task="mlm" \
--streaming=false \
--depth=1 \
--do_train \
--output_dir=tmp \
--save_strategy="steps" \
--evaluation_strategy="steps" \
--save_steps=100 \
--eval_steps=100 \
--logging_steps=10 \
--dataloader_num_workers=4 \
--max_steps=1000 \
--optim="adamw_torch" \
--learning_rate="1e-4" \
--lr_scheduler_type="linear" \
--warmup_ratio=0.05 \
--weight_decay=0.01 \
--max_grad_norm=0.5 \
--save_total_limit=3 \
--model_name_or_path="hrrformer" \
--max_length=512 \
--ft_freeze_positional_embeddings=false \
--ft_duplicate_positional_embeddings=false \
--ft_initialize_positional_embeddings=false \
--per_device_train_batch_size=512 \
--per_device_eval_batch_size=512 \
--gradient_accumulation_steps=1 \
--fp16 \
--fp16_full_eval \
--debug=underflow_overflow #\
#--resume_from_checkpoint="/home/lk3591/Documents/code/RawByteClf/output/hrrformer/512/mlm/1/checkpoints/checkpoint-700" # \
# --tf32=true --eval_accumulation_steps=16  \   \