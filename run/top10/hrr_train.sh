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
--root="./output/top10" \
--task="clf" \
--streaming=false \
--depth=1 \
--do_train \
--output_dir=tmp \
--overwrite_output_dir=true \
--save_strategy="epoch" \
--evaluation_strategy="epoch" \
--logging_steps=10 \
--dataloader_num_workers=4 \
--num_train_epochs=25 \
--optim="adamw_torch" \
--learning_rate="1e-4" \
--lr_scheduler_type="inverse_sqrt" \
--warmup_steps=100 \
--weight_decay=0.01 \
--max_grad_norm=1.0 \
--save_total_limit=3 \
--model_name_or_path="hrrformer" \
--max_length=4096 \
--ft_freeze_positional_embeddings=false \
--ft_duplicate_positional_embeddings=false \
--ft_initialize_positional_embeddings=false \
--per_device_train_batch_size=64 \
--per_device_eval_batch_size=64 \
--gradient_accumulation_steps=1 \
--eval_accumulation_steps=16 \
--fp16 \
--fp16_full_eval