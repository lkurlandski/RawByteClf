#!/bin/bash -l

#SBATCH --job-name=pretrain_hrr_512
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=00-06:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=9
#SBATCH --mem=64G
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
--arch_config_file="./config/hrrformer.json" \
--task="mlm" \
--streaming=false \
--depth=1 \
--do_train \
--output_dir=tmp \
--save_strategy="epoch" \
--evaluation_strategy="epoch" \
--save_steps=2 \
--eval_steps=2 \
--logging_steps=100 \
--dataloader_num_workers=8 \
--num_train_epochs=20 \
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
--per_device_train_batch_size=1536 \
--per_device_eval_batch_size=1536 \
--gradient_accumulation_steps=2 \
--fp16 \
--fp16_full_eval \
--tf32=true 
