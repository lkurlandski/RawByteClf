#!/bin/bash -l

#SBATCH --job-name=mamba_clm_8192
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=01-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=5
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1

source ~/anaconda3/etc/profile.d/conda.sh
conda activate RawByteClf
conda activate RawByteClf2
module unload blindfold

python \
src/learn/train.py \
--root="./output" \
--arch_config_file="./config/mamba.json" \
--metric_for_best_model="eval_loss" \
--task="clm" \
--streaming=false \
--vl_size=16384 \
--ts_size=16384 \
--depth=1 \
--do_train \
--do_eval \
--output_dir=tmp \
--save_strategy="epoch" \
--evaluation_strategy="epoch" \
--logging_steps=10 \
--dataloader_num_workers=4 \
--num_train_epochs=20 \
--optim="adamw_torch" \
--learning_rate="1e-4" \
--lr_scheduler_type="linear" \
--warmup_ratio=0.05 \
--weight_decay=0.05 \
--adam_beta2=0.98 \
--max_grad_norm=0.5 \
--save_total_limit=3 \
--model_name_or_path="mamba" \
--max_length=8192 \
--ft_freeze_positional_embeddings=false \
--ft_duplicate_positional_embeddings=false \
--ft_initialize_positional_embeddings=false \
--per_device_train_batch_size=64 \
--per_device_eval_batch_size=64 \
--gradient_accumulation_steps=64 \
--eval_accumulation_steps=4 \
--bf16 \
--bf16_full_eval \
--tf32=true
