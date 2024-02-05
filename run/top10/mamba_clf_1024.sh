#!/bin/bash -l

#SBATCH --job-name=mamba_clf_1024
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=01-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=3
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1

source ~/anaconda3/etc/profile.d/conda.sh
conda activate RawByteClf
conda activate RawByteClf2
module unload blindfold

python \
src/learn/train.py \
--root="./output" \
--task="clf" \
--arch_config_file="./config/mamba.json" \
--metric_for_best_model="eval_loss" \
--streaming=false \
--bodmas_top_k=10 \
--depth=1 \
--do_train \
--do_eval \
--output_dir=tmp \
--overwrite_output_dir=true \
--save_strategy="epoch" \
--evaluation_strategy="epoch" \
--logging_steps=10 \
--dataloader_num_workers=2 \
--num_train_epochs=25 \
--optim="adamw_torch" \
--learning_rate="1e-4" \
--lr_scheduler_type="inverse_sqrt" \
--warmup_ratio=0.05 \
--weight_decay=0.01 \
--max_grad_norm=1.0 \
--save_total_limit=3 \
--model_name_or_path="mamba" \
--max_length=1024 \
--ft_freeze_positional_embeddings=false \
--ft_duplicate_positional_embeddings=false \
--ft_initialize_positional_embeddings=false \
--per_device_train_batch_size=64 \
--per_device_eval_batch_size=64 \
--gradient_accumulation_steps=1 \
--eval_accumulation_steps=8 \
--fp16 \
--fp16_full_eval \
--tf32=true
