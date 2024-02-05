#!/bin/bash -l

#SBATCH --job-name=mamba_ftclf_8192
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=00-06:00:00
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
--model_name_or_path="output/mamba/8192/clm/1/d_model--512/n_layer--2/mlp_hidden_size--512/per_device_train_batch_size--64/gradient_accumulation_steps--64/learning_rate--0.0001/weight_decay--0.05/adam_beta1--0.9/adam_beta2--0.98/adam_epsilon--1e-08/max_grad_norm--0.5/lr_scheduler_type--linear/warmup_ratio--0.05/bf16--True/fp16--False/tf32--True/optim--adamw_torch/checkpoints/checkpoint-2566" \
--max_length=8192 \
--ft_freeze_positional_embeddings=false \
--ft_duplicate_positional_embeddings=false \
--ft_initialize_positional_embeddings=false \
--per_device_train_batch_size=64 \
--per_device_eval_batch_size=64 \
--gradient_accumulation_steps=1 \
--eval_accumulation_steps=8 \
--bf16 \
--bf16_full_eval \
--tf32=true
