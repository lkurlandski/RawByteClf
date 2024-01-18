#!/bin/bash -l

#SBATCH --job-name=hrrformer
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=05-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --ntasks=3
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1

source ~/anaconda3/etc/profile.d/conda.sh
conda activate RawByteClf

python \
src/learn/train.py \
--root="./output" \
--task="clf" \
--streaming=false \
--depth=1 \
--do_train \
--do_eval \
--output_dir=tmp \
--overwrite_output_dir \
--save_strategy="epoch" \
--evaluation_strategy="epoch" \
--logging_steps=10 \
--dataloader_num_workers=4 \
--num_train_epochs=50 \
--optim="adamw_torch" \
--learning_rate="1e-4" \
--weight_decay=0.01 \
--max_grad_norm=0.75 \
--save_total_limit=3 \
--fp16 \
--fp16_full_eval \
--model_name_or_path="hrrformer" \
--max_length=65536 \
--ft_freeze_positional_embeddings=false \
--ft_duplicate_positional_embeddings=false \
--ft_initialize_positional_embeddings=false \
--per_device_train_batch_size=8 \
--per_device_eval_batch_size=12 \
--gradient_accumulation_steps=16 \
--eval_accumulation_steps=32
