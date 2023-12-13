#!/bin/bash -l

#SBATCH --job-name=train_final
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=03-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1

source ~/anaconda3/etc/profile.d/conda.sh
conda activate RawByteClf

python \
src/learn/train.py \
--root="./output" \
--task="clf" \
--depth=4 \
--do_train \
--do_eval \
--output_dir=tmp \
--overwrite_output_dir \
--load_best_model_at_end \
--save_strategy="epoch" \
--evaluation_strategy="epoch" \
--logging_steps=10 \
--dataloader_num_workers=2 \
--num_train_epochs=50 \
--optim="adamw_torch" \
--learning_rate="5e-4" \
--weight_decay=0.005 \
--warmup_steps=100 \
--save_total_limit=5 \
--fp16 \
--fp16_full_eval \
--tf32=true \
--model_name_or_path="longformer" \
--max_length=16384 \
--ft_freeze_positional_embeddings=false \
--ft_duplicate_positional_embeddings=false \
--ft_initialize_positional_embeddings=false \
--per_device_train_batch_size=16 \
--per_device_eval_batch_size=64 \
--gradient_accumulation_steps=12 \
--eval_accumulation_steps=16
