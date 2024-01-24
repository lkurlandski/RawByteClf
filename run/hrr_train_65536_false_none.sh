#!/bin/bash -l

#SBATCH --job-name=top10_hrr_16384
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=01-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=5
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1

source ~/anaconda3/etc/profile.d/conda.sh
conda activate RawByteClf

# torchrun --no-python --nnodes=1 --nproc_per_node=2 \
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
--max_length=16384 \
--ft_freeze_positional_embeddings=false \
--ft_duplicate_positional_embeddings=false \
--ft_initialize_positional_embeddings=false \
--per_device_train_batch_size=32 \
--per_device_eval_batch_size=32 \
--gradient_accumulation_steps=2 \
--eval_accumulation_steps=16 \
--fp16 \
--fp16_full_eval \
--tf32=true