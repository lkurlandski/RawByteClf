#!/bin/bash -l

#SBATCH --job-name=hrrformer_pretrain
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=05-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks=5
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:2


source ~/anaconda3/etc/profile.d/conda.sh
conda activate RawByteClf

# export CUDA_VISIBLE_DEVICES=1
# export CUDA_LAUNCH_BLOCKING=1
# export TRANSFORMERS_VERBOSITY="info"

# torchrun --no-python --nnodes=1 --nproc_per_node=2 \
python \
src/learn/train.py \
--root="./output" \
--task="mlm" \
--streaming=true \
--depth=1 \
--do_train \
--do_eval \
--output_dir=tmp \
--overwrite_output_dir \
--save_strategy="steps" \
--evaluation_strategy="steps" \
--save_steps=100 \
--eval_steps=100 \
--logging_steps=10 \
--max_steps=10000 \
--dataloader_num_workers=4 \
--optim="adamw_torch" \
--learning_rate="1e-4" \
--weight_decay=0.01 \
--max_grad_norm=0.75 \
--save_total_limit=3 \
--tf32=true \
--fp16 \
--fp16_full_eval \
--model_name_or_path="hrrformer" \
--max_length=65536 \
--ft_freeze_positional_embeddings=false \
--ft_duplicate_positional_embeddings=false \
--ft_initialize_positional_embeddings=false \
--per_device_train_batch_size=8 \
--per_device_eval_batch_size=8 \
--gradient_accumulation_steps=16 \
--eval_accumulation_steps=8
