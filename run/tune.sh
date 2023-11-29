#!/bin/bash -l

#SBATCH --job-name=tune
#SBATCH --account=admalware
#SBATCH --partition=debug
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=01-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks=3
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1

source ~/anaconda3/etc/profile.d/conda.sh
conda activate RawByteClf


# export CUDA_VISIBLE_DEVICES=1
# export CUDA_LAUNCH_BLOCKING=1
# export TRANSFORMERS_VERBOSITY="info"


python \
src/learn/train.py \
--root="./output" \
--model_name_or_path="longformer" \
--max_length=1024 \
--task="mlm" \
--do_train=false \
--do_eval=false \
--do_tune \
--output_dir=tmp \
--overwrite_output_dir \
--dataloader_num_workers=2 \
--evaluation_strategy="steps" \
--eval_steps=100 \
--num_train_epochs=2 \
--per_device_train_batch_size=8 \
--per_device_eval_batch_size=8 \
--gradient_accumulation_steps=32 \
--optim="adamw_torch" \
--fp16 \
--fp16_full_eval \
--tf32=true \
--group_by_length
