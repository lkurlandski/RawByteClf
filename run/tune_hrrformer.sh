#!/bin/bash -l

#SBATCH --job-name=tune
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=05-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks=9
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:4

source ~/anaconda3/etc/profile.d/conda.sh
conda activate RawByteClf


# export CUDA_VISIBLE_DEVICES=1
# export CUDA_LAUNCH_BLOCKING=1
# export TRANSFORMERS_VERBOSITY="info"


python -u \
src/learn/train.py \
--root="./output" \
--model_name_or_path="hrrformer" \
--max_length=16384 \
--task="mlm" \
--do_train=false \
--do_eval=false \
--do_tune \
--output_dir=tmp \
--overwrite_output_dir \
--dataloader_num_workers=1 \
--evaluation_strategy="steps" \
--per_device_train_batch_size=8 \
--per_device_eval_batch_size=8 \
--gradient_accumulation_steps=32 \
--optim="adamw_torch" \
--eval_steps=50 \
--max_steps=100 \
--fp16 \
--tf32=true \
--group_by_length
