#!/bin/bash -l

#SBATCH --job-name=train
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=01-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1

source ~/anaconda3/etc/profile.d/conda.sh
conda activate RawByteClf

# export CUDA_VISIBLE_DEVICES=1
# export CUDA_LAUNCH_BLOCKING=1
# export TRANSFORMERS_VERBOSITY="debug"

strategy="epoch"
steps=10

#torchrun --standalone --nnodes=1 --nproc_per_node=1 \
python \
src/learn/train.py \
--root="./output" \
--model_name_or_path="reformer" \
--max_length=4096 \
--task="clf" \
--do_train \
--do_eval \
--output_dir=tmp \
--overwrite_output_dir \
--load_best_model_at_end \
--save_strategy=$strategy \
--save_steps=$steps \
--evaluation_strategy=$strategy \
--eval_steps=$steps \
--logging_steps=$steps \
--dataloader_num_workers=8 \
--num_train_epochs=200 \
--per_device_train_batch_size=64 \
--per_device_eval_batch_size=64 \
--gradient_accumulation_steps=4 \
--optim="adamw_torch" \
--learning_rate="1e-4" \
--weight_decay=0.01 \
--save_total_limit=5 \
--fp16 \
--auto_find_batch_size
#--group_by_length
#--tf32=true  # REQUIRES =true
