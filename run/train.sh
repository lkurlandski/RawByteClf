#!/bin/bash -l

#SBATCH --job-name=train
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=01-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks=3
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1

source ~/anaconda3/etc/profile.d/conda.sh
conda activate RawByteClf

# export CUDA_VISIBLE_DEVICES=1
# export CUDA_LAUNCH_BLOCKING=1
# export TRANSFORMERS_VERBOSITY="info"

strategy="epoch"
save_eval_steps=100
logging_steps=10

torchrun --no-python --standalone --nnodes=1 --nproc_per_node=4 \
python \
src/learn/train.py \
--root="./output_test_overfitting" \  # FIXME: select output directory!!!
--model_name_or_path="longformer" \
--max_length=16384 \
--task="mlm" \
--do_train \
--do_eval \
--output_dir=tmp \
--overwrite_output_dir \
--load_best_model_at_end \
--save_strategy=$strategy \
--save_steps=$save_eval_steps \
--evaluation_strategy=$strategy \
--eval_steps=$save_eval_steps \
--logging_steps=$logging_steps \
--dataloader_num_workers=2 \
--per_device_train_batch_size=8 \
--per_device_eval_batch_size=8 \
--gradient_accumulation_steps=32 \
--max_steps=10000 \
--optim="adamw_torch" \
--learning_rate="1e-4" \
--weight_decay=0.01 \
--warmup_steps=500 \
--save_total_limit=5 \
--fp16 \
--fp16_full_eval \
--tf32=true
#--optim="adamw_8bit"
#--fsdp="shard_grad_op"
#--group_by_length
# REQUIRES =true
