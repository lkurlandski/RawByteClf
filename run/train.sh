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

if [ $# -eq 0 ]; then
  echo "Please provide the log2 of the vocab size as an argument."
  exit 1
fi

export vocab_size=$((2**$1 + 6))
export steps=100
export strategy="epoch"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate RawByteClf

# export CUDA_VISIBLE_DEVICES=0
# export CUDA_LAUNCH_BLOCKING=1

#torchrun --standalone --nnodes=1 --nproc_per_node=1 \
python \
src/train.py \
--dataset_name="all" \
--vocab_size=$vocab_size \
--num_tok=1000 \
--num=1 \
--algorithm="Raw" \
--model="longformer" \
--max_length=10000 \
--scale_numerator=2 \
--scale_denominator=3 \
--task="clf" \
--pretrain_task="clf" \
--early_stopping \
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
--dataloader_num_workers=14 \
--num_train_epochs=25 \
--per_device_train_batch_size=8 \
--per_device_eval_batch_size=16 \
--gradient_accumulation_steps=16 \
--optim="adamw_torch" \
--learning_rate="5e-5" \
--weight_decay=0.01 \
--group_by_length \
--save_total_limit=5 \
--fp16 \
--auto_find_batch_size \
--tf32=true  # REQUIRES =true
