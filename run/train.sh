#!/bin/bash -l

#SBATCH --job-name=train
#SBATCH --account=admalware
#SBATCH --partition=debug
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1


export ntasks=4
export algorithm="SentencePieceBPE"
export vocab_size=$((2**$1+6))

source ~/anaconda3/etc/profile.d/conda.sh
conda activate RawByteClf

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 python src/train.py \
--vocab_size=$vocab_size \
--num=0.1 \
--algorithm="SentencePieceBPE" \
--model=longformer \
--max_length=1000000 \
--scale=12 \
--do_train \
--output_dir=tmp \
--overwrite_output_dir=true \
--load_best_model_at_end=true \
--save_strategy="epoch" \
--evaluation_strategy="epoch" \
--dataloader_num_workers=$ntasks \
--num_train_epochs=2 \
--per_device_train_batch_size=1 \
--per_device_eval_batch_size=1 \
--tf32=false
