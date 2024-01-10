#!/bin/bash -l

#SBATCH --job-name=malconv
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=05-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=5
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1

source ~/anaconda3/etc/profile.d/conda.sh
conda activate RawByteClf

# export CUDA_VISIBLE_DEVICES=1
# export CUDA_LAUNCH_BLOCKING=1
# export TRANSFORMERS_VERBOSITY="info"

strategy="epoch"
save_eval_steps=100
logging_steps=10

python \
src/learn/train.py \
--root="./output" \
--model_name_or_path="mymalconv" \
--max_length=1048576 \
--task="clf" \
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
--dataloader_num_workers=4 \
--per_device_train_batch_size=192 \
--per_device_eval_batch_size=192 \
--num_train_epochs=50 \
--optim="adamw_torch" \
--learning_rate="5e-4" \
--save_total_limit=3 \
--fp16 \
--fp16_full_eval \
--tf32=true
