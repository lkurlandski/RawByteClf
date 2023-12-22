#!/bin/bash -l

#SBATCH --job-name=train
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=05-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks=9
#SBATCH --mem=256G
#SBATCH --gres=gpu:a100:4

source ~/anaconda3/etc/profile.d/conda.sh
conda activate RawByteClf

export CUDA_VISIBLE_DEVICES=1
# export CUDA_LAUNCH_BLOCKING=1
# export TRANSFORMERS_VERBOSITY="info"

strategy="epoch"
save_eval_steps=100
logging_steps=10

python \
src/learn/train.py \
--root="./output" \
--model_name_or_path="malconv" \
--max_length=65536 \
--task="clf" \
--do_tune \
--output_dir=tmp \
--overwrite_output_dir \
--load_best_model_at_end \
--save_strategy=$strategy \
--save_steps=$save_eval_steps \
--evaluation_strategy=$strategy \
--eval_steps=$save_eval_steps \
--logging_steps=$logging_steps \
--dataloader_num_workers=2 \
--per_device_train_batch_size=256 \
--per_device_eval_batch_size=256 \
--num_train_epochs=1 \
--optim="adamw_torch" \
--learning_rate="5e-4" \
--save_total_limit=5 \
--fp16

python \
src/learn/train.py \
--root="./output" \
--model_name_or_path="malconvgct" \
--max_length=65536 \
--task="clf" \
--do_tune \
--output_dir=tmp \
--overwrite_output_dir \
--load_best_model_at_end \
--save_strategy=$strategy \
--save_steps=$save_eval_steps \
--evaluation_strategy=$strategy \
--eval_steps=$save_eval_steps \
--logging_steps=$logging_steps \
--dataloader_num_workers=2 \
--per_device_train_batch_size=256 \
--per_device_eval_batch_size=256 \
--num_train_epochs=1 \
--optim="adamw_torch" \
--learning_rate="5e-4" \
--save_total_limit=5 \
--fp16
