#!/bin/bash -l

# Weight Analysis
# ---------------
# > bash run/hrr_weight_analysis_runner.sh
# or with arguments
# > bash run/hrr_weight_analysis_runner.sh {learning_rate} {weight_decay} {max_grad_norm} {warmup_ratio}

# Command line arguments. Default values are transformers defaults.
learning_rate=${1:-"5e-5"}
weight_decay=${2:-"0.0"}
max_grad_norm=${3:-"1.0"}
warmup_ratio=${4:-"0.0"}

# Environment variables
export ANALYSIS_ID="${learning_rate}/${weight_decay}/${max_grad_norm}/${warmup_ratio}"
export CUDA_VISIBLE_DEVICES=0
source ~/anaconda3/etc/profile.d/conda.sh
conda activate RawByteClf

echo "--------------------------------------------------------------------------------"
echo "Learning Rate: $learning_rate"
echo "Weight Decay: $weight_decay"
echo "Max Grad Norm: $max_grad_norm"
echo "Warmup Ratio: $warmup_ratio"
echo "ANALYSIS_ID: ${ANALYSIS_ID}"
echo "--------------------------------------------------------------------------------"
python \
src/learn/train.py \
--root="./output/weight_analysis/${ANALYSIS_ID}" \
--task="clf" \
--depth=4 \
--do_train \
--output_dir=tmp \
--overwrite_output_dir \
--save_strategy="steps" \
--save_steps=100 \
--evaluation_strategy="steps" \
--eval_steps=1000 \
--logging_steps=1 \
--dataloader_num_workers=2 \
--save_total_limit=3 \
--model_name_or_path="hrrformer" \
--max_length=65536 \
--ft_freeze_positional_embeddings=false \
--ft_duplicate_positional_embeddings=false \
--ft_initialize_positional_embeddings=false \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=16 \
--fp16 \
--max_steps=500 \
--optim="adamw_torch" \
--learning_rate=$learning_rate \
--weight_decay=$weight_decay \
--warmup_ratio=$warmup_ratio \
--max_grad_norm=$max_grad_norm
echo "--------------------------------------------------------------------------------"
