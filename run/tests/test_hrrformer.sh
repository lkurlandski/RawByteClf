#!/bin/bash -l

source ~/anaconda3/etc/profile.d/conda.sh
conda activate RawByteClf
conda activate RawByteClf2

DATALOADER_NUM_WORKERS=$1
STREAMING=$2

MODELNAME="hrrformer"
BATCHSIZE="512"
FINETUNECHECKPOINT="./output/test/hrrformer/512/mlm/1/hidden_size--512/intermediate_size--2048/num_hidden_layers--1/num_attention_heads--8/superposition_scale_factor--1.0/tensor_logging--True/attention_score_scale_factor--1.0/norm--forward/per_device_train_batch_size--512/gradient_accumulation_steps--1/learning_rate--0.0001/weight_decay--0.01/adam_beta1--0.9/adam_beta2--0.999/adam_epsilon--1e-08/max_grad_norm--0.5/lr_scheduler_type--linear/warmup_ratio--0.05/bf16--False/fp16--True/tf32--None/optim--adamw_torch/checkpoints/checkpoint-6/"

test="Training $MODELNAME for clf"
echo $test
python \
src/learn/train.py \
--root="./output/test" \
--arch_config_file="./config/tests/$MODELNAME.json" \
--task="clf" \
--streaming=$STREAMING \
--depth=1 \
--bodmas_top_k=10 \
--do_train \
--do_eval \
--output_dir=tmp \
--save_strategy="steps" \
--evaluation_strategy="steps" \
--save_steps=2 \
--eval_steps=2 \
--logging_steps=1 \
--max_steps=4 \
--dataloader_num_workers=$DATALOADER_NUM_WORKERS \
--optim="adamw_torch" \
--learning_rate="1e-4" \
--lr_scheduler_type="linear" \
--warmup_ratio=0.05 \
--weight_decay=0.01 \
--max_grad_norm=0.5 \
--save_total_limit=3 \
--metric_for_best_model="eval_loss" \
--model_name_or_path=$MODELNAME \
--max_length=512 \
--ft_freeze_positional_embeddings=false \
--ft_duplicate_positional_embeddings=false \
--ft_initialize_positional_embeddings=false \
--per_device_train_batch_size=$BATCHSIZE \
--per_device_eval_batch_size=$BATCHSIZE \
--fp16 \
--fp16_full_eval
if [ $? -ne 0 ]; then
    echo "Error: $test"
    exit 1
fi

test="Resume training $MODELNAME for clf"
echo $test
python \
src/learn/train.py \
--root="./output/test" \
--arch_config_file="./config/tests/$MODELNAME.json" \
--resume_from_checkpoint=true \
--task="clf" \
--streaming=$STREAMING \
--depth=1 \
--bodmas_top_k=10 \
--do_train \
--do_eval \
--output_dir=tmp \
--save_strategy="steps" \
--evaluation_strategy="steps" \
--save_steps=2 \
--eval_steps=2 \
--logging_steps=1 \
--max_steps=6 \
--dataloader_num_workers=$DATALOADER_NUM_WORKERS \
--optim="adamw_torch" \
--learning_rate="1e-4" \
--lr_scheduler_type="linear" \
--warmup_ratio=0.05 \
--weight_decay=0.01 \
--max_grad_norm=0.5 \
--save_total_limit=3 \
--metric_for_best_model="eval_loss" \
--model_name_or_path=$MODELNAME \
--max_length=512 \
--ft_freeze_positional_embeddings=false \
--ft_duplicate_positional_embeddings=false \
--ft_initialize_positional_embeddings=false \
--per_device_train_batch_size=$BATCHSIZE \
--per_device_eval_batch_size=$BATCHSIZE \
--fp16 \
--fp16_full_eval
if [ $? -ne 0 ]; then
    echo "Error: $test"
    exit 2
fi

test="Training $MODELNAME for mlm"
echo $test
python \
src/learn/train.py \
--root="./output/test" \
--arch_config_file="./config/tests/$MODELNAME.json" \
--task="mlm" \
--streaming=$STREAMING \
--subset=1000 \
--depth=1 \
--do_train \
--do_eval \
--output_dir=tmp \
--save_strategy="steps" \
--evaluation_strategy="steps" \
--save_steps=2 \
--eval_steps=2 \
--logging_steps=1 \
--max_steps=4 \
--dataloader_num_workers=$DATALOADER_NUM_WORKERS \
--optim="adamw_torch" \
--learning_rate="1e-4" \
--lr_scheduler_type="linear" \
--warmup_ratio=0.05 \
--weight_decay=0.01 \
--max_grad_norm=0.5 \
--save_total_limit=3 \
--metric_for_best_model="eval_loss" \
--model_name_or_path=$MODELNAME \
--max_length=512 \
--ft_freeze_positional_embeddings=false \
--ft_duplicate_positional_embeddings=false \
--ft_initialize_positional_embeddings=false \
--per_device_train_batch_size=$BATCHSIZE \
--per_device_eval_batch_size=$BATCHSIZE \
--fp16 \
--fp16_full_eval
if [ $? -ne 0 ]; then
    echo "Error: $test"
    exit 3
fi

test="Resume training $MODELNAME for mlm"
echo $test
python \
src/learn/train.py \
--root="./output/test" \
--arch_config_file="./config/tests/$MODELNAME.json" \
--resume_from_checkpoint=true \
--subset=1000 \
--task="mlm" \
--streaming=$STREAMING \
--depth=1 \
--do_train \
--do_eval \
--output_dir=tmp \
--save_strategy="steps" \
--evaluation_strategy="steps" \
--save_steps=2 \
--eval_steps=2 \
--logging_steps=1 \
--max_steps=6 \
--dataloader_num_workers=$DATALOADER_NUM_WORKERS \
--optim="adamw_torch" \
--learning_rate="1e-4" \
--lr_scheduler_type="linear" \
--warmup_ratio=0.05 \
--weight_decay=0.01 \
--max_grad_norm=0.5 \
--save_total_limit=3 \
--metric_for_best_model="eval_loss" \
--model_name_or_path=$MODELNAME \
--max_length=512 \
--ft_freeze_positional_embeddings=false \
--ft_duplicate_positional_embeddings=false \
--ft_initialize_positional_embeddings=false \
--per_device_train_batch_size=$BATCHSIZE \
--per_device_eval_batch_size=$BATCHSIZE \
--fp16 \
--fp16_full_eval
if [ $? -ne 0 ]; then
    echo "Error: $test"
    exit 4
fi

test="Fine-tuning $MODELNAME for clf"
echo $test
python \
src/learn/train.py \
--root="./output/test" \
--arch_config_file="./config/tests/$MODELNAME.json" \
--task="clf" \
--streaming=$STREAMING \
--depth=1 \
--bodmas_top_k=10 \
--do_train \
--do_eval \
--output_dir=tmp \
--save_strategy="steps" \
--evaluation_strategy="steps" \
--save_steps=2 \
--eval_steps=2 \
--logging_steps=1 \
--max_steps=4 \
--dataloader_num_workers=$DATALOADER_NUM_WORKERS \
--optim="adamw_torch" \
--learning_rate="1e-4" \
--lr_scheduler_type="linear" \
--warmup_ratio=0.05 \
--weight_decay=0.01 \
--max_grad_norm=0.5 \
--save_total_limit=3 \
--metric_for_best_model="eval_loss" \
--model_name_or_path=$FINETUNECHECKPOINT \
--max_length=512 \
--ft_freeze_positional_embeddings=false \
--ft_duplicate_positional_embeddings=false \
--ft_initialize_positional_embeddings=false \
--per_device_train_batch_size=$BATCHSIZE \
--per_device_eval_batch_size=$BATCHSIZE \
--fp16 \
--fp16_full_eval
if [ $? -ne 0 ]; then
    echo "Error: $test"
    exit 5
fi
