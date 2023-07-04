export CUDA_VISIBLE_DEVICES="1"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate RawByteClf
torchrun \
--standalone \
--nnodes=1 \
--nproc-per-node 1 \
main.py \
--vocab_size=1024 \
--n_tok=100 \
--use_saved_tokenizer=false \
--overwrite_saved_tokenizer=true \
--max_size=10000 \
--n_dat=1000 \
--min_bytes=1000 \
--max_bytes=1000000 \
--model=longformer \
--downscale=4 \
--do_train \
--do_eval \
--overwrite_output_dir \
--output_dir="./models/longformer/clf" \
--dataloader_num_workers=1 \
--gradient_accumulation_steps=4 \
--per_device_train_batch_size=2 \
--per_device_eval_batch_size=2 \
--num_train_epochs=100 \
--evaluation_strategy="epoch" \
--save_strategy="epoch" \
--save_total_limit=5 \
--load_best_model_at_end=true \
--optim="adamw_torch" \
--group_by_length

