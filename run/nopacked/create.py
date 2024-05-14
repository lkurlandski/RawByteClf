"""
"""

import os
from pathlib import Path


SEEDS = [0, 1, 2, 3, 4]


ROOT = "./output/nopacked"
MODEL_NAME_OR_PATH = "mamba"
ARCH_CONFIG = '{"mode": "uni", "num_hidden_layers": 8, "hidden_size": 256, "embedding_size": 256}'
PER_DEVICE_TRAIN_BATCH_SIZE = 64
PER_DEVICE_EVAL_BATCH_SIZE = 64
SEED = "SEED"
MAX_LENGTH = 2 ** 16
DATA_READ_BYTES = 2 ** 16
BF_OR_FP = "fp"
TF32 = "true"
LOGGING_STEPS = 10

LM_TR_SIZE = 2 ** 21  # 2097152
LM_VL_SIZE = 2 ** 14  # 16384
LM_SAVE_EVAL_STEPS = 512
LM_GRADIENT_ACCUMULATION_STEPS = 8

CLF_GRADIENT_ACCUMULATION_STEPS = 1

BODY_LM = f"""#!/bin/bash -l

#SBATCH --job-name=JOB_NAME
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=02-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks={6}
#SBATCH --mem={64}G
#SBATCH --gres=gpu:a100:1


source ~/anaconda3/etc/profile.d/conda.sh
conda activate RawByteClf
conda activate RawByteClf2
module unload blindfold


python -u \\
src/learn/train.py \\
--root="{ROOT}" \\
--arch_config='{ARCH_CONFIG}' \\
--metric_for_best_model="eval_loss" \\
--task="clm" \\
--seed=0 \\
--remove_packed \\
--streaming=true \\
--skip_eval_check=false \\
--dataset_backend="HF" \\
--representation=8 \\
--algorithm="Raw" \\
--vocab_size=256 \\
--tr_size={LM_TR_SIZE} \\
--vl_size={LM_VL_SIZE} \\
--ts_size=0 \\
--do_train \\
--output_dir=tmp \\
--save_strategy="steps" \\
--evaluation_strategy="steps" \\
--num_train_epochs=1 \\
--logging_steps={LOGGING_STEPS} \\
--save_steps={LM_SAVE_EVAL_STEPS} \\
--eval_steps={LM_SAVE_EVAL_STEPS} \\
--dataloader_num_workers={5} \\
--optim="adamw_torch" \\
--learning_rate="1e-3" \\
--lr_scheduler_type="linear" \\
--warmup_ratio=0.05 \\
--weight_decay=0.10 \\
--adam_beta1=0.900 \\
--adam_beta2=0.990 \\
--max_grad_norm=1.0 \\
--save_total_limit=-1 \\
--model_name_or_path={MODEL_NAME_OR_PATH} \\
--max_length={MAX_LENGTH} \\
--data_read_bytes={DATA_READ_BYTES} \\
--per_device_train_batch_size={PER_DEVICE_TRAIN_BATCH_SIZE} \\
--per_device_eval_batch_size={PER_DEVICE_EVAL_BATCH_SIZE} \\
--gradient_accumulation_steps={LM_GRADIENT_ACCUMULATION_STEPS} \\
--load_best_model_at_end \\
--early_stopping=false \\
--auto_find_batch_size_and_gradient_accumulation_steps \\
--{BF_OR_FP}16 \\
--{BF_OR_FP}16_full_eval \\
--tf32={TF32} \\
--gradient_checkpointing=true
"""


BODY_CLF = f"""#!/bin/bash -l

#SBATCH --job-name=JOB_NAME
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=00-6:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks={4}
#SBATCH --mem={64}G
#SBATCH --gres=gpu:a100:1


source ~/anaconda3/etc/profile.d/conda.sh
conda activate RawByteClf
conda activate RawByteClf2
module unload blindfold


python -u \\
src/learn/train.py \\
--root="{ROOT}" \\
--arch_config='{ARCH_CONFIG}' \\
--metric_for_best_model="eval_accuracy" \\
--task="clf-bod" \\
--seed=SEED \\
--pretraining_task=PRETRAINING_TASK \\
--remove_packed \\
--streaming=false \\
--skip_eval_check=true \\
--top_k=10 \\
--dataset_backend="HF" \\
--representation=8 \\
--algorithm="Raw" \\
--vocab_size=256 \\
--tr_size=0.85 \\
--vl_size=0.15 \\
--ts_size=0.0 \\
--do_train \\
--output_dir=tmp \\
--save_strategy="steps" \\
--evaluation_strategy="steps" \\
--num_train_epochs=5 \\
--logging_steps={LOGGING_STEPS} \\
--save_steps=2 \\
--eval_steps=2 \\
--dataloader_num_workers={3} \\
--optim="adamw_torch" \\
--learning_rate="1e-3" \\
--lr_scheduler_type="linear" \\
--warmup_ratio=0.00 \\
--weight_decay=0.01 \\
--adam_beta2=0.999 \\
--max_grad_norm=1.0 \\
--save_total_limit=3 \\
--model_name_or_path={MODEL_NAME_OR_PATH} \\
--max_length={MAX_LENGTH} \\
--data_read_bytes={DATA_READ_BYTES} \\
--per_device_train_batch_size={PER_DEVICE_TRAIN_BATCH_SIZE} \\
--per_device_eval_batch_size={PER_DEVICE_EVAL_BATCH_SIZE} \\
--gradient_accumulation_steps={CLF_GRADIENT_ACCUMULATION_STEPS} \\
--load_best_model_at_end \\
--early_stopping=false \\
--auto_find_batch_size_and_gradient_accumulation_steps \\
--{BF_OR_FP}16 \\
--{BF_OR_FP}16_full_eval \\
--tf32={TF32} \\
--gradient_checkpointing=true
"""


OUTPUT = Path(os.path.realpath(__file__)).parent


body = BODY_LM.replace("JOB_NAME", "nopack-clm.sh")
with open(OUTPUT / "nopack-clm.sh", "w") as fp:
    fp.write(body)


for seed in SEEDS:
    jobname = f"nopack-clf-{seed}"
    body = BODY_CLF \
        .replace("JOB_NAME", jobname) \
        .replace("SEED", str(seed)) \
        .replace("PRETRAINING_TASK", "None")
    with open((OUTPUT / jobname).with_suffix(".sh"), "w") as fp:
        fp.write(body)

    jobname = f"nopack-ft-{seed}"
    body = BODY_CLF \
        .replace("JOB_NAME", jobname) \
        .replace("SEED", str(seed)) \
        .replace("PRETRAINING_TASK", "clm")
    with open((OUTPUT / jobname).with_suffix(".sh"), "w") as fp:
        fp.write(body)

