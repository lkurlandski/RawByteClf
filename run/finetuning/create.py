from dataclasses import dataclass
import math
import os
from pathlib import Path
from pprint import pformat
from typing import Optional


ROOT = "./output/finetuning"
MODEL_NAME_OR_PATH = "mamba"
ARCH_CONFIG = '{"mode": "bi", "d_model": 128, "n_layer": 6, "mlp_hidden_size": 512}'
PER_DEVICE_TRAIN_BATCH_SIZE = 16
PER_DEVICE_EVAL_BATCH_SIZE = 32
REPRESENTATION = 8
ALGORITHM = "Raw"
VOCAB_SIZE = 256
MAX_LENGTH = 2 ** 14
DATA_READ_BYTES = 2 ** 14
TOP_K = 10
BF_OR_FP = "fp"
TF32 = "false"


BODY_CLM = f"""#!/bin/bash -l

#SBATCH --job-name=ftCLM
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=01-00:00:00
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
--metric_for_best_model="eval_loss" \\
--task="clm" \\
--streaming=true \\
--skip_eval_check=true \\
--dataset_backend="HF" \\
--representation={REPRESENTATION} \\
--algorithm={ALGORITHM} \\
--vocab_size={VOCAB_SIZE} \\
--tr_size=1000000 \\
--vl_size=10000 \\
--ts_size=0 \\
--do_train \\
--output_dir=tmp \\
--save_strategy="steps" \\
--evaluation_strategy="steps" \\
--num_train_epochs=1 \\
--logging_steps=10 \\
--save_steps=100 \\
--eval_steps=100 \\
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
--gradient_accumulation_steps=16 \\
--load_best_model_at_end \\
--early_stopping=false \\
--auto_find_batch_size_and_gradient_accumulation_steps \\
--{BF_OR_FP}16 \\
--{BF_OR_FP}16_full_eval \\
--tf32={TF32}
"""


BODY_CLF = f"""#!/bin/bash -l

#SBATCH --job-name=ftCLF
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=01-00:00:00
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
--task="clf" \\
--streaming=false \\
--skip_eval_check=false \\
--top_k={TOP_K} \\
--dataset_backend="HF" \\
--representation={REPRESENTATION} \\
--algorithm={ALGORITHM} \\
--vocab_size={VOCAB_SIZE} \\
--tr_size=0.85 \\
--vl_size=0.15 \\
--ts_size=0.0 \\
--do_train \\
--output_dir=tmp \\
--save_strategy="epoch" \\
--evaluation_strategy="epoch" \\
--num_train_epochs=10 \\
--logging_steps=10 \\
--save_steps=100 \\
--eval_steps=100 \\
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
--gradient_accumulation_steps=1 \\
--load_best_model_at_end \\
--early_stopping=false \\
--auto_find_batch_size_and_gradient_accumulation_steps \\
--{BF_OR_FP}16 \\
--{BF_OR_FP}16_full_eval \\
--tf32={TF32}
"""


# BODY_FT = BODY_CLF.replace("ftCLF", "ftFT").replace("mamba", "")


OUTPUT = Path(os.path.realpath(__file__)).parent

with open(OUTPUT / "clm.sh", "w") as f:
    f.write(BODY_CLM)
with open(OUTPUT / "clf.sh", "w") as f:
    f.write(BODY_CLF)
# with open("ft.sh", "w") as f:
#     ...
