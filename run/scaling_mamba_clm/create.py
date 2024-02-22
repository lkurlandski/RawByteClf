import math
import os
from pathlib import Path
from pprint import pformat

BODY = """#!/bin/bash -l

#SBATCH --job-name=JOB_NAME
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=DD-HH:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=3
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1

source ~/anaconda3/etc/profile.d/conda.sh
conda activate RawByteClf
conda activate RawByteClf2
module unload blindfold

python -u \\
src/learn/train.py \\
--root="./output/scaling" \\
--arch_config='{"d_model": D_MODEL, "n_layer": N_LAYER, "mlp_hidden_size": 512}' \\
--metric_for_best_model="eval_loss" \\
--task="clm" \\
--streaming=true \\
--tr_size=7000000 \\
--vl_size=50000 \\
--ts_size=2 \\
--depth=1 \\
--do_train \\
--do_eval \\
--output_dir=tmp \\
--save_strategy="steps" \\
--evaluation_strategy="steps" \\
--num_train_epochs=1 \\
--save_steps=100 \\
--eval_steps=100 \\
--logging_steps=10 \\
--dataloader_num_workers=2 \\
--optim="adamw_torch" \\
--learning_rate="1e-3" \\
--lr_scheduler_type="linear" \\
--warmup_ratio=0.05 \\
--weight_decay=0.01 \\
--adam_beta2=0.98 \\
--max_grad_norm=1.0 \\
--save_total_limit=3 \\
--model_name_or_path="mamba" \\
--max_length=MAX_LENGTH \\
--per_device_train_batch_size=1024 \\
--per_device_eval_batch_size=1024 \\
--gradient_accumulation_steps=1 \\
--load_best_model_at_end \\
--early_stopping=true \\
--early_stopping_patience=1 \\
--early_stopping_threshold=0 \\
--auto_find_batch_size_and_gradient_accumulation_steps \\
--bf16 \\
--bf16_full_eval \\
--tf32=true
"""


MAX_LENGTH = 16384


CONFIGS = [
    (4, 192, 1.057728),
    (4, 256, 1.819904),
    (6, 192, 1.561152),
    (6, 256, 2.695936),
    (6, 384, 5.887104),
    (8, 256, 3.571968),
    (8, 384, 7.815552),
    (8, 512, 13.697536),
    (12, 384, 11.672448),
    (12, 512, 20.478464),
    (12, 768, 45.463296),
    (16, 512, 27.259392),
    (16, 768, 60.549888),
    (24, 768, 90.723072),
]


def dd_hh(p: int) -> str:
    hours = math.ceil((p / 5e6) * 6)  # 6 hours for every 5 million parameters
    days = 0
    while hours >= 24:
        hours -= 24
        days += 1
    if days > 5:
        days = 5
        hours = 0
    return f"0{days}-{hours}"


OUTPUT = Path(os.path.realpath(__file__)).parent


for n_layer, d_model, params in CONFIGS:
    jobname = f"clm_{n_layer}-{d_model}"
    text = BODY \
        .replace("JOB_NAME", jobname) \
        .replace("D_MODEL", str(d_model)) \
        .replace("N_LAYER", str(n_layer)) \
        .replace("DD-HH", "05-00") \
        .replace("MAX_LENGTH", str(MAX_LENGTH))
    with open(OUTPUT / f"{jobname}.sh", "w") as fp:
        fp.write(text)
