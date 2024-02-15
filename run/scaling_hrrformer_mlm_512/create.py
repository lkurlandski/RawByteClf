import math
from pathlib import Path


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
--arch_config='{"hidden_size": HIDDEN_SIZE, "intermediate_size": INTERMEDIATE_SIZE, "num_hidden_layers": NUM_HIDDEN_LAYERS, "norm": "backward", "num_attention_heads": 8, "superposition_scale_factor": 1.0, "tensor_logging": false, "attention_score_scale_factor": 1.0}' \\
--metric_for_best_model="eval_loss" \\
--task="mlm" \\
--streaming=false \\
--vl_size=50000 \\
--ts_size=50000 \\
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
--model_name_or_path="hrrformer" \\
--max_length=512 \\
--per_device_train_batch_size=2048 \\
--per_device_eval_batch_size=2048 \\
--gradient_accumulation_steps=1 \\
--load_best_model_at_end \\
--early_stopping=true \\
--early_stopping_patience=1 \\
--early_stopping_threshold=0 \\
--auto_find_batch_size_and_gradient_accumulation_steps \\
--tf32=true
"""


CONFIGS = [
    (1, 512, 3815688),
    (1, 768, 8279304),
    (1, 1024, 14446856),
    (1, 2048, 56156424),
    (2, 512, 6968072),
    (2, 768, 15367176),
    (2, 1024, 27043080),
    (4, 256, 3425288),
    (4, 512, 13272840),
    (4, 768, 29542920),
    (4, 1024, 52235528),
    (6, 256, 5004808),
    (6, 512, 19577608),
    (6, 768, 43718664),
    (6, 1024, 77427976),
    (8, 256, 6584328),
    (8, 512, 25882376),
    (8, 768, 57894408),
    (10, 256, 8163848),
    (10, 512, 32187144),
    (10, 768, 72070152),
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


OUTPUT = Path("run/scaling_hrrformer_mlm_512/")
for num_hidden_layers, hidden_size, params in CONFIGS:
    intermediate_size = hidden_size * 4
    jobname = f"scl_hrr_{num_hidden_layers}-{hidden_size}"
    text = BODY \
        .replace("JOB_NAME", jobname) \
        .replace("HIDDEN_SIZE", str(hidden_size)) \
        .replace("INTERMEDIATE_SIZE", str(intermediate_size)) \
        .replace("NUM_HIDDEN_LAYERS", str(num_hidden_layers)) \
        .replace("DD-HH", dd_hh(params))

    with open(OUTPUT / f"{jobname}.sh", "w") as fp:
        fp.write(text)
