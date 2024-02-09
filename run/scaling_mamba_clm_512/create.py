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
--arch_config='{"d_model": D_MODEL, "n_layer": N_LAYER, "mlp_hidden_size": 512}' \\
--metric_for_best_model="eval_loss" \\
--task="clm" \\
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
--model_name_or_path="mamba" \\
--max_length=512 \\
--per_device_train_batch_size=2048 \\
--per_device_eval_batch_size=2048 \\
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

CONFIGS = [
    (2, 512, 3526144),
    (2, 768, 7746816),
    (4, 384, 3958656),
    (4, 512, 6916608),
    (4, 768, 15290112),
    (6, 384, 5887104),
    (6, 512, 10307072),
    (6, 768, 22833408),
    (8, 256, 3571968),
    (8, 384, 7815552),
    (8, 512, 13697536),
    (8, 768, 30376704),
    (12, 192, 3071424),
    (12, 256, 5324032),
    (12, 384, 11672448),
    (12, 512, 20478464),
    (12, 768, 45463296),
    (16, 192, 4078272),
    (16, 256, 7076096),
    (16, 384, 15529344),
    (16, 512, 27259392),
    (16, 768, 60549888),
    (24, 192, 6091968),
    (24, 256, 10580224),
    (24, 384, 23243136),
    (24, 512, 40821248),
    (24, 768, 90723072),
    (28, 128, 3298944),
    (28, 192, 7098816),
    (28, 256, 12332288),
    (28, 384, 27100032),
    (28, 512, 47602176),
    (32, 128, 3765376),
    (32, 192, 8105664),
    (32, 256, 14084352),
    (32, 384, 30956928),
    (32, 512, 54383104),
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


OUTPUT = Path("run/scaling_mamba_clm_512/")
for n_layer, d_model, params in CONFIGS:
    jobname = f"scl_{n_layer}-{d_model}"
    text = BODY \
        .replace("JOB_NAME", jobname) \
        .replace("D_MODEL", str(d_model)) \
        .replace("N_LAYER", str(n_layer)) \
        .replace("DD-HH", dd_hh(params))

    with open(OUTPUT / f"{jobname}.sh", "w") as fp:
        fp.write(text)
