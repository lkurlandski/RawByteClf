import math
import os
from pathlib import Path
from pprint import pformat
import sys


# We need some utilities from src to get this going
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data.utils import _select_k_for_each_class
from src.data.loaders_pt import get_bodmas_file_label_map



BODY = """#!/bin/bash -l

#SBATCH --job-name=JOB_NAME
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=DD-HH:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1

source ~/anaconda3/etc/profile.d/conda.sh
conda activate RawByteClf
conda activate RawByteClf2
module unload blindfold


if [ $# -eq 0 ]; then
    echo "Error: No argument supplied."
    exit 1
elif [ "$1" != "clf" ] && [ "$1" != "ftclf" ]; then
    echo "Error: Invalid argument. Argument must be 'clf' or 'ftclf'."
    exit 1
elif [ "$1" = "clf" ]; then
    MODEL_NAME_OR_PATH="mamba"
elif [ "$1" = "ftclf" ]; then
    MODEL_NAME_OR_PATH="PRETRAINED_MODEL_PATH"
fi


python -u \\
src/learn/train.py \\
--root="./output/scaling" \\
--arch_config='{"d_model": D_MODEL, "n_layer": N_LAYER, "mlp_hidden_size": 512}' \\
--metric_for_best_model="eval_f1-macro" \\
--task="clf" \\
--streaming=false \\
--bodmas_min_freq=MIN_FREQ \\
--depth=1 \\
--do_train \\
--output_dir=tmp \\
--save_strategy="steps" \\
--evaluation_strategy="steps" \\
--save_steps=50 \\
--eval_steps=50 \\
--max_steps=30000 \\
--logging_steps=10 \\
--dataloader_num_workers=0 \\
--optim="adamw_torch" \\
--learning_rate="1e-3" \\
--lr_scheduler_type="linear" \\
--warmup_ratio=0.00 \\
--weight_decay=0.01 \\
--adam_beta2=0.999 \\
--max_grad_norm=1.0 \\
--save_total_limit=3 \\
--model_name_or_path=$MODEL_NAME_OR_PATH \\
--max_length=512 \\
--per_device_train_batch_size=128 \\
--per_device_eval_batch_size=2048 \\
--gradient_accumulation_steps=1 \\
--load_best_model_at_end \\
--early_stopping=false \\
--early_stopping_patience=10 \\
--early_stopping_threshold=0 \\
--bf16 \\
--bf16_full_eval \\
--tf32=true
"""

# n_layers, d_model, n_params, time DD:HH
CONFIGS = [
    (8, 384,   7815552, "06-00"),
    (12, 384, 11672448, "06-00"),
    (12, 512, 20478464, "12-00"),
    (16, 512, 27259392, "12-00"),
    (24, 512, 40821248, "01-00"),
    (24, 768, 90723072, "01-00"),
]


MIN_FREQS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9,
    10, 12, 14, 16, 18, 20,
    24, 28, 32, 36, 40,
    45, 50, 55, 60,
    70, 80, 90, 100,
    125, 150, 175, 200,
    250, 300, 350, 400, 450, 500,
]


OUTPUT = Path(os.path.realpath(__file__)).parent
PRETRAINED_MODELS_ROOT = Path("/home/lk3591/Documents/code/RawByteClf/output/scaling/mamba/512/clm/3000000/1/")


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


def get_pretrained_model_path(d_model: int, n_layer: int) -> Path:
    root = PRETRAINED_MODELS_ROOT / f"d_model--{d_model}/n_layer--{n_layer}/mlp_hidden_size--512"
    pretrained_model_path = None
    pretrained_model_paths = list(root.rglob("checkpoints"))
    pretrained_model_paths = [p for p in pretrained_model_paths if "clf" not in p.as_posix()]

    if len(pretrained_model_paths) > 2:
        raise FileExistsError(root)
    if len(pretrained_model_paths) == 0:
        raise FileNotFoundError(root)
    if len(pretrained_model_paths) == 1:
        pretrained_model_path = pretrained_model_paths[0]
    if len(pretrained_model_paths) == 2:
        if "clf" not in str(pretrained_model_paths[0]):
            pretrained_model_path = pretrained_model_paths[0]
        elif "clf" not in str(pretrained_model_paths[1]):
            pretrained_model_path = pretrained_model_paths[1]

    pretrained_model_path: Path = pretrained_model_path / "checkpoint-1400"
    return pretrained_model_path


outfiles = []

for n_layer, d_model, params, req_time in CONFIGS:
    for min_freq in MIN_FREQS:
        jobname = f"tr_fq_bl_{min_freq}_{n_layer}-{d_model}"
        print(jobname)

        pretrained_model_path = get_pretrained_model_path(d_model, n_layer)
        assert pretrained_model_path.exists(), pretrained_model_path

        # must replace PRETRAINED_MODEL_PATH before D_MODEL!
        text = BODY \
            .replace("JOB_NAME", jobname) \
            .replace("PRETRAINED_MODEL_PATH", pretrained_model_path.as_posix()) \
            .replace("D_MODEL", str(d_model)) \
            .replace("N_LAYER", str(n_layer)) \
            .replace("DD-HH", req_time) \
            .replace("MIN_FREQ", str(min_freq))

        outfile = OUTPUT / f"{jobname}.sh"
        with open(outfile, "w") as fp:
            fp.write(text)
        outfiles.append(outfile)


with open(OUTPUT / "run.sh", "w") as fp:
    for outfile in sorted(outfiles, key=lambda p: int(p.as_posix().split("_")[6])):
        fp.write(f"sbatch {outfile.as_posix()} clf\n")
        fp.write(f"sbatch {outfile.as_posix()} ftclf\n")

