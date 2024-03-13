import math
import os
from pathlib import Path
from pprint import pformat
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils import get_highest_path


BODY_CLM = """#!/bin/bash -l

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
--per_device_train_batch_size=256 \\
--per_device_eval_batch_size=256 \\
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


BODY_CLF = """#!/bin/bash -l

#SBATCH --job-name=JOB_NAME
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=DD-HH:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=3
#SBATCH --mem=32G
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
--metric_for_best_model="eval_accuracy" \\
--task="clf" \\
--streaming=false \\
--bodmas_top_k=10 \\
--tr_size=0.875 \\
--vl_size=0.100 \\
--ts_size=0.025 \\
--do_train \\
--output_dir=tmp \\
--save_strategy="epoch" \\
--evaluation_strategy="epoch" \\
--num_train_epochs=25 \\
--logging_steps=10 \\
--dataloader_num_workers=2 \\
--optim="adamw_torch" \\
--learning_rate="1e-3" \\
--lr_scheduler_type="linear" \\
--warmup_ratio=0.00 \\
--weight_decay=0.01 \\
--adam_beta2=0.999 \\
--max_grad_norm=1.0 \\
--save_total_limit=3 \\
--model_name_or_path=$MODEL_NAME_OR_PATH \\
--max_length=MAX_LENGTH \\
--per_device_train_batch_size=64 \\
--per_device_eval_batch_size=64 \\
--gradient_accumulation_steps=1 \\
--load_best_model_at_end \\
--early_stopping=false \\
--auto_find_batch_size_and_gradient_accumulation_steps \\
--bf16 \\
--bf16_full_eval \\
--tf32=true
"""


PRETRAINED_MODELS_ROOT = Path(
    "/home/lk3591/Documents/code/RawByteClf/"
    "output/scaling/mamba/16384/clm/7000000/1/"
)
OUTPUT = Path(os.path.realpath(__file__)).parent
MAX_LENGTH = 16384
CONFIGS = [
    (4, 192, 1.057728, "00-06"),
    (4, 256, 1.819904, "00-06"),
    (6, 192, 1.561152, "00-06"),
    (6, 256, 2.695936, "00-08"),
    (6, 384, 5.887104, "00-08"),
    (8, 256, 3.571968, "00-12"),
    (8, 384, 7.815552, "00-12"),
    (12, 384, 11.672448, "00-16"),
    (12, 512, 20.478464, "00-20"),
    (16, 512, 27.259392, "01-00"),
]


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

    pretrained_model_path: Path = get_highest_path(pretrained_model_path, lstrip="checkpoint-")
    return pretrained_model_path


outfiles_clm = []
outfiles_clf = []
for n_layer, d_model, params, time_clf in CONFIGS:
    jobname = f"clm_{n_layer}-{d_model}"
    text = BODY_CLM \
        .replace("JOB_NAME", jobname) \
        .replace("D_MODEL", str(d_model)) \
        .replace("N_LAYER", str(n_layer)) \
        .replace("DD-HH", "05-00") \
        .replace("MAX_LENGTH", str(MAX_LENGTH))
    outfile = OUTPUT / f"{jobname}.sh"
    outfiles_clm.append(outfile)
    with open(outfile, "w") as fp:
        fp.write(text)

    jobname = f"clf_{n_layer}-{d_model}"
    pretrained_model_path = get_pretrained_model_path(d_model, n_layer)
    if not pretrained_model_path.exists():
        raise FileNotFoundError(pretrained_model_path.as_posix())
    pretrained_model_path = pretrained_model_path.as_posix()

    text = BODY_CLF \
        .replace("PRETRAINED_MODEL_PATH", pretrained_model_path) \
        .replace("JOB_NAME", jobname) \
        .replace("D_MODEL", str(d_model)) \
        .replace("N_LAYER", str(n_layer)) \
        .replace("DD-HH", time_clf) \
        .replace("MAX_LENGTH", str(MAX_LENGTH))
    outfile = OUTPUT / f"{jobname}.sh"
    outfiles_clf.append(outfile)
    with open(outfile, "w") as fp:
        fp.write(text)


with open(OUTPUT / "run_clm.sh", "w") as fp:
    for outfile in outfiles_clm:
        fp.write(f"sbatch {outfile.as_posix()} \n")

with open(OUTPUT / "run_clf.sh", "w") as fp:
    for outfile in outfiles_clf:
        fp.write(f"sbatch {outfile.as_posix()} clf\n")
        fp.write(f"sbatch {outfile.as_posix()} ftclf\n")

