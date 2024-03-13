import math
from pathlib import Path
from pprint import pformat
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils import get_highest_path


BODY = """#!/bin/bash -l

#SBATCH --job-name=JOB_NAME
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=01-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1

source ~/anaconda3/etc/profile.d/conda.sh
conda activate RawByteClf
conda activate RawByteClf2
module unload blindfold


python -u \\
src/learn/train.py \\
--root="./output/scaling" \\
--arch_config='{"d_model": D_MODEL, "n_layer": N_LAYER, "mlp_hidden_size": 512}' \\
--metric_for_best_model="eval_accuracy" \\
--task="clf" \\
--streaming=false \\
--bodmas_top_k=BODMAS_TOP_K \\
--tr_size=TR_SIZE \\
--vl_size=5000 \\
--ts_size=2 \\
--depth=1 \\
--do_train \\
--do_eval \\
--output_dir=tmp \\
--save_strategy="epoch" \\
--evaluation_strategy="epoch" \\
--max_steps=20000 \\
--logging_steps=10 \\
--dataloader_num_workers=0 \\
--optim="adamw_torch" \\
--learning_rate="1e-3" \\
--lr_scheduler_type="inverse_sqrt" \\
--warmup_ratio=0.01 \\
--weight_decay=0.01 \\
--adam_beta2=0.999 \\
--max_grad_norm=1.0 \\
--save_total_limit=3 \\
--model_name_or_path=MODEL_NAME_OR_PATH \\
--max_length=MAX_LENGTH \\
--per_device_train_batch_size=64 \\
--per_device_eval_batch_size=64 \\
--gradient_accumulation_steps=1 \\
--load_best_model_at_end \\
--early_stopping=false \\
--early_stopping_patience=3 \\
--early_stopping_threshold=0 \\
--auto_find_batch_size_and_gradient_accumulation_steps \\
--bf16 \\
--bf16_full_eval \\
--tf32=true
"""


CONFIGS = [
    (4, 192, 1.057728, "00-02"),
    (8, 384, 7.815552, "00-06"),
    (12, 384, 11.672448, "00-06"),
    (16, 512, 27.259392, "00-12"),
]
TR_SIZES = [50, 100, 150, 200, 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000]


MAX_LENGTH=16384
OUTPUT = Path("run/train_sizes")
PRETRAINED_MODELS_ROOT = Path(
    "/home/lk3591/Documents/code/RawByteClf/output/"
    f"scaling/mamba/{MAX_LENGTH}/clm/7000000/1/"
)


def save_eval_steps(size: int) -> int:
    if size > 1000:
        return int(size / 100)
    return 


def early_stopping_patience(size: int) -> int:
    if size < 50:
        return 50
    if size < 100:
        return 40
    if size < 200:
        return 30
    if size < 500:
        return 20
    return 10


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


outfiles = []
for n_layer, d_model, params, dd_hh in CONFIGS:
    for tr_size in TR_SIZES:
        for finetune in [True, False]:
            jobname = f"tr_sz_bl_{MAX_LENGTH}_{finetune}_{tr_size}_{n_layer}_{d_model}"
            print(jobname)

            pretrained_model_path = get_pretrained_model_path(d_model, n_layer)
            assert pretrained_model_path.exists(), pretrained_model_path

            model_name_or_path = pretrained_model_path.as_posix() if finetune else "mamba"
            text = BODY \
            .replace("JOB_NAME", jobname) \
            .replace("MODEL_NAME_OR_PATH", pretrained_model_path.as_posix()) \
            .replace("D_MODEL", str(d_model)) \
            .replace("N_LAYER", str(n_layer)) \
            .replace("TR_SIZE", str(tr_size)) \
            .replace("BODMAS_TOP_K", str(10)) \
            .replace("MAX_LENGTH", str(MAX_LENGTH)) \
            .replace("DD-HH", dd_hh)

            outfile = OUTPUT / f"{jobname}.sh"
            with open(outfile, "w") as fp:
                fp.write(text)
            outfiles.append(outfile)


with open(OUTPUT / "run.sh", "w") as fp:
    for outfile in sorted(outfiles, key=lambda p: int(p.as_posix().split("_")[7])):
        fp.write(f"sbatch {outfile.as_posix()}\n")

