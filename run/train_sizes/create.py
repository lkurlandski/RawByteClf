import math
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
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1

source ~/anaconda3/etc/profile.d/conda.sh
conda activate RawByteClf
conda activate RawByteClf2
module unload blindfold


if [ "$1" = "clf" ]; then
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
--bodmas_top_k=BODMAS_TOP_K \\
--tr_size=TR_SIZE \\
--vl_size=5000 \\
--ts_size=2 \\
--depth=1 \\
--do_train \\
--do_eval \\
--output_dir=tmp \\
--save_strategy="steps" \\
--evaluation_strategy="steps" \\
--save_steps=250 \\
--eval_steps=250 \\
--max_steps=17625 \\
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
--max_length=512 \\
--per_device_train_batch_size=64 \\
--per_device_eval_batch_size=64 \\
--gradient_accumulation_steps=1 \\
--load_best_model_at_end \\
--early_stopping=false \\
--early_stopping_patience=2 \\
--early_stopping_threshold=0 \\
--auto_find_batch_size_and_gradient_accumulation_steps \\
--bf16 \\
--bf16_full_eval \\
--tf32=true
"""

CONFIGS = [
    (4, 384, 3958656),
    (6, 384, 5887104),
    (8, 384, 7815552),
    (12, 512, 20478464),
    (16, 512, 27259392),
    (24, 768, 90723072),
]

TR_SIZES = [250, 500, 750, 1000, 2000, 4000, 8000, 12000, 16000, 20000]


OUTPUT = Path("run/train_sizes")
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


for n_layer, d_model, params in CONFIGS:
    for tr_size in TR_SIZES:
        jobname = f"tr_size_{tr_size}_{n_layer}-{d_model}"
        print(jobname)

        pretrained_model_path = get_pretrained_model_path(d_model, n_layer)
        assert pretrained_model_path.exists(), pretrained_model_path

        # must replace PRETRAINED_MODEL_PATH before D_MODEL!
        text = BODY \
            .replace("JOB_NAME", jobname) \
            .replace("PRETRAINED_MODEL_PATH", pretrained_model_path.as_posix()) \
            .replace("D_MODEL", str(d_model)) \
            .replace("N_LAYER", str(n_layer)) \
            .replace("DD-HH", dd_hh(params)) \
            .replace("TR_SIZE", str(tr_size)) \
            .replace("BODMAS_TOP_K", str(10))

        outfile = OUTPUT / f"{jobname}.sh"
        with open(outfile, "w") as fp:
            fp.write(text)
        outfiles.append(outfile)


with open(OUTPUT / "run.sh", "w") as fp:
    for outfile in sorted(outfiles, key=lambda p: int(p.as_posix().split("_")[3])):
        fp.write(f"sbatch {outfile.as_posix()} clf\n")
        fp.write(f"sbatch {outfile.as_posix()} ftclf\n")

