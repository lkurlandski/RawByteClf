from dataclasses import dataclass
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
#SBATCH --ntasks=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1


source ~/anaconda3/etc/profile.d/conda.sh
conda activate RawByteClf
conda activate RawByteClf2
module unload blindfold


python -u \\
src/learn/train.py \\
--root="./output/length_extrap_sorel_categories" \\
--arch_config=ARCH_CONFIG \\
--metric_for_best_model="eval_accuracy" \\
--task="clf" \\
--streaming=false \\
--group_by_length \\
--tr_length_cutoff=TR_LENGTH_CUTOFF \\
--do_train \\
--do_eval \\
--output_dir=tmp \\
--save_strategy="epoch" \\
--evaluation_strategy="epoch" \\
--num_train_epochs=10 \\
--logging_steps=100 \\
--dataloader_num_workers=3 \\
--optim="adamw_torch" \\
--learning_rate="1e-3" \\
--lr_scheduler_type="linear" \\
--warmup_ratio=0.00 \\
--weight_decay=0.01 \\
--adam_beta2=0.999 \\
--max_grad_norm=1.0 \\
--save_total_limit=3 \\
--model_name_or_path=MODEL_NAME_OR_PATH \\
--max_length=1048576 \\
--per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE \\
--per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE \\
--gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS \\
--eval_accumulation_steps=EVAL_ACCUMULATION_STEPS \\
--load_best_model_at_end \\
--early_stopping=false \\
--auto_find_batch_size_and_gradient_accumulation_steps \\
--bf16 \\
--bf16_full_eval \\
--tf32=true
"""


@dataclass
class Config:
    model_name_or_path: str
    arch_config: str
    threshold: int
    time: str
    per_device_train_batch_size: int = 64
    per_device_eval_batch_size: int = 64
    gradient_accumulation_steps: int = 1
    eval_accumulation_steps: int = 4096


ARCH_CONFIG = {
    "mymalconv": '{"hidden_size": 512}',
    "mamba": '{"d_model": 64, "n_layer": 2, "mlp_hidden_size": 512}'
}


THRESHOLDS = [
    2 ** 17,
    2 ** 18,
    (2 ** 18) + (2 ** 17),
    2 ** 19,
    (2 ** 19) + (2 ** 17),
    (2 ** 19) + (2 ** 18) + (2 ** 17),
    (2 ** 20),
]


CONFIGS = [
    Config("mymalconv", ARCH_CONFIG["mymalconv"], THRESHOLDS[0], "01-00"),
    Config("mymalconv", ARCH_CONFIG["mymalconv"], THRESHOLDS[1], "01-00"),
    Config("mymalconv", ARCH_CONFIG["mymalconv"], THRESHOLDS[2], "01-00"),
    Config("mymalconv", ARCH_CONFIG["mymalconv"], THRESHOLDS[3], "01-00"),
    Config("mymalconv", ARCH_CONFIG["mymalconv"], THRESHOLDS[4], "02-00"),
    Config("mymalconv", ARCH_CONFIG["mymalconv"], THRESHOLDS[5], "02-00"),
    Config("mymalconv", ARCH_CONFIG["mymalconv"], THRESHOLDS[6], "02-00"),
    Config("mamba", ARCH_CONFIG["mamba"], THRESHOLDS[0], "05-00"),
    Config("mamba", ARCH_CONFIG["mamba"], THRESHOLDS[1], "05-00"),
    Config("mamba", ARCH_CONFIG["mamba"], THRESHOLDS[2], "05-00"),
    Config("mamba", ARCH_CONFIG["mamba"], THRESHOLDS[3], "05-00"),
    Config("mamba", ARCH_CONFIG["mamba"], THRESHOLDS[4], "05-00"),
    Config("mamba", ARCH_CONFIG["mamba"], THRESHOLDS[5], "05-00"),
    Config("mamba", ARCH_CONFIG["mamba"], THRESHOLDS[6], "05-00"),
]



OUTPUT = Path(os.path.realpath(__file__)).parent


outfiles = []
for config in CONFIGS:
    jobname = f"lxs_{config.model_name_or_path}_{config.threshold}"
    print(jobname)

    text = BODY \
        .replace("JOB_NAME", jobname) \
        .replace("MODEL_NAME_OR_PATH", config.model_name_or_path) \
        .replace("ARCH_CONFIG", f"'{config.arch_config}'") \
        .replace("TR_LENGTH_CUTOFF", str(config.threshold)) \
        .replace("DD-HH", config.time) \
        .replace("PER_DEVICE_EVAL_BATCH_SIZE", str(config.per_device_eval_batch_size)) \
        .replace("PER_DEVICE_TRAIN_BATCH_SIZE", str(config.per_device_train_batch_size)) \
        .replace("GRADIENT_ACCUMULATION_STEPS", str(config.gradient_accumulation_steps)) \
        .replace("EVAL_ACCUMULATION_STEPS", str(config.eval_accumulation_steps))


    outfile = OUTPUT / f"{jobname}.sh"
    with open(outfile, "w") as fp:
        fp.write(text)
    outfiles.append(outfile)


with open(OUTPUT / "run.sh", "w") as fp:
    for outfile in sorted(outfiles):
        fp.write(f"sbatch {outfile.as_posix()}\n")
