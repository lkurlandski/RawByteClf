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
#SBATCH --ntasks=5
#SBATCH --mem=MEMORY
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
--representation=8 \\
--tr_size=120000 \\
--ts_size=12000 \\
--streaming=false \\
--group_by_length \\
--tr_length_cutoff=TR_LENGTH_CUTOFF \\
--do_train \\
--output_dir=tmp \\
--save_strategy="epoch" \\
--evaluation_strategy="epoch" \\
--num_train_epochs=10 \\
--logging_steps=100 \\
--dataloader_num_workers=4 \\
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
--auto_find_batch_size_and_gradient_accumulation_steps \\
--per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE \\
--per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE \\
--gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS \\
--eval_accumulation_steps=EVAL_ACCUMULATION_STEPS \\
--load_best_model_at_end \\
--early_stopping=false \\
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
    memory: str = "192G"
    per_device_train_batch_size: int = 64
    gradient_accumulation_steps: int = 1
    per_device_eval_batch_size: int = 64
    eval_accumulation_steps: int = 4096


ARCH_CONFIG = {
    "mymalconv": '{"hidden_size": 512}', # 0.5M
    "mamba-tiny": '{"d_model": 64, "n_layer": 2, "mlp_hidden_size": 512}',  # 0.1M
    "mamba-small": '{"d_model": 192, "n_layer": 8, "mlp_hidden_size": 512}',  # 3.5M
}


# [28.311552, 44.040192, 59.768832, 75.497472, 91.226112, 122.683392, 138.412032]
THRESHOLDS = [
    (2 ** 17),                            # 0
    (2 ** 18),                            # 1
    (2 ** 18) + (2 ** 17),                # 2
    (2 ** 19),                            # 3
    (2 ** 19) + (2 ** 17),                # 4
    (2 ** 19) + (2 ** 18),                # 5
    (2 ** 19) + (2 ** 18) + (2 ** 17),    # 6
    (2 ** 20),                            # 7
]


CONFIGS = [
    Config("mymalconv", ARCH_CONFIG["mymalconv"], THRESHOLDS[0], "01-00", "96G"),
    Config("mymalconv", ARCH_CONFIG["mymalconv"], THRESHOLDS[1], "01-00", "96G"),
    Config("mymalconv", ARCH_CONFIG["mymalconv"], THRESHOLDS[2], "02-00", "128G"),
    Config("mymalconv", ARCH_CONFIG["mymalconv"], THRESHOLDS[3], "02-00", "128G"),
    Config("mymalconv", ARCH_CONFIG["mymalconv"], THRESHOLDS[4], "03-00", "128G"),
    Config("mymalconv", ARCH_CONFIG["mymalconv"], THRESHOLDS[5], "03-00", "144G"),
    Config("mymalconv", ARCH_CONFIG["mymalconv"], THRESHOLDS[6], "03-00", "160G"),
    Config("mymalconv", ARCH_CONFIG["mymalconv"], THRESHOLDS[7], "03-00", "192G"),

    Config("mamba", ARCH_CONFIG["mamba-tiny"], THRESHOLDS[0], "01-00", "96G", 32, 2, 8),
    Config("mamba", ARCH_CONFIG["mamba-tiny"], THRESHOLDS[1], "01-00", "96G", 16, 4, 8),
    Config("mamba", ARCH_CONFIG["mamba-tiny"], THRESHOLDS[2], "02-00", "128G", 8, 8, 8),
    Config("mamba", ARCH_CONFIG["mamba-tiny"], THRESHOLDS[3], "02-00", "128G", 8, 8, 8),
    Config("mamba", ARCH_CONFIG["mamba-tiny"], THRESHOLDS[4], "03-00", "128G", 4, 16, 8),
    Config("mamba", ARCH_CONFIG["mamba-tiny"], THRESHOLDS[5], "03-00", "144G", 4, 16, 8),
    Config("mamba", ARCH_CONFIG["mamba-tiny"], THRESHOLDS[6], "03-00", "160G", 4, 16, 8),
    Config("mamba", ARCH_CONFIG["mamba-tiny"], THRESHOLDS[7], "03-00", "192G", 4, 16, 8),

    Config("mamba", ARCH_CONFIG["mamba-small"], THRESHOLDS[0], "03-00", "96G", 8, 8, 2),
    Config("mamba", ARCH_CONFIG["mamba-small"], THRESHOLDS[1], "03-00", "96G", 8, 8, 2),
    Config("mamba", ARCH_CONFIG["mamba-small"], THRESHOLDS[2], "03-00", "128G", 8, 8, 2),
    Config("mamba", ARCH_CONFIG["mamba-small"], THRESHOLDS[3], "04-00", "128G", 8, 8, 2),
    Config("mamba", ARCH_CONFIG["mamba-small"], THRESHOLDS[4], "04-00", "128G", 8, 8, 2),
    Config("mamba", ARCH_CONFIG["mamba-small"], THRESHOLDS[5], "04-00", "144G", 8, 8, 2),
    Config("mamba", ARCH_CONFIG["mamba-small"], THRESHOLDS[6], "05-00", "160G", 8, 8, 2),
    Config("mamba", ARCH_CONFIG["mamba-small"], THRESHOLDS[7], "05-00", "192G", 8, 8, 2),
]



OUTPUT = Path(os.path.realpath(__file__)).parent


outfiles = []
for config in CONFIGS:
    model_name = [k for k, v in ARCH_CONFIG.items() if v == config.arch_config][0]
    jobname = f"lxs_{model_name}_{config.threshold}"
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
        .replace("EVAL_ACCUMULATION_STEPS", str(config.eval_accumulation_steps)) \
        .replace("MEMORY", config.memory)

    outfile = OUTPUT / f"{jobname}.sh"
    with open(outfile, "w") as fp:
        fp.write(text)
    outfiles.append(outfile)


with open(OUTPUT / "run.sh", "w") as fp:
    for outfile in sorted(outfiles, key=lambda p: int(p.stem.split("_")[2]), reverse=True):
        fp.write(f"sbatch {outfile.as_posix()}\n")
