from dataclasses import dataclass
import math
import os
from pathlib import Path
from pprint import pformat
from typing import Optional


BODY = """#!/bin/bash -l

#SBATCH --job-name=JOB_NAME
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=00-08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=CPUS_PER_TASK
#SBATCH --ntasks=NTASKS
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1


source ~/anaconda3/etc/profile.d/conda.sh
conda activate RawByteClf
conda activate RawByteClf2
module unload blindfold


python -u \\
src/learn/train.py \\
--root="./output/representations" \\
--arch_config='ARCH_CONFIG' \\
--metric_for_best_model="eval_accuracy" \\
--task="clf" \\
--streaming=false \\
--top_k=10 \\
--dataset_backend="HF" \\
--representation=REPRESENTATION \\
--algorithm=ALGORITHM \\
--vocab_size=VOCAB_SIZE \\
--tr_size=0.85 \\
--vl_size=0.15 \\
--ts_size=0.0 \\
--do_train \\
--do_eval \\
--output_dir=tmp \\
--save_strategy="steps" \\
--evaluation_strategy="steps" \\
--max_steps=4000 \\
--logging_steps=10 \\
--save_steps=100 \\
--eval_steps=100 \\
--dataloader_num_workers=DATALOADER_NUM_WORKERS \\
--optim="adamw_torch" \\
--learning_rate="1e-3" \\
--lr_scheduler_type="linear" \\
--warmup_ratio=0.00 \\
--weight_decay=0.01 \\
--adam_beta2=0.999 \\
--max_grad_norm=1.0 \\
--save_total_limit=3 \\
--model_name_or_path=MODEL_NAME_OR_PATH \\
--max_length=MAX_LENGTH \\
--data_read_bytes=DATA_READ_BYTES \\
--per_device_train_batch_size=32 \\
--per_device_eval_batch_size=64 \\
--gradient_accumulation_steps=1 \\
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
    algorithm: str
    max_length: int = 65536
    data_read_bytes: Optional[int] = None
    representation: str = 8
    vocab_size: Optional[int] = None
    ntasks: int = 1
    cpus_per_task: int = 1

    def __post_init__(self):
        self.representation = str(self.representation)
        self.vocab_size = int(self.vocab_size) if self.vocab_size is not None else None
        self.data_read_bytes = self.max_length if self.data_read_bytes is None else self.data_read_bytes


OUTPUT = Path(os.path.realpath(__file__)).parent

MAX_LENGTH = 65536
DATA_READ_BYTES = 65536

ARCH_CONFIG = {
    "mamba": '{"mode": "bi", "d_model": 192, "n_layer": 8, "mlp_hidden_size": 512}',
    "mymalconv": '{"hidden_size": 512}',
}

CONFIGS: list[Config] = []
for model_name_or_path in ["mamba", "mymalconv"]:
    for rep in [8, 12, 16]:
        CONFIGS.append(
            Config(
                model_name_or_path,
                ARCH_CONFIG[model_name_or_path],
                "Raw",
                MAX_LENGTH,
                DATA_READ_BYTES,
                rep,
            )
        )
    for alg in ["BPE", "Unigram"]:
        for vs in [2 ** 10, 2 ** 12, 2 ** 14, 2 ** 16]:
            CONFIGS.append(
                Config(
                    model_name_or_path,
                    ARCH_CONFIG[model_name_or_path],
                    alg,
                    MAX_LENGTH,
                    DATA_READ_BYTES,
                    8,
                    vs,
                    ntasks=4,
                    cpus_per_task=2,
                )
            )
    for alg in ["gzip", "bzip2", "lzma", "zlib", "7z"]:
        CONFIGS.append(
            Config(
                model_name_or_path,
                ARCH_CONFIG[model_name_or_path],
                alg,
                MAX_LENGTH,
                DATA_READ_BYTES,
                8,
                ntasks=4,
            )
        )


outfiles = []
for config in CONFIGS:

    jobname = f"rep_{config.model_name_or_path}_{config.algorithm[:3]}_{config.representation}"
    jobname = jobname + f"_{config.vocab_size}" if config.vocab_size is not None else jobname

    text = BODY \
        .replace("JOB_NAME", jobname) \
        .replace("MODEL_NAME_OR_PATH", config.model_name_or_path) \
        .replace("ARCH_CONFIG", config.arch_config) \
        .replace("REPRESENTATION", config.representation) \
        .replace("ALGORITHM", config.algorithm) \
        .replace("MAX_LENGTH", str(config.max_length)) \
        .replace("DATA_READ_BYTES", str(config.data_read_bytes)) \
        .replace("DATALOADER_NUM_WORKERS", str(config.ntasks - 1)) \
        .replace("NTASKS", str(config.ntasks)) \
        .replace("CPUS_PER_TASK", str(config.cpus_per_task))

    if config.vocab_size is None:
        text = text.replace("--vocab_size=VOCAB_SIZE \\\n", "")
    else:
        text = text.replace("VOCAB_SIZE", str(config.vocab_size))

    outfile = OUTPUT / f"{jobname}.sh"
    with open(outfile, "w") as fp:
        fp.write(text)


with open(OUTPUT / "run.sh", "w") as fp:
    for outfile in sorted(outfiles):
        fp.write(f"sbatch {outfile.as_posix()}\n")
