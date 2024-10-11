"""
Create sbatch scripts.
"""

from __future__ import annotations
from argparse import ArgumentParser
from dataclasses import dataclass
import datetime
from itertools import product
from enum import Enum
import json
from pathlib import Path
import os
import sys
from typing import Any, Optional
import warnings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))  #pylint: disable=wrong-import-position
from src.enums import LiftLevel, Task, TokenizationAlgorithm


DEBUG    = False
ARMITAGE = False
OUTPATH  = Path("run/esp/sbatch")


def bool_to_str(b: bool) -> str:
    if not isinstance(b, bool):
        raise TypeError(type(b))
    return 'true' if b else 'false'

def seconds_to_slurm_time(seconds: int) -> str:
    days = seconds // (24 * 3600)
    seconds %= (24 * 3600)
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return f"{int(days):02d}-{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

def get_body(
    job: str,
    tim: str,
    cpu: int,
    mem: int,
    gpu: int,
    streaming: bool,
    model_name: ModelName,
    arch_config: dict,
    max_length: int,
    lift_level: LiftLevel,
    tokenization_algorithm: TokenizationAlgorithm,
    vocab_size: int,
    pretraining_task: Optional[Task],
    task: Task,
    num_train_epochs: int,
    saves_and_evals_per_epochs: int,
    dataloader_num_workers: int,
    learning_rate: float,
    batch_size: int,
    bf16: bool,
    seed: int,
) -> str: return (
f"""
#!/bin/bash -l

#SBATCH --job-name={job}
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time={tim}
#SBATCH --nodes=1
#SBATCH --cpus-per-task={cpu}
#SBATCH --ntasks=1
#SBATCH --mem={mem}G
#SBATCH --gres=gpu:a100:{gpu}

source ~/anaconda3/etc/profile.d/conda.sh
conda activate {"RawByteClf" if ARMITAGE else "RawByteClf2"}
{"" if ARMITAGE else "module unload blindfold"}

python -u \\
src/learn/train.py \\
--root='./output/'esp{"-test" if DEBUG else ""}' \\
--streaming={bool_to_str(streaming)} \\
--auto_find_batch_size_and_gradient_accumulation_steps='true' \\
--dataset_backend="HF" \\
--model_name_or_path='{model_name.value}' \\
--arch_config='{json.dumps(arch_config)}' \\
--max_length={max_length} \\
--lift_level='{lift_level.value}' \\
--tokenization_algorithm='{tokenization_algorithm.value}' \\
--vocab_size={vocab_size} \\
--task='{task.value}' \\
{"--pretraining_task='" + pretraining_task.value + "'" if pretraining_task else ""} \\
--pretraining_checkpoint=-1 \\
--seed={seed} \\
--do_train \\
--output_dir='/tmp' \\
--save_strategy='epoch' \\
--evaluation_strategy='epoch' \\
--num_train_epochs={num_train_epochs} \\
--logging_steps=1 \\
--saves_per_epoch={saves_and_evals_per_epochs} \\
--evals_per_epoch={saves_and_evals_per_epochs} \\
--dataloader_num_workers={dataloader_num_workers} \\
--optim="adamw_torch" \\
--learning_rate={learning_rate} \\
--lr_scheduler_type="linear" \\
--weight_decay=0.01 \\
--adam_beta1=0.950 \\
--adam_beta2=0.999 \\
--max_grad_norm=1.0 \\
--save_total_limit=-1 \\
--per_device_train_batch_size={batch_size} \\
--per_device_eval_batch_size={batch_size} \\
--gradient_accumulation_steps=1 \\
--eval_accumulation_steps=64 \\
--load_best_model_at_end \\
--tf32='true' \\
--bf16={'true' if bf16 else 'false'} \\
--gradient_checkpointing='true'
""".replace("\n \\", "").strip() + "\n"
)

class ModelName(Enum):
    HRR = "hrrformer"
    MAM = "mamba"

class ModelSize(Enum):
    TN = "tn"
    SM = "sm"
    MD = "md"
    LG = "lg"
    HG = "hg"

class ModelMode(Enum):
    UN = "un"
    BI = "bi"

DATASET_SIZES = {
    Task.CLM: 1000000,
    Task.MLM: 1000000,
    Task.DET: 40000,
    Task.FAM: 80000,
    Task.BEH: 35000,
}

MODEL_TIME_PER_SAMPLE = {
    ModelName.HRR: {
        ModelSize.TN: {},
        ModelSize.SM: {},
        ModelSize.MD: {},
        ModelSize.LG: {},
        ModelSize.HG: {},
    },
    ModelName.MAM: {
        ModelSize.TN: {},
        ModelSize.SM: {},
        ModelSize.MD: {},
        ModelSize.LG: {},
        ModelSize.HG: {},
    },
}

COMPRESSION_RATIOS = {
    TokenizationAlgorithm.BPE: {
        1024: 1,
        4096: 1,
        16384: 1,
    },
    TokenizationAlgorithm.UNIGRAM: {
        1024: 1,
        4096: 1,
        16384: 1,
    },
}


@dataclass
class Configuration:
    model_name: ModelName
    model_size: ModelSize
    model_mode: ModelMode
    max_length: int
    lift_level: LiftLevel
    tokenization_algorithm: TokenizationAlgorithm
    vocab_size: int
    pretraining_task: Optional[Task]
    task: Task
    seed: int

    def __postinit__(self) -> None:
        if not (self.cpu / self.gpu).is_integer():
            raise ValueError(f"CPU/GPU ratio must be an integer: {self.cpu} / {self.gpu}")

    @property
    def do(self) -> bool:
        # Unidirectional models cannot do MLM.
        if self.model_mode == ModelMode.UN and Task.MLM in (self.task, self.pretraining_task):
            return False
        # Bidirectional models cannot do CLM.
        if self.model_mode == ModelMode.BI and Task.CLM in (self.task, self.pretraining_task):
            return False
        # Only some tokenizers were trained.
        if self.tokenization_algorithm not in (TokenizationAlgorithm.BPE, TokenizationAlgorithm.UNIGRAM):
            return False
        # Pretraining and detection does not required random runs.
        if self.task in (Task.CLM, Task.MLM, Task.DET) and self.seed != 0:
            return False
        # Pretraining cannot be pretrained.
        if self.task in (Task.CLM, Task.MLM) and self.pretraining_task is not None:
            return False
        # At the moment, none of the ESP loaders support different seeds.
        if self.seed != 0:
            return False

        # Adjust as desired
        # if self.model_name != ModelName.MAM:
        #     return False
        # if self.model_mode != ModelMode.BI:
        #     return False
        # if self.model_size != ModelSize.TN:
        #     return False
        # if self.max_length != 1024:
        #     return False
        # if self.lift_level != LiftLevel.RAW:
        #     return False
        # if self.tokenization_algorithm != TokenizationAlgorithm.BPE:
        #     return False
        # if self.vocab_size != 16384:
        #     return False
        # if self.seed != 0:
        #     return False

        return True

    @property
    def job(self) -> str:
        return "-".join([
            f"{self.model_name.value}"[0:3],
            f"{self.model_size.value}",
            f"{self.model_mode.value}",
            f"{'0' * (6 - len(str(self.max_length))) + str(self.max_length)}",
            f"{self.lift_level.value}",
            f"{self.tokenization_algorithm.value}",
            f"{'0' * (5 - len(str(self.vocab_size))) + str(self.vocab_size)}",
            f"{self.pretraining_task.value if self.pretraining_task else 'nop'}",
            f"{self.task.value}",
            f"{self.seed}",
        ])

    @property
    def tim(self) -> str:
        try:
            t = MODEL_TIME_PER_SAMPLE[self.model_name][self.model_size][self.max_length]
        except KeyError:
            warnings.warn(f"Time/sample unknown for {self.model_name.value} {self.model_size.value} {self.max_length}.")
            t = 0.00001
        s = t * DATASET_SIZES[self.task] * self.num_train_epochs
        s = s + 3600
        return seconds_to_slurm_time(s)

    @property
    def cpu(self) -> int:
        if self.task in (Task.CLM, Task.MLM):
            return 8
        return 4

    @property
    def mem(self) -> int:
        if self.streaming:
            return 64

        if self.vocab_size not in COMPRESSION_RATIOS[self.tokenization_algorithm]:
            warnings.warn(f"Compression ratio unknown for {self.tokenization_algorithm.value} {self.vocab_size}.")
        c = COMPRESSION_RATIOS[self.tokenization_algorithm].get(self.vocab_size, 1)
        t = c * DATASET_SIZES[self.task] * self.max_length
        b = t * 4
        b = b + (16 * 1024**3)
        return b // 1024**3

    @property
    def gpu(self) -> int:
        if self.task in (Task.CLM, Task.MLM):
            return 2
        return 1

    @property
    def streaming(self) -> bool:
        if self.task in (Task.CLM, Task.MLM):
            return True
        return False

    @property
    def arch_config(self) -> dict:
        if self.model_name == ModelName.MAM:
            if self.model_size == ModelSize.TN:
                d = {"num_hidden_layers": 2,  "hidden_size": 64 }
            if self.model_size == ModelSize.SM:
                d = {"num_hidden_layers": 4,  "hidden_size": 128}
            if self.model_size == ModelSize.MD:
                d = {"num_hidden_layers": 8,  "hidden_size": 256}
            if self.model_size == ModelSize.LG:
                d = {"num_hidden_layers": 12, "hidden_size": 384}
            if self.model_size == ModelSize.HG:
                d = {"num_hidden_layers": 16, "hidden_size": 512}
            d["mode"] = "uni" if self.model_mode == ModelMode.UN else "bi"
            d["embedding_size"] = d["hidden_size"]
            return d

        if self.model_name == ModelName.HRR:
            if self.model_size == ModelSize.TN:
                d = {"num_hidden_layers": 1, "hidden_size": 64,  "intermediate_size": 128,  "num_attention_heads": 1}
            if self.model_size == ModelSize.SM:
                d = {"num_hidden_layers": 2, "hidden_size": 128, "intermediate_size": 256,  "num_attention_heads": 2}
            if self.model_size == ModelSize.MD:
                d = {"num_hidden_layers": 4, "hidden_size": 256, "intermediate_size": 512, "num_attention_heads": 4}
            if self.model_size == ModelSize.LG:
                d = {"num_hidden_layers": 6, "hidden_size": 384, "intermediate_size": 768, "num_attention_heads": 6}
            if self.model_size == ModelSize.HG:
                d = {"num_hidden_layers": 8, "hidden_size": 512, "intermediate_size": 1024, "num_attention_heads": 8}
            d["is_decoder"] = self.model_mode == ModelMode.UN
            d["embedding_size"] = d["hidden_size"]
            return d

        raise ValueError(self.model_name)


    @property
    def num_train_epochs(self) -> int:
        if DEBUG: return 1

        if self.task in (Task.CLM, Task.MLM):
            return 1
        return 10

    @property
    def saves_and_evals_per_epochs(self) -> int:
        if DEBUG: return 2

        if self.task in (Task.CLM, Task.MLM):
            return 16
        return 1

    @property
    def dataloader_num_workers(self) -> int:
        return self.cpu // self.gpu - 1

    @property
    def learning_rate(self) -> float:
        if self.task in (Task.CLM, Task.MLM):
            return 1e-3
        return 1e-4

    @property
    def batch_size(self) -> int:
        if self.task in (Task.CLM, Task.MLM):
            return 1024
        return 256

    @property
    def bf16(self) -> bool:
        if self.model_name == ModelName.MAM:
            return True
        return False

    @property
    def outfile(self) -> Path:
        return OUTPATH / f"{self.job}.sh"


def main():

    global DEBUG
    global ARMITAGE
    global OUTPATH

    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--armitage", action="store_true")
    args = parser.parse_args()

    DEBUG    = args.debug
    ARMITAGE = args.armitage
    OUTPATH  = Path("run/esp/test") if DEBUG else OUTPATH

    OUTPATH.mkdir(parents=True, exist_ok=True)
    for file in OUTPATH.iterdir():
        file.unlink()

    configurations = product(
        ModelName,
        ModelSize,
        ModelMode,
        (1024, 4096, 16384, 65536),
        LiftLevel,
        TokenizationAlgorithm,
        (1024, 4096, 16384),
        [None, Task.CLM, Task.MLM],
        Task,
        (0, 1, 2, 3, 4),
    )
    configurations = [Configuration(*config) for config in configurations]

    for config in configurations:
        if not config.do:
            continue
        body = get_body(
            job=config.job,
            tim=config.tim,
            cpu=config.cpu,
            mem=config.mem,
            gpu=config.gpu,
            streaming=config.streaming,
            model_name=config.model_name,
            arch_config=config.arch_config,
            max_length=config.max_length,
            lift_level=config.lift_level,
            tokenization_algorithm=config.tokenization_algorithm,
            vocab_size=config.vocab_size,
            task=config.task,
            pretraining_task=config.pretraining_task,
            num_train_epochs=config.num_train_epochs,
            saves_and_evals_per_epochs=config.saves_and_evals_per_epochs,
            dataloader_num_workers=config.dataloader_num_workers,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            bf16=config.bf16,
            seed=config.seed,
        )
        if config.outfile.exists():
            raise FileExistsError(config.outfile)
        config.outfile.write_text(body)


if __name__ == "__main__":
    main()
