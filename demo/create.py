"""
Create sbatch scripts.
"""

from __future__ import annotations
from argparse import ArgumentParser
from dataclasses import dataclass
from enum import Enum
from itertools import product, chain
import json
import math
from pathlib import Path
import os
import sys
from typing import Optional

from tqdm import tqdm

# pylint: disable=wrong-import-position
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# pylint: disable=wrong-import-position

from src.enums import LiftLevel, Task, TokenizationAlgorithm, WeightedLossAlgorithm


ROOT = Path(os.path.dirname(os.path.realpath(__file__)))
GB   = 1024 ** 3


def outpath() -> Path:
    return ROOT / "sbatch"


def bool_to_str(b: bool) -> str:
    if not isinstance(b, bool):
        raise TypeError(type(b))
    return "true" if b else "false"


def seconds_to_slurm_time(s: int, r: int = 3600) -> str:
    s = int(math.ceil(s / r) * r)
    days = s // (24 * 3600)
    s %= (24 * 3600)
    hours = s // 3600
    s %= 3600
    minutes = s // 60
    s %= 60
    return f"{int(days):02d}-{int(hours):02d}:{int(minutes):02d}:{int(s):02d}"


def bytes_to_slurm_mem(b: int, r: int = 4 * GB) -> str:
    b = int(math.ceil(b / r) * r)
    g = int(b // GB)
    return f"{g}G"


def get_body(
    job: str,
    tim: str,
    cpu: int,
    mem: str,
    gpu: int,
    streaming: bool,
    sync_batch_size: bool,
    exit_after_map: bool,
    skip_eval_check: bool,
    model_name: ModelName,
    arch_config: dict,
    max_length: int,
    lift_level: LiftLevel,
    lift_level_ddp: Optional[LiftLevel],
    tokenization_algorithm: TokenizationAlgorithm,
    vocab_size: int,
    task: Task,
    pretraining_task: Optional[Task],
    weighted_loss: Optional[WeightedLossAlgorithm],
    beta: Optional[float],
    pretraining_checkpoint: Optional[str],
    num_train_epochs: int,
    max_steps: Optional[int],
    save_steps: Optional[int],
    eval_steps: Optional[int],
    saves_per_epoch: int,
    evals_per_epoch: int,
    dataloader_num_workers: int,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    tr_per_device_batch_size: int,
    vl_per_device_batch_size: int,
    gradient_accumulation_steps: int,
    eval_accumulation_steps: int,
    tf32: bool,
    fp16: bool,
    fp16_full_eval: bool,
    bf16: bool,
    bf16_full_eval: bool,
    gradient_checkpointing: bool,
    sort_when_making_archives_contiguous: bool,
    seed: int,
) -> str: return (
f"""
#!/bin/bash -l

#SBATCH --job-name=exe-{job}
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time={tim}
#SBATCH --nodes=1
#SBATCH --cpus-per-task={cpu}
#SBATCH --ntasks=1
#SBATCH --mem={mem}
{"#SBATCH --gres=gpu:a100:" + str(gpu) if gpu > 0 else ""}

{"" if sort_when_making_archives_contiguous else "export SORT_WHEN_MAKING_ARCHIVES_CONTIGUOUS=0"}

python -u \\
src/learn/train.py \\
--root='./output/esp-exe' \\
--streaming={bool_to_str(streaming)} \\
--sync_batch_size={bool_to_str(sync_batch_size)} \\
--exit_after_map={bool_to_str(exit_after_map)} \\
{"--do_compute_unigram_probabilities" if task in (Task.CLM, Task.MLM) else ""} \\
--skip_eval_check={bool_to_str(skip_eval_check)} \\
--dataset_backend="HF" \\
--model_name_or_path='{model_name.value}' \\
--arch_config='{json.dumps(arch_config)}' \\
--max_length={max_length} \\
--lift_level='{lift_level.value}' \\
{"--lift_level_ddp='" + lift_level_ddp.value + "'" if lift_level_ddp is not None else ""} \\
--tokenization_algorithm='{tokenization_algorithm.value}' \\
--vocab_size={vocab_size} \\
--task='{task.value}' \\
{"--pretraining_task='" + pretraining_task.value + "'" if pretraining_task is not None else ""} \\
{"--weighted_loss='" + weighted_loss.value + "'" if weighted_loss is not None else ""} \\
{f"--beta={beta}" if beta is not None else ""} \\
{f"--pretraining_checkpoint='{pretraining_checkpoint}'" if pretraining_checkpoint is not None else ""} \\
--seed={seed} \\
--do_train \\
--output_dir='/tmp' \\
--save_strategy='{"epoch" if save_steps is None else "steps"}' \\
--eval_strategy='{"epoch" if eval_steps is None else "steps"}' \\
{"--max_steps=" + str(max_steps) if max_steps is not None else ""} \\
{"--save_steps=" + str(save_steps) if save_steps is not None else ""} \\
{"--eval_steps=" + str(eval_steps) if eval_steps is not None else ""} \\
{"--num_train_epochs=" + str(num_train_epochs) if max_steps is None else ""} \\
{"--saves_per_epoch=" + str(saves_per_epoch) if save_steps is None else ""} \\
{"--evals_per_epoch=" + str(evals_per_epoch) if eval_steps is None else ""} \\
--logging_steps=1 \\
--dataloader_num_workers={dataloader_num_workers} \\
--optim="adamw_torch" \\
--learning_rate={learning_rate:.6f} \\
--weight_decay={weight_decay:.6f} \\
--warmup_ratio={warmup_ratio:.6f} \\
--lr_scheduler_type="linear" \\
--adam_beta1=0.900 \\
--adam_beta2=0.999 \\
--max_grad_norm=1.0 \\
--save_total_limit=1 \\
--per_device_train_batch_size={tr_per_device_batch_size} \\
--per_device_eval_batch_size={vl_per_device_batch_size} \\
--gradient_accumulation_steps={gradient_accumulation_steps} \\
--eval_accumulation_steps={eval_accumulation_steps} \\
{"--use_cpu" if gpu <= 0 else ""} \\
--tf32={bool_to_str(tf32)} \\
--fp16={bool_to_str(fp16)} \\
--fp16_full_eval={bool_to_str(fp16_full_eval)} \\
--bf16={bool_to_str(bf16)} \\
--bf16_full_eval={bool_to_str(bf16_full_eval)} \\
--gradient_checkpointing={bool_to_str(gradient_checkpointing)} \\
""".replace("\n \\", "").strip() + "\n"
)


class ModelName(Enum):
    HRR = "hrrformer"
    MAM = "mamba"
    MAL = "malconv2"


class ModelSize(Enum):
    TN = "tn"  # tiny
    SM = "sm"  # small
    MD = "md"  # medium
    LG = "lg"  # large
    HG = "hg"  # huge
    CO = "co"  # colossal


class ModelMode(Enum):
    UN = "un"
    BI = "bi"


@dataclass
class Configuration:
    model_name: ModelName
    model_size: ModelSize
    model_mode: ModelMode
    max_length: int
    lift_level: LiftLevel
    lift_level_ddp: Optional[LiftLevel]
    tokenization_algorithm: TokenizationAlgorithm
    vocab_size: int
    task: Task
    pretraining_task: Optional[Task]
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
            if self.lift_level != LiftLevel.NOP:
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
        # Cannot pretrain with multi representations
        if self.lift_level == LiftLevel.ALL and self.task in (Task.CLM, Task.MLM):
            return False
        # MalConv is only implemented under these conditions
        if self.model_name == ModelName.MAL:
            if self.task not in (Task.BEH, Task.DET, Task.FAM):
                return False
            if self.pretraining_task is not None:
                return False
            if self.model_mode != ModelMode.BI:
                return False

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
            f"{self.lift_level_ddp.value if self.lift_level_ddp else 'nop'}",
            f"{self.seed}",
        ]).replace("--", "-").rstrip("-")

    @property
    def tim(self) -> str:
        return seconds_to_slurm_time(3600 * 24)

    @property
    def cpu(self) -> int:
        return 1

    @property
    def mem(self) -> int:
        return bytes_to_slurm_mem(64 * GB)

    @property
    def gpu(self) -> int:
        return 0

    @property
    def sort_when_making_archives_contiguous(self) -> bool:
        if self.task == Task.DET and self.max_length > 65536:
            return False
        return True

    @property
    def streaming(self) -> bool:
        if self.task == Task.DET and self.max_length <= 65536:
            return False
        return True

    @property
    def sync_batch_size(self) -> bool:
        if self.task in (Task.DET, Task.FAM, Task.BEH):
            return False
        return True

    @property
    def exit_after_map(self) -> bool:
        return False

    @property
    def skip_eval_check(self) -> bool:
        return False

    @property
    def arch_config(self) -> dict:

        if self.model_name == ModelName.MAL:
            d = {
                "mode": "gcg",
                "channels": 256,
                "stride": 64,
                "kernel_size": 64,
                "embedding_size": 8,
            }
            return d

        NUM_BLOCKS = {
            ModelSize.TN:  2,
            ModelSize.SM:  4,
            ModelSize.MD:  8,
            ModelSize.LG: 12,
            ModelSize.HG: 16,
            ModelSize.CO: 32,
        }

        HIDDEN_SIZES = {
            ModelSize.TN: 32,
            ModelSize.SM: 64,
            ModelSize.MD: 96,
            ModelSize.LG: 128,
            ModelSize.HG: 192,
            ModelSize.CO: 384,
        }

        d = {
            "is_decoder": self.model_mode == ModelMode.UN,
            "num_hidden_layers": NUM_BLOCKS[self.model_size],
            "embedding_size": HIDDEN_SIZES[self.model_size],
            "hidden_size": HIDDEN_SIZES[self.model_size],
            "head_num_hidden_layers": 0,
            "head_hidden_size": 0,
        }

        if self.model_name == ModelName.HRR:
            d |= {
                "num_attention_heads": 1,
                "hidden_dropout_prob": 0.1,
                "position_embedding_type": "rotary",
            }
        elif self.model_name == ModelName.MAM:
            d |= {
                "hidden_dropout_prob": 0.0,
                "bi_tie_directions": False,
            }
            if self.model_mode == ModelMode.UN:
                d["num_hidden_layers"] *= 2

        ORDER = [
            "is_decoder",
            "num_hidden_layers",
            "embedding_size",
            "hidden_size",
            "num_attention_heads",
            "hidden_dropout_prob",
            "head_num_hidden_layers",
            "head_hidden_size",
            "position_embedding_type",
            "bi_tie_directions",
        ]
        return {k: d[k] for k in ORDER if k in d}

    @property
    def num_train_epochs(self) -> int:
        if self.task in (Task.CLM, Task.MLM):
            return None
        return 1

    @property
    def max_steps(self) -> Optional[int]:
        if self.task in (Task.CLM, Task.MLM):
            return 16
        return None

    @property
    def save_steps(self) -> Optional[int]:
        if self.task in (Task.CLM, Task.MLM):
            return 8
        return None

    @property
    def eval_steps(self) -> Optional[int]:
        if self.task in (Task.CLM, Task.MLM):
            return 8
        return None

    @property
    def saves_per_epoch(self) -> int:
        if self.task in (Task.CLM, Task.MLM):
            return None
        return 1

    @property
    def evals_per_epoch(self) -> int:
        if self.task in (Task.CLM, Task.MLM):
            return None
        return 1

    @property
    def dataloader_num_workers(self) -> int:
        return 0

    @property
    def learning_rate(self) -> float:
        if self.task in (Task.CLM, Task.MLM):
            return 1e-3
        if self.model_name == ModelName.MAL:
            return 1e-3
        if self.pretraining_task is None:
            return 1e-4
        return 1e-5

    @property
    def weight_decay(self) -> float:
        if self.task in (Task.CLM, Task.MLM):
            return 0.10
        return 0.01

    @property
    def warmup_ratio(self) -> float:
        return 0.05

    @property
    def tr_batch_size(self) -> int:
        return 4

    @property
    def tr_per_device_batch_size(self) -> int:
        return 2

    @property
    def vl_per_device_batch_size(self) -> int:
        return 2

    @property
    def gradient_accumulation_steps(self) -> int:
        return 2

    @property
    def eval_accumulation_steps(self) -> int:
        return 2

    @property
    def tf32(self) -> bool:
        return False

    @property
    def fp16(self) -> bool:
        return False

    @property
    def fp16_full_eval(self) -> bool:
        return self.fp16

    @property
    def bf16(self) -> bool:
        return False

    @property
    def bf16_full_eval(self) -> bool:
        return self.bf16

    @property
    def gradient_checkpointing(self) -> bool:
        return False

    @property
    def weighted_loss(self) -> Optional[WeightedLossAlgorithm]:
        if self.task == Task.FAM:
            return WeightedLossAlgorithm.SAMPLE_REWEIGHTING
        if self.task == Task.BEH:
            return WeightedLossAlgorithm.FOCAL_LOSS
        return None

    @property
    def beta(self) -> Optional[float]:
        if self.task in (Task.FAM, Task.BEH):
            return 0.990
        return None

    @property
    def pretraining_checkpoint(self) -> Optional[str]:
        return None

    @property
    def outfile(self) -> Path:
        return outpath() / f"{self.job}.sh"


def sort_configurations_key(c: Configuration) -> tuple:
    return (
        c.lift_level.value,
        c.model_name.value,
        c.model_size.value,
        c.model_mode.value,
        "" if c.pretraining_task is None else c.pretraining_task.value,
        "a" + c.task.value if c.task in (Task.CLM, Task.MLM) else "z" + c.task.value,
    )


def main():

    outpath().mkdir(parents=True, exist_ok=True)
    for f in outpath().glob("*.sh"):
        if f.is_file():
            f.unlink()

    configurations = product(
        [ModelName.HRR, ModelName.MAM],
        [ModelSize.TN],
        [ModelMode.UN, ModelMode.BI],
        [512],
        [LiftLevel.RAW, LiftLevel.DIS, LiftLevel.DEC],
        [LiftLevel.DEC],
        [TokenizationAlgorithm.BPE],
        [16384],
        [Task.CLM, Task.MLM, Task.DET, Task.FAM, Task.BEH],
        [Task.CLM, Task.MLM, None],
        [0],
    )
    configurations = chain(configurations, product(
        [ModelName.MAL],
        [ModelSize.TN],
        [ModelMode.BI],
        [2 ** 20],
        [LiftLevel.NOP],
        [LiftLevel.DEC],
        [TokenizationAlgorithm.WORDLEVEL],
        [256],
        [Task.DET, Task.FAM, Task.BEH],
        [None],
        [0],
    ))
    configurations = [Configuration(*config) for config in tqdm(configurations)]
    configurations = [config for config in configurations if config.do]
    configurations = sorted(configurations, key=sort_configurations_key)

    for config in tqdm(configurations):
        body = get_body(
            job=config.job,
            tim=config.tim,
            cpu=config.cpu,
            mem=config.mem,
            gpu=config.gpu,
            streaming=config.streaming,
            sync_batch_size=config.sync_batch_size,
            exit_after_map=config.exit_after_map,
            skip_eval_check=config.skip_eval_check,
            model_name=config.model_name,
            arch_config=config.arch_config,
            max_length=config.max_length,
            lift_level=config.lift_level,
            lift_level_ddp=config.lift_level_ddp,
            tokenization_algorithm=config.tokenization_algorithm,
            vocab_size=config.vocab_size,
            task=config.task,
            pretraining_task=config.pretraining_task,
            weighted_loss=config.weighted_loss,
            beta=config.beta,
            pretraining_checkpoint=config.pretraining_checkpoint,
            num_train_epochs=config.num_train_epochs,
            max_steps=config.max_steps,
            save_steps=config.save_steps,
            eval_steps=config.eval_steps,
            saves_per_epoch=config.saves_per_epoch,
            evals_per_epoch=config.evals_per_epoch,
            dataloader_num_workers=config.dataloader_num_workers,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            warmup_ratio=config.warmup_ratio,
            tr_per_device_batch_size=config.tr_per_device_batch_size,
            vl_per_device_batch_size=config.vl_per_device_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            eval_accumulation_steps=config.eval_accumulation_steps,
            tf32=config.tf32,
            fp16=config.fp16,
            fp16_full_eval=config.fp16_full_eval,
            bf16=config.bf16,
            bf16_full_eval=config.bf16_full_eval,
            gradient_checkpointing=config.gradient_checkpointing,
            sort_when_making_archives_contiguous=config.sort_when_making_archives_contiguous,
            seed=config.seed,
        )
        if config.outfile.exists():
            raise FileExistsError(config.outfile)
        print(f"{config.outfile.relative_to(outpath())}")
        config.outfile.write_text(body)


if __name__ == "__main__":
    main()
