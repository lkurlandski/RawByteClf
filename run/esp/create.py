"""
Create sbatch scripts.
"""

from __future__ import annotations
from argparse import ArgumentParser
from dataclasses import dataclass
import datetime
from enum import Enum
from itertools import product
import json
import math
from pathlib import Path
import os
import sys
from typing import Any, Optional
import warnings

from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))  #pylint: disable=wrong-import-position
from src.enums import (
    LiftLevel, Task,
    TokenizationAlgorithm, WeightedLossAlgorithm,
    ExplanationMethod, ExplanationAlgorithm,
)


ARMITAGE = False
ROOT     = Path(os.path.dirname(os.path.realpath(__file__)))


GB = 1024 ** 3


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


def torchrun_str(gpu: int) -> str:
    return (
f"""
OMP_NUM_THREADS=1 \\
torchrun \\
--no-python \\
--nnodes=1 \\
--nproc_per_node={gpu} \\
--rdzv-backend=c10d \\
--rdzv-endpoint=localhost:0 \\
""".replace("\n \\", "").strip()
)


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
    at_per_device_batch_size: int,
    gradient_accumulation_steps: int,
    eval_accumulation_steps: int,
    tf32: bool,
    fp16: bool,
    fp16_full_eval: bool,
    bf16: bool,
    bf16_full_eval: bool,
    gradient_checkpointing: bool,
    sort_when_making_archives_contiguous: bool,
    xai_method: ExplanationMethod,
    xai_algorithm: ExplanationAlgorithm,
    xai_chunk_size: int,
    seed: int,
) -> str: return (
f"""
#!/bin/bash -l

#SBATCH --job-name=att-{job}
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time={tim}
#SBATCH --nodes=1
#SBATCH --cpus-per-task={cpu}
#SBATCH --ntasks=1
#SBATCH --mem={mem}
{"#SBATCH --gres=gpu:a100:" + str(gpu) if gpu > 0 else ""}

source ~/anaconda3/etc/profile.d/conda.sh
conda activate {"RawByteClf" if ARMITAGE else "RawByteClf2"}
{"" if ARMITAGE else "module unload blindfold"}
{"" if ARMITAGE else "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"}
{"" if sort_when_making_archives_contiguous else "export SORT_WHEN_MAKING_ARCHIVES_CONTIGUOUS=0"}

{"" if gpu <= 1 else torchrun_str(gpu)}
python -u \\
src/learn/train.py \\
--root='./output/esp-exe' \\
--streaming={bool_to_str(streaming)} \\
--sync_batch_size={bool_to_str(sync_batch_size)} \\
--exit_after_map={bool_to_str(exit_after_map)} \\
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
--do_attribute \\
--xai_method={xai_method.value} \\
--xai_algorithm={xai_algorithm.value} \\
{f"--xai_chunk_size={xai_chunk_size}" if xai_chunk_size is not None else ""} \\
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
--per_device_attr_batch_size={at_per_device_batch_size} \\
--gradient_accumulation_steps={gradient_accumulation_steps} \\
--eval_accumulation_steps={eval_accumulation_steps} \\
{"--use_cpu" if gpu <= 0 else ""} \\
--tf32={bool_to_str(tf32)} \\
--fp16={bool_to_str(fp16)} \\
--fp16_full_eval={bool_to_str(fp16_full_eval)} \\
--bf16={bool_to_str(bf16)} \\
--bf16_full_eval={bool_to_str(bf16_full_eval)} \\
--gradient_checkpointing={bool_to_str(gradient_checkpointing)} \\
--auto_find_batch_size_and_gradient_accumulation_steps=true \\
""".replace("\n \\", "").strip() + "\n"
)


class ModelName(Enum):
    HRR = "hrrformer"
    MAM = "mamba"
    MAL = "malconv"


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


DATASET_SIZES: dict[Task, tuple[int, int]] = {
    Task.CLM: (820000, 4096),
    Task.MLM: (820000, 4096),
    Task.DET: (29000, 14000),
    Task.FAM: (64000, 16000),
    Task.BEH: (28000, 7000),
}


# This is a nice idea, but I've found that this can be very unreliable.
MODEL_SAMPLES_PER_SECOND: dict[tuple, tuple[float, float]] = {

}


COMPRESSION_RATIOS: dict[tuple[TokenizationAlgorithm, int], float] = {

}


CHECKPOINTS = {
    (LiftLevel.DEC, ModelName.HRR, Task.MLM): "/home/lk3591/Documents/code/RawByteClf/output/esp-exe/lift_level--dec/lift_level_ddp--dec/packing_protocol--any/bits_in_byte--8/tokenization_algorithm--bpe/vocab_size--16384/max_length--65536/model_name--hrrformer/is_decoder--False/num_hidden_layers--32/embedding_size--384/hidden_size--384/hidden_dropout_prob--0.1/head_num_hidden_layers--0/head_hidden_size--0/position_embedding_type--rotary/task--mlm/weighted_loss--none/split_mode--none/max_grad_norm--1.0/weight_decay--0.1/learning_rate--0.001/lr_scheduler_type--linear/warmup_ratio--0.05/optim--adamw_torch/adam_beta1--0.9/adam_beta2--0.999/adam_epsilon--1e-08/max_steps---1/num_train_epochs--1.0/world_size--4/per_device_train_batch_size--1/gradient_accumulation_steps--256/tf32--True/fp16--False/bf16--True/seed--0/results/checkpoints/checkpoint-799",
    (LiftLevel.DEC, ModelName.HRR, Task.CLM): "/home/lk3591/Documents/code/RawByteClf/output/esp-exe/lift_level--dec/lift_level_ddp--dec/packing_protocol--any/bits_in_byte--8/tokenization_algorithm--bpe/vocab_size--16384/max_length--65536/model_name--hrrformer/is_decoder--True/num_hidden_layers--32/embedding_size--384/hidden_size--384/hidden_dropout_prob--0.1/head_num_hidden_layers--0/head_hidden_size--0/position_embedding_type--rotary/task--clm/weighted_loss--none/split_mode--none/max_grad_norm--1.0/weight_decay--0.1/learning_rate--0.001/lr_scheduler_type--linear/warmup_ratio--0.05/optim--adamw_torch/adam_beta1--0.9/adam_beta2--0.999/adam_epsilon--1e-08/max_steps---1/num_train_epochs--1.0/world_size--4/per_device_train_batch_size--1/gradient_accumulation_steps--256/tf32--True/fp16--False/bf16--True/seed--0/results/checkpoints/checkpoint-799",
    (LiftLevel.DEC, ModelName.MAM, Task.MLM): "/home/lk3591/Documents/code/RawByteClf/output/esp-exe/lift_level--dec/lift_level_ddp--dec/packing_protocol--any/bits_in_byte--8/tokenization_algorithm--bpe/vocab_size--16384/max_length--65536/model_name--mamba/is_decoder--False/num_hidden_layers--32/embedding_size--384/hidden_size--384/hidden_dropout_prob--0.0/head_num_hidden_layers--0/head_hidden_size--0/bi_tie_directions--False/task--mlm/weighted_loss--none/split_mode--none/max_grad_norm--1.0/weight_decay--0.1/learning_rate--0.001/lr_scheduler_type--linear/warmup_ratio--0.05/optim--adamw_torch/adam_beta1--0.9/adam_beta2--0.999/adam_epsilon--1e-08/max_steps---1/num_train_epochs--1.0/world_size--4/per_device_train_batch_size--2/gradient_accumulation_steps--128/tf32--True/fp16--False/bf16--True/seed--0/results/checkpoints/checkpoint-799",
    (LiftLevel.DEC, ModelName.MAM, Task.CLM): "/home/lk3591/Documents/code/RawByteClf/output/esp-exe/lift_level--dec/lift_level_ddp--dec/packing_protocol--any/bits_in_byte--8/tokenization_algorithm--bpe/vocab_size--16384/max_length--65536/model_name--mamba/is_decoder--True/num_hidden_layers--64/embedding_size--384/hidden_size--384/hidden_dropout_prob--0.0/head_num_hidden_layers--0/head_hidden_size--0/bi_tie_directions--False/task--clm/weighted_loss--none/split_mode--none/max_grad_norm--1.0/weight_decay--0.1/learning_rate--0.001/lr_scheduler_type--linear/warmup_ratio--0.05/optim--adamw_torch/adam_beta1--0.9/adam_beta2--0.999/adam_epsilon--1e-08/max_steps---1/num_train_epochs--1.0/world_size--4/per_device_train_batch_size--2/gradient_accumulation_steps--128/tf32--True/fp16--False/bf16--True/seed--0/results/checkpoints/checkpoint-799",
    (LiftLevel.DIS, ModelName.HRR, Task.MLM): "/home/lk3591/Documents/code/RawByteClf/output/esp-exe/lift_level--dis/lift_level_ddp--dec/packing_protocol--any/bits_in_byte--8/tokenization_algorithm--bpe/vocab_size--16384/max_length--65536/model_name--hrrformer/is_decoder--False/num_hidden_layers--32/embedding_size--384/hidden_size--384/hidden_dropout_prob--0.1/head_num_hidden_layers--0/head_hidden_size--0/position_embedding_type--rotary/task--mlm/weighted_loss--none/split_mode--none/max_grad_norm--1.0/weight_decay--0.1/learning_rate--0.001/lr_scheduler_type--linear/warmup_ratio--0.05/optim--adamw_torch/adam_beta1--0.9/adam_beta2--0.999/adam_epsilon--1e-08/max_steps---1/num_train_epochs--1.0/world_size--4/per_device_train_batch_size--1/gradient_accumulation_steps--256/tf32--True/fp16--False/bf16--True/seed--0/results/checkpoints/checkpoint-799",
    (LiftLevel.DIS, ModelName.HRR, Task.CLM): "/home/lk3591/Documents/code/RawByteClf/output/esp-exe/lift_level--dis/lift_level_ddp--dec/packing_protocol--any/bits_in_byte--8/tokenization_algorithm--bpe/vocab_size--16384/max_length--65536/model_name--hrrformer/is_decoder--True/num_hidden_layers--32/embedding_size--384/hidden_size--384/hidden_dropout_prob--0.1/head_num_hidden_layers--0/head_hidden_size--0/position_embedding_type--rotary/task--clm/weighted_loss--none/split_mode--none/max_grad_norm--1.0/weight_decay--0.1/learning_rate--0.001/lr_scheduler_type--linear/warmup_ratio--0.05/optim--adamw_torch/adam_beta1--0.9/adam_beta2--0.999/adam_epsilon--1e-08/max_steps---1/num_train_epochs--1.0/world_size--4/per_device_train_batch_size--1/gradient_accumulation_steps--256/tf32--True/fp16--False/bf16--True/seed--0/results/checkpoints/checkpoint-799",
    (LiftLevel.DIS, ModelName.MAM, Task.MLM): "/home/lk3591/Documents/code/RawByteClf/output/esp-exe/lift_level--dis/lift_level_ddp--dec/packing_protocol--any/bits_in_byte--8/tokenization_algorithm--bpe/vocab_size--16384/max_length--65536/model_name--mamba/is_decoder--False/num_hidden_layers--32/embedding_size--384/hidden_size--384/hidden_dropout_prob--0.0/head_num_hidden_layers--0/head_hidden_size--0/bi_tie_directions--False/task--mlm/weighted_loss--none/split_mode--none/max_grad_norm--1.0/weight_decay--0.1/learning_rate--0.001/lr_scheduler_type--linear/warmup_ratio--0.05/optim--adamw_torch/adam_beta1--0.9/adam_beta2--0.999/adam_epsilon--1e-08/max_steps---1/num_train_epochs--1.0/world_size--4/per_device_train_batch_size--2/gradient_accumulation_steps--128/tf32--True/fp16--False/bf16--True/seed--0/results/checkpoints/checkpoint-799",
    (LiftLevel.DIS, ModelName.MAM, Task.CLM): "/home/lk3591/Documents/code/RawByteClf/output/esp-exe/lift_level--dis/lift_level_ddp--dec/packing_protocol--any/bits_in_byte--8/tokenization_algorithm--bpe/vocab_size--16384/max_length--65536/model_name--mamba/is_decoder--True/num_hidden_layers--64/embedding_size--384/hidden_size--384/hidden_dropout_prob--0.0/head_num_hidden_layers--0/head_hidden_size--0/bi_tie_directions--False/task--clm/weighted_loss--none/split_mode--none/max_grad_norm--1.0/weight_decay--0.1/learning_rate--0.001/lr_scheduler_type--linear/warmup_ratio--0.05/optim--adamw_torch/adam_beta1--0.9/adam_beta2--0.999/adam_epsilon--1e-08/max_steps---1/num_train_epochs--1.0/world_size--4/per_device_train_batch_size--2/gradient_accumulation_steps--128/tf32--True/fp16--False/bf16--True/seed--0/results/checkpoints/checkpoint-799",
    (LiftLevel.RAW, ModelName.HRR, Task.MLM): "/home/lk3591/Documents/code/RawByteClf/output/esp-exe/lift_level--raw/lift_level_ddp--dec/packing_protocol--any/bits_in_byte--8/tokenization_algorithm--bpe/vocab_size--16384/max_length--65536/model_name--hrrformer/is_decoder--False/num_hidden_layers--32/embedding_size--384/hidden_size--384/hidden_dropout_prob--0.1/head_num_hidden_layers--0/head_hidden_size--0/position_embedding_type--rotary/task--mlm/weighted_loss--none/split_mode--none/max_grad_norm--1.0/weight_decay--0.1/learning_rate--0.001/lr_scheduler_type--linear/warmup_ratio--0.05/optim--adamw_torch/adam_beta1--0.9/adam_beta2--0.999/adam_epsilon--1e-08/max_steps---1/num_train_epochs--1.0/world_size--4/per_device_train_batch_size--1/gradient_accumulation_steps--256/tf32--True/fp16--False/bf16--True/seed--0/results/checkpoints/checkpoint-799",
    (LiftLevel.RAW, ModelName.HRR, Task.CLM): "/home/lk3591/Documents/code/RawByteClf/output/esp-exe/lift_level--raw/lift_level_ddp--dec/packing_protocol--any/bits_in_byte--8/tokenization_algorithm--bpe/vocab_size--16384/max_length--65536/model_name--hrrformer/is_decoder--True/num_hidden_layers--32/embedding_size--384/hidden_size--384/hidden_dropout_prob--0.1/head_num_hidden_layers--0/head_hidden_size--0/position_embedding_type--rotary/task--clm/weighted_loss--none/split_mode--none/max_grad_norm--1.0/weight_decay--0.1/learning_rate--0.001/lr_scheduler_type--linear/warmup_ratio--0.05/optim--adamw_torch/adam_beta1--0.9/adam_beta2--0.999/adam_epsilon--1e-08/max_steps---1/num_train_epochs--1.0/world_size--4/per_device_train_batch_size--1/gradient_accumulation_steps--256/tf32--True/fp16--False/bf16--True/seed--0/results/checkpoints/checkpoint-799",
    (LiftLevel.RAW, ModelName.MAM, Task.MLM): "/home/lk3591/Documents/code/RawByteClf/output/esp-exe/lift_level--raw/lift_level_ddp--dec/packing_protocol--any/bits_in_byte--8/tokenization_algorithm--bpe/vocab_size--16384/max_length--65536/model_name--mamba/is_decoder--False/num_hidden_layers--32/embedding_size--384/hidden_size--384/hidden_dropout_prob--0.0/head_num_hidden_layers--0/head_hidden_size--0/bi_tie_directions--False/task--mlm/weighted_loss--none/split_mode--none/max_grad_norm--1.0/weight_decay--0.1/learning_rate--0.001/lr_scheduler_type--linear/warmup_ratio--0.05/optim--adamw_torch/adam_beta1--0.9/adam_beta2--0.999/adam_epsilon--1e-08/max_steps---1/num_train_epochs--1.0/world_size--4/per_device_train_batch_size--2/gradient_accumulation_steps--128/tf32--True/fp16--False/bf16--True/seed--0/results/checkpoints/checkpoint-799",
    (LiftLevel.RAW, ModelName.MAM, Task.CLM): "/home/lk3591/Documents/code/RawByteClf/output/esp-exe/lift_level--raw/lift_level_ddp--dec/packing_protocol--any/bits_in_byte--8/tokenization_algorithm--bpe/vocab_size--16384/max_length--65536/model_name--mamba/is_decoder--True/num_hidden_layers--64/embedding_size--384/hidden_size--384/hidden_dropout_prob--0.0/head_num_hidden_layers--0/head_hidden_size--0/bi_tie_directions--False/task--clm/weighted_loss--none/split_mode--none/max_grad_norm--1.0/weight_decay--0.1/learning_rate--0.001/lr_scheduler_type--linear/warmup_ratio--0.05/optim--adamw_torch/adam_beta1--0.9/adam_beta2--0.999/adam_epsilon--1e-08/max_steps---1/num_train_epochs--1.0/world_size--4/per_device_train_batch_size--2/gradient_accumulation_steps--128/tf32--True/fp16--False/bf16--True/seed--0/results/checkpoints/checkpoint-799",
}
# for path in CHECKPOINTS.values():
    # if not os.path.exists(path):
    #     raise FileNotFoundError(path)

def add_ensemble_checkpoints():
    for model_name in (ModelName.HRR, ModelName.MAM):
        for task in (Task.CLM, Task.MLM):
            CHECKPOINTS[(LiftLevel.ALL, model_name, task)] = {
                lift_level.value: CHECKPOINTS[(lift_level, model_name, task)]
                for lift_level in (LiftLevel.RAW, LiftLevel.DIS, LiftLevel.DEC)
            }

add_ensemble_checkpoints()


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
    xai_method: ExplanationMethod
    xai_algorithm: ExplanationAlgorithm
    xai_chunk_size: int
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
            if self.model_size != ModelSize.CO:
                return False

        # Adjust as desired.
        if self.tokenization_algorithm != TokenizationAlgorithm.WORDLEVEL:
            return False
        if self.vocab_size != 256:
            return False
        if self.lift_level_ddp != LiftLevel.DEC:
            return False
        if self.max_length != 2**20:
            return False
        if self.model_name != ModelName.MAL:
            return False
        if self.model_size != ModelSize.CO:
            return False
        if self.lift_level != LiftLevel.NOP:
            return False
        if self.pretraining_task is not None:
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
            f"{self.xai_method.value}",
            f"{self.xai_algorithm.value}",
            f"{self.xai_chunk_size if self.xai_chunk_size is not None else 'nop'}",
            f"{self.seed}",
        ]).replace("--", "-").rstrip("-")

    @property
    def tim(self) -> str:
        if self.model_size != ModelSize.CO:
            raise NotImplementedError()

        if self.task in (Task.CLM, Task.MLM):

            if self.lift_level == LiftLevel.RAW:
                h = 520
            if self.lift_level == LiftLevel.DIS:
                h = 208
            if self.lift_level == LiftLevel.DEC:
                h = 260
            if self.lift_level == LiftLevel.ALL:
                h = 520 + 208 + 260  # TODO: measure
            if self.lift_level == LiftLevel.NOP:
                h = 1000  # TODO: measure

            h /= (65536 / self.max_length)
            h /= self.gpu

            return seconds_to_slurm_time(h * 3600)

        if self.model_name == ModelName.MAL:
            k = self.max_length // 65536
            z = 3 if self.lift_level == LiftLevel.ALL else 1
            if self.task == Task.DET:
                h = 3
            if self.task == Task.FAM:
                h = 6
            if self.task == Task.BEH:
                h = 2
            return seconds_to_slurm_time(3600 * h * z * k)

        if self.task == Task.DET:
            n_tr = 24614
            n_vl = 8322
        if self.task == Task.FAM:
            n_tr = 49168
            n_vl = 12631
        if self.task == Task.BEH:
            n_tr = 13369
            n_vl = 3343

        n_tr *= self.num_train_epochs / max(self.gpu, 1)
        n_vl *= self.num_train_epochs / max(self.gpu, 1)

        if self.lift_level == LiftLevel.RAW:
            tr_f = 1.0
            vl_f = 1.0
        if self.lift_level == LiftLevel.DIS:
            tr_f = 0.760 / 3.617
            vl_f = 3.192 / 16.163
        if self.lift_level == LiftLevel.DEC:
            tr_f = 0.760 / 2.978
            vl_f = 3.192 / 13.02
        if self.lift_level == LiftLevel.ALL:
            tr_f = 1.0 + 0.760 / 3.617 + 0.760 / 2.978   # TODO: measure
            vl_f = 1.0 + 3.192 / 16.163 + 3.192 / 13.02  # TODO: measure
        if self.lift_level == LiftLevel.NOP:
            tr_f = 2.0  # TODO: measure
            vl_f = 2.0  # TODO: measure

        if self.model_name == ModelName.HRR and self.model_mode == ModelMode.BI:
            tr_samples_per_second = 0.766
            vl_samples_per_second = 3.226
        if self.model_name == ModelName.HRR and self.model_mode == ModelMode.UN:
            tr_samples_per_second = 0.760
            vl_samples_per_second = 3.192
        if self.model_name == ModelName.MAM and self.model_mode == ModelMode.BI:
            tr_samples_per_second = 0.766
            vl_samples_per_second = 3.226
        if self.model_name == ModelName.MAM and self.model_mode == ModelMode.UN:
            tr_samples_per_second = 0.760
            vl_samples_per_second = 3.192

        tr_samples_per_second /= tr_f
        vl_samples_per_second /= vl_f

        s_tr = n_tr / tr_samples_per_second * 1.50
        s_vl = n_vl / vl_samples_per_second * 1.50

        return seconds_to_slurm_time(s_tr + s_vl)

    @property
    def cpu(self) -> int:
        if self.streaming:
            return 4 * self.gpu
        return 2 * self.gpu

    @property
    def mem(self) -> int:
        if self.streaming:
            return bytes_to_slurm_mem(90 * self.gpu * GB)

        # The memory required for the classification tasks can be estimated.
        k = (self.tokenization_algorithm, self.vocab_size)
        c = COMPRESSION_RATIOS.get(k, 1.0)
        t = c * sum(DATASET_SIZES[self.task]) * self.max_length
        b = t * 8
        b = b + (16 * GB)
        return bytes_to_slurm_mem(b)

    @property
    def gpu(self) -> int:
        if self.model_name == ModelName.MAL:
            return 1
        if self.task in (Task.CLM, Task.MLM):
            if self.max_length >= 32768:
                return 4
            if self.max_length >= 8192:
                return 2
            return 1
        if self.task == Task.FAM:
            return 4
        if self.task == Task.DET:
            return 2
        if self.task == Task.BEH:
            return 1
        raise RuntimeError()

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

        # Unfortunately, tie-ing the forward and backward directions is really
        # not going to work without substantial refactoring when doing DDP.

        # Baseline MalConv2 from original authors.
        if self.model_name == ModelName.MAL:
            d = {
                "channels": 256,
                "stride": 64,
                "kernel_size": 64,
                "head_num_hidden_layers": 1,
                "head_hidden_size": 128,
            }
            if self.vocab_size == 256:
                d["embedding_size"] = 8
            elif self.vocab_size == 16384:
                d["embedding_size"] = 384
            else:
                raise NotImplementedError(f"{self.vocab_size}")
            return d

        # When using gradient checkpointing, the hidden_size is much more impactful
        # for causing CUDA OOM errors, hence the increasing depth of these architectures.
        NUM_BLOCKS = {
            ModelSize.TN:  2,
            ModelSize.SM:  4,
            ModelSize.MD:  8,
            ModelSize.LG: 12,
            ModelSize.HG: 16,
            ModelSize.CO: 32,
        }

        d = {
            "is_decoder": self.model_mode == ModelMode.UN,
            "num_hidden_layers": NUM_BLOCKS[self.model_size],
            "embedding_size": 384,
            "hidden_size": 384,
            "head_num_hidden_layers": 0,
            "head_hidden_size": 0,
        }

        if self.model_name == ModelName.HRR:
            d |= {
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
            return 1
        return 5

    @property
    def max_steps(self) -> Optional[int]:
        return None

    @property
    def save_steps(self) -> Optional[int]:
        return None

    @property
    def eval_steps(self) -> Optional[int]:
        return None

    @property
    def saves_per_epoch(self) -> int:
        if self.task in (Task.CLM, Task.MLM):
            return 32
        return 1

    @property
    def evals_per_epoch(self) -> int:
        if self.task in (Task.CLM, Task.MLM):
            return 8
        return 1

    @property
    def dataloader_num_workers(self) -> int:
        # One additional process will engage the prefetching.
        # When streaming, we can rely on tokenizers' parallelization for speed.
        # Fuck. Now I'm getting the stupid parallel tokenizers warning. Just make it 0. Don't care.
        if self.gpu == 0:
            return 0
        # When using byte-level embeddings, we don't use tokenizers, so we can use multiple loaders.
        if self.streaming and not self.vocab_size == 256:
            return 0
        return self.cpu // self.gpu - 1

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
        if self.task in (Task.CLM, Task.MLM):
            return 0.05
        return 0.05

    @property
    def tr_batch_size(self) -> int:
        if self.task in (Task.CLM, Task.MLM):
            return 1024
        return 64

    # @property
    # def per_device_batch_size(self) -> int:
    #     return 2

    # Mamba can handle two samples during eval, but only one during training.
    # HRRFormer can handle two samples during training, but only one during eval.
    # Furthermore the CUDA OOMs only pop up when training in multi GPU setting.
    # Wow, fuck dude you're stupid. You had sync_batch_size=True, so it was all
    # getting set to 1 anyway.
    # Jesus christ, apparently, these models can fit like 8x larger batch sizes
    # for classification? WTF. Don't care.

    @property
    def tr_per_device_batch_size(self) -> int:
        if self.model_name == ModelName.MAL:
            return 64

        f = int(65536 / self.max_length)

        if self.task in (Task.DET, Task.FAM, Task.BEH):
            if self.model_name == ModelName.MAM and self.lift_level == LiftLevel.ALL:
                return 3 * f
            return 4 * f
        if self.model_name == ModelName.HRR:
            return 2 * f
        if self.model_name == ModelName.MAM:
            return 1 * f

        raise NotImplementedError(f"{self.model_name=}")

    @property
    def vl_per_device_batch_size(self) -> int:
        if self.model_name == ModelName.MAL:
            return 64

        f = int(65536 / self.max_length)

        if self.task in (Task.DET, Task.FAM, Task.BEH):
            if self.model_name == ModelName.MAM and self.lift_level == LiftLevel.ALL:
                return 12 * f
            return 16 * f
        if self.model_name == ModelName.HRR:
            return 1 * f
        if self.model_name == ModelName.MAM:
            return 2 * f

        raise NotImplementedError(f"{self.model_name=}")

    @property
    def at_per_device_batch_size(self) -> int:
        if self.model_name != ModelName.MAL:
            raise NotImplementedError(f"{self.model_name=}")
        if self.xai_algorithm == ExplanationAlgorithm.IGRD:
            return 4
        if self.xai_algorithm == ExplanationAlgorithm.GSHP:
            return 32
        return 256

    @property
    def gradient_accumulation_steps(self) -> int:
        return self.tr_batch_size // (self.tr_per_device_batch_size * self.gpu)

    @property
    def eval_accumulation_steps(self) -> int:
        return 64 // self.vl_per_device_batch_size

    @property
    def tf32(self) -> bool:
        if self.gpu == 0:
            return False
        return True

    @property
    def fp16(self) -> bool:
        if self.gpu == 0:
            return False
        if self.model_name == ModelName.MAL:
            return False
        return False

    @property
    def fp16_full_eval(self) -> bool:
        return self.fp16

    @property
    def bf16(self) -> bool:
        if self.gpu == 0:
            return False
        if self.model_name == ModelName.MAL:
            return False
        return True

    @property
    def bf16_full_eval(self) -> bool:
        return self.bf16

    @property
    def gradient_checkpointing(self) -> bool:
        if self.gpu == 0:
            return False
        if self.model_name == ModelName.MAL:
            return False
        return True

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
        # If we use a different dropout for finetuning, it is challenging to resolve
        # the pretraining path were we to pass a "-1" to the program. This is even more complex
        # when we try and finetune the ensemble of pretrained models, so we just pass raw paths.
        if self.pretraining_task is not None:
            checkpoint = CHECKPOINTS[(self.lift_level, self.model_name, self.pretraining_task)]
            if isinstance(checkpoint, dict):
                checkpoint = json.dumps(checkpoint)
            return checkpoint
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


class Product:

    def __init__(self, *args):
        self.args = args

    def __iter__(self):
        return product(*self.args)

    def __len__(self):
        n = 1
        for a in self.args:
            n *= len(a)
        return n


def main():

    global ARMITAGE  # pylint: disable=global-statement

    parser = ArgumentParser()
    parser.add_argument("--armitage", action="store_true")
    parser.add_argument("--no_remove", action="store_true")
    parser.add_argument("--dependencies", action="store_true")
    args = parser.parse_args()

    ARMITAGE = args.armitage

    if not args.no_remove:
        for f in outpath().iterdir():
            if f.name[0] != ".":
                f.unlink()
    outpath().mkdir(parents=True, exist_ok=True)

    configurations = Product(
        ModelName,
        ModelSize,
        ModelMode,
        # (4096, 8192, 16384, 32768, 65536),
        (2 ** 20,),
        LiftLevel,
        # list(LiftLevel) + [None],
        [LiftLevel.DEC],
        TokenizationAlgorithm,
        (256, 1024, 4096, 16384),
        Task,
        [None, Task.CLM, Task.MLM],
        [ExplanationMethod.FUN, ExplanationMethod.NUM],
        list(ExplanationAlgorithm),
        [None],
        (0,),
    )
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
            at_per_device_batch_size=config.at_per_device_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            eval_accumulation_steps=config.eval_accumulation_steps,
            tf32=config.tf32,
            fp16=config.fp16,
            fp16_full_eval=config.fp16_full_eval,
            bf16=config.bf16,
            bf16_full_eval=config.bf16_full_eval,
            gradient_checkpointing=config.gradient_checkpointing,
            sort_when_making_archives_contiguous=config.sort_when_making_archives_contiguous,
            xai_method=config.xai_method,
            xai_algorithm=config.xai_algorithm,
            xai_chunk_size=config.xai_chunk_size,
            seed=config.seed,
        )
        if config.outfile.exists():
            raise FileExistsError(config.outfile)
        config.outfile.write_text(body)

    with open(ROOT / "execute.sh", "w") as fp:
        for i, config in enumerate(configurations):
            if args.dependencies:
                var = "j"
                dep = ""

                newlines = 1 if config.task == Task.MLM and i > 0 else 0

                if config.pretraining_task is None:
                    var = f"j_{config.task.value[0]}"
                if config.pretraining_task is not None:
                    dep = f"--dependency=\"afterok:$j_{config.pretraining_task.value[0]}:$j_{config.task.value[0]}\""

                f = config.outfile.as_posix().replace("/home/lk3591/Documents/code/RawByteClf/", "./")
                awk = "awk '{print $4}'"
                new = "\n" * newlines
                s = f"{new}{var}=$(sbatch{' ' + dep if dep else ''} '{f}' | {awk})\n"
                fp.write(s)
            else:
                f = config.outfile.as_posix().replace("/home/lk3591/Documents/code/RawByteClf/", "./")
                s = f"sbatch {f}\n"
                fp.write(s)


if __name__ == "__main__":
    main()
