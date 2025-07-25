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
import shutil
import sys
from typing import Any, Optional
import warnings

from tqdm import tqdm

# pylint: disable=wrong-import-position
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# pylint: enable=wrong-import-position

from src.enums import LiftLevel, Task, TokenizationAlgorithm, WeightedLossAlgorithm


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
    probability_to_pack: float,
    probability_to_unpack: float,
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

source ~/anaconda3/etc/profile.d/conda.sh
conda activate {"RawByteClf" if ARMITAGE else "RawByteClf2"}
{"" if ARMITAGE else "module unload blindfold"}
{"" if ARMITAGE else "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"}
{"" if sort_when_making_archives_contiguous else "export SORT_WHEN_MAKING_ARCHIVES_CONTIGUOUS=0"}

export LMLM_PACK_AND_UNPACK=1

python -u \\
src/learn/train.py \\
--root='./output/esp-pck' \\
--streaming={bool_to_str(streaming)} \\
--sync_batch_size={bool_to_str(sync_batch_size)} \\
--exit_after_map={bool_to_str(exit_after_map)} \\
--skip_eval_check={bool_to_str(skip_eval_check)} \\
--dataset_backend="HF" \\
--model_name_or_path='{model_name.value}' \\
--arch_config='{json.dumps(arch_config)}' \\
--data_read_bytes={sys.maxsize} \\
--max_length={max_length} \\
--probability_to_pack={probability_to_pack} \\
--probability_to_unpack={probability_to_unpack} \\
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
--do_eval \\
--output_dir='/tmp' \\
--save_strategy='{"epoch" if save_steps is None else "steps"}' \\
{"--max_steps=" + str(max_steps) if max_steps is not None else ""} \\
{"--save_steps=" + str(save_steps) if save_steps is not None else ""} \\
{"--num_train_epochs=" + str(num_train_epochs) if max_steps is None else ""} \\
{"--saves_per_epoch=" + str(saves_per_epoch) if save_steps is None else ""} \\
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
--save_total_limit=-1 \\
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


CHECKPOINTS = {
    (Task.CLM, None, None): "",
    (Task.CLM, None, None): "",
    (Task.CLM, None, None): "",
    (Task.MLM, None, None): "",
    (Task.MLM, None, None): "",
    (Task.MLM, None, None): "",
}


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
    probability_to_pack: float
    probability_to_unpack: float
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

        if self.probability_to_pack == 0.67 and self.probability_to_unpack != 0.5:
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
            f"{self.probability_to_pack}{'0' * (4 - len(str(self.probability_to_pack)))}",
            f"{self.probability_to_unpack}{'0' * (4 - len(str(self.probability_to_unpack)))}",
            f"{self.seed}",
        ]).replace("--", "-").rstrip("-")

    @property
    def tim(self) -> str:
        if self.task in (Task.CLM, Task.MLM):
            return seconds_to_slurm_time(3600 * 54)
        return seconds_to_slurm_time(3600 * 18)

    @property
    def cpu(self) -> int:
        if self.streaming:
            return 4 * self.gpu
        return 2 * self.gpu

    @property
    def mem(self) -> int:
        return bytes_to_slurm_mem(64 * self.gpu * GB)

    @property
    def gpu(self) -> int:
        return 1

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
        return True

    @property
    def arch_config(self) -> dict:

        # Unfortunately, tie-ing the forward and backward directions is really
        # not going to work without substantial refactoring when doing DDP.

        # Baseline MalConv2 from original authors.
        if self.model_name == ModelName.MAL:
            d = {
                "mode": "gcg",
                "channels": 256,
                "stride": 64,
                "kernel_size": 64,
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
        HIDDEN_SIZE = {
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
            "embedding_size": HIDDEN_SIZE[self.model_size],
            "hidden_size": HIDDEN_SIZE[self.model_size],
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
    def num_train_epochs(self) -> int | float:
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
        return 1

    @property
    def evals_per_epoch(self) -> int:
        return 1

    @property
    def dataloader_num_workers(self) -> int:
        # One additional process will engage the prefetching.
        # When streaming, we can rely on tokenizers' parallelization for speed.
        # Fuck. Now I'm getting the stupid parallel tokenizers warning. Just make it 0. Don't care.
        if self.gpu == 0:
            return 0
        # When using byte-level embeddings, we don't use tokenizers, so we can use multiple loaders.
        if self.streaming and self.vocab_size != 256:
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

    @property
    def tr_per_device_batch_size(self) -> int:
        return 16

    @property
    def vl_per_device_batch_size(self) -> int:
        return 16

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
            checkpoint = CHECKPOINTS[(self.pretraining_task, self.probability_to_pack, self.probability_to_unpack)]
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
            if not f.name[0] == ".":
                f.unlink()
    outpath().mkdir(parents=True, exist_ok=True)

    configurations = product(
        [ModelName.MAM],
        [ModelSize.HG],
        # [ModelMode.UN, ModelMode.BI],
        [ModelMode.BI],
        [65536],
        [LiftLevel.RAW],
        [LiftLevel.DEC],
        [TokenizationAlgorithm.BPE],
        [16384],
        [Task.DET],
        # [None, Task.CLM, Task.MLM],
        [None],
        [0.0, 0.67, 1.0],
        [0.0, 0.50, 1.0],
        [0],
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
            probability_to_pack=config.probability_to_pack,
            probability_to_unpack=config.probability_to_unpack,
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
        config.outfile.write_text(body)


if __name__ == "__main__":
    main()
