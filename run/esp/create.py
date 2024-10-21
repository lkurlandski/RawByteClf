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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))  #pylint: disable=wrong-import-position
from src.enums import LiftLevel, Task, TokenizationAlgorithm


ACTION   = None
ARMITAGE = False
ROOT     = Path(os.path.dirname(os.path.realpath(__file__)))

GB = 1024 ** 3


class Action(Enum):
    DEBUG   = "dbg"
    PREPARE = "pre"
    TIME    = "tim"
    EXECUTE = "exe"
    OVERFIT = "oft"


def outpath() -> Path:
    return ROOT / "sbatch" / ACTION.value


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


def torchrun_str(cpu: int, gpu: int) -> str:
    return (
f"""
OMP_NUM_THREADS={cpu // gpu} \\
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
    exit_after_map: bool,
    model_name: ModelName,
    arch_config: dict,
    max_length: int,
    lift_level: LiftLevel,
    tokenization_algorithm: TokenizationAlgorithm,
    vocab_size: int,
    pretraining_task: Optional[Task],
    task: Task,
    num_train_epochs: int,
    max_steps: Optional[int],
    save_steps: Optional[int],
    eval_steps: Optional[int],
    saves_and_evals_per_epochs: int,
    dataloader_num_workers: int,
    learning_rate: float,
    tr_batch_size: int,
    vl_batch_size: int,
    tf32: bool,
    bf16: bool,
    bf16_full_eval: bool,
    seed: int,
) -> str: return (
f"""
#!/bin/bash -l

#SBATCH --job-name={ACTION.value}-{job}
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

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


{"" if gpu <= 1 else torchrun_str(cpu, gpu)}
python -u \\
src/learn/train.py \\
--root='./output/esp-{ACTION.value}' \\
--streaming={bool_to_str(streaming)} \\
--exit_after_map={bool_to_str(exit_after_map)} \\
--auto_find_batch_size_and_gradient_accumulation_steps=true \\
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
--save_strategy='{"epoch" if save_steps is None else "steps"}' \\
--eval_strategy='{"epoch" if eval_steps is None else "steps"}' \\
{"--max_steps=" + str(max_steps) if max_steps is not None else ""} \\
{"--save_steps=" + str(save_steps) if save_steps is not None else ""} \\
{"--eval_steps=" + str(eval_steps) if eval_steps is not None else ""} \\
{"--num_train_epochs=" + str(num_train_epochs) if max_steps is None else ""} \\
{"--saves_per_epoch=" + str(saves_and_evals_per_epochs) if save_steps is None else ""} \\
{"--evals_per_epoch=" + str(saves_and_evals_per_epochs) if eval_steps is None else ""} \\
--logging_steps=1 \\
--dataloader_num_workers={dataloader_num_workers} \\
--optim="adamw_torch" \\
--learning_rate={learning_rate} \\
--lr_scheduler_type="linear" \\
--weight_decay=0.01 \\
--adam_beta1=0.900 \\
--adam_beta2=0.999 \\
--max_grad_norm=1.0 \\
--save_total_limit=-1 \\
--per_device_train_batch_size={tr_batch_size} \\
--per_device_eval_batch_size={vl_batch_size} \\
--gradient_accumulation_steps=1 \\
--eval_accumulation_steps=64 \\
--load_best_model_at_end \\
--tf32={bool_to_str(tf32)} \\
--bf16={bool_to_str(bf16)} \\
--bf16_full_eval={bool_to_str(bf16_full_eval)} \\
--gradient_checkpointing=true
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


DATASET_SIZES: dict[Task, tuple[int, int]] = {
    Task.CLM: (820000, 4096),
    Task.MLM: (820000, 4096),
    Task.DET: (29000, 14000),
    Task.FAM: (64000, 16000),
    Task.BEH: (28000, 7000),
}


MODEL_SAMPLES_PER_SECOND: dict[tuple, tuple[float, float]] = {

    (ModelName.HRR, ModelSize.TN, ModelMode.BI, 16384, "cf"): (06.075, 045.082),
    (ModelName.HRR, ModelSize.TN, ModelMode.BI, 16384, "lm"): (14.721, 016.544),
    (ModelName.HRR, ModelSize.TN, ModelMode.UN, 16384, "cf"): (06.574, 043.032),
    (ModelName.HRR, ModelSize.TN, ModelMode.UN, 16384, "lm"): (09.022, 013.255),
    (ModelName.MAM, ModelSize.TN, ModelMode.BI, 16384, "cf"): (13.481, 117.289),
    (ModelName.MAM, ModelSize.TN, ModelMode.BI, 16384, "lm"): (13.873, 022.896),
    (ModelName.MAM, ModelSize.TN, ModelMode.UN, 16384, "cf"): (17.266, 090.022),
    (ModelName.MAM, ModelSize.TN, ModelMode.UN, 16384, "lm"): (12.384, 019.876),

    # TODO: these A) contian some (unreliable) estimates and B) computed the
    # cf times for Task.DET not Task.BEH.
    (ModelName.HRR, ModelSize.MD, ModelMode.BI, 16384, "cf"): (2.443, 41.315),
    (ModelName.HRR, ModelSize.MD, ModelMode.BI, 16384, "lm"): (0.757, 13.339), # est tr
    (ModelName.HRR, ModelSize.MD, ModelMode.UN, 16384, "cf"): (1.971, 34.166),
    (ModelName.HRR, ModelSize.MD, ModelMode.UN, 16384, "lm"): (0.448, 07.884), # est tr
    (ModelName.MAM, ModelSize.MD, ModelMode.BI, 16384, "cf"): (0.918, 16.834),
    (ModelName.MAM, ModelSize.MD, ModelMode.BI, 16384, "lm"): (0.221, 03.885), # est tr, vl
    (ModelName.MAM, ModelSize.MD, ModelMode.UN, 16384, "cf"): (1.533, 27.511),
    (ModelName.MAM, ModelSize.MD, ModelMode.UN, 16384, "lm"): (0.360, 06.349), # est tr, vl

}


COMPRESSION_RATIOS: dict[tuple[TokenizationAlgorithm, int], float] = {

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

        # Adjust as desired.
        if self.tokenization_algorithm != TokenizationAlgorithm.BPE:
            return False
        if self.vocab_size != 16384:
            return False

        if ACTION == Action.DEBUG:
            if self.max_length != 1024:
                return False
            if self.task in (Task.DET, Task.FAM):
                return False
            if self.model_size != ModelSize.TN:
                return False
            if self.lift_level != LiftLevel.RAW:
                return False

        if ACTION == Action.PREPARE:
            if self.max_length != 65536:
                return False
            if self.model_size != ModelSize.TN:
                return False
            if self.model_mode != ModelMode.BI:
                return False
            if self.pretraining_task is not None:
                return False
            if self.task == Task.CLM:
                return False

        if ACTION == Action.TIME:
            if self.max_length != 65536:
                return False
            if self.model_size != ModelSize.TN:
                return False
            if self.task not in (Task.CLM, Task.MLM, Task.BEH):
                return False
            if self.pretraining_task is not None:
                return False
            if self.lift_level != LiftLevel.RAW:
                return False

        if ACTION == Action.EXECUTE:
            if self.max_length != 65536:
                return False
            if self.model_size != ModelSize.MD:
                return False

        if ACTION == Action.OVERFIT:
            if self.max_length != 4096:
                return False
            if self.lift_level != LiftLevel.RAW:
                return False
            if self.model_size not in (ModelSize.TN, ModelSize.SM, ModelSize.MD):
                return False
            if self.task != Task.DET:
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
            f"{self.seed}",
            f"{self.num_train_epochs if ACTION == Action.OVERFIT else ''}",
        ])

    @property
    def tim(self) -> str:
        if ACTION == Action.PREPARE:
            if self.task in (Task.CLM, Task.MLM):
                return seconds_to_slurm_time(7200)
            if self.task == Task.FAM:
                return seconds_to_slurm_time(3600)
            return seconds_to_slurm_time(1800)

        if ACTION == Action.TIME:
            f = {
                ModelSize.TN: 1.0,
                ModelSize.SM: 2.0,
                ModelSize.MD: 3.0,
                ModelSize.LG: 4.0,
                ModelSize.HG: 5.0,
            }
            s = 3600 if self.task in (Task.CLM, Task.MLM) else 1800
            return seconds_to_slurm_time(s * f[self.model_size])

        t = "lm" if self.task in (Task.CLM, Task.MLM) else "cf"
        k = (self.model_name, self.model_size, self.model_mode, self.vocab_size, t)
        if k in MODEL_SAMPLES_PER_SECOND:
            t_tr, t_vl = MODEL_SAMPLES_PER_SECOND[k]
        else:
            warnings.warn(f"Samples/second unknown for {k=}.")
            t_tr = 10
            t_vl = 25

        n_tr, n_vl = DATASET_SIZES[self.task]
        s_tr = n_tr * self.num_train_epochs / t_tr / self.gpu
        s_vl = n_vl * self.num_train_epochs / t_vl / self.gpu
        s = s_tr + s_vl + 3600
        return seconds_to_slurm_time(s)

    @property
    def cpu(self) -> int:
        if ACTION == Action.PREPARE:
            return 16
        if ACTION == Action.TIME:
            return 4
        if self.task in (Task.CLM, Task.MLM):
            return 8
        return 4

    @property
    def mem(self) -> int:
        if ACTION == Action.PREPARE:
            return bytes_to_slurm_mem(64 * GB)
        if self.streaming:
            return bytes_to_slurm_mem(self.gpu * 64 * GB)

        k = (self.tokenization_algorithm, self.vocab_size)
        if k in COMPRESSION_RATIOS:
            c = COMPRESSION_RATIOS[k]
        else:
            warnings.warn(f"Compression ratio unknown for {k=}.")
            c = 1.0

        t = c * sum(DATASET_SIZES[self.task]) * self.max_length
        b = t * 8
        b = b + (16 * GB)
        return bytes_to_slurm_mem(b)

    @property
    def gpu(self) -> int:
        if ACTION == Action.PREPARE:
            return 0
        if ACTION == Action.TIME:
            return 1
        if ACTION == Action.DEBUG:
            return 1 if ARMITAGE else 2  # multi GPU seems to hang on armitage
        if self.task in (Task.CLM, Task.MLM):
            return 1
        return 1

    @property
    def streaming(self) -> bool:
        if ACTION == Action.DEBUG:  # FIXME: remove
            return False
        if self.task in (Task.CLM, Task.MLM):
            return True
        return False

    @property
    def exit_after_map(self) -> bool:
        if ACTION == Action.PREPARE:
            return True
        return False

    @property
    def arch_config(self) -> dict:

        d = {"is_decoder": self.model_mode == ModelMode.UN}

        if self.model_name == ModelName.MAM:
            if self.model_size == ModelSize.TN:
                d |= {"num_hidden_layers": 2,  "hidden_size":  64}
            if self.model_size == ModelSize.SM:
                d |= {"num_hidden_layers": 4,  "hidden_size": 128}
            if self.model_size == ModelSize.MD:
                d |= {"num_hidden_layers": 8,  "hidden_size": 256}
            if self.model_size == ModelSize.LG:
                d |= {"num_hidden_layers": 12, "hidden_size": 384}
            if self.model_size == ModelSize.HG:
                d |= {"num_hidden_layers": 16, "hidden_size": 512}

        elif self.model_name == ModelName.HRR:
            if self.model_size == ModelSize.TN:
                d |= {"num_hidden_layers": 1, "hidden_size":  64}
            if self.model_size == ModelSize.SM:
                d |= {"num_hidden_layers": 2, "hidden_size": 128}
            if self.model_size == ModelSize.MD:
                d |= {"num_hidden_layers": 4, "hidden_size": 256}
            if self.model_size == ModelSize.LG:
                d |= {"num_hidden_layers": 6, "hidden_size": 384}
            if self.model_size == ModelSize.HG:
                d |= {"num_hidden_layers": 8, "hidden_size": 512}

        d |= {"embedding_size": 64, "head_num_hidden_layers": 1}

        if ACTION == Action.DEBUG:
            # Check the multiheaded attention.
            if self.model_name == ModelName.HRR:
                d["num_attention_heads"] = d.get("num_attention_heads", 2)
            # Check the weight-tying with embedding projection.
            d["embedding_size"]   = d["hidden_size"] // 2

        d["head_hidden_size"] = d["embedding_size"]

        return d

    @property
    def num_train_epochs(self) -> int:
        if ACTION == Action.OVERFIT:
            return int(os.environ["NUM_TRAIN_EPOCHS"])
        if self.task in (Task.CLM, Task.MLM):
            return 1
        return 5

    @property
    def max_steps(self) -> Optional[int]:
        if ACTION == Action.TIME:
            return 16
        if ACTION == Action.DEBUG:
            return 2
        return None

    @property
    def save_steps(self) -> Optional[int]:
        if ACTION == Action.TIME:
            return 16
        if ACTION == Action.DEBUG:
            return 1
        return None

    @property
    def eval_steps(self) -> Optional[int]:
        if ACTION == Action.TIME:
            return 16
        if ACTION == Action.DEBUG:
            return 1
        return None

    @property
    def saves_and_evals_per_epochs(self) -> int:
        if ACTION == Action.DEBUG:
            return 1
        if self.task in (Task.CLM, Task.MLM):
            return 16
        return 1

    @property
    def dataloader_num_workers(self) -> int:
        if self.gpu == 0:
            return 0
        # If we're streaming, we rely on thread-based parallelism,
        # so let there only be a single process for each GPU.
        if self.streaming:
            return 0
        return self.cpu // self.gpu - 1

    @property
    def learning_rate(self) -> float:
        if self.task in (Task.CLM, Task.MLM):
            return 1e-3
        return 1e-3

    @property
    def tr_batch_size(self) -> int:
        if ACTION == Action.DEBUG:
            return 64
        if self.task in (Task.CLM, Task.MLM):
            return 1024
        return 64

    @property
    def vl_batch_size(self) -> int:
        if self.model_size == ModelSize.TN:
            return 256
        if self.model_size == ModelSize.SM:
            return 128
        if self.model_size == ModelSize.MD:
            return 64
        if self.model_size == ModelSize.LG:
            return 32
        if self.model_size == ModelSize.HG:
            return 16
        raise RuntimeError()

    @property
    def tf32(self) -> bool:
        if ACTION == Action.PREPARE:
            return False
        return True

    @property
    def bf16(self) -> bool:
        if ACTION == Action.PREPARE:
            return False
        if self.model_name == ModelName.MAM:
            return True
        return False

    @property
    def bf16_full_eval(self) -> bool:
        if not self.bf16:
            return False
        if self.task in (Task.CLM, Task.MLM):
            return True
        return False

    @property
    def outfile(self) -> Path:
        return outpath() / f"{self.job}.sh"


def sort_configurations_key(c: Configuration) -> tuple:
    return (
        c.lift_level.value,
        c.model_name.value,
        c.model_size.value,
        # "a" + c.model_mode.value if c.task in (Task.CLM, Task.MLM) else "z" + c.model_mode.value,
        c.model_mode.value,
        "" if c.pretraining_task is None else c.pretraining_task.value,
        "a" + c.task.value if c.task in (Task.CLM, Task.MLM) else "z" + c.task.value,
    )


def main():

    global ACTION
    global ARMITAGE

    parser = ArgumentParser()
    parser.add_argument("--action", type=Action)
    parser.add_argument("--armitage", action="store_true")
    parser.add_argument("--no_remove", action="store_true")
    args = parser.parse_args()

    ACTION   = args.action
    ARMITAGE = args.armitage

    if not args.no_remove:
        shutil.rmtree(outpath(), ignore_errors=True)
    outpath().mkdir(parents=True, exist_ok=True)

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
            exit_after_map=config.exit_after_map,
            model_name=config.model_name,
            arch_config=config.arch_config,
            max_length=config.max_length,
            lift_level=config.lift_level,
            tokenization_algorithm=config.tokenization_algorithm,
            vocab_size=config.vocab_size,
            task=config.task,
            pretraining_task=config.pretraining_task,
            num_train_epochs=config.num_train_epochs,
            max_steps=config.max_steps,
            save_steps=config.save_steps,
            eval_steps=config.eval_steps,
            saves_and_evals_per_epochs=config.saves_and_evals_per_epochs,
            dataloader_num_workers=config.dataloader_num_workers,
            learning_rate=config.learning_rate,
            tr_batch_size=config.tr_batch_size,
            vl_batch_size=config.vl_batch_size,
            tf32=config.tf32,
            bf16=config.bf16,
            bf16_full_eval=config.bf16_full_eval,
            seed=config.seed,
        )
        if config.outfile.exists():
            raise FileExistsError(config.outfile)
        config.outfile.write_text(body)

    if ACTION == Action.EXECUTE:
        with open(ROOT / "execute.sh", "w") as fp:
            for i, config in enumerate(configurations):
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


if __name__ == "__main__":
    main()
