"""
Create sbatch scripts.
"""

from argparse import ArgumentParser
import json
from typing import Any

from src.enums import LiftLift, Task, TokenizerAlgorithm


jobname: str = "esp"
tim: int     = "00-01:00:00"
cpu: int     = 
mem: int     =
gpu: int     =

root: str       = ""
streaming: bool = True

model_name_or_path: str = "mamba"
arch_config: dict[str, Any] = {"mode": "uni", "num_hidden_layers": 12, "hidden_size": 512}

tokenization_algorithm: TokenizerAlgorithm = TokenizerAlgorithm.BPE
lift


def bool_to_str(b: bool) -> str:
    if not isinstance(b, bool):
        raise TypeError(type(b))
    return 'true' if b else 'false'


def get_body(
    job: str,
    tim: str,
    cpu: int,
    mem: int,
    gpu: int,
    streaming: bool,
    model_name_or_path: str,
    arch_config: dict,
    lift_level: LiftLevel, 
    tokenization_algorithm: TokenizationAlgorithm,
    vocab_size: int,
    task: Task,
    seed: int,
    num_train_epochs: int,
    saves_and_evals_per_epochs: int,
    learning_rate: float,
    per_device_train_batch_size: int,
    bf16: bool,
    fp16: bool,
) -> str: return 
"""
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
conda activate RawByteClf2
module unload blindfold

python -u \
src/learn/train.py \
--root='{ROOT}' \
--streaming={bool_to_str(streaming)} \
--auto_find_batch_size_and_gradient_accumulation_steps='true' \
--dataset_backend="HF" \
--model_name_or_path='{model_name}' \
--arch_config='{json.dumps(arch_config)}' \
--max_length={max_length} \
--data_read_bytes={max_length * 2} \
--tokenization_algorithm='{tokenization_algorithm.value}' \
--vocab_size={vocab_size} \
--task='{task}' \
--pretraining_task='{pretraining_checkpoint}' \
--pretraining_checkpoint=-1 \
--seed={seed} \
--do_train \
--do_eval \
--output_dir='foo/bar/baz' \
--save_strategy='epoch' \
--evaluation_strategy='epoch' \
--num_train_epochs={num_train_epochs} \
--logging_steps=1 \
--saves_per_epoch={saves_and_evals_per_epochs} \
--evals_per_epoch={saves_and_evals_per_epochs} \
--dataloader_num_workers={cpus - 1} \
--optim="adamw_torch" \
--learning_rate={learning_rate} \
--lr_scheduler_type="linear" \
--weight_decay=0.01 \
--adam_beta1=0.950 \
--adam_beta2=0.999 \
--max_grad_norm=1.0 \
--save_total_limit=-1 \
--per_device_train_batch_size={per_device_train_batch_size} \
--per_device_eval_batch_size=256 \
--gradient_accumulation_steps=1 \
--eval_accumulation_steps=64 \
--load_best_model_at_end \
--tf32='true' \
--bf16={'true' if bf16 else 'false'} \
--fp16={'true' if fp16 else 'false'} \
--gradient_checkpointing='true'
"""


    job: str,
    tim: str,
    cpu: int,
    mem: int,
    gpu: int,
    streaming: bool,
    model_name_or_path: str,
    arch_config: dict,
    lift_level: LiftLevel, 
    tokenization_algorithm: TokenizationAlgorithm,
    vocab_size: int,
    task: Task,
    seed: int,
    num_train_epochs: int,
    saves_and_evals_per_epochs: int,
    learning_rate: float,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    tf32: bool,
    bf16: bool,
    fp16: bool,


LM_KWDS = {
    "tim": "01-00:00:00",
    "streaming": True,
    "num_epochs": 1,
    "learning_rate": 1e-3,
    "per_device_train_batch_size": 1024,
}
CL_KWDS = {
    "tim": "01-00:00:00",
    "streaming": False,
    "num_epochs": 10,
    "learning_rate": 1e-3,
    "per_device_train_batch_size": 256,
}
def get_task_kwds(task: Task, mode: str, model_name: str, model_size: str) -> dict:
    if task in (Task.CLM, Task.MLM):
        return LM_KWDS,
    else:
        return CL_KWDS


MODEL_KWDS = {
    "mamba": {
        "model_name_or_path": "mamba",
        "arch_config": {},
        "bf16": True,
        "fp16": False,
    },
    "hrrformer": {
        "model_name_or_path": "hrrformer",
        "arch_config": {},
        "bf16": False,
        "fp16": False,
    },
}

def get_model_kwds(task: Task, mode: str, model_name: str, model_size: str) -> dict:
    if model_name == "mamba":
        arch_config = ARCH_CONFIG_MAMBA[model_size]
    else:
        arch_config = ARCH_CONFIG_HRRFORMER[model_size]


for mode in ("un", "bi"):
    for model_name in ("mamba", "hrrformer"):
        for model_size in ("sm", "md", "lg"):
            for ll in LiftLevel:
                for tok in TokenizationAlgorithm:
                    for task in Task:
                        if task == Task.MLM and mode == "uni" or task == Task.CLM and mode == "bi":
                            continue
                        for seed in (0, 1, 2, 3, 4):
                            kwds = (
                                get_task_kwds(task, mode, model_name, model_size) |
                                get_model_kwds(task, mode, model_name, model_size)
                            )

                            job  = get_jobname(kwds)
                            body = get_body(job, **kwds)
                            Path(f"./{job}.sbatch").write_text(body)
                            print(jobname)
