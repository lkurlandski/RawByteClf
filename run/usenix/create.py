"""
Create pretraining, classification, and finetuning bash scripts.
"""

from argparse import ArgumentParser
from enum import Enum
import os
from pathlib import Path
from pprint import pprint
import sys


DOUBLE_BACKSLASH = """\\"""


class System(Enum):
    RC = "RC"
    ARMITAGE = "ARMITAGE"


parser = ArgumentParser()
parser.add_argument("--system", type=System, required=True)
parser.add_argument("--clm_ngpus", type=int, default=1)
parser.add_argument("--clf_ngpus", type=int, default=1)
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()


CLM_NTASKS = 4
CLF_NTASKS = 4
CLM_MEM = "64G"
CLF_MEM = "MEMORY"


SEEDS = [0, 1, 2, 3, 4]
TASKS = ["clf-bod"] + [f"clf-sor-{s}" for s in ("class_", "file", "fam", "beh", "pack")]
PRETRAINING_TASKS = ["None", "clm-sor"]

ROOT = "./output/test" if args.debug else "./output/usenix"
MODEL_NAME_OR_PATHS = ["mamba", "malconv", "malconv2"]
ARCH_CONFIGS = {
    "mamba": '{"mode": "uni", "num_hidden_layers": 8, "hidden_size": 256, "embedding_size": 256}',
    "malconv": '{"channels": 128, "stride": 512, "kernel_size": 512, "embedding_size": 256}',
    "malconv2": '{"mode": "gcg", "channels": 256, "stride": 64, "kernel_size": 64, "embedding_size": 256}',
}
MAX_LENGTH = 2 ** 12
DATA_READ_BYTES = 2 ** 12

CLM_TR_SIZE = 2 ** 15 if args.debug else 2 ** 21
CLM_VL_SIZE = 2 ** 14
CLM_TRAIN_BATCH_SIZE = 1024
CLM_PER_DEVICE_TRAIN_BATCH_SIZE = CLM_TRAIN_BATCH_SIZE // args.clm_ngpus
CLM_GRADIENT_ACCUMULATION_STEPS = 1
CLM_PER_DEVICE_EVAL_BATCH_SIZE = 1024
CLM_SAVE_EVAL_STEPS = 1 if args.debug else 512
CLM_NUM_TRAIN_EPOCHS = 0.1 if args.debug else 1

CLF_TRAIN_BATCH_SIZE = 64
CLF_PER_DEVICE_TRAIN_BATCH_SIZE = CLF_TRAIN_BATCH_SIZE // args.clf_ngpus
CLF_GRADIENT_ACCUMULATION_STEPS = 1
CLF_PER_DEVICE_EVAL_BATCH_SIZE = "CLF_PER_DEVICE_EVAL_BATCH_SIZE"
CLF_SAVE_EVAL_STRATEGY = "steps" if args.debug else "epoch"
CLF_NUM_TRAIN_EPOCHS = 0.1 if args.debug else 10

CLF_ALLOC_TIME: dict[tuple[str, str, int, str], str] = {
    ("mamba", "clf-bod"): "00-01:00:00",
    ("mamba", "clf-sor-class_"): "00-01:00:00",
    ("mamba", "clf-sor-file"): "00-01:00:00",
    ("mamba", "clf-sor-fam"): "00-01:00:00",
    ("mamba", "clf-sor-beh"): "00-01:00:00",
    ("mamba", "clf-sor-pack"): "00-01:00:00",

    ("malconv", "clf-bod"): "00-01:00:00",
    ("malconv", "clf-sor-class_"): "00-01:00:00",
    ("malconv", "clf-sor-file"): "00-01:00:00",
    ("malconv", "clf-sor-fam"): "00-01:00:00",
    ("malconv", "clf-sor-beh"): "00-01:00:00",
    ("malconv", "clf-sor-pack"): "00-01:00:00",

    ("malconv2", "clf-bod"): "00-01:00:00",
    ("malconv2", "clf-sor-class_"): "00-01:00:00",
    ("malconv2", "clf-sor-file"): "00-01:00:00",
    ("malconv2", "clf-sor-fam"): "00-01:00:00",
    ("malconv2", "clf-sor-beh"): "00-01:00:00",
    ("malconv2", "clf-sor-pack"): "00-01:00:00",
}
    
CLF_ALLOC_MEM: dict[tuple[str, str], str] = {
    ("clf-bod",): "32G",
    ("clf-sor-class_",): "64G",
    ("clf-sor-file",): "32G",
    ("clf-sor-fam",): "64G",
    ("clf-sor-beh",): "64G",
    ("clf-sor-pack",): "16G",
}


BODY_CLM = f"""#!/bin/bash -l

#SBATCH --job-name={'debug-' if args.debug else ''}JOB_NAME
#SBATCH --account=admalware
#SBATCH --partition={'debug' if args.debug else 'tier3'}
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time={'00-01:00:00' if args.debug else '02-00:00:00'}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks={1 if args.debug else CLM_NTASKS}
#SBATCH --mem={'16G' if args.debug else CLM_MEM}
#SBATCH --gres=gpu:a100:{args.clm_ngpus}


source ~/anaconda3/etc/profile.d/conda.sh
conda activate RawByteClf{2 if args.system == System.RC else ""}
{"module unload blindfold" if args.system == System.RC else ""}


{"torchrun --no-python --nnodes=1 --nproc_per_node=" + str(args.clm_ngpus) + " " + DOUBLE_BACKSLASH if args.clm_ngpus > 1 else ""}
python -u \\
src/learn/train.py \\
--root="{ROOT}" \\
--arch_config='ARCH_CONFIG' \\
--metric_for_best_model="eval_loss" \\
--task=DOWNSTREAM_TASK \\
--seed=0 \\
--packing_protocol="any" \\
--streaming={'true' if args.debug else 'true'} \\
--skip_eval_check={'true' if args.debug else 'false'} \\
--dataset_backend="HF" \\
--representation=8 \\
--algorithm="Raw" \\
--vocab_size=256 \\
--tr_size={CLM_TR_SIZE} \\
--vl_size={CLM_VL_SIZE} \\
--ts_size=0 \\
--do_train \\
--output_dir=tmp \\
--save_strategy="steps" \\
--evaluation_strategy="steps" \\
--num_train_epochs={CLM_NUM_TRAIN_EPOCHS} \\
--logging_steps=1 \\
--save_steps={CLM_SAVE_EVAL_STEPS} \\
--eval_steps={CLM_SAVE_EVAL_STEPS} \\
--dataloader_num_workers={0 if args.debug else CLM_NTASKS - 1} \\
--optim="adamw_torch" \\
--learning_rate="1e-3" \\
--lr_scheduler_type="linear" \\
--warmup_ratio=0.05 \\
--weight_decay=0.10 \\
--adam_beta1=0.900 \\
--adam_beta2=0.990 \\
--max_grad_norm=1.0 \\
--save_total_limit=-1 \\
--model_name_or_path=MODEL_NAME_OR_PATH \\
--max_length={MAX_LENGTH} \\
--data_read_bytes={DATA_READ_BYTES} \\
--per_device_train_batch_size={CLM_PER_DEVICE_TRAIN_BATCH_SIZE} \\
--per_device_eval_batch_size={CLM_PER_DEVICE_EVAL_BATCH_SIZE} \\
--gradient_accumulation_steps={CLM_GRADIENT_ACCUMULATION_STEPS} \\
--load_best_model_at_end \\
--early_stopping=false \\
--auto_find_batch_size_and_gradient_accumulation_steps \\
--tf32=true \\
--bf16=true \\
--fp16=false \\
--gradient_checkpointing=true
"""


BODY_CLF = f"""#!/bin/bash -l

#SBATCH --job-name={'debug-' if args.debug else ''}JOB_NAME
#SBATCH --account=admalware
#SBATCH --partition={'debug' if args.debug else 'tier3'}
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time={'00-01:00:00' if args.debug else 'ALLOC_TIME'}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks={1 if args.debug else CLF_NTASKS}
#SBATCH --mem={'16G' if args.debug else CLF_MEM}
#SBATCH --gres=gpu:a100:{args.clf_ngpus}


source ~/anaconda3/etc/profile.d/conda.sh
conda activate RawByteClf{2 if args.system == System.RC else ""}
{"module unload blindfold" if args.system == System.RC else ""}


{"torchrun --no-python --nnodes=1 --nproc_per_node=" + str(args.clf_ngpus) + " " + DOUBLE_BACKSLASH if args.clf_ngpus > 1 else ""}
python -u \\
src/learn/train.py \\
--root="{ROOT}" \\
--arch_config='ARCH_CONFIG' \\
--metric_for_best_model="eval_accuracy" \\
--task=DOWNSTREAM_TASK \\
--seed=SEED \\
--pretraining_task=PRETRAINING_TASK \\
--packing_protocol="any" \\
--streaming={'true' if args.debug else 'true'} \\
--skip_eval_check={'false' if args.debug else 'false'} \\
--dataset_backend="HF" \\
--representation=8 \\
--algorithm="Raw" \\
--vocab_size=256 \\
--tr_size=0.85 \\
--vl_size=0.15 \\
--ts_size=0.0 \\
--do_train \\
--output_dir=tmp \\
--save_strategy="{CLF_SAVE_EVAL_STRATEGY}" \\
--evaluation_strategy="{CLF_SAVE_EVAL_STRATEGY}" \\
--num_train_epochs={CLF_NUM_TRAIN_EPOCHS} \\
--logging_steps=1 \\
--save_steps=1 \\
--eval_steps=1 \\
--saves_per_epoch=10 \\
--evals_per_epoch=10 \\
--dataloader_num_workers={0 if args.debug else CLF_NTASKS - 1} \\
--optim="adamw_torch" \\
--learning_rate="1e-4" \\
--lr_scheduler_type="linear" \\
--warmup_ratio=0.05 \\
--weight_decay=0.01 \\
--adam_beta1=0.900 \\
--adam_beta2=0.999 \\
--max_grad_norm=1.0 \\
--save_total_limit=2 \\
--model_name_or_path=MODEL_NAME_OR_PATH \\
--max_length={MAX_LENGTH} \\
--data_read_bytes={DATA_READ_BYTES} \\
--per_device_train_batch_size={CLF_PER_DEVICE_TRAIN_BATCH_SIZE} \\
--per_device_eval_batch_size={CLF_PER_DEVICE_EVAL_BATCH_SIZE} \\
--gradient_accumulation_steps={CLF_GRADIENT_ACCUMULATION_STEPS} \\
--eval_accumulation_steps=64 \\
--load_best_model_at_end \\
--early_stopping=false \\
--auto_find_batch_size_and_gradient_accumulation_steps \\
--tf32=true \\
--bf16=false \\
--fp16=false \\
--gradient_checkpointing=GRADIENT_CHECKPOINTING
"""


OUTPUT = Path(os.path.realpath(__file__)).parent
for f in OUTPUT.glob("*.sh"):
    if f.name[0:3] == "run":
        continue
    f.unlink()


outfiles = []


def get_jobname(
    model_name: str,
    pretraining_task: str,
    downstream_task: str,
    seed: int,
) -> str:
    args = [
        model_name,
        pretraining_task,
        downstream_task,
        str(seed),
    ]
    return "--".join(args)


for model_name in MODEL_NAME_OR_PATHS:
    arch_config = ARCH_CONFIGS[model_name]
    gradient_checkpointing = "true" if model_name == "mamba" else "false"
    per_device_eval_batch_size = 512 if model_name in ("malconv", "malconv2") else 64

    for pretraining_task in PRETRAINING_TASKS:
        if pretraining_task == "None" or model_name != "mamba":
            continue

        jobname = get_jobname(model_name, "None", pretraining_task, 0)
        body = BODY_CLM \
            .replace("JOB_NAME", jobname) \
            .replace("MODEL_NAME_OR_PATH", model_name) \
            .replace("ARCH_CONFIG", arch_config) \
            .replace("CLM_PER_DEVICE_EVAL_BATCH_SIZE", str(per_device_eval_batch_size)) \
            .replace("GRADIENT_CHECKPOINTING", gradient_checkpointing) \
            .replace("DOWNSTREAM_TASK", pretraining_task)
        outfile = (OUTPUT / jobname).with_suffix(".sh")
        with open(outfile, "w") as fp:
            fp.write(body)
        outfiles.append(outfile)

    # Classification
    for task in TASKS:
        top_k = 10 if task == "clf-bod" else None
        min_freq = None if task == "clf-bod" else 2
        alloc_time = CLF_ALLOC_TIME[(model_name, task)]
        memory = CLF_ALLOC_MEM[(task,)]

        for pretraining_task in PRETRAINING_TASKS:
            if model_name in ("malconv", "malconv2") and pretraining_task != "None":
                continue
            for seed in SEEDS:
                jobname = get_jobname(model_name, pretraining_task, task, seed)
                body = BODY_CLF \
                    .replace("JOB_NAME", jobname) \
                    .replace("MODEL_NAME_OR_PATH", model_name) \
                    .replace("ARCH_CONFIG", arch_config) \
                    .replace("CLF_PER_DEVICE_EVAL_BATCH_SIZE", str(per_device_eval_batch_size)) \
                    .replace("GRADIENT_CHECKPOINTING", gradient_checkpointing) \
                    .replace("DOWNSTREAM_TASK", task) \
                    .replace("ALLOC_TIME", alloc_time) \
                    .replace("MEMORY", memory) \
                    .replace("SEED", str(seed)) \
                    .replace("PRETRAINING_TASK", pretraining_task)
                outfile = (OUTPUT / jobname).with_suffix(".sh")
                with open(outfile, "w") as fp:
                    fp.write(body)
                outfiles.append(outfile)


def key(s: str) -> tuple:
    out = s.split("-")
    for i in range(len(out)):
        o = out[i]
        if o.isdigit():
            out[i] == float(o)
    return out


with open(OUTPUT / "run.sh", "w") as fp:
    for f in sorted(outfiles, key=lambda p: key(str(p.name))):
        if args.system == System.RC:
            pre = "sbatch"
            pos = ""
        else:
            if "clm" in f.name:
                gpus = [str(i) for i in range(args.clm_ngpus)]
            else:
                gpus = [str(i) for i in range(args.clf_ngpus)]
            pre = f"CUDA_VISIBLE_DEVICES={','.join(gpus)} bash"
            pos = f"&> ./logs/{f.stem}.out"   
        fp.write(f"{pre} {str(f)} {pos}\n")


test_files = [
    "mamba--None--clm-sor--0.sh",
    "mamba--None--clf-bod--0.sh",
    "mamba--None--clf-sor-pack--0.sh",
    "mamba--clm-sor--clf-bod--0.sh",
    "mamba--clm-sor--clf-sor-pack--0.sh",
    "malconv2--None--clf-bod--0.sh",
    "malconv2--None--clf-sor-pack--0.sh",
]
with open(OUTPUT / "debug.sh", "w") as fp:
    for f in test_files:
        f = OUTPUT / f
        if args.system == System.RC:
            pre = "sbatch"
            pos = ""
        else:
            if "clm" in f.name:
                gpus = [str(i) for i in range(args.clm_ngpus)]
            else:
                gpus = [str(i) for i in range(args.clf_ngpus)]
            pre = f"CUDA_VISIBLE_DEVICES={','.join(gpus)} bash"
            pos = f"&> ./logs/{f.stem}.out"   
        fp.write(f"{pre} {str(f)} {pos}\n")
