"""
Create pretraining, classification, and finetuning bash scripts.
"""

from argparse import ArgumentParser
from enum import Enum
import os
from pathlib import Path
from pprint import pprint
import sys
from typing import Optional


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
MODEL_NAME_OR_PATHS = ["mamba", "malconv2"]
ARCH_CONFIGS = {
    "mamba": '{"mode": "uni", "num_hidden_layers": 4, "hidden_size": 128, "embedding_size": 8}',  # 0.50 M
    # "mamba": '{"mode": "uni", "num_hidden_layers": 4, "hidden_size": 128, "embedding_size": 128}',  # 0.53 M
    # "mamba": '{"mode": "uni", "num_hidden_layers": 8, "hidden_size": 256, "embedding_size": 8}',  # 3.57 M
    # "mamba": '{"mode": "uni", "num_hidden_layers": 8, "hidden_size": 256, "embedding_size": 256}',  # 3.64 M
    # "mamba": '{"mode": "uni", "num_hidden_layers": 12, "hidden_size": 384, "embedding_size": 8}',  # 11.7 M
    # "mamba": '{"mode": "uni", "num_hidden_layers": 12, "hidden_size": 384, "embedding_size": 384}',  # 11.8 M
    # "mamba": '{"mode": "uni", "num_hidden_layers": 16, "hidden_size": 512, "embedding_size": 8}',  # 27.3 M
    # "mamba": '{"mode": "uni", "num_hidden_layers": 16, "hidden_size": 512, "embedding_size": 512}',  # 27.4 M
    "malconv2": '{"mode": "gcg", "channels": 256, "stride": 64, "kernel_size": 64, "embedding_size": 8}',  # 2.56 M
    # "malconv2": '{"mode": "gcg", "channels": 256, "stride": 64, "kernel_size": 64, "embedding_size": 256}',  # 67.7 M
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
CLF_PER_DEVICE_EVAL_BATCH_SIZE = 256
CLF_SAVE_EVAL_STRATEGY = "steps" if args.debug else "epoch"
CLF_NUM_TRAIN_EPOCHS = 0.1 if args.debug else 10

CLF_ALLOC_TIME: dict[tuple[str, str, int, str], str] = {
    ("mamba", "clf-bod"): "00-01:00:00",
    ("mamba", "clf-sor-class_"): "00-01:00:00",
    ("mamba", "clf-sor-file"): "00-01:00:00",
    ("mamba", "clf-sor-fam"): "00-01:00:00",
    ("mamba", "clf-sor-beh"): "00-01:00:00",
    ("mamba", "clf-sor-pack"): "00-01:00:00",
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


MAX_IMBALANCE_RATIO = 1000


def get_body_lm(
    jobname: str,
    downstream_task: str,
    model_name_or_path: str,
    arch_config: str,
) -> str:

   return f"""#!/bin/bash -l

    #SBATCH --job-name={'debug-' if args.debug else ''}{jobname}
    #SBATCH --account=admalware
    #SBATCH --partition={'debug' if args.debug else 'tier3'}
    #SBATCH --output=./logs/%x_%j.out
    #SBATCH --time={'00-01:00:00' if args.debug else '05-00:00:00'}
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
    --model_name_or_path='{model_name_or_path}' \\
    --arch_config='{arch_config}' \\
    --task={downstream_task} \\
    --seed=0 \\
    --packing_protocol="any" \\
    --streaming='true' \\
    --skip_eval_check='false' \\
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
    --num_train_epochs=1 \\
    --logging_steps=1 \\
    --save_steps={CLM_SAVE_EVAL_STEPS} \\
    --eval_steps={CLM_SAVE_EVAL_STEPS} \\
    --dataloader_num_workers={0 if args.debug else CLM_NTASKS - 1} \\
    --optim="adamw_torch" \\
    --learning_rate="1e-3" \\
    --lr_scheduler_type="linear" \\
    --warmup_ratio=0.05 \\
    --weight_decay=0.01 \\
    --adam_beta1=0.900 \\
    --adam_beta2=0.990 \\
    --max_grad_norm=1.0 \\
    --save_total_limit=-1 \\
    --model_name_or_path={model_name_or_path} \\
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
    """.replace("    ", "").replace("\n\n", "\n")


def get_body_clf(
    jobname: str,
    alloc_time: str,
    alloc_memory: str,
    downstream_task: str,
    pretraining_task: str,
    model_name_or_path: str,
    arch_config: str,
    seed: int,
    gradient_checkpointing: bool,
    *,
    top_k: Optional[int] = None,
    tr_samples_per_class: Optional[int] = None,
    weighted_loss: Optional[str] = None,
) -> str:

    return f"""#!/bin/bash -l

    #SBATCH --job-name={'debug-' if args.debug else ''}{jobname}
    #SBATCH --account=admalware
    #SBATCH --partition={'debug' if args.debug else 'tier3'}
    #SBATCH --output=./logs/%x_%j.out
    #SBATCH --time={'00-01:00:00' if args.debug else alloc_time}
    #SBATCH --nodes=1
    #SBATCH --cpus-per-task=1
    #SBATCH --ntasks={1 if args.debug else CLF_NTASKS}
    #SBATCH --mem={'16G' if args.debug else alloc_memory}
    #SBATCH --gres=gpu:a100:{args.clf_ngpus}


    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate RawByteClf{2 if args.system == System.RC else ""}
    {"module unload blindfold" if args.system == System.RC else ""}


    {"torchrun --no-python --nnodes=1 --nproc_per_node=" + str(args.clf_ngpus) + " " + DOUBLE_BACKSLASH if args.clf_ngpus > 1 else ""}
    python -u \\
    src/learn/train.py \\
    --root='{ROOT}' \\
    --streaming='true' \\
    --skip_eval_check='false' \\
    --dataset_backend="HF" \\

    --model_name_or_path='{model_name_or_path}' \\
    --arch_config='{arch_config}' \\

    --max_length={MAX_LENGTH} \\
    --data_read_bytes={DATA_READ_BYTES} \\
    --packing_protocol='any' \\
    --representation=8 \\
    --algorithm='Raw' \\
    --vocab_size=256 \\

    --weighted_loss='{weighted_loss}' \\
    --beta=0.85 \\
    --early_stopping=false \\

    --task='{downstream_task}' \\
    --pretraining_task='{pretraining_task}' \\
    --tr_size=0.85 \\
    --vl_size=0.15 \\
    --ts_size=0.0 \\
    --min_freq=2 \\
    --top_k={top_k} \\
    --tr_samples_per_class={tr_samples_per_class} \\
    --max_imbalance_ratio={MAX_IMBALANCE_RATIO} \\

    --seed={seed} \\
    --auto_find_batch_size_and_gradient_accumulation_steps \\

    --do_train \\
    --output_dir=tmp \\
    --save_strategy='{CLF_SAVE_EVAL_STRATEGY}' \\
    --evaluation_strategy='{CLF_SAVE_EVAL_STRATEGY}' \\
    --num_train_epochs={CLF_NUM_TRAIN_EPOCHS} \\
    --logging_steps=1 \\
    --save_steps=1 \\
    --eval_steps=1 \\
    --saves_per_epoch={10 if args.debug else 2} \\
    --evals_per_epoch={10 if args.debug else 2} \\
    --dataloader_num_workers={0 if args.debug else CLF_NTASKS - 1} \\
    --optim="adamw_torch" \\
    --learning_rate="1e-3" \\
    --lr_scheduler_type="linear" \\
    --weight_decay=0.01 \\
    --adam_beta1=0.900 \\
    --adam_beta2=0.999 \\
    --max_grad_norm=1.0 \\
    --save_total_limit=2 \\
    --per_device_train_batch_size={CLF_PER_DEVICE_TRAIN_BATCH_SIZE} \\
    --per_device_eval_batch_size={CLF_PER_DEVICE_EVAL_BATCH_SIZE} \\
    --gradient_accumulation_steps={CLF_GRADIENT_ACCUMULATION_STEPS} \\
    --eval_accumulation_steps=64 \\
    --load_best_model_at_end \\
    --tf32=true \\
    --bf16=false \\
    --fp16=false \\
    --gradient_checkpointing={'true' if gradient_checkpointing else 'false'}
    """.replace("    ", "").replace("\n\n", "\n")


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
    tr_samples_per_class: Optional[int],
    seed: int,
) -> str:
    args = [
        model_name,
        pretraining_task,
        downstream_task,
        str(tr_samples_per_class),
        str(seed),
    ]
    return "--".join(args)


for model_name in MODEL_NAME_OR_PATHS:
    arch_config = ARCH_CONFIGS[model_name]
    gradient_checkpointing = True if model_name == "mamba" else False

    # Pretraining
    for pretraining_task in PRETRAINING_TASKS:
        if pretraining_task == "None" or model_name != "mamba":
            continue

        jobname = get_jobname(model_name, "None", pretraining_task, None, 0)
        body = get_body_lm(jobname, pretraining_task, model_name, arch_config)
        outfile = (OUTPUT / jobname).with_suffix(".sh")
        with open(outfile, "w") as fp:
            fp.write(body)
        outfiles.append(outfile)

    # Classification
    for task in TASKS:
        weighted_loss = "sample_reweighting" if task in ("clf-bod", "clf-sor-fam", "clf-sor-file") else None
        alloc_time = CLF_ALLOC_TIME[(model_name, task)]
        alloc_memory = CLF_ALLOC_MEM[(task,)]

        for pretraining_task in PRETRAINING_TASKS:
            if model_name in ("malconv", "malconv2") and pretraining_task != "None":
                continue

            for tr_samples_per_class in (None, 1, 5):
                if tr_samples_per_class is not None and task not in ("clf-bod", "clf-sor-fam", "clf-sor-file"):
                    continue

                for seed in SEEDS:
                    jobname = get_jobname(model_name, pretraining_task, task, tr_samples_per_class, seed)
                    body = get_body_clf(
                        jobname,
                        alloc_time,
                        alloc_memory,
                        task,
                        pretraining_task,
                        model_name,
                        arch_config,
                        seed,
                        gradient_checkpointing,
                        tr_samples_per_class=tr_samples_per_class,
                        weighted_loss=weighted_loss,
                    )
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
    "mamba--None--clm-sor--None--0.sh",
    "mamba--None--clf-bod--None--0.sh",
    "mamba--None--clf-sor-pack--None--0.sh",
    "mamba--clm-sor--clf-bod--5--0.sh",
    "mamba--clm-sor--clf-sor-pack--None--0.sh",
    "malconv2--None--clf-bod--None--1.sh",
    "malconv2--None--clf-sor-pack--None--0.sh",
]
with open(OUTPUT / "debug.sh", "w") as fp:
    for f in test_files:
        f = OUTPUT / f
        if not f.exists():
            raise FileNotFoundError(f)
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
