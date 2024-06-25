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

OUTPUT = Path(os.path.realpath(__file__)).parent
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.cfg import System, SYSTEM


DOUBLE_BACKSLASH = """\\"""


parser = ArgumentParser()
parser.add_argument("--clm_ngpus", type=int, default=1)
parser.add_argument("--clf_ngpus", type=int, default=1)
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()


# Universal configuration for experments
ROOT = "./output/test" if args.debug else "./output/usenix"
CLM_NTASKS = 4
CLF_NTASKS = 4
MAX_LENGTH = 2 ** 12
DATA_READ_BYTES = 2 ** 12

# Parameters to vary for the experiments.
MODEL_NAME_AND_ARCH_CONFIGS = {
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
PRETRAINING_TASKS = ["None", "clm-sor"]
TASKS = ["clf-bod"] + [f"clf-sor-{s}" for s in ("class_", "file", "fam", "beh", "pack")]
TR_SAMPLES_PER_CLASS = [None, 1, 5]
SEEDS = [0, 1, 2, 3, 4]


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
    #SBATCH --mem={'16G' if args.debug else '64G'}
    #SBATCH --gres=gpu:a100:{args.clm_ngpus}


    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate RawByteClf{2 if SYSTEM == System.RC else ""}
    {"module unload blindfold" if SYSTEM == System.RC else ""}


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
    --tr_size={2 ** 14 if args.debug else 2 ** 21} \\
    --vl_size={2 ** 12 if args.debug else 2 ** 14} \\
    --ts_size=0 \\
    --do_train \\
    --output_dir=tmp \\
    --save_strategy="steps" \\
    --evaluation_strategy="steps" \\
    --num_train_epochs=1 \\
    --logging_steps=1 \\
    --save_steps={1 if args.debug else 512} \\
    --eval_steps={1 if args.debug else 512} \\
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
    --per_device_train_batch_size={1024 // args.clm_ngpus} \\
    --per_device_eval_batch_size={1024} \\
    --gradient_accumulation_steps={1} \\
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
    num_train_epochs: int,
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
    conda activate RawByteClf{2 if SYSTEM == System.RC else ""}
    {"module unload blindfold" if SYSTEM == System.RC else ""}


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
    --max_imbalance_ratio=1000 \\

    --seed={seed} \\
    --auto_find_batch_size_and_gradient_accumulation_steps \\

    --do_train \\
    --output_dir=tmp \\
    --save_strategy='{"steps" if args.debug else "epoch"}' \\
    --evaluation_strategy='{"steps" if args.debug else "epoch"}' \\
    --num_train_epochs={num_train_epochs} \\
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
    --per_device_train_batch_size={64 // args.clf_ngpus} \\
    --per_device_eval_batch_size={256} \\
    --gradient_accumulation_steps={1} \\
    --eval_accumulation_steps=64 \\
    --load_best_model_at_end \\
    --tf32=true \\
    --bf16=false \\
    --fp16=false \\
    --gradient_checkpointing={'true' if gradient_checkpointing else 'false'}
    """.replace("    ", "").replace("\n\n", "\n")


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


def get_clf_alloc_time_and_mem(
    model_name: str,
    task: str,
    tr_samples_per_class: Optional[int],
) -> tuple[str, str]:
    mem = None
    tim = None

    if tr_samples_per_class is None:
        if task == "clf-bod":
            mem = "32G"
        elif task == "clf-sor-class_":
            mem = "32G"
        elif task == "clf-sor-file":
            mem = "32G"
        elif task == "clf-sor-fam":
            mem = "32G"
        elif task == "clf-sor-beh":
            mem = "32G"
        elif task == "clf-sor-pack":
            mem = "32G"
    elif tr_samples_per_class == 1:
        mem = "16G"
    elif tr_samples_per_class == 5:
        mem = "16G"

    if model_name == "mamba":
        if task == "clf-bod":
            tim = "00-01:00:00"
        elif task == "clf-sor-class_":
            tim = "00-01:00:00"
        elif task == "clf-sor-file":
            tim = "00-01:00:00"
        elif task == "clf-sor-fam":
            tim = "00-01:00:00"
        elif task == "clf-sor-beh":
            tim = "00-01:00:00"
        elif task == "clf-sor-pack":
            tim = "00-01:00:00"
    elif model_name == "malconv2":
        if task == "clf-bod":
            tim = "00-01:00:00"
        elif task == "clf-sor-class_":
            tim = "00-01:00:00"
        elif task == "clf-sor-file":
            tim = "00-01:00:00"
        elif task == "clf-sor-fam":
            tim = "00-01:00:00"
        elif task == "clf-sor-beh":
            tim = "00-01:00:00"
        elif task == "clf-sor-pack":
            tim = "00-01:00:00"

    if tim is None or mem is None:
        raise RuntimeError(f"{model_name=} {task=} {tr_samples_per_class=}")

    return tim, mem


def main():

    for f in OUTPUT.glob("*.sh"):
        if f.name[0:3] == "run":
            continue
        f.unlink()

    outfiles = []

    for model_name, arch_config in MODEL_NAME_AND_ARCH_CONFIGS.items():
        gradient_checkpointing = True if model_name == "mamba" else False

        # Pretraining
        for pretraining_task in PRETRAINING_TASKS:
            if pretraining_task == "None" or model_name != "mamba":
                continue

            jobname = get_jobname(model_name, "None", pretraining_task, None, 0)
            body = get_body_lm(
                jobname=jobname,
                downstream_task=pretraining_task,
                model_name_or_path=model_name,
                arch_config=arch_config,
            )
            outfile = (OUTPUT / jobname).with_suffix(".sh")
            with open(outfile, "w") as fp:
                fp.write(body)
            outfiles.append(outfile)

        # Classification
        for task in TASKS:
            weighted_loss = "sample_reweighting" if task in ("clf-bod", "clf-sor-fam", "clf-sor-file") else None

            for pretraining_task in PRETRAINING_TASKS:
                if model_name in ("malconv", "malconv2") and pretraining_task != "None":
                    continue

                for tr_samples_per_class in TR_SAMPLES_PER_CLASS:
                    if tr_samples_per_class is not None and task not in ("clf-bod", "clf-sor-fam", "clf-sor-file"):
                        continue
                    alloc_time, alloc_memory = get_clf_alloc_time_and_mem(model_name, task, tr_samples_per_class)

                    if tr_samples_per_class is None:
                        num_train_epochs = 5
                    elif tr_samples_per_class == 5:
                        num_train_epochs = 10
                    elif tr_samples_per_class == 1:
                        num_train_epochs = 50
                    else:
                        raise ValueError(tr_samples_per_class)

                    for seed in SEEDS:
                        jobname = get_jobname(model_name, pretraining_task, task, tr_samples_per_class, seed)
                        body = get_body_clf(
                            jobname=jobname,
                            alloc_time=alloc_time,
                            alloc_memory=alloc_memory,
                            downstream_task=task,
                            pretraining_task=pretraining_task,
                            model_name_or_path=model_name,
                            arch_config=arch_config,
                            seed=seed,
                            gradient_checkpointing=gradient_checkpointing,
                            num_train_epochs=num_train_epochs,
                            top_k=None,
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
            if SYSTEM == System.RC:
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
            if SYSTEM == System.RC:
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


if __name__ == "__main__":
    main()
