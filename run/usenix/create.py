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
parser.add_argument("--clm_ntasks", type=int, default=4)
parser.add_argument("--clm_ndataloaderworkers", type=int, default=3)
parser.add_argument("--clf_ngpus", type=int, default=1)
parser.add_argument("--clf_ntasks", type=int, default=4)
parser.add_argument("--clf_ndataloaderworkers", type=int, default=3)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--dependencies", action="store_true")
args = parser.parse_args()


# Universal configuration for experments
ROOT = "./output/test" if args.debug else "./output/usenix"
MAX_LENGTH = 2 ** 14
DATA_READ_BYTES = 2 ** 14

# Parameters to vary for the experiments.
MODEL_NAME_AND_ARCH_CONFIGS = {
    # "mamba": '{"mode": "uni", "num_hidden_layers": 4, "hidden_size": 128, "embedding_size": 8}',  # 0.50 M
    # "mamba": '{"mode": "uni", "num_hidden_layers": 4, "hidden_size": 128, "embedding_size": 128}',  # 0.53 M
    "mamba": '{"mode": "uni", "num_hidden_layers": 6, "hidden_size": 256, "embedding_size": 8}',  # 2.70 M
    # "mamba": '{"mode": "uni", "num_hidden_layers": 6, "hidden_size": 256, "embedding_size": 128}',  # 2.76 M
    # "mamba": '{"mode": "uni", "num_hidden_layers": 8, "hidden_size": 256, "embedding_size": 8}',  # 3.57 M
    # "mamba": '{"mode": "uni", "num_hidden_layers": 8, "hidden_size": 256, "embedding_size": 256}',  # 3.64 M
    # "mamba": '{"mode": "uni", "num_hidden_layers": 12, "hidden_size": 384, "embedding_size": 8}',  # 11.7 M
    # "mamba": '{"mode": "uni", "num_hidden_layers": 12, "hidden_size": 384, "embedding_size": 384}',  # 11.8 M
    # "mamba": '{"mode": "uni", "num_hidden_layers": 16, "hidden_size": 512, "embedding_size": 8}',  # 27.3 M
    # "mamba": '{"mode": "uni", "num_hidden_layers": 16, "hidden_size": 512, "embedding_size": 512}',  # 27.4 M
    "malconv2": '{"mode": "gcg", "channels": 256, "stride": 64, "kernel_size": 64, "embedding_size": 8}',  # 2.56 M
    # "malconv2": '{"mode": "gcg", "channels": 256, "stride": 64, "kernel_size": 64, "embedding_size": 256}',  # 67.7 M
}
PRETRAINING_TASKS = [None, "clm-sor"]
TASKS_SCMF = ["clf-bod", "clf-sor-fam", "clf-sor-file"]
TASKS_MCMF = ["clf-sor-class_", "clf-sor-beh", "clf-sor-pack"]
TASKS = TASKS_SCMF + TASKS_MCMF
MIN_FREQ = [None, 100]
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
    #SBATCH --time={'00-01:00:00' if args.debug else '01-00:00:00'}
    #SBATCH --nodes=1
    #SBATCH --cpus-per-task=1
    #SBATCH --ntasks={1 if args.debug else args.clm_ntasks}
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
    --save_steps={1 if args.debug else 256} \\
    --eval_steps={1 if args.debug else 256} \\
    --dataloader_num_workers={0 if args.debug else args.clm_ndataloaderworkers} \\
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
    pretraining_task: Optional[str],
    model_name_or_path: str,
    arch_config: str,
    seed: int,
    gradient_checkpointing: bool,
    num_train_epochs: int,
    top_k: Optional[int],
    min_freq: Optional[int],
    tr_samples_per_class: Optional[int],
    weighted_loss: Optional[str],
) -> str:

    return f"""#!/bin/bash -l

    #SBATCH --job-name={'debug-' if args.debug else ''}{jobname}
    #SBATCH --account=admalware
    #SBATCH --partition={'debug' if args.debug else 'tier3'}
    #SBATCH --output=./logs/%x_%j.out
    #SBATCH --time={'00-01:00:00' if args.debug else alloc_time}
    #SBATCH --nodes=1
    #SBATCH --cpus-per-task=1
    #SBATCH --ntasks={1 if args.debug else args.clf_ntasks}
    #SBATCH --mem={'16G' if args.debug else alloc_memory}
    #SBATCH --gres=gpu:a100:{args.clf_ngpus}


    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate RawByteClf{2 if SYSTEM == System.RC else ""}
    {"module unload blindfold" if SYSTEM == System.RC else ""}


    {"torchrun --no-python --nnodes=1 --nproc_per_node=" + str(args.clf_ngpus) + " " + DOUBLE_BACKSLASH if args.clf_ngpus > 1 else ""}
    python -u \\
    src/learn/train.py \\
    --root='{ROOT}' \\
    --streaming='false' \\
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
    --beta=0.80 \\
    --early_stopping=false \\

    --task='{downstream_task}' \\
    --pretraining_task='{pretraining_task}' \\
    --tr_size=0.85 \\
    --vl_size=0.15 \\
    --ts_size=0.0 \\
    --min_freq={min_freq} \\
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
    --saves_per_epoch={10 if args.debug else 1} \\
    --evals_per_epoch={10 if args.debug else 1} \\
    --dataloader_num_workers={0 if args.debug else args.clf_ndataloaderworkers} \\
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
    pretraining_task: Optional[str],
    downstream_task: str,
    min_freq: Optional[int],
    tr_samples_per_class: Optional[int],
    seed: int,
) -> str:
    args = [
        model_name,
        str(pretraining_task),
        downstream_task,
        str(min_freq),
        str(tr_samples_per_class),
        str(seed),
    ]
    return "--".join(args)


def compute_mem(
    tr_num_samples: int,
    vl_num_samples: int,
    max_length: int,
    bytes_per_token: int = 4,
) -> str:
    """Compute the memory requirements of the raw data, then add a bit of buffer."""
    num_samples = tr_num_samples + vl_num_samples
    num_tokens = num_samples * max_length
    b = num_tokens * bytes_per_token
    g = b / 1e9
    e = 16 + (g / 8)
    t = int(round(g + e))
    return f"{t}G"


def compute_time(
    tr_num_samples: int,
    vl_num_samples: int,
    max_length: int,
    num_train_epochs: int,
    model_name: str,
) -> str:
    """Compute an estimation of time, then add a bit of buffer. 

    NOTE: the stats were computed in streaming mode with num_data_loaders=0.
    """
    if model_name == "mamba":
        if max_length == 2 ** 14:
            tr_time_per_sample = 0.0548155737704918000
            vl_time_per_sample = 0.0146341463414634150
        if max_length == 2 ** 16:
            tr_time_per_sample = None
            vl_time_per_sample = None
    if model_name == "malconv2":
        if max_length == 2 ** 14:
            tr_time_per_sample = 0.0048668032786885250
            vl_time_per_sample = 0.0040172166427546625
        if max_length == 2 ** 16:
            tr_time_per_sample = None
            vl_time_per_sample = None


    tr_time = tr_time_per_sample * tr_num_samples * num_train_epochs
    vl_time = vl_time_per_sample * vl_num_samples * num_train_epochs

    total_time = tr_time + vl_time
    total_time = (30 * 60) + (.05 * total_time) + total_time

    total_hours = total_time // 3600
    if total_hours < 1:
        days, hours = 0, 1
    elif total_hours >= 24:
        days, hours = divmod(total_hours, 24)
    else:
        days, hours = 0, round(total_hours)

    days, hours = int(days), int(hours)
    return f"0{days}-{hours}:00:00"


# dict[(task, tr_samples_per_class, min_freq), (tr_samples, vl_samples)]
TR_VL_SIZES = {
    ("clf-bod", None, None): (39191, 6970),
    ("clf-bod", 1, None): (382, 2509),
    ("clf-bod", 5, None): (1215, 1920),
    ("clf-sor-fam", None, None): (283445, 50566),
    ("clf-sor-fam", 1, None): (2348, 14255),
    ("clf-sor-fam", 5, None): (6700, 10692),
    ('clf-sor-file', None, None): (13388, 2372),
    ('clf-sor-file', 1, None): (34, 256),
    ('clf-sor-file', 5, None): (125, 229),
    ('clf-sor-beh', None, None): (36866, 23209),
    ('clf-sor-class_', None, None): (12635, 8666),
    ('clf-sor-pack', None, None): (4866, 2524),

    ('clf-bod', None, 10): (47707, 8452),
    ('clf-sor-beh', None, 10): (215788, 29780),
    ('clf-sor-class_', None, 10): (187847, 43363),
    ('clf-sor-fam', None, 10): (839871, 148328),
    ('clf-sor-file', None, 10): (78011, 13777),
    ('clf-sor-pack', None, 10): (28034, 7808),

    ('clf-bod', None, 100): (44102, 7797),
    ('clf-sor-beh', None, 100): (676087, 124691),
    ('clf-sor-class_', None, 100): (501607, 84007),
    ('clf-sor-fam', None, 100): (1652519, 291536),
    ('clf-sor-file', None, 100): (241791, 42543),
    ('clf-sor-pack', None, 100): (78858, 13884),
}


def get_clf_alloc_time_and_mem(
    model_name: str,
    task: str,
    min_freq: Optional[int],
    tr_samples_per_class: Optional[int],
    max_length: int,
    num_train_epochs: int,
) -> tuple[str, str]:
    key = (task, tr_samples_per_class, min_freq)
    tr_num_samples, vl_num_samples = TR_VL_SIZES[key]

    mem = compute_mem(
        tr_num_samples,
        vl_num_samples,
        max_length,
    )
    tim = compute_time(
        tr_num_samples,
        vl_num_samples,
        max_length,
        num_train_epochs,
        model_name,
    )
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
            if pretraining_task is None or model_name != "mamba":
                continue

            jobname = get_jobname(
                model_name,
                None,
                pretraining_task,
                None,
                None,
                0,
            )
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
            weighted_loss = "sample_reweighting" if task in TASKS_SCMF else None

            for pretraining_task in PRETRAINING_TASKS:
                if model_name in ("malconv", "malconv2") and pretraining_task is not None:
                    continue

                for tr_samples_per_class in TR_SAMPLES_PER_CLASS:
                    if tr_samples_per_class is not None and task not in TASKS_SCMF:
                        continue
                    
                    if tr_samples_per_class is None:
                        num_train_epochs = 5
                    elif tr_samples_per_class == 5:
                        num_train_epochs = 10
                    elif tr_samples_per_class == 1:
                        num_train_epochs = 50
                    else:
                        raise ValueError(tr_samples_per_class)

                    for min_freq in MIN_FREQ:
                        if min_freq is not None and tr_samples_per_class is not None:
                            continue

                        alloc_time, alloc_memory = get_clf_alloc_time_and_mem(
                            model_name=model_name,
                            task=task,
                            min_freq=min_freq,
                            tr_samples_per_class=tr_samples_per_class,
                            max_length=MAX_LENGTH,
                            num_train_epochs=num_train_epochs,
                        )

                        for seed in SEEDS:
                            jobname = get_jobname(
                                model_name,
                                pretraining_task,
                                task,
                                min_freq,
                                tr_samples_per_class,
                                seed,
                            )
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
                                min_freq=min_freq,
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
            f = f.relative_to("/home/lk3591/Documents/code/RawByteClf")
            if SYSTEM == System.RC:
                if args.dependencies:
                    if f.stem[-1] == "0":
                        pre = "jobid=$(sbatch"
                        pos = "| awk '{print $4}')"
                    else:
                        pre = "sbatch --dependency=afterok:$jobid"
                        pos = ""
                else:
                    pre = "sbatch"
                    pos = ""
            else:
                if "clm" in f.name:
                    gpus = [str(i) for i in range(args.clm_ngpus)]
                else:
                    gpus = [str(i) for i in range(args.clf_ngpus)]
                pre = f"CUDA_VISIBLE_DEVICES={','.join(gpus)} bash"
                pos = f"&> ./logs/{f.stem}.out"   
            if f.stem[-1] == "0":
                fp.write(f"# {f.stem[0:-3]}\n")
            fp.write(f"{pre} {str(f)} {pos}\n")

    sys.exit()

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
                # raise FileNotFoundError(f)
                print(f"FileNorFoundError: {str(f)}")
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
