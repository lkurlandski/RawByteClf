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
REPRESENTATIONS = [8, 16]
TASKS = ["clf-bod", "clf-sor-nam", "clf-elf-nam"]
PRETRAINING_TASKS = ["None", "clm", "clm-elf"]
PACKING_PROTOCOLS = ["yes", "no", "any"]

ROOT = "./output/main"
MODEL_NAME_OR_PATHS = ["mamba", "malconv", "malconv2"]
ARCH_CONFIGS = {
    "mamba": '{"mode": "uni", "num_hidden_layers": 8, "hidden_size": 256, "embedding_size": EMBEDDING_SIZE}',
    "malconv": '{"channels": 128, "stride": 512, "kernel_size": 512, "embedding_size": EMBEDDING_SIZE}',
    "malconv2": '{"mode": "gcg", "channels": 256, "stride": 64, "kernel_size": 64, "embedding_size": EMBEDDING_SIZE}',
}
MAX_LENGTH = 2 ** 16 if args.system == System.RC else 2 ** 12
DATA_READ_BYTES = 2 ** 16 if args.system == System.RC else 2 ** 12

CLM_TR_SIZE = 2 ** 21
CLM_VL_SIZE = 2 ** 14
CLM_TRAIN_BATCH_SIZE = 512
CLM_PER_DEVICE_TRAIN_BATCH_SIZE = CLM_TRAIN_BATCH_SIZE // args.clm_ngpus
CLM_GRADIENT_ACCUMULATION_STEPS = 1
CLM_PER_DEVICE_EVAL_BATCH_SIZE = "CLM_PER_DEVICE_EVAL_BATCH_SIZE"
CLM_SAVE_EVAL_STEPS = 512 if args.system == System.RC else 128

CLF_TRAIN_BATCH_SIZE = 64
CLF_PER_DEVICE_TRAIN_BATCH_SIZE = CLF_TRAIN_BATCH_SIZE // args.clf_ngpus
CLF_GRADIENT_ACCUMULATION_STEPS = 1
CLF_PER_DEVICE_EVAL_BATCH_SIZE = "CLF_PER_DEVICE_EVAL_BATCH_SIZE"


# no - # tr_size ~= 350000; vl_size ~= 60000
# yes - # tr_size == 336828; vl_size == 59687
# any - # tr_size == 651339; vl_size == 115334


CLF_ALLOC_TIME: dict[tuple[str, str, int, str], str] = {
    ("mamba", "no", 8, "clf-bod"): "00-01:00:00",
    ("mamba", "no", 16, "clf-bod"): "00-00:40:00",
    ("mamba", "no", 8, "clf-sor-nam"): "01-12:00:00",
    ("mamba", "no", 16, "clf-sor-nam"): "01-00:00:00",
    ("mamba", "yes", 8, "clf-bod"): "00-06:00:00",
    ("mamba", "yes", 16, "clf-bod"): "00-04:00:00",
    ("mamba", "yes", 8, "clf-sor-nam"): "01-12:00:00",
    ("mamba", "yes", 16, "clf-sor-nam"): "01-00:00:00",
    ("mamba", "any", 8, "clf-bod"): "00-06:00:00",
    ("mamba", "any", 16, "clf-bod"): "00-04:00:00",
    ("mamba", "any", 8, "clf-sor-nam"): "03-00:00:00",
    ("mamba", "any", 16, "clf-sor-nam"): "02-00:00:00",

    ("malconv", "no", 8, "clf-bod"): "00-00:30:00",
    ("malconv", "no", 16, "clf-bod"): "00-00:30:00",
    ("malconv", "no", 8, "clf-sor-nam"): "00-06:00:00",
    ("malconv", "no", 16, "clf-sor-nam"): "00-06:00:00",
    ("malconv", "yes", 8, "clf-bod"): "00-00:30:00",
    ("malconv", "yes", 16, "clf-bod"): "00-00:30:00",
    ("malconv", "yes", 8, "clf-sor-nam"): "00-06:00:00",
    ("malconv", "yes", 16, "clf-sor-nam"): "00-06:00:00",
    ("malconv", "any", 8, "clf-bod"): "00-00:30:00",
    ("malconv", "any", 16, "clf-bod"): "00-00:30:00",
    ("malconv", "any", 8, "clf-sor-nam"): "00-12:00:00",
    ("malconv", "any", 16, "clf-sor-nam"): "00-12:00:00",
}
for k, v in list(CLF_ALLOC_TIME.items()):
    m, p, r, t = k
    CLF_ALLOC_TIME[("malconv2", p, r, t)] = v
    
CLF_ALLOC_MEM: dict[tuple[str, str], str] = {
    ("no", "clf-bod"): "16G",
    ("no", "clf-sor-nam"): "48G",
    ("yes", "clf-bod"): "32G",
    ("yes", "clf-sor-nam"): "48G",
    ("any", "clf-bod"): "32G",
    ("any", "clf-sor-nam"): "64G",
}


# ELF dataset is about same size as BODMAS, so we're just going to update the
# structures for clf-elf-name to be the same as clf-bod.
for k in list(CLF_ALLOC_TIME.keys()):
    m, p, r, t = k
    if t == "clf-bod":
        CLF_ALLOC_TIME[(m, p, r, "clf-elf-nam")] = CLF_ALLOC_TIME[k]
for k in list(CLF_ALLOC_MEM.keys()):
    p, t = k
    if t == "clf-bod":
        CLF_ALLOC_MEM[(p, "clf-elf-nam")] = CLF_ALLOC_MEM[k]


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
--packing_protocol=PACKING_PROTOCOL \\
--streaming={'true' if args.debug else 'true'} \\
--skip_eval_check={'true' if args.debug else 'false'} \\
--dataset_backend="HF" \\
--representation=REPRESENTATION \\
--algorithm="Raw" \\
--vocab_size=VOCAB_SIZE \\
--tr_size={CLM_TR_SIZE} \\
--vl_size={CLM_VL_SIZE} \\
--ts_size=0 \\
--do_train \\
--output_dir=tmp \\
--save_strategy="steps" \\
--evaluation_strategy="steps" \\
--num_train_epochs=2 \\
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
--packing_protocol=PACKING_PROTOCOL \\
--streaming={'true' if args.debug else 'true'} \\
--skip_eval_check={'true' if args.debug else 'false'} \\
--top_k=TOP_K \\
--min_freq=MIN_FREQ \\
--dataset_backend="HF" \\
--representation=REPRESENTATION \\
--algorithm="Raw" \\
--vocab_size=VOCAB_SIZE \\
--tr_size=0.85 \\
--vl_size=0.15 \\
--ts_size=0.0 \\
--do_train \\
--output_dir=tmp \\
--save_strategy="epoch" \\
--evaluation_strategy="epoch" \\
--num_train_epochs=NUM_TRAIN_EPOCHS \\
--logging_steps=1 \\
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
    packing_protocol: str,
    model_name: str,
    pretraining_task: str,
    downstream_task: str,
    representation: int,
    seed: int,
) -> str:
    args = [
        packing_protocol,
        model_name,
        pretraining_task,
        downstream_task,
        str(representation),
        str(seed),
    ]
    return "--".join(args)


for model_name in MODEL_NAME_OR_PATHS:
    arch_config = ARCH_CONFIGS[model_name]
    gradient_checkpointing = "true" if model_name == "mamba" else "false"
    per_device_eval_batch_size = 512 if model_name in ("malconv", "malconv2") else 64

    for packing_protocol in PACKING_PROTOCOLS:

        for representation in REPRESENTATIONS:
            vocab_size = int(2 ** representation)
            embedding_size = max(8, int(256 / (2 ** (representation - 8))))

            for pretraining_task in PRETRAINING_TASKS:
                if pretraining_task == "None" or model_name != "mamba":
                    continue

                jobname = get_jobname(packing_protocol, model_name, "None", pretraining_task, representation, 0)
                body = BODY_CLM \
                    .replace("JOB_NAME", jobname) \
                    .replace("MODEL_NAME_OR_PATH", model_name) \
                    .replace("ARCH_CONFIG", arch_config) \
                    .replace("PACKING_PROTOCOL", packing_protocol) \
                    .replace("REPRESENTATION", str(representation)) \
                    .replace("VOCAB_SIZE", str(vocab_size)) \
                    .replace("EMBEDDING_SIZE", str(embedding_size)) \
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
                alloc_time = CLF_ALLOC_TIME[(model_name, packing_protocol, representation, task)]
                memory = CLF_ALLOC_MEM[(packing_protocol, task)]
                num_train_epochs = 5 if task == "clf-elf-nam" else 1

                for pretraining_task in PRETRAINING_TASKS:
                    if model_name in ("malconv", "malconv2") and pretraining_task != "None":
                        continue
                    for seed in SEEDS:
                        jobname = get_jobname(packing_protocol, model_name, pretraining_task, task, representation, seed)
                        body = BODY_CLF \
                            .replace("JOB_NAME", jobname) \
                            .replace("MODEL_NAME_OR_PATH", model_name) \
                            .replace("ARCH_CONFIG", arch_config) \
                            .replace("PACKING_PROTOCOL", packing_protocol) \
                            .replace("REPRESENTATION", str(representation)) \
                            .replace("VOCAB_SIZE", str(vocab_size)) \
                            .replace("EMBEDDING_SIZE", str(embedding_size)) \
                            .replace("CLF_PER_DEVICE_EVAL_BATCH_SIZE", str(per_device_eval_batch_size)) \
                            .replace("GRADIENT_CHECKPOINTING", gradient_checkpointing) \
                            .replace("DOWNSTREAM_TASK", task) \
                            .replace("TOP_K", str(top_k)) \
                            .replace("MIN_FREQ", str(min_freq)) \
                            .replace("ALLOC_TIME", alloc_time) \
                            .replace("MEMORY", memory) \
                            .replace("NUM_TRAIN_EPOCHS", str(num_train_epochs)) \
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

