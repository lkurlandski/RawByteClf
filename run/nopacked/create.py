"""
"""

from enum import Enum
import os
from pathlib import Path
from pprint import pprint
import sys


class System(Enum):
    RC = 0
    ARMITAGE = 1


SYSTEM = System(int(sys.argv[1]))

CLM_NGPUS = 1 if SYSTEM == System.RC else 2
CLF_NGPUS = 1 if SYSTEM == System.RC else 2
CLM_NTASKS = 4
CLF_NTASKS = 4
CLM_MEM = "64G"
CLF_MEM = "MEMORY"

SEEDS = [0, 1, 2, 3, 4]
REPRESENTATIONS = [8, 16]
TASKS = ["clf-bod", "clf-sor-nam"]
PRETRAINING_TASKS = ["None", "clm"]

ROOT = "./output/nopacked"
MODEL_NAME_OR_PATH = "mamba"
ARCH_CONFIG = '{"mode": "uni", "num_hidden_layers": 8, "hidden_size": 256, "embedding_size": EMBEDDING_SIZE}'
MAX_LENGTH = 2 ** 16 if SYSTEM == System.RC else 2 ** 12
DATA_READ_BYTES = 2 ** 16 if SYSTEM == System.RC else 2 ** 12

CLM_TR_SIZE = 2 ** 21 if SYSTEM == System.RC else 350000
CLM_VL_SIZE = 2 ** 14 if SYSTEM == System.RC else 14868
CLM_TRAIN_BATCH_SIZE = 512
CLM_PER_DEVICE_TRAIN_BATCH_SIZE = CLM_TRAIN_BATCH_SIZE // CLM_NGPUS
CLM_GRADIENT_ACCUMULATION_STEPS = 1
CLM_PER_DEVICE_EVAL_BATCH_SIZE = 64
CLM_SAVE_EVAL_STEPS = 512 if SYSTEM == System.RC else 128

CLF_TRAIN_BATCH_SIZE = 64
CLF_PER_DEVICE_TRAIN_BATCH_SIZE = CLF_TRAIN_BATCH_SIZE // CLF_NGPUS
CLF_GRADIENT_ACCUMULATION_STEPS = 1
CLF_PER_DEVICE_EVAL_BATCH_SIZE = "CLF_PER_DEVICE_EVAL_BATCH_SIZE"


BODY_CLM = f"""#!/bin/bash -l

#SBATCH --job-name=JOB_NAME
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=02-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks={CLM_NTASKS}
#SBATCH --mem={CLM_MEM}
#SBATCH --gres=gpu:a100:{CLM_NGPUS}


source ~/anaconda3/etc/profile.d/conda.sh
conda activate RawByteClf{2 if SYSTEM == System.RC else ""}
{"module unload blindfold" if SYSTEM == System.RC else ""}


torchrun --no-python --nnodes=1 --nproc_per_node={CLM_NGPUS} \\
python -u \\
src/learn/train.py \\
--root="{ROOT}" \\
--arch_config='{ARCH_CONFIG}' \\
--metric_for_best_model="eval_loss" \\
--task="clm" \\
--seed=0 \\
--remove_packed \\
--streaming=false \\
--skip_eval_check=false \\
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
--dataloader_num_workers={3} \\
--optim="adamw_torch" \\
--learning_rate="1e-3" \\
--lr_scheduler_type="linear" \\
--warmup_ratio=0.05 \\
--weight_decay=0.10 \\
--adam_beta1=0.900 \\
--adam_beta2=0.990 \\
--max_grad_norm=1.0 \\
--save_total_limit=-1 \\
--model_name_or_path={MODEL_NAME_OR_PATH} \\
--max_length={MAX_LENGTH} \\
--data_read_bytes={DATA_READ_BYTES} \\
--per_device_train_batch_size={CLM_PER_DEVICE_TRAIN_BATCH_SIZE} \\
--per_device_eval_batch_size={CLM_PER_DEVICE_EVAL_BATCH_SIZE} \\
--gradient_accumulation_steps={CLM_GRADIENT_ACCUMULATION_STEPS} \\
--load_best_model_at_end \\
--early_stopping=false \\
--auto_find_batch_size_and_gradient_accumulation_steps \\
--bf16 \\
--bf16_full_eval \\
--tf32=true \\
--gradient_checkpointing=false
"""


BODY_CLF = f"""#!/bin/bash -l

#SBATCH --job-name=JOB_NAME
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=ALLOC_TIME
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks={CLF_NTASKS}
#SBATCH --mem={CLF_MEM}
#SBATCH --gres=gpu:a100:{CLF_NGPUS}


source ~/anaconda3/etc/profile.d/conda.sh
conda activate RawByteClf{2 if SYSTEM == System.RC else ""}
{"module unload blindfold" if SYSTEM == System.RC else ""}


torchrun --no-python --nnodes=1 --nproc_per_node={CLF_NGPUS} \\
python -u \\
src/learn/train.py \\
--root="{ROOT}" \\
--arch_config='{ARCH_CONFIG}' \\
--metric_for_best_model="eval_accuracy" \\
--task=DOWNSTREAM_TASK \\
--seed=SEED \\
--pretraining_task=PRETRAINING_TASK \\
--remove_packed \\
--streaming=false \\
--skip_eval_check=false \\
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
--num_train_epochs=1 \\
--logging_steps=1 \\
--saves_per_epoch=10 \\
--evals_per_epoch=10 \\
--dataloader_num_workers={3} \\
--optim="adamw_torch" \\
--learning_rate="1e-4" \\
--lr_scheduler_type="linear" \\
--warmup_ratio=0.05 \\
--weight_decay=0.01 \\
--adam_beta1=0.900 \\
--adam_beta2=0.999 \\
--max_grad_norm=1.0 \\
--save_total_limit=2 \\
--model_name_or_path={MODEL_NAME_OR_PATH} \\
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
--gradient_checkpointing=false
"""


OUTPUT = Path(os.path.realpath(__file__)).parent
for f in OUTPUT.glob("*.sh"):
    if f.name[0:3] == "run":
        continue
    f.unlink()


outfiles = []


for representation in REPRESENTATIONS:
    jobname = f"nopack-clm-{representation}-0"
    vocab_size = int(2 ** representation)
    embedding_size = max(8, int(256 / (2 ** (representation - 8))))
    body = BODY_CLM \
        .replace("JOB_NAME", jobname) \
        .replace("REPRESENTATION", str(representation)) \
        .replace("VOCAB_SIZE", str(vocab_size)) \
        .replace("EMBEDDING_SIZE", str(embedding_size))
    outfile = (OUTPUT / jobname).with_suffix(".sh")
    with open(outfile, "w") as fp:
        fp.write(body)
    outfiles.append(outfile)


for representation in REPRESENTATIONS:
    vocab_size = int(2 ** representation)
    embedding_size = max(8, int(256 / (2 ** (representation - 8))))
    per_device_eval_batch_size = 32 if representation == 16 else 64
    for task in TASKS:
        top_k = 10 if task == "clf-bod" else None
        min_freq = None if task == "clf-bod" else 2
        alloc_time = "00-00:30:00" if task == "clf-bod" else "01-00:00:00"
        memory = "16G" if task == "clf-bod" else "48G"
        for pretraining_task in PRETRAINING_TASKS:
            name = "clf" if pretraining_task == "None" else "ft"
            for seed in SEEDS:
                jobname = f"nopack-{name}-{task}-{representation}-{seed}"
                body = BODY_CLF \
                    .replace("JOB_NAME", jobname) \
                    .replace("REPRESENTATION", str(representation)) \
                    .replace("VOCAB_SIZE", str(vocab_size)) \
                    .replace("EMBEDDING_SIZE", str(embedding_size)) \
                    .replace("CLF_PER_DEVICE_EVAL_BATCH_SIZE", str(per_device_eval_batch_size)) \
                    .replace("DOWNSTREAM_TASK", task) \
                    .replace("TOP_K", str(top_k)) \
                    .replace("MIN_FREQ", str(min_freq)) \
                    .replace("ALLOC_TIME", alloc_time) \
                    .replace("MEMORY", memory) \
                    .replace("SEED", str(seed)) \
                    .replace("PRETRAINING_TASK", pretraining_task)
                outfile = (OUTPUT / jobname).with_suffix(".sh")
                with open(outfile, "w") as fp:
                    fp.write(body)
                outfiles.append(outfile)


with open(OUTPUT / "run.sh", "w") as fp:
    for f in sorted(outfiles, key=lambda p: str(p).split("-")[1]):
        if SYSTEM == System.RC:
            pre = "sbatch"
            pos = ""
        else:
            if "clm" in f.name:
                gpus = [str(i) for i in range(CLM_NGPUS)]
            else:
                gpus = [str(i) for i in range(CLF_NGPUS)]
            pre = f"CUDA_VISIBLE_DEVICES={','.join(gpus)} bash"
            pos = f"&> ./logs/{f.stem}.out"   
        fp.write(f"{pre} {str(f)} {pos}\n")

