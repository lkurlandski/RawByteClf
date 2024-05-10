from dataclasses import dataclass
import math
import os
from pathlib import Path
from pprint import pformat
from typing import Optional


ALGORITHMS = ["Raw", "BPE", "Unigram"]
VOCAB_SIZES = [256, 16384]
TOP_KS = [10]
MODES = ["uni", "bi"]
TASKS = ["clm", "mlm"]
SEEDS = [42, 0, 1, 2, 3]


JOB_NAME = "JOB_NAME"
ROOT = "./output/finetuning"
MODEL_NAME_OR_PATH = "mamba"
ARCH_CONFIG = '{"mode": "uni", "num_hidden_layers": 8, "hidden_size": 256, "embedding_size": 256}'
PER_DEVICE_TRAIN_BATCH_SIZE = 64
PER_DEVICE_EVAL_BATCH_SIZE = 64
REPRESENTATION = 8
ALGORITHM = "ALGORITHM"
VOCAB_SIZE = "VOCAB_SIZE"
SEED = "SEED"
MAX_LENGTH = 2 ** 16
DATA_READ_BYTES = 2 ** 16
TOP_K = "TOP_K"
BF_OR_FP = "bf"
TF32 = "true"

LM_TR_SIZE = 1000000
LM_VL_SIZE = 10000
LM_SAVE_EVAL_STEPS = 200
LM_GRADIENT_ACCUMULATION_STEPS = 8

CLF_GRADIENT_ACCUMULATION_STEPS = 1

BODY_LM = f"""#!/bin/bash -l

#SBATCH --job-name={JOB_NAME}
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=02-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks={4}
#SBATCH --mem={64}G
#SBATCH --gres=gpu:a100:1


source ~/anaconda3/etc/profile.d/conda.sh
conda activate RawByteClf
conda activate RawByteClf2
module unload blindfold


python -u \\
src/learn/train.py \\
--root="{ROOT}" \\
--arch_config='{ARCH_CONFIG}' \\
--metric_for_best_model="eval_loss" \\
--task="TASK" \\
--seed=42 \\
--streaming=true \\
--skip_eval_check=false \\
--dataset_backend="HF" \\
--representation={REPRESENTATION} \\
--algorithm={ALGORITHM} \\
--vocab_size={VOCAB_SIZE} \\
--tr_size={LM_TR_SIZE} \\
--vl_size={LM_VL_SIZE} \\
--ts_size=0 \\
--do_train \\
--output_dir=tmp \\
--save_strategy="steps" \\
--evaluation_strategy="steps" \\
--num_train_epochs=1 \\
--logging_steps=10 \\
--save_steps={LM_SAVE_EVAL_STEPS} \\
--eval_steps={LM_SAVE_EVAL_STEPS} \\
--dataloader_num_workers={3} \\
--optim="adamw_torch" \\
--learning_rate="1e-3" \\
--lr_scheduler_type="linear" \\
--warmup_ratio=0.05 \\
--weight_decay=0.01 \\
--adam_beta2=0.999 \\
--max_grad_norm=1.0 \\
--save_total_limit=-1 \\
--model_name_or_path={MODEL_NAME_OR_PATH} \\
--max_length={MAX_LENGTH} \\
--data_read_bytes={DATA_READ_BYTES} \\
--per_device_train_batch_size={PER_DEVICE_TRAIN_BATCH_SIZE} \\
--per_device_eval_batch_size={PER_DEVICE_EVAL_BATCH_SIZE} \\
--gradient_accumulation_steps={LM_GRADIENT_ACCUMULATION_STEPS} \\
--load_best_model_at_end \\
--early_stopping=false \\
--auto_find_batch_size_and_gradient_accumulation_steps \\
--{BF_OR_FP}16 \\
--{BF_OR_FP}16_full_eval \\
--tf32={TF32} \\
--gradient_checkpointing=true
"""


BODY_CLF = f"""#!/bin/bash -l

#SBATCH --job-name={JOB_NAME}
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=00-6:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks={4}
#SBATCH --mem={64}G
#SBATCH --gres=gpu:a100:1


source ~/anaconda3/etc/profile.d/conda.sh
conda activate RawByteClf
conda activate RawByteClf2
module unload blindfold


python -u \\
src/learn/train.py \\
--root="{ROOT}" \\
--arch_config='{ARCH_CONFIG}' \\
--metric_for_best_model="eval_accuracy" \\
--task="clf" \\
--seed={SEED} \\
--streaming=false \\
--skip_eval_check=false \\
--top_k={TOP_K} \\
--dataset_backend="HF" \\
--representation={REPRESENTATION} \\
--algorithm={ALGORITHM} \\
--vocab_size={VOCAB_SIZE} \\
--tr_size=0.85 \\
--vl_size=0.15 \\
--ts_size=0.0 \\
--do_train \\
--output_dir=tmp \\
--save_strategy="epoch" \\
--evaluation_strategy="epoch" \\
--num_train_epochs=5 \\
--logging_steps=10 \\
--save_steps=100 \\
--eval_steps=100 \\
--dataloader_num_workers={3} \\
--optim="adamw_torch" \\
--learning_rate="1e-3" \\
--lr_scheduler_type="linear" \\
--warmup_ratio=0.00 \\
--weight_decay=0.01 \\
--adam_beta2=0.999 \\
--max_grad_norm=1.0 \\
--save_total_limit=3 \\
--model_name_or_path={MODEL_NAME_OR_PATH} \\
--max_length={MAX_LENGTH} \\
--data_read_bytes={DATA_READ_BYTES} \\
--per_device_train_batch_size={PER_DEVICE_TRAIN_BATCH_SIZE} \\
--per_device_eval_batch_size={PER_DEVICE_EVAL_BATCH_SIZE} \\
--gradient_accumulation_steps={CLF_GRADIENT_ACCUMULATION_STEPS} \\
--load_best_model_at_end \\
--early_stopping=false \\
--auto_find_batch_size_and_gradient_accumulation_steps \\
--{BF_OR_FP}16 \\
--{BF_OR_FP}16_full_eval \\
--tf32={TF32} \\
--gradient_checkpointing=true
"""


OUTPUT = Path(os.path.realpath(__file__)).parent
PRETRAINED_MODEL_PATH = {  # FIXME
    "clm": Path("/home/lk3591/Documents/code/RawByteClf/output/finetuning/mamba/seed-42/representation--8/algorithm--Raw/vocab_size--256/max_length--65536/task--clm/tr_size--1000000/depth--1/mode--uni/num_hidden_layers--8/hidden_size--256/embedding_size--256/per_device_train_batch_size--8/gradient_accumulation_steps--64/learning_rate--0.001/weight_decay--0.01/adam_beta1--0.9/adam_beta2--0.999/adam_epsilon--1e-08/max_grad_norm--1.0/lr_scheduler_type--linear/warmup_ratio--0.0/bf16--True/fp16--False/tf32--True/optim--adamw_torch/checkpoints/"),
    "mlm": Path("/home/lk3591/Documents/code/RawByteClf/output/finetuning/mamba/seed-42/representation--8/algorithm--Raw/vocab_size--256/max_length--65536/task--mlm/tr_size--1000000/depth--1/mode--bi/num_hidden_layers--8/hidden_size--256/embedding_size--256/per_device_train_batch_size--4/gradient_accumulation_steps--128/learning_rate--0.001/weight_decay--0.01/adam_beta1--0.9/adam_beta2--0.999/adam_epsilon--1e-08/max_grad_norm--1.0/lr_scheduler_type--linear/warmup_ratio--0.0/bf16--True/fp16--False/tf32--True/optim--adamw_torch/checkpoints/"),
}


for algorithm in ALGORITHMS:
    for vocab_size in VOCAB_SIZES:
        if algorithm == "Raw" and vocab_size != 256:
            continue
        elif algorithm != "Raw" and vocab_size == 256:
            continue

        for mode, task in zip(MODES, TASKS):
            job_name = f"ft-{task}-{algorithm[0]}-{vocab_size}"
            body = BODY_LM \
                .replace("JOB_NAME", job_name) \
                .replace("ALGORITHM", algorithm) \
                .replace("VOCAB_SIZE", str(vocab_size)) \
                .replace("TASK", task) \
                .replace('"mode": "uni"', f'"mode": "{mode}"')
            print(job_name)
            with open(OUTPUT / f"{job_name}.sh", "w") as fp:
                fp.write(body)

            for top_k in TOP_KS:
                for seed in SEEDS:
                    job_name = f"ft-{'clf'}-{mode}-{algorithm[0]}-{vocab_size}-{top_k}-{seed}"
                    body = BODY_CLF \
                        .replace("JOB_NAME", job_name) \
                        .replace("ALGORITHM", algorithm) \
                        .replace("VOCAB_SIZE", str(vocab_size)) \
                        .replace("TOP_K", str(top_k)) \
                        .replace("SEED", str(seed))
                    print(job_name)
                    with open(OUTPUT / f"{job_name}.sh", "w") as fp:
                        fp.write(body)

                    if not PRETRAINED_MODEL_PATH[task].exists():
                        continue
                    checkpoints = list(PRETRAINED_MODEL_PATH[task].iterdir())
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.name.split("-")[1]))
                    for f in checkpoints:
                        job_name = f"ft-{'ft'}-{f.name.replace('checkpoint-', '')}-{mode}-{algorithm[0]}-{vocab_size}-{top_k}-{seed}"
                        body = BODY_CLF \
                            .replace("JOB_NAME", job_name) \
                            .replace("ALGORITHM", algorithm) \
                            .replace("VOCAB_SIZE", str(vocab_size)) \
                            .replace("TOP_K", str(top_k)) \
                            .replace("SEED", str(seed)) \
                            .replace("mamba", f'"{f.as_posix()}"') \
                            .replace('"mode": "uni"', f'"mode": "{mode}"')
                        print(job_name)
                        with open(OUTPUT / f"{job_name}.sh", "w") as fp:
                            fp.write(body)
