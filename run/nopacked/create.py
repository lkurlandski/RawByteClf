"""

FIXME:
- The casual language models, especially with 16-bit byte representation,
 require more than 2 days. The 8-bit and 16-bit representations likely require
 02-20:00:00 and 04-12:00:00 respectively.
- The causal language models probably do not need this much data to converge
 at the current model sizes. I'd guess that they'd converge to the same loss
 with half as much data.
- The classifiers pretty much converge after a single epoch. 10 epochs is simply
 not nessecary.
"""

import os
from pathlib import Path


SEEDS = [0, 1, 2, 3, 4]
REPRESENTATIONS = [8, 16]
TASKS = ["clf-bod", "clf-sor-nam"]


ROOT = "./output/nopacked"
MODEL_NAME_OR_PATH = "mamba"
ARCH_CONFIG = '{"mode": "uni", "num_hidden_layers": 8, "hidden_size": 256, "embedding_size": EMBEDDING_SIZE}'
PER_DEVICE_TRAIN_BATCH_SIZE = 64
PER_DEVICE_EVAL_BATCH_SIZE = 64
SEED = "SEED"
MAX_LENGTH = 2 ** 16
DATA_READ_BYTES = 2 ** 16
LOGGING_STEPS = 10

LM_TR_SIZE = 2 ** 21  # 2097152
LM_VL_SIZE = 2 ** 14  # 16384
LM_SAVE_EVAL_STEPS = 512
LM_GRADIENT_ACCUMULATION_STEPS = 8

CLF_GRADIENT_ACCUMULATION_STEPS = 1

BODY_LM = f"""#!/bin/bash -l

#SBATCH --job-name=JOB_NAME
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
--task="clm" \\
--seed=0 \\
--remove_packed \\
--streaming=true \\
--skip_eval_check=false \\
--dataset_backend="HF" \\
--representation=REPRESENTATION \\
--algorithm="Raw" \\
--vocab_size=VOCAB_SIZE \\
--tr_size={LM_TR_SIZE} \\
--vl_size={LM_VL_SIZE} \\
--ts_size=0 \\
--do_train \\
--output_dir=tmp \\
--save_strategy="steps" \\
--evaluation_strategy="steps" \\
--num_train_epochs=1 \\
--logging_steps={LOGGING_STEPS} \\
--save_steps={LM_SAVE_EVAL_STEPS} \\
--eval_steps={LM_SAVE_EVAL_STEPS} \\
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
--per_device_train_batch_size={PER_DEVICE_TRAIN_BATCH_SIZE} \\
--per_device_eval_batch_size={PER_DEVICE_EVAL_BATCH_SIZE} \\
--gradient_accumulation_steps={LM_GRADIENT_ACCUMULATION_STEPS} \\
--load_best_model_at_end \\
--early_stopping=false \\
--auto_find_batch_size_and_gradient_accumulation_steps \\
--bf16 \\
--bf16_full_eval \\
--tf32=true \\
--gradient_checkpointing=true
"""


BODY_CLF = f"""#!/bin/bash -l

#SBATCH --job-name=JOB_NAME
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=00-2:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks={4}
#SBATCH --mem={16}G
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
--logging_steps={LOGGING_STEPS} \\
--saves_per_epoch=10 \\
--evals_per_epoch=10 \\
--dataloader_num_workers={3} \\
--optim="adamw_torch" \\
--learning_rate="1e-3" \\
--lr_scheduler_type="linear" \\
--warmup_ratio=0.00 \\
--weight_decay=0.01 \\
--adam_beta1=0.900 \\
--adam_beta2=0.999 \\
--max_grad_norm=1.0 \\
--save_total_limit=2 \\
--model_name_or_path={MODEL_NAME_OR_PATH} \\
--max_length={MAX_LENGTH} \\
--data_read_bytes={DATA_READ_BYTES} \\
--per_device_train_batch_size={PER_DEVICE_TRAIN_BATCH_SIZE} \\
--per_device_eval_batch_size={PER_DEVICE_EVAL_BATCH_SIZE} \\
--gradient_accumulation_steps={CLF_GRADIENT_ACCUMULATION_STEPS} \\
--load_best_model_at_end \\
--early_stopping=false \\
--auto_find_batch_size_and_gradient_accumulation_steps \\
--bf16 \\
--bf16_full_eval \\
--tf32=true \\
--gradient_checkpointing=true
"""


OUTPUT = Path(os.path.realpath(__file__)).parent
for f in OUTPUT.glob("*.sh"):
    f.unlink()


outfiles = []


for representation in REPRESENTATIONS:
    jobname = f"nopack-clm-{representation}-0"
    vocab_size = int(2 ** representation)
    embedding_size = max(8, int(256 / (2 ** (representation - 8))))
    body = BODY_LM \
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
    for task in TASKS:
        top_k = 10 if task == "clf-bod" else None
        min_freq = None if task == "clf-bod" else 2
        for seed in SEEDS:
            jobname = f"nopack-clf-{task}-{representation}-{seed}"
            body = BODY_CLF \
                .replace("JOB_NAME", jobname) \
                .replace("REPRESENTATION", str(representation)) \
                .replace("VOCAB_SIZE", str(vocab_size)) \
                .replace("EMBEDDING_SIZE", str(embedding_size)) \
                .replace("DOWNSTREAM_TASK", task) \
                .replace("TOP_K", str(top_k)) \
                .replace("MIN_FREQ", str(min_freq)) \
                .replace("SEED", str(seed)) \
                .replace("PRETRAINING_TASK", "None")
            outfile = (OUTPUT / jobname).with_suffix(".sh")
            with open(outfile, "w") as fp:
                fp.write(body)
            outfiles.append(outfile)

            jobname = f"nopack-ft-{task}-{representation}-{seed}"
            body = BODY_CLF \
                .replace("JOB_NAME", jobname) \
                .replace("REPRESENTATION", str(representation)) \
                .replace("VOCAB_SIZE", str(vocab_size)) \
                .replace("EMBEDDING_SIZE", str(embedding_size)) \
                .replace("DOWNSTREAM_TASK", task) \
                .replace("TOP_K", str(top_k)) \
                .replace("MIN_FREQ", str(min_freq)) \
                .replace("SEED", str(seed)) \
                .replace("PRETRAINING_TASK", "clm")
            outfile = (OUTPUT / jobname).with_suffix(".sh")
            with open(outfile, "w") as fp:
                fp.write(body)
            outfiles.append(outfile)


outfiles.sort(key=lambda p: str(p).split("-")[1])


with open(OUTPUT / "run.sh", "w") as fp:
    for f in outfiles:
        fp.write(f"sbatch {str(f)}\n")
