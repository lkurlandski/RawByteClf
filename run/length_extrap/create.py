import math
from pathlib import Path
from pprint import pformat

BODY = """#!/bin/bash -l

#SBATCH --job-name=JOB_NAME
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=DD-HH:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=4
#SBATCH --mem=96G
#SBATCH --gres=gpu:a100:1


source ~/anaconda3/etc/profile.d/conda.sh
conda activate RawByteClf
conda activate RawByteClf2
module unload blindfold


python -u \\
src/learn/train.py \\
--root="./output/length_extrap" \\
--arch_config=ARCH_CONFIG \\
--metric_for_best_model="eval_accuracy" \\
--task="clf" \\
--streaming=false \\
--tr_length_cutoff=TR_LENGTH_CUTOFF \\
--enforce_cutoff=ENFORCE_CUTOFF \\
--do_train \\
--do_eval \\
--output_dir=tmp \\
--save_strategy="steps" \\
--evaluation_strategy="steps" \\
--max_steps=MAX_STEPS \\
--save_steps=SAVE_EVAL_STEPS \\
--eval_steps=SAVE_EVAL_STEPS \\
--logging_steps=10 \\
--dataloader_num_workers=3 \\
--optim="adamw_torch" \\
--learning_rate="1e-3" \\
--lr_scheduler_type="linear" \\
--warmup_ratio=0.00 \\
--weight_decay=0.01 \\
--adam_beta2=0.999 \\
--max_grad_norm=1.0 \\
--save_total_limit=3 \\
--model_name_or_path=MODEL_NAME_OR_PATH \\
--max_length=1048576 \\
--per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE \\
--per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE \\
--gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS \\
--load_best_model_at_end \\
--early_stopping=false \\
--auto_find_batch_size_and_gradient_accumulation_steps \\
--bf16 \\
--bf16_full_eval \\
--tf32=true
"""


CONFIGS = [
    ("mymalconv", '{"hidden_size": 512}'),
    ("mamba", '{"d_model": 128, "n_layer": 4, "mlp_hidden_size": 512}'),
    # ("mamba", '{"d_model": 256, "n_layer": 8, "mlp_hidden_size": 512}'),
    # ("mamba", '{"d_model": 384, "n_layer": 12, "mlp_hidden_size": 512}'),
    # ("mamba", '{"d_model": 512, "n_layer": 16, "mlp_hidden_size": 512}'),
    # ("mamba", '{"d_model": 768, "n_layer": 24, "mlp_hidden_size": 512}'),
]


# Validation with batch size of 4 for mamba takes six minutes
# 200 validation cycles takes 1200 minutes, or 20 hours...
# So we might have to extend these times substantially...
# Training with enforce=true for mamba @65536 only takes 6 hours though
times = {
    "mymalconv": {2 ** 16: "00-06", 2 ** 17: "01-00", 2 ** 18: "03-00"},
    "mamba": {2 ** 16: "05-00", 2 ** 17: "05-00", 2 ** 18: "05-00"},
}

# tr sizes: 15930, 25815, 31520
max_steps = {2 ** 16: 10000, 2 ** 17: 20000, 2 ** 18: 30000}
save_eval_steps = {2 ** 16: 50, 2 ** 17: 100, 2 ** 18: 150}
per_device_eval_batch_size = {"mymalconv": 64, "mamba": 4}
per_device_train_batch_size = {
    "mymalconv": {
        2 ** 16: {True: 64, False: 64}, 
        2 ** 17: {True: 64, False: 64},
        2 ** 18: {True: 64, False: 64},
    },
    "mamba": {
        2 ** 16: {True: 32, False: 4},
        2 ** 17: {True: 16, False: 4},
        2 ** 18: {True: 8, False: 4},
     },
}


TR_THRESHOLDS = [2 ** 16, 2 ** 17, 2 ** 18]


OUTPUT = Path("run/length_extrap")

outfiles = []
for model_name, arch_config in CONFIGS:
    for thrsh in TR_THRESHOLDS:
        for enforce in [True, False]:
            jobname = f"lx_{thrsh}_{model_name}_{enforce}"
            print(jobname)
            train_batch_size = per_device_train_batch_size[model_name][thrsh][enforce]

            # must replace PRETRAINED_MODEL_PATH before D_MODEL!
            text = BODY \
            .replace("JOB_NAME", jobname) \
            .replace("MODEL_NAME_OR_PATH", model_name) \
            .replace("ARCH_CONFIG", f"'{arch_config}'") \
            .replace("TR_LENGTH_CUTOFF", str(thrsh)) \
            .replace("DD-HH", times[model_name][thrsh]) \
            .replace("MAX_STEPS", str(max_steps[thrsh])) \
            .replace("SAVE_EVAL_STEPS", str(save_eval_steps[thrsh])) \
            .replace("PER_DEVICE_EVAL_BATCH_SIZE", str(per_device_eval_batch_size[model_name])) \
            .replace("PER_DEVICE_TRAIN_BATCH_SIZE", str(train_batch_size)) \
            .replace("GRADIENT_ACCUMULATION_STEPS", str(int(64 / train_batch_size))) \
            .replace("ENFORCE_CUTOFF", str(enforce).lower())


            outfile = OUTPUT / f"{jobname}.sh"
            with open(outfile, "w") as fp:
                fp.write(text)
            outfiles.append(outfile)


with open(OUTPUT / "run.sh", "w") as fp:
    for outfile in sorted(outfiles):
        fp.write(f"sbatch {outfile.as_posix()}\n")
