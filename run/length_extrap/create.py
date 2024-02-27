import math
from pathlib import Path
from pprint import pformat

BODY = """#!/bin/bash -l

#SBATCH --job-name=JOB_NAME
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=05-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1


source ~/anaconda3/etc/profile.d/conda.sh
conda activate RawByteClf
conda activate RawByteClf2
module unload blindfold


if [ $# -eq 0 ]; then
    echo "Error: No argument supplied."
    exit 1
elif [ "$1" != "true" ] && [ "$1" != "false" ]; then
    echo "Error: Invalid argument. Argument must be 'true' or 'false'."
    exit 1
elif [ "$1" = "true" ]; then
    ENFORCE_CUTOFF="true"
elif [ "$1" = "false" ]; then
    ENFORCE_CUTOFF="false"
fi


python -u \\
src/learn/train.py \\
--root="./output/length_extrap" \\
--arch_config=ARCH_CONFIG \\
--metric_for_best_model="eval_accuracy" \\
--task="clf" \\
--streaming=false \\
--tr_length_cutoff=TR_LENGTH_CUTOFF \\
--enforce_cutoff=$ENFORCE_CUTOFF \\
--do_train \\
--do_eval \\
--output_dir=tmp \\
--save_strategy="epoch" \\
--evaluation_strategy="epoch" \\
--num_train_epochs=50 \\
--logging_steps=10 \\
--dataloader_num_workers=0 \\
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
--per_device_train_batch_size=64 \\
--per_device_eval_batch_size=64 \\
--gradient_accumulation_steps=1 \\
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



TR_THRESHOLDS = [2 ** 16, 2 ** 17, 2 ** 18]


OUTPUT = Path("run/length_extrap")

outfiles = []
for model_name, arch_config in CONFIGS:
    for thrsh in TR_THRESHOLDS:
        jobname = f"lx_{thrsh}_{model_name}"  # edit if using multiple mambas
        print(jobname)

        # must replace PRETRAINED_MODEL_PATH before D_MODEL!
        text = BODY \
            .replace("JOB_NAME", jobname) \
            .replace("MODEL_NAME_OR_PATH", model_name) \
            .replace("ARCH_CONFIG", f"'{arch_config}'") \
            .replace("TR_LENGTH_CUTOFF", str(thrsh))

        outfile = OUTPUT / f"{jobname}.sh"
        with open(outfile, "w") as fp:
            fp.write(text)
        outfiles.append(outfile)


with open(OUTPUT / "run.sh", "w") as fp:
    for outfile in sorted(outfiles):
        fp.write(f"sbatch {outfile.as_posix()} true")
        fp.write(f"sbatch {outfile.as_posix()} false")
