"""
Create pretraining, classification, and finetuning bash scripts.
"""

from argparse import ArgumentParser
from enum import Enum
import json
import math
import os
from pathlib import Path
from pprint import pprint
import sys
from typing import Optional
import warnings


# Recommend setting ntasks and ndataloaderworkers such that 
#  ntasks = ngpus * (1 + ndataloaderworkers)
parser = ArgumentParser()
parser.add_argument("--lm_ngpus", type=int, default=1)
parser.add_argument("--lm_ntasks", type=int, default=4)
parser.add_argument("--lm_ndataloaderworkers", type=int, default=3)
parser.add_argument("--clf_ngpus", type=int, default=1)
parser.add_argument("--clf_ntasks", type=int, default=4)
parser.add_argument("--clf_ndataloaderworkers", type=int, default=3)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--dependencies", action="store_true")
args = parser.parse_args()


class System(Enum):
    ARMITAGE = "ARMITAGE"
    RC = "RC"


SYSTEM = System(Path("./config/.system").read_text().strip())
OUTPUT = Path(os.path.realpath(__file__)).parent
DOUBLE_BACKSLASH = """\\"""


# Universal configuration for experments
ROOT = "./output/test" if args.debug else "./output/usenix"
MAX_LENGTH = 2 ** 14
DATA_READ_BYTES = 2 ** 14

# Parameters to vary for the experiments.

# At the moment, gradient checkpointing with distributed data parallel does not work
# with the bidrectional mamba weight-tying, so we need to disable this.
MODELS = [
    ("hrr-tn-uni",   "hrrformer", '{"num_hidden_layers": 1, "hidden_size": 128, "embedding_size": 128, "num_attention_heads": 1, "intermediate_size": 256, "is_decoder": true}'),
    ("hrr-tn-bi",    "hrrformer", '{"num_hidden_layers": 1, "hidden_size": 128, "embedding_size": 128, "num_attention_heads": 1, "intermediate_size": 256, "is_decoder": false}'),
    ("hrr-sm-uni",   "hrrformer", '{"num_hidden_layers": 2, "hidden_size": 256, "embedding_size": 256, "num_attention_heads": 2, "intermediate_size": 512, "is_decoder": true}'),
    ("hrr-sm-bi",    "hrrformer", '{"num_hidden_layers": 2, "hidden_size": 256, "embedding_size": 256, "num_attention_heads": 2, "intermediate_size": 512, "is_decoder": false}'),
    ("hrr-md-uni",   "hrrformer", '{"num_hidden_layers": 3, "hidden_size": 384, "embedding_size": 384, "num_attention_heads": 4, "intermediate_size": 768, "is_decoder": true}'),
    ("hrr-md-bi",    "hrrformer", '{"num_hidden_layers": 3, "hidden_size": 384, "embedding_size": 384, "num_attention_heads": 4, "intermediate_size": 768, "is_decoder": false}'),
    ("hrr-lg-uni",   "hrrformer", '{"num_hidden_layers": 4, "hidden_size": 512, "embedding_size": 512, "num_attention_heads": 8, "intermediate_size": 1024, "is_decoder": true}'),
    ("hrr-lg-bi",    "hrrformer", '{"num_hidden_layers": 4, "hidden_size": 512, "embedding_size": 512, "num_attention_heads": 8, "intermediate_size": 1024, "is_decoder": false}'),
    ("hrr-hg-uni",   "hrrformer", '{"num_hidden_layers": 6, "hidden_size": 768, "embedding_size": 768, "num_attention_heads": 12, "intermediate_size": 2048, "is_decoder": true}'),
    ("hrr-hg-bi",    "hrrformer", '{"num_hidden_layers": 6, "hidden_size": 768, "embedding_size": 768, "num_attention_heads": 12, "intermediate_size": 2048, "is_decoder": false}'),
    ("mamba-tn-uni", "mamba",     '{"mode": "uni", "num_hidden_layers": 3, "hidden_size": 128, "embedding_size": 128, "tie_directions": false}'),  #
    ("mamba-tn-bi",  "mamba",     '{"mode": "bi", "num_hidden_layers": 3, "hidden_size": 128, "embedding_size": 128, "tie_directions": false}'),  #
    ("mamba-sm-uni", "mamba",     '{"mode": "uni", "num_hidden_layers": 6, "hidden_size": 256, "embedding_size": 256, "tie_directions": false}'),  #
    ("mamba-sm-bi",  "mamba",     '{"mode": "bi", "num_hidden_layers": 6, "hidden_size": 256, "embedding_size": 256, "tie_directions": false}'),  #
    ("mamba-md-uni", "mamba",     '{"mode": "uni", "num_hidden_layers": 9, "hidden_size": 384, "embedding_size": 384, "tie_directions": false}'),  #
    ("mamba-md-bi",  "mamba",     '{"mode": "bi", "num_hidden_layers": 9, "hidden_size": 384, "embedding_size": 384, "tie_directions": false}'),  #
    ("mamba-lg-uni", "mamba",     '{"mode": "uni", "num_hidden_layers": 12, "hidden_size": 512, "embedding_size": 512, "tie_directions": false}'),  #
    ("mamba-lg-bi",  "mamba",     '{"mode": "bi", "num_hidden_layers": 12, "hidden_size": 512, "embedding_size": 512, "tie_directions": false}'),  #
    ("malconv",      "malconv2",  '{"mode": "gcg", "channels": 256, "stride": 64, "kernel_size": 64, "embedding_size": 8}'),  # 2.56 M
]
PRETRAINING_TASKS = [None, "clm-sor", "mlm-sor"]
PRETRAINING_CHECKPOINTS = [None, -1, 0]
TASKS_SCMF = ["clf-bod", "clf-sor-fam", "clf-sor-file"]
TASKS_MCMF = ["clf-sor-class_", "clf-sor-beh", "clf-sor-pack"]
TASKS = TASKS_SCMF + TASKS_MCMF
MIN_FREQ = [None, 100]
TR_SAMPLES_PER_CLASS = [None, 1, 5]
WEIGHTED_LOSSES = [None, "sample_reweighting"]
CLF_LEARNING_RATES = [1e-3]
FT_LEARNING_RATES = [1e-3]
LEARNING_RATES = sorted(set(CLF_LEARNING_RATES + FT_LEARNING_RATES))
SEEDS = [0, 1, 2]

# Adjust these frequently to configure which experiments to actually run.
# This is simpler than adding a complex CLI.
# MODELS = list(filter(lambda x: "lg" in x[0], MODELS))
MODELS = list(filter(lambda x: x[0] in ("hrr-lg-bi", "hrr-lg-uni"), MODELS))
PRETRAINING_TASKS = [None, "clm-sor", "mlm-sor"]
TASKS = ["clf-bod"]
WEIGHTED_LOSSES = [None]
PRETRAINING_CHECKPOINTS = [None]


def get_body_lm(
    jobname: str,
    downstream_task: str,
    model_name_or_path: str,
    arch_config: str,
    model_nickname: str,
) -> str:

    parts = model_nickname.split("-")
    if len(parts) == 2:
        name, size, mode = parts[0], parts[1], parts[2]
    if len(parts) == 3:
        name, size, mode = parts[0], parts[1], parts[2]

    bf16 = "false" if name == "hrr" else "true"

    if name == "mamba":
        if size == "sm":
            hours = 48
        if size == "lg":
            hours = 96
        if mode == "bi":
            hours *= 2
    if name == "hrr":
        if size == "sm":
            hours = 12
        if size == "lg":
            hours = 36

    if hours is None:
        warnings.warn(f"Don't know how much time to allocate to {model_nickname=} for {downstream_task=}.")
        tim = 8

    hours /= args.lm_ngpus
    tim = get_slurm_time(hours)

    mem = 32
    mem *= args.lm_ngpus
    mem = f"{mem}G"

    return f"""#!/bin/bash -l

    #SBATCH --job-name={'debug-' if args.debug else ''}{jobname}
    #SBATCH --account=admalware
    #SBATCH --partition={'debug' if args.debug else 'tier3'}
    #SBATCH --output=./logs/%x_%j.out
    #SBATCH --time={'00-01:00:00' if args.debug else tim}
    #SBATCH --nodes=1
    #SBATCH --cpus-per-task=1
    #SBATCH --ntasks={1 if args.debug else args.lm_ntasks}
    #SBATCH --mem={'16G' if args.debug else mem}
    #SBATCH --gres=gpu:a100:{args.lm_ngpus}


    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate RawByteClf{2 if SYSTEM == System.RC else ""}
    {"module unload blindfold" if SYSTEM == System.RC else ""}


    {"torchrun --no-python --nnodes=1 --nproc_per_node=" + str(args.lm_ngpus) + " " + DOUBLE_BACKSLASH if args.lm_ngpus > 1 else ""}
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
    --save_steps={1 if args.debug else 128} \\
    --eval_steps={1 if args.debug else 128} \\
    --dataloader_num_workers={0 if args.debug else args.lm_ndataloaderworkers} \\
    --optim="adamw_torch" \\
    --learning_rate="1e-3" \\
    --lr_scheduler_type="linear" \\
    --warmup_ratio=0.05 \\
    --weight_decay=0.10 \\
    --adam_beta1=0.900 \\
    --adam_beta2=0.990 \\
    --max_grad_norm=1.0 \\
    --save_total_limit=-1 \\
    --model_name_or_path={model_name_or_path} \\
    --max_length={MAX_LENGTH} \\
    --data_read_bytes={DATA_READ_BYTES} \\
    --per_device_train_batch_size={1024 // args.lm_ngpus} \\
    --per_device_eval_batch_size={1024} \\
    --gradient_accumulation_steps={1} \\
    --early_stopping=false \\
    --auto_find_batch_size_and_gradient_accumulation_steps \\
    --tf32=true \\
    --bf16={bf16} \\
    --fp16=false \\
    --gradient_checkpointing=true
    """.replace("    ", "").replace("\n\n", "\n")


def get_body_clf(
    jobname: str,
    alloc_time: str,
    alloc_memory: str,
    streaming: bool,
    downstream_task: str,
    pretraining_task: Optional[str],
    pretraining_checkpoint: Optional[int],
    model_name_or_path: str,
    arch_config: str,
    seed: int,
    gradient_checkpointing: bool,
    num_train_epochs: int,
    top_k: Optional[int],
    min_freq: Optional[int],
    tr_samples_per_class: Optional[int],
    weighted_loss: Optional[str],
    learning_rate: float,
) -> str:

    bf16 = "false" if "hrr" in model_name_or_path else "true"

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
    --streaming='{'true' if streaming else 'false'}' \\
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
    --pretraining_checkpoint={pretraining_checkpoint} \\
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
    --learning_rate="{learning_rate}" \\
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
    --bf16={bf16} \\
    --fp16=false \\
    --gradient_checkpointing={'true' if gradient_checkpointing else 'false'}
    """.replace("    ", "").replace("\n\n", "\n")


def get_jobname(
    model_name: str,
    pretraining_task: Optional[str],
    pretraining_checkpoint: Optional[int],
    downstream_task: str,
    min_freq: Optional[int],
    tr_samples_per_class: Optional[int],
    weighted_loss: Optional[str],
    learning_rate: float,
    seed: int,
) -> str:
    learning_rate = math.log10(learning_rate)
    if not learning_rate.is_integer():
        raise ValueError()
    learning_rate = f"1e{int(learning_rate)}"
    if weighted_loss is not None:
        weighted_loss = weighted_loss[0:3]
    args = [
        model_name,
        str(pretraining_task),
        str(pretraining_checkpoint),
        downstream_task,
        str(min_freq),
        str(tr_samples_per_class),
        str(weighted_loss),
        learning_rate,
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


def get_slurm_time(total_hours: int | float) -> str:
    if total_hours < 1:
        days, hours = 0, 1
    elif total_hours >= 24:
        days, hours = divmod(total_hours, 24)
    else:
        days, hours = 0, round(total_hours)

    days, hours = int(days), int(hours)
    return f"0{days}-{hours}:00:00"


def compute_time(
    tr_num_samples: int,
    vl_num_samples: int,
    max_length: int,
    num_train_epochs: int,
    model_nickname: str,
) -> str:
    """Compute an estimation of time, then add a bit of buffer. 
    """
    parts = model_nickname.split("-")
    if len(parts) == 2:
        name, size, mode = parts[0], parts[1], parts[2]
    if len(parts) == 3:
        name, size, mode = parts[0], parts[1], parts[2]

    if name == "mamba": # this are just the unidrectional times; for bidirection, multiply by two
        if size == "sm":
            if max_length == 2 ** 14:
                tr_time_per_sample = 0.0548155737704918000
                vl_time_per_sample = 0.0146341463414634150
            if max_length == 2 ** 16:
                tr_time_per_sample = None
                vl_time_per_sample = None
        if size == "lg":
            if max_length == 2 ** 14:
                tr_time_per_sample = 0.2441406250000
                vl_time_per_sample = 0.0457763671875
            if max_length == 2 ** 16:
                tr_time_per_sample = None
                vl_time_per_sample = None

        if mode == "bi":
            tr_time_per_sample *= 2
            vl_time_per_sample *= 2

    if name == "malconv2":
        if max_length == 2 ** 14:
            tr_time_per_sample = 0.0048668032786885250
            vl_time_per_sample = 0.0040172166427546625
        if max_length == 2 ** 16:
            tr_time_per_sample = None
            vl_time_per_sample = None

    if name == "hrr":
        if size == "sm":
            if max_length == 2 ** 14:
                tr_time_per_sample = 0.01074219
                vl_time_per_sample = 0.00430416
            if max_length == 2 ** 16:
                tr_time_per_sample = None
                vl_time_per_sample = None
        if size == "lg":
            if max_length == 2 ** 14:
                tr_time_per_sample = 0.012451171875
                vl_time_per_sample = 0.049804687500
            if max_length == 2 ** 16:
                tr_time_per_sample = None
                vl_time_per_sample = None

    tr_time = tr_time_per_sample * tr_num_samples * num_train_epochs
    vl_time = vl_time_per_sample * vl_num_samples * num_train_epochs

    total_time = tr_time + vl_time
    total_time = (30 * 60) + (.05 * total_time) + total_time

    total_hours = total_time / 3600
    return get_slurm_time(total_hours)


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
    model_nickname: str,
    task: str,
    min_freq: Optional[int],
    tr_samples_per_class: Optional[int],
    max_length: int,
    num_train_epochs: int,
) -> tuple[str, str]:
    key = (task, tr_samples_per_class, min_freq)
    tr_num_samples, vl_num_samples = TR_VL_SIZES[key]

    if "mamba" in model_nickname:
        name, size, mode = model_nickname.split("-")
    else:
        name, size, mode = model_nickname, None, None

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
        model_nickname,
    )

    # Special cases:
    if name == "mamba" and size == "sm":
        if key == ("clf-sor-class_", None, None):
            if mode == "bi":
                tim = "00-04:00:00"
            else:
                tim = "00-02:00:00"
        if key == ("clf-sor-beh", None, None):
            if mode == "bi":
                tim = "00-10:00:00"
            else:
                tim = "00-05:00:00"
    if name == "malconv2" and key == ("clf-sor-beh", None, 100):
        tim = "00-01:30:00"

    return tim, mem


def key_for_sorting_jobnames(s: str) -> tuple:
    out = s.split("-")
    for i in range(len(out)):
        o = out[i]
        if o.isdigit():
            out[i] == float(o)
    return out


def main():

    for f in OUTPUT.glob("*.sh"):
        f.unlink()

    outfiles_lm = []
    outfiles_clf = []

    for model_nickname, model_name, arch_config in MODELS:
        arch_config_dict: dict = json.loads(arch_config)
        gradient_checkpointing = True if model_name == "mamba" else False

        # Pretraining
        for pretraining_task in PRETRAINING_TASKS:
            # pretraining is not implemented in these configurations
            if pretraining_task is None:
                continue
            if model_name not in ("mamba", "hrrformer"):
                continue
            if pretraining_task == "mlm-sor" and  model_name == "mamba" and arch_config_dict["mode"] == "uni":
                continue
            if pretraining_task == "mlm-sor" and model_name == "hrrformer" and arch_config_dict["is_decoder"]:
                continue
            if pretraining_task == "clm-sor" and model_name == "mamba" and arch_config_dict["mode"] == "bi":
                continue
            if pretraining_task == "clm-sor" and model_name == "hrrformer" and not arch_config_dict["is_decoder"]:
                continue

            jobname = get_jobname(
                model_name=model_nickname,
                pretraining_task=None,
                pretraining_checkpoint=None,
                downstream_task=pretraining_task,
                min_freq=None,
                tr_samples_per_class=None,
                weighted_loss=None,
                learning_rate=1e-3,
                seed=0,
            )
            body = get_body_lm(
                jobname=jobname,
                downstream_task=pretraining_task,
                model_name_or_path=model_name,
                arch_config=arch_config,
                model_nickname=model_nickname,
            )
            outfile = OUTPUT / (jobname + ".sh")
            with open(outfile, "w") as fp:
                fp.write(body)
            outfiles_lm.append(outfile)

        # Classification
        for task in TASKS:

            for pretraining_task in PRETRAINING_TASKS:
                # pretraining is not implemented in these configurations
                if pretraining_task is not None and model_name not in ("mamba", "hrrformer"):
                    continue
                if pretraining_task == "mlm-sor" and model_name == "mamba" and arch_config_dict["mode"] == "uni":
                    continue
                if pretraining_task == "mlm-sor" and model_name == "hrrformer" and arch_config_dict["is_decoder"]:
                    continue
                if pretraining_task == "clm-sor" and model_name == "mamba" and arch_config_dict["mode"] == "bi":
                    continue
                if pretraining_task == "clm-sor" and model_name == "hrrformer" and not arch_config_dict["is_decoder"]:
                    continue

                for pretraining_checkpoint in PRETRAINING_CHECKPOINTS:
                    # pretraining_checkpoint does not make sense for classification from scratch
                    if pretraining_task is None and pretraining_checkpoint is not None:
                        continue
                    # need to specify a pretraining checkpoint if finetuning
                    if pretraining_task is not None and pretraining_checkpoint is None:
                        continue

                    for tr_samples_per_class in TR_SAMPLES_PER_CLASS:
                        # tr_samples_per_class is not implemented for multilabel tasks
                        if tr_samples_per_class is not None and task not in TASKS_SCMF:
                            continue
                        
                        for min_freq in MIN_FREQ:
                            # using tr_samples_per_class overrides min_freq, so running when both are active is redundant
                            if min_freq is not None and tr_samples_per_class is not None:
                                continue

                            # set the number of training epochs for the various scenarios
                            num_train_epochs = None
                            if tr_samples_per_class is None:
                                if min_freq is None:
                                    num_train_epochs = 5
                                elif min_freq == 10:
                                    num_train_epochs = 2
                                elif min_freq == 100:
                                    num_train_epochs = 1
                            elif tr_samples_per_class == 5:
                                num_train_epochs = 10
                            elif tr_samples_per_class == 1:
                                num_train_epochs = 50
                            if num_train_epochs is None:
                                raise AttributeError("num_train_epochs not set")

                            # get the time and memory requirements for the classification task
                            alloc_time, alloc_memory = get_clf_alloc_time_and_mem(
                                model_nickname=model_nickname,
                                task=task,
                                min_freq=min_freq,
                                tr_samples_per_class=tr_samples_per_class,
                                max_length=MAX_LENGTH,
                                num_train_epochs=num_train_epochs,
                            )

                            # if memory requirements are over 64G, reduce to 64G and set streaming to True
                            if int(alloc_memory[:-1]) > 64:
                                warnings.warn(
                                    f"LargeMemoryWarning: {alloc_memory=} {task=} {min_freq=} "
                                    f"{tr_samples_per_class=}. Reducing alloc_memory to 64G."
                                )
                                streaming = True
                                alloc_memory = "64G"
                            else:
                                streaming = False

                            # if time requirements are over 5 days, reduce to 5 days
                            if int(alloc_time[0:2]) >= 5:
                                warnings.warn(
                                    f"LargeRuntimeWarning: {alloc_time=} {task=} {min_freq=} "
                                    f"{tr_samples_per_class=}. Reducing alloc_time to 05-00:00:00."
                                )
                                alloc_time = "05-00:00:00"

                            for weighted_loss in WEIGHTED_LOSSES:
                                # weighted loss is only implemented for single class tasks;
                                # only needed for imbalanced tasks
                                if weighted_loss is not None:
                                    if task not in TASKS_SCMF or tr_samples_per_class is not None:
                                        continue

                                for learning_rate in LEARNING_RATES:
                                    # we don't need to try many learning rates if training from scratch
                                    if pretraining_task is None and learning_rate not in CLF_LEARNING_RATES:
                                        continue
                                    if pretraining_task is not None and learning_rate not in FT_LEARNING_RATES:
                                        continue

                                    for seed in SEEDS:
                                        jobname = get_jobname(
                                            model_name=model_nickname,
                                            pretraining_task=pretraining_task,
                                            pretraining_checkpoint=pretraining_checkpoint,
                                            downstream_task=task,
                                            min_freq=min_freq,
                                            tr_samples_per_class=tr_samples_per_class,
                                            weighted_loss=weighted_loss,
                                            learning_rate=learning_rate,
                                            seed=seed,
                                        )
                                        body = get_body_clf(
                                            jobname=jobname,
                                            alloc_time=alloc_time,
                                            alloc_memory=alloc_memory,
                                            streaming=streaming,
                                            downstream_task=task,
                                            pretraining_task=pretraining_task,
                                            pretraining_checkpoint=pretraining_checkpoint,
                                            model_name_or_path=model_name,
                                            arch_config=arch_config,
                                            seed=seed,
                                            gradient_checkpointing=gradient_checkpointing,
                                            num_train_epochs=num_train_epochs,
                                            top_k=None,
                                            min_freq=min_freq,
                                            tr_samples_per_class=tr_samples_per_class,
                                            weighted_loss=weighted_loss,
                                            learning_rate=learning_rate,
                                        )
                                        outfile = OUTPUT / (jobname + ".sh")
                                        with open(outfile, "w") as fp:
                                            fp.write(body)
                                        outfiles_clf.append(outfile)


    # did not bother adding seeds or dependencies etc.
    with open(OUTPUT / "run.sh", "w") as fp:
        fp.write("# lm\n")
        for f in sorted(outfiles_lm, key=lambda p: key_for_sorting_jobnames(str(p.name))):
            f = f.relative_to("/home/lk3591/Documents/code/RawByteClf")
            f_parts = f.stem.split("--")
            if SYSTEM == System.RC:
                pre = "sbatch"
                pos = ""
            else:
                gpus = [str(i) for i in range(args.lm_ngpus)]
                pre = f"CUDA_VISIBLE_DEVICES={','.join(gpus)} bash"
                pos = f"&> ./logs/{f.stem}.out"
            fp.write(f"{pre} {str(f)} {pos}\n")



    f_previous_parts = []
    # the dependency logic is dependent on the format of the jobname
    with open(OUTPUT / "run.sh", "a") as fp:
        for f in sorted(outfiles_clf, key=lambda p: key_for_sorting_jobnames(str(p.name))):
            f = f.relative_to("/home/lk3591/Documents/code/RawByteClf")
            f_parts = f.stem.split("--")
            learning_rate = float(f_parts[-2])
            seed = int(f_parts[-1])

            if f_previous_parts[0:-2] != f_parts[0:-2]:
                new_dependency_chain = True
            else:
                new_dependency_chain = False

            if SYSTEM == System.RC:
                # jobs that differ only by learning rate and seed will succeed or fail together, so add a dependency
                if args.dependencies:
                    if new_dependency_chain:
                        pre = "jobid=$(sbatch"
                        pos = "| awk '{print $4}')"
                    else:
                        pre = "sbatch --dependency=afterok:$jobid"
                        pos = ""
                else:
                    pre = "sbatch"
                    pos = ""
            else:
                gpus = [str(i) for i in range(args.clf_ngpus)]
                pre = f"CUDA_VISIBLE_DEVICES={','.join(gpus)} bash"
                pos = f"&> ./logs/{f.stem}.out"

            # add a little comment to make reading the run file easier.
            if new_dependency_chain:
                fp.write(f"# {'--'.join(f_parts[0:-2])}\n")
            fp.write(f"{pre} {str(f)} {pos}\n")

            f_previous_parts = f_parts


if __name__ == "__main__":
    main()
