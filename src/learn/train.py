"""
Train and evaluate the models for malware family classification.
# TODO: drop bytes after converting to text!

# TODO: investigate increasing the batch size and number of processes for the
# Dataset.map() calls. Experiment with malconv, as the impact of disk access is
# more pronounced there than with the heavy transformers models.

# TODO: does the attention mask need to be handled by the tokenizer? Can it be
# handled by the dataloader instead? If so, we could skip the entire second map 
# process, which would greatly enhance the speed of the entire process...

# TODO: figure out a more intelligent way for the models to return their
# hidden states. This will probably entail overrriding the Trainer.eval_loop
# to compute the metrics more frequently and free the logits from memory.

# TODO: memory profiling is useless on the cluster as it is for the entire node,
# not just me. Need to figure out a way to profile just one user.

# NOTE 0: storing token class probabilities for every token in a long input sequence requires
    an infeasible amount of memory (hundreds of GBs) for long sequences with a reasonbly-
    sized test dataset.
"""

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime
from functools import partial, reduce
import gc
import inspect
import json
import math
from pathlib import Path
from pprint import pformat, pprint
import random
import time
from typing import Any, Callable, Literal, Optional, Protocol
import os
import sys
import warnings

# pylint: disable=wrong-import-position
print(f"Entered {__file__=}")
if __name__ == "__main__":
    print(f"STARTING @{datetime.now()}\n{'-' * 88}", flush=True)
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# pylint: enable=wrong-import-position

from accelerate.utils import DistributedDataParallelKwargs
import datasets
from datasets import (
    DatasetDict,
    Dataset,
    IterableDataset,
    IterableDatasetDict,
)
import numpy as np
import torch
from torch import tensor, Tensor  # pylint: disable=no-name-in-module
from torch.nn import CrossEntropyLoss, Embedding
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Subset
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    DefaultDataCollator,
    HfArgumentParser,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerFast,
    Trainer,
    LongformerConfig,
    LongformerPreTrainedModel,
    ReformerConfig,
    ReformerPreTrainedModel,
    NystromformerConfig,
    NystromformerPreTrainedModel,
    FNetConfig,
    FNetPreTrainedModel,
    RwkvPreTrainedModel,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    EarlyStoppingCallback,
)
from transformers import TrainingArguments as HfTrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.training_args import ParallelMode
from transformers.trainer_utils import (
    BestRun,
    PredictionOutput,
    TrainOutput,
    PREFIX_CHECKPOINT_DIR,
    IntervalStrategy,
)
from transformers.trainer_pt_utils import AcceleratorConfig  # pylint: disable=no-name-in-module
from transformers.models.reformer.modeling_reformer import _get_least_common_mult_chunk_len
from transformers.utils import (
    CONFIG_NAME,
    is_ninja_available,
    is_torch_bf16_gpu_available,
    is_torch_tf32_available,
    is_torch_neuroncore_available,
)
from tqdm import tqdm
try:
    from ray import tune
    from ray.tune.search.hyperopt import HyperOptSearch
    from ray.tune import TuneError
except (ModuleNotFoundError, ImportError) as _err:
    print(f"{_err.__class__.__name__}: ray")

from src.cfg import BR, System, SYSTEM
from src.enums import (
    CompressionAlgorithm,
    EncryptionAlgorithm,
    LiftLevel,
    Task,
    TokenizationAlgorithm,
    WeightedLossAlgorithm,
)
from src.utils import (
    getattr_recursively,
    count_parameters,
    get_highest_path,
    object_from_superset_of_constructor_kwds,
    to_long_tensor,
    compress,
    encrypt,
    compose_functions,
    remove_empty_directories,
    check_model_parameters,
)
from src.architectures.head_utils import Head, check_for_anomalous_weights
from src.architectures.other import FocalLoss
from src.architectures.malconv_hf import (
    MalConvConfig,
    MalConvForSequenceClassification,
    MalConvPreTrainedModel,
)
from src.architectures.malconv2 import (
    MalConv2Config,
    MalConv2ForSequenceClassification,
    MalConv2PreTrainedModel,
    MalConv2EnsembleForSequenceClassification,
)
from src.architectures.hrrformer import (
    HRRConfig,
    HRRForSequenceClassification,
    HRRForMaskedLM,
    HRRForCausalLM,
    HRRPreTrainedModel,
    HRREnsembleForSequenceClassification,
)
from src.architectures.mamba_hf import (
    MambaConfig,
    MambaForSequenceClassification,
    MambaForCausalLM,
    MambaForMaskedLM,
    MambaPreTrainedModel,
    MambaEnsembleForSequenceClassification,
)
from src.architectures.rwkv import (
    RwkvConfig,
    RwkvForSequenceClassification,
)
from src.data.loaders_core import (
    Materials,
    get_materials_esp_clm,
    get_materials_esp_mlm,
    get_materials_esp_det,
    get_materials_esp_fam,
    get_materials_esp_beh,
)
from src.data.loaders_hf import get_dataset_hf, print_dataset_hf, is_dataset_empty, merge_raw_dis_dec_datasets
from src.data.loaders_pt import get_dataset_pt, print_dataset_pt, MapBinaryDatasetDict, IterableBinaryDatasetDict
from src.learn.class_weighting import sample_reweighting, inverse_class_frequency
from src.learn.collators import EnsembleDataCollatorWithPadding
from src.learn.helpers import Args, OutputHelper
from src.learn.evaluation import (
    CLMComputeMetrics,
    MLMComputeMetrics,
    CLFComputeMetricsBinary,
    CLFComputeMetricsSingleLabel,
    CLFComputeMetricsMultiLabel,
)
from src.learn.preprocessing import (
    hf_bytes_to_input_ids,
    hf_tokenize_bytes,
    hf_tokenize_str,
    hf_multilabel_encode,
    bytes_to_input_ids,
    tokenize_bytes,
    hf_compress_bytes,
    hf_encrypt_bytes,
)
from src.learn.printers import print_tokenizer, print_data_collator, print_config, print_model
from src.learn.utils import (
    pad_to_multiple_of_fn,
    find_two_largest_factors,
    float_to_int,
    compute_total_steps,
    str_or_bool_to_str,
    get_mem,
    find_executable_batch_size,
    find_executable_batch_size_and_gradient_accumulation_steps,
    interpret_bytes_as_integers,
    chunk_mask,
    optimizer_to_,
    clear_cuda_caches,
)
from src.tokenization.api import get_fast_tokenizer, load_unigrams, save_unigrams


random.seed(0)
np.random.seed(0)
torch.random.manual_seed(0)


def _hp_space(trial: Any) -> dict[str, float | int]:
    raise NotImplementedError()

hp_space_malconv = hp_space_longformer = hp_space_hrrformer = _hp_space


PAD_TO = 8

# Some random temporary flags.
MOVE_IN_MEMORY = False

# Default variables for the datasets.Dataset.map() and datasets.IterableDataset.map()
BATCH_SIZE_MAP: Optional[int] = 1024
BATCH_SIZE_ITR: Optional[int] = 128
WRITER_BATCH_SIZE: Optional[int] = 1000
CACHE_FILE_NAME: Optional[str] = None
# NOTE: if this is None, the map from bytes directly to integers will not be parallelized at all.
# NUM_PROC: Optional[int] = len(os.sched_getaffinity(0)) if len(os.sched_getaffinity(0)) != 40 else 20
NUM_PROC = None
KEEP_IN_MEMORY = False

# Variables for hyperparameter tuning.
N_INITIAL_POINTS = 16
N_TRIALS = 64
TUNE_RESOURCES_PER_TRIAL = {
    "cpu": 4,
    "gpu": 1,
}
RAISE_ON_FAILED_TRIAL = False


MODEL_NAMES = [
    "longformer",
    "reformer",
    "nystromformer",
    "fnet",
    "malconv",
    "malconv2",
    "hrrformer",
    "rwkv",
    "mamba",
]

REQ_ATTENTION_MASK = [
    "longformer",
    "reformer",
    "nystromformer",
    "hrrformer",
]

REQ_TOKEN_TYPE_IDS = [
    "longformer",
    "reformer",
    "nystromformer",
    "hrrformer",
]

MODEL_NAME_TO_CONFIG_CLASS = {
    "longformer": LongformerConfig,
    "reformer": ReformerConfig,
    "nystromformer": NystromformerConfig,
    "fnet": FNetConfig,
    "malconv": MalConvConfig,
    "malconv2": MalConv2Config,
    "hrrformer": HRRConfig,
    "rwkv": RwkvConfig,
    "mamba": MambaConfig,
}


@dataclass
class TrainingArguments(HfTrainingArguments):

    # Analogue to save_steps and eval_steps.
    saves_per_epoch: Optional[float] = field(default=None)
    evals_per_epoch: Optional[float] = field(default=None)

    def __post_init__(self) -> None:
        # If do_eval flag is not passed, as would be the case when passing do_tune, it is set to
        # True by the transformers.TrainingArguments. When we pass do_tune, we do not want the
        # do_eval flag to be set to True, so we adjust the value as needed at the end of __init__.
        do_eval = self.do_eval

        # If resume_from_checkpoint is passed as a flag without an argument, it is not set to True
        # by the transformers.TrainingArguments. This let's us pass in "true" from the command line,
        # and have the flag be set to True, i.e., resume training from the last checkpoint.
        if isinstance(self.resume_from_checkpoint, str) and self.resume_from_checkpoint.lower() == "true":  # pylint: disable=no-member
            self.resume_from_checkpoint = True

        # If no metric is supplied, eval_loss is a good one :)
        if self.metric_for_best_model is None:
            self.metric_for_best_model = "eval_loss"

        # Pytorch recommends setting this parameter to False.
        # https://pytorch.org/docs/stable/checkpoint.html
        # Huggingface recommends setting it to False when using DDP
        # https://huggingface.co/docs/trl/en/sft_trainer
        # I don't really understand how its going to impact training with gradient checkpointing,
        # but it seems to work in the non-DDP scenario, so let's just keep it as a default.
        # I also cannot figure out how to properly set this from the CLI, so its got to be done
        # within the Python code itself.
        if self.gradient_checkpointing:
            if self.gradient_checkpointing_kwargs is None:
                ...
                # This is now causing an error:
                  # mamba_ssm/ops/selective_scan_interface.py, line 235, in backward
                  # conv1d_out, delta, A, B, C, D, delta_bias, scan_intermediates, out) = ctx.saved_tensors
                  # RuntimeError: !grad_accumulator_.expired() INTERNAL ASSERT FAILED at "/opt/conda/conda-bld/pytorch_1682343995026/work/torch/csrc/autograd/saved_variable.cpp":226, please report a bug to PyTorch. No grad accumulator for a saved leaf
                # self.gradient_checkpointing_kwargs = {"use_reentrant": False}

        if self.use_cpu or self.no_cuda:
            self.fp16 = False
            self.fp16_full_eval = False
            self.bf16 = False
            self.bf16_full_eval = False
            self.tf32 = False

        # Assumes we will not be using mixed precision on the CPU.
        if not is_torch_bf16_gpu_available():
            if self.bf16:
                warnings.warn("UnavailableNumericType: Requested bf16. Using fp16 instead.")
                self.fp16 = True
                self.bf16 = False
            if self.bf16_full_eval:
                self.fp16_full_eval = True
                self.bf16_full_eval = False
                warnings.warn("UnavailableNumericType: Requested bf16_full_eval. Using fp16 instead.")
        if not is_torch_tf32_available():
            if self.tf32:
                warnings.warn("UnavailableNumericType: Requested tf32. Using fp32 instead.")
                self.tf32 = False

        super().__post_init__()

        # When training with multiple GPUs, if the number of steps is not divisible by the number of
        # devices, the sequence lengths for a batch prepared for one device might not equal the
        # sequence length for a batch prepared for another device. Due to a bug in accelerate, this
        # causes an error when concatenating tensors. The issue is documented here:
        # https://github.com/huggingface/transformers/issues/26548. The temporary fix is to either
        # use a number of steps that is divisible by the number of devices and the
        # per_device_train_batch or simply set the dispatch_batches flag to false.
        if self.world_size > 1:
            if hasattr(self, "accelerator_config"):
                if isinstance(self.accelerator_config, AcceleratorConfig):
                    self.accelerator_config.dispatch_batches = False
                elif isinstance(self.accelerator_config, dict):
                    self.accelerator_config["dispatch_batches"] = False
                else:
                    raise TypeError(f"Invalid type for accelerator_config: {type(self.accelerator_config)}")
            else:  # Place this in the else block to avoid a deprecation warning
                self.dispatch_batches = False

        self.do_eval = do_eval

    def epochs_to_steps(self, num_train_examples: int) -> tuple[Optional[int], Optional[int], Optional[int]]:
        """
        If epoch information, e.g., saves_per_epoch, evals_per_epoch, num_train_epochs is provided, this
         function will calculate the corresponding values for save_steps, eval_steps, and max_steps.
         The epoch information for saves_per_epoch and evals_per_epoch is given priority over the save_steps
         and eval_steps information. If the max_steps is provided, it is given priority over
         num_train_epochs. This is roughly in-line with the natural behavior of the Trainer.

        Returns
        -------
        tuple[Optional[int], Optional[int], Optional[int]]: max_steps, save_steps, eval_steps
        """
        max_steps = self.max_steps
        save_steps = self.save_steps
        eval_steps = self.eval_steps

        if self.max_steps is None or self.max_steps == -1:
            max_steps = compute_total_steps(
                num_train_examples,
                self.num_train_epochs,
                per_device_batch_size=self.per_device_train_batch_size,
                n_accumulation_steps=self.gradient_accumulation_steps,
                n_devices=self.world_size,
            )

        if self.saves_per_epoch is not None:
            if self.saves_per_epoch > max_steps:
                raise ValueError(f"{self.saves_per_epoch=} must be less than or equal to {max_steps=}.")
            num_evals = math.ceil(self.num_train_epochs * self.saves_per_epoch)
            save_steps = max(int(math.floor(max_steps / num_evals)), 1)

        if self.evals_per_epoch is not None:
            if self.evals_per_epoch > max_steps:
                raise ValueError(f"{self.evals_per_epoch=} must be less than or equal to {max_steps=}.")
            num_saves = math.ceil(self.num_train_epochs * self.evals_per_epoch)
            eval_steps = max(int(math.floor(max_steps / num_saves)), 1)

        return max_steps, save_steps, eval_steps

    def hf_training_arguments_object(self) -> HfTrainingArguments:
        return object_from_superset_of_constructor_kwds(HfTrainingArguments, **self.__dict__)


class SaveConfigToCheckpointCallback(TrainerCallback):
    def on_save(self, args: HfTrainingArguments, state: TrainerState, control: TrainerControl, **kwds) -> None:
        checkpoint_folder = f"{args.output_dir}/{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        kwds["model"].config.save_pretrained(checkpoint_folder)
        with open(f"{checkpoint_folder}/{CONFIG_NAME}", "r") as fp:
            config = json.load(fp)
        config["architectures"] = type(kwds["model"]).__name__
        with open(f"{checkpoint_folder}/{CONFIG_NAME}", "w") as fp:
            json.dump(config, fp, indent=4, sort_keys=True)


class UtilCallback(TrainerCallback):

    def __init__(self, *args, do_print: bool = True, **kwds) -> None:
        super().__init__(*args, **kwds)
        self.do_print = do_print
        self._time_step_start = -1.0
        self._time_step_end = -1.0
        self._time_step_deltas: list[float] = []

    def on_log(self, args: HfTrainingArguments, state: TrainerState, control: TrainerControl, **kwds) -> None:  # pylint: disable=unused-argument
        m = get_mem(unit="B")
        d = {"mem_used": m[2], "mem_avail": m[1], "mem_total": m[0]}
        state.log_history[-1].update(d)
        # even if this is added to the log_history, the ProgressCallback will not print it, so we
        # need to print it manually. Also, end="" doesn't seem to work for some reason.
        if self.do_print:
            print(d, end="\n", flush=True)

        if len(self._time_step_deltas) > 0:
            d = {"mean_time_per_step": np.mean(np.array(self._time_step_deltas))}
            state.log_history[-1].update(d)
            self._time_step_deltas = []

    def on_step_begin(self, args: HfTrainingArguments, state: TrainerState, control: TrainerControl, **kwds) -> None:  # pylint: disable=unused-argument
        self._time_step_start = time.time()

    def on_step_end(self, args: HfTrainingArguments, state: TrainerState, control: TrainerControl, **kwds) -> None:  # pylint: disable=unused-argument
        self._time_step_end = time.time()
        self._time_step_deltas.append(self._time_step_end - self._time_step_start)


class RobustEpochCallback(TrainerCallback):
    """
    Ensures that the final model is logged, saved, and evaluated when using "steps" as the
    save or evaluation strategy.
    """

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):

        condition = True  # detect if training has occurred since last log and training will stop after this epoch.
        condition &= control.should_training_stop  # training will stop after this epoch
        condition &= args.save_strategy == IntervalStrategy.STEPS  # save strategy is steps
        condition &= args.eval_strategy == IntervalStrategy.STEPS  # eval strategy is steps
        # condition &= state.global_step > state._globalstep_last_logged  # _globalstep_last_logged not available
        condition &= state.global_step % args.save_steps != 0  # indicates that training has occurred since last save

        # Log
        if condition or args.logging_strategy == IntervalStrategy.EPOCH:
            control.should_log = True

        # Evaluate
        if condition or args.eval_strategy == IntervalStrategy.EPOCH and args.eval_delay <= state.epoch:
            control.should_evaluate = True

        # Save
        if condition or args.save_strategy == IntervalStrategy.EPOCH:
            control.should_save = True

        return control


class ComputeLossFunction(Protocol):

    def __call__(self, outputs: tuple | dict, labels: Tensor, num_items_in_batch: Optional[int] = None) -> Tensor:
        ...


class CustomComputeLossFunction:

    def __init__(self, num_labels: int, problem_type: str, loss_fn: torch.nn.Module) -> None:
        self.num_labels = num_labels
        self.problem_type = problem_type
        self.loss_fn = loss_fn
        if self.problem_type not in ("single_label_classification", "multi_label_classification", "regression"):
            raise ValueError(f"{self.problem_type}")

    def __call__(self, outputs: SequenceClassifierOutput, labels: Tensor, num_items_in_batch: Optional[int] = None) -> Tensor:
        self.loss_fn = self.loss_fn.to(labels.device)

        inputs  = outputs.logits
        targets = labels
        if self.problem_type == "single_label_classification":
            inputs = inputs.view(-1, self.num_labels)

        loss = self.loss_fn(inputs, targets)

        # if torch.isnan(loss).any():
        #     raise RuntimeError("NaN loss.")
        # if torch.isinf(loss).any():
        #     raise RuntimeError("Inf loss.")

        return loss

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"CustomComputeLossFunction({self.num_labels}, {self.problem_type}, {self.loss_fn.__class__.__name__})"


def hp_model_init(
    trial: Optional[dict[str, Any]],
    task: Task,
    model_name_or_path: str,
    tokenizer: Optional[PreTrainedTokenizerFast] = None,
    max_length: Optional[int] = None,
    num_labels: Optional[int] = None,
    id2label: Optional[dict[int, str]] = None,
    label2id: Optional[dict[str, int]] = None,
) -> PreTrainedModel:
    """
    Create new model for Optuna hyperaparameter tuning.

    Args:
        trial (Optional[Trial | dict[str, Any]]): None when first called, optuna.Trial if
            tuning with optuna, otherwise the parameters for the trial if tuning with ray.

    Cannot use kwds because huggingface checks for a function that takes one and only one parameter.
    """

    kwds = {
        k: v
        for k, v in {
            "num_labels": num_labels,
            "id2label": id2label,
            "label2id": label2id,
        }.items()
        if v
    }
    if isinstance(trial, dict):
        hparams = trial
    else:
        hparams = {}

    config = get_config(
        model_name_or_path,
        tokenizer,
        max_length,
        arch_config=None,
        **(hparams | kwds),
    )
    model = get_model(task, None, config, **kwds)
    if model is None:
        raise RuntimeError("Model is None for some reason.")
    return model


def hp_compute_objective(metrics: dict[str, float]) -> float:
    return metrics["eval_loss"]


def object_to_model_name(obj: PretrainedConfig | PreTrainedModel | str | Path) -> str:
    if obj in MODEL_NAMES:
        return obj

    if isinstance(obj, (FNetConfig, FNetPreTrainedModel)):
        return "fnet"
    if isinstance(obj, (NystromformerConfig, NystromformerPreTrainedModel)):
        return "nystromformer"
    if isinstance(obj, (ReformerConfig, ReformerPreTrainedModel)):
        return "reformer"
    if isinstance(obj, (LongformerConfig, LongformerPreTrainedModel)):
        return "longformer"
    if isinstance(obj, (HRRConfig, HRRPreTrainedModel)):
        return "hrrformer"
    if isinstance(obj, (RwkvConfig, RwkvPreTrainedModel)):
        return "rwkv"
    if isinstance(obj, (MambaConfig, MambaPreTrainedModel)):
        return "mamba"
    if isinstance(obj, (MalConvConfig, MalConvPreTrainedModel)):
        return "malconv"
    if isinstance(obj, (MalConv2Config, MalConv2PreTrainedModel)):
        return "malconv2"

    possible_model_names = []
    for model_name in MODEL_NAMES:
        if model_name in str(obj).lower():
            possible_model_names.append(model_name)
    if len(possible_model_names) == 1:
        return possible_model_names[0]
    if len(possible_model_names) > 1:
        if set(possible_model_names) == {"malconv", "malconv2"}:
            return "malconv2"
        raise RuntimeError(f"Multiple possible model names: {possible_model_names} for {obj=}")

    raise RuntimeError(f"Could not determine a model name for {obj=}")


def get_model_type(model_name_or_path: str | Path) -> Literal["HF", "MC"]:
    model_name_or_path = str(model_name_or_path)
    if "malconv" in model_name_or_path:
        return "MC"
    return "HF"


def longformer_max_position_embeddings(max_length: int, attention_window: int) -> int:
    return pad_to_multiple_of_fn(attention_window + max_length, PAD_TO)


def _modify_positional_embeddings(
    current_embeddings: Embedding,
    max_position_embeddings: int,
    freeze: bool = False,
    duplicate: bool = False,
    initialize: bool = False,
    initalizer_range: float = None,
) -> Embedding:
    def _extra_duplicate_embeddings(length: int, X: Tensor) -> PreTrainedModel:
        return torch.cat([X for _ in range(0, length // X.shape[0] + 1)], dim=0)[:length]

    def _extra_random_embeddings(shape: tuple[int]) -> PreTrainedModel:
        return torch.normal(mean=0.0, std=initalizer_range, size=shape)

    if duplicate and initialize:
        raise ValueError("Must specify either `duplicate` (bool) or `initalize` (bool).")
    if initialize and initalizer_range is None:
        raise ValueError("Must specify `initalizer_range` (float) when `initialize` is True.")

    if duplicate or initialize:
        extra_embeddings_shape = (
            max_position_embeddings - current_embeddings.num_embeddings,
            current_embeddings.embedding_dim,
        )
        if duplicate:
            extra_embeddings = _extra_duplicate_embeddings(
                extra_embeddings_shape[0], current_embeddings.weight
            )
        elif initialize:
            extra_embeddings = _extra_random_embeddings(extra_embeddings_shape)
        _new_embeddings = torch.cat([current_embeddings.weight, extra_embeddings], dim=0)
    else:
        _new_embeddings = current_embeddings.weight

    return Embedding.from_pretrained(
        _new_embeddings, freeze=freeze, padding_idx=current_embeddings.padding_idx
    )


def modify_positional_embeddings(
    model: PreTrainedModel,
    max_position_embeddings: int,
    freeze: bool = False,
    duplicate: bool = False,
    initialize: bool = False,
    initalizer_range: float = None,
) -> PreTrainedModel:
    if not any([freeze, duplicate, initialize]):
        return model
    if not isinstance(model, (transformers.LongformerForSequenceClassification,)):
        raise TypeError(f"Cannot add additional positional embeddings to this model: {type(model)}")

    if isinstance(model, transformers.LongformerForSequenceClassification):
        current_embeddings = model.longformer.embeddings.position_embeddings

    print(f"{current_embeddings=}")
    new_embeddings = _modify_positional_embeddings(
        current_embeddings,
        max_position_embeddings,
        freeze,
        duplicate,
        initialize,
        initalizer_range,
    )
    print(f"{new_embeddings=}")
    print(BR, flush=True)

    if isinstance(model, transformers.LongformerForSequenceClassification):
        model.longformer.embeddings.position_embeddings = new_embeddings

    return model


def modify_positional_embeddings_allowed(model: Any) -> bool:
    incompatible = [
        RwkvPreTrainedModel,
        MambaPreTrainedModel,
        MalConvPreTrainedModel,
        MalConv2PreTrainedModel,
    ]
    if isinstance(model, tuple(incompatible)):
        return False
    if isinstance(model, PreTrainedModel):
        return True
    return False


def get_config_from_path(model_name_or_path: str | Path, **kwds) -> PretrainedConfig:

    config_file = f"{model_name_or_path}/{CONFIG_NAME}"
    with open(config_file, "r") as fp:
        config = json.load(fp)
    architecture = config["architectures"][0]

    possible = []
    for m in MODEL_NAMES:
        if m in architecture.lower():
            possible.append(m)
        # TODO: rename HRRForMaskedLM and HRRForSequenceClassification then remove this.
        # Or rename the `model_name` key from "hrrformer" to "hrr".
        elif m == "hrrformer" and "hrr" in architecture.lower():
            possible.append(m)

    if len(possible) == 1:
        return MODEL_NAME_TO_CONFIG_CLASS[possible[0]](**(config | kwds))
    if len(possible) > 1:
        raise RuntimeError(
            f"Tried to associate the {architecture=} from the {config_file=} with a model_name."
            f"Multiple possibilities were found, so we cannot proceed: {possible=}."
        )
    if len(possible) == 0:
        raise RuntimeError(
            f"Tried to associate the {architecture=} from the {config_file=} with a model_name."
            f"No possibilities were found, so we cannot proceed: {possible=}."
        )

    raise RuntimeError()


def get_config(
    model_name_or_path: str | dict[LiftLevel, str],
    tokenizer: Optional[PreTrainedTokenizerFast] = None,
    max_length: Optional[int] = None,
    arch_config: Optional[dict[str, Any]] = None,
    **kwds,
) -> PretrainedConfig:
    """
    Get the configuration for the model.

    Precedence (highest to lowest):
        - config from the tokenizer, max_length
        - kwds
        - arch_config
        - default config from the architecture itself

    Args:
        model_name_or_path (str): name of the model or path to the model
        tokenizer (PreTrainedTokenizerFast): tokenizer to use
        max_length (int): maximum length of the input sequence
        arch_config (dict): architecture-specific configuration
        id2label (dict[int, str]): use for classification
        label2id (dict[str, int]): use for classification
        num_labels (int): use for classification
    """
    # PretrainedConfig doesn't like None values for num_labels id2label and label2id.
    for k in ["num_labels", "id2label", "label2id", "problem_type"]:
        if k in kwds and kwds[k] is None:
            kwds.pop(k)

    if isinstance(model_name_or_path, dict):
        configs = [get_config(p, tokenizer, max_length, arch_config, **kwds) for p in model_name_or_path.values()]
        if not all(c == configs[0] for c in configs):
            raise RuntimeError()
        return configs[0]

    if Path(model_name_or_path).exists():
        print("Getting config from disk.")
        config = get_config_from_path(model_name_or_path, **kwds)
        if arch_config is not None:
            for k, v in arch_config.items():
                if hasattr(config, k) and getattr(config, k) != v:
                    print(f"Modifying the config found on disk. Field={k} Value={getattr(config, k)} New Value={v}")
                    setattr(config, k, v)
        return config

    print("Creating new config.")

    # When hyperparameter tuning, some of these can get turned into floats for some weird reason.
    float_to_int_kwds = [
        "num_hidden_layers",
        "num_attention_heads",
        "hidden_size",
        "intermediate_size",
        "attention_window",
    ]
    for k in [k for k in kwds if k in float_to_int_kwds]:
        kwds[k] = float_to_int(kwds[k])

    vocab_size = pad_to_multiple_of_fn(len(tokenizer), PAD_TO)
    max_posititional_embeddings = pad_to_multiple_of_fn(max_length, 8) if max_length else None

    arch_config = {} if arch_config is None else arch_config
    kwds = arch_config | kwds

    # pylint: disable=use-dict-literal

    if model_name_or_path.lower() == "longformer":
        kwds = kwds | dict(
            vocab_size=vocab_size,
            max_position_embeddings=longformer_max_position_embeddings(max_length, kwds.pop("attention_window", 512)),
            attention_window=kwds.pop("attention_window", 512),
            sep_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        return LongformerConfig(**kwds)

    if model_name_or_path.lower() == "reformer":
        kwds = kwds | dict(
            vocab_size=vocab_size,
            max_position_embeddings=max_posititional_embeddings,
            hidden_size=kwds.pop("hidden_size", 256),
            feed_forward_size=kwds.pop("feed_forward_size", kwds.pop("intermediate_size", 512)),
            attn_layers=["local" if i % 2 == 0 else "lsh" for i in range(kwds.pop("num_hidden_layers", 6))],
            axial_pos_shape=list(find_two_largest_factors(max_length)),
            axial_pos_embds_dim=[kwds.pop("hidden_size", 256) // 2, kwds.pop("hidden_size", 256) // 2],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        return ReformerConfig(**kwds)

    if model_name_or_path.lower() == "nystromformer":
        kwds = kwds | dict(
            vocab_size=vocab_size,
            max_position_embeddings=max_posititional_embeddings,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        return NystromformerConfig(**kwds)

    if model_name_or_path.lower() == "fnet":
        kwds = kwds | dict(
            vocab_size=vocab_size,
            max_position_embeddings=max_posititional_embeddings,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        return FNetConfig(**kwds)

    if model_name_or_path.lower() == "hrrformer":
        kwds = kwds | dict(
            vocab_size=vocab_size,
            max_position_embeddings=max_posititional_embeddings,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            cls_token_id=tokenizer.bos_token_id,
            sep_token_id=tokenizer.eos_token_id,
        )
        return HRRConfig(**kwds)

    if model_name_or_path.lower() == "rwkv":
        if not is_ninja_available():
            raise RuntimeError("Ninja is required to use RWKV. Without it, the model is too slow.")
        kwds = kwds | dict(
            vocab_size=vocab_size,
            context_length=max_posititional_embeddings,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        return RwkvConfig(**kwds)

    if model_name_or_path.lower() == "mamba":
        kwds = kwds | dict(
            vocab_size=vocab_size,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_vocab_size_multiple=PAD_TO,
        )
        return MambaConfig(**kwds)

    if model_name_or_path.lower() == "malconv":
        kwds = kwds | dict(
            vocab_size=vocab_size,
            pad_token_id=tokenizer.pad_token_id,
        )
        return MalConvConfig(**kwds)

    if model_name_or_path.lower() == "malconv2":
        kwds = kwds | dict(
            vocab_size=vocab_size,
            pad_token_id=tokenizer.pad_token_id,
        )
        return MalConv2Config(**kwds)

    # pylint: enable=use-dict-literal

    raise ValueError(f"Invalid model name or path: {model_name_or_path}")


def get_model(
    task: Task,
    model_name_or_path: Optional[str | dict[LiftLevel, str]] = None,
    config: Optional[PretrainedConfig] = None,
    ensemble: bool = False,
    **kwds,
) -> PreTrainedModel:

    # PreTrainedModel doesn't like None values for num_labels id2label and label2id.
    for k in ["num_labels", "id2label", "label2id"]:
        if k in kwds and kwds[k] is None:
            kwds.pop(k)

    # Raise an exception if the arguments lead to ambiguous behavior to ensure we're taking the correct action.
    should_load_from_disk = None
    should_load_from_conf = None
    if model_name_or_path is not None:
        if isinstance(model_name_or_path, str) and model_name_or_path.lower() in MODEL_NAMES:
            should_load_from_disk = False
            should_load_from_conf = True
        elif isinstance(model_name_or_path, str) and Path(model_name_or_path).exists():
            should_load_from_disk = True
            should_load_from_conf = False
        elif isinstance(model_name_or_path, dict) and all(Path(f).exists() for f in model_name_or_path.values()):
            should_load_from_disk = True
            should_load_from_conf = False
            # The EnsembleModelForSequenceClassification requires a dictionary with str keys.
            model_name_or_path = deepcopy({lift_level.value: f for lift_level, f in model_name_or_path.items()})
        else:
            raise ValueError(
                "A value for model_name_or_path was provided, but its not in the list of MODEL_NAMES or a valid path. "
                f"For the record, {model_name_or_path=} and MODEL_NAMES={'|'.join(MODEL_NAMES)}."
            )
    else:
        should_load_from_disk = False
        should_load_from_conf = True

    if should_load_from_conf is None or should_load_from_disk is None:
        raise RuntimeError(f"Invalid state: {should_load_from_disk=} {should_load_from_conf=}")

    # Get model from disk.
    if should_load_from_disk:
        print("Getting model from disk.")
        model_name = object_to_model_name(model_name_or_path)
        # This is an extensive procedure to ensure the weights of the classification head are
        # initialized correctly. Previously, loading the model from disk was performed like this:
        # >>> ModelNameForSequenceClassification.from_pretrained(model_name_or_path, **kwds)
        # Namely, the classification config was not used and the config found in the checkpoint path
        # (e.g., the config for language modeling) was used instead. This resulted in the weights
        # not being initialized correctly (they were set with massive floating point values). It
        # appears that loading the checkpoint like this:
        # >>> ModelNameForSequenceClassification.from_pretrained(model_name_or_path, config=config),
        # where `config` is a config for classification, prevents this from happening.
        if task in (Task.DET, Task.FAM, Task.BEH):
            if model_name == "hrrformer":
                if ensemble:
                    model = HRREnsembleForSequenceClassification.from_pretrained(model_name_or_path, config=config)
                else:
                    model = HRRForSequenceClassification.from_pretrained(model_name_or_path, config=config)
                if isinstance(model_name_or_path, str):
                    _config = HRRConfig.from_pretrained(model_name_or_path)
                else:
                    _config = HRRConfig.from_pretrained(next(iter(model_name_or_path.values())))
                _head_names = ["head_clf"]
            elif model_name == "rwkv":
                if ensemble:
                    raise NotImplementedError()
                model = RwkvForSequenceClassification.from_pretrained(model_name_or_path, config=config)
                _config = RwkvConfig.from_pretrained(model_name_or_path)
                _head_names = ["head_clf"]
            elif model_name == "mamba":
                if ensemble:
                    model = MambaEnsembleForSequenceClassification.from_pretrained(model_name_or_path, config=config)
                    if not config.is_decoder and config.bi_tie_directions:
                        for backbone in [model.raw_backbone, model.dis_backbone, model.dec_backbone]:
                            backbone.tie_forward_and_backward_weights(tie=True, clone=False)
                    _config = MambaConfig.from_pretrained(next(iter(model_name_or_path.values())))
                else:
                    model = MambaForSequenceClassification.from_pretrained(model_name_or_path, config=config)
                    if not config.is_decoder and config.bi_tie_directions:
                        model.backbone.tie_forward_and_backward_weights(tie=True, clone=False)
                    _config = MambaConfig.from_pretrained(model_name_or_path)
                _head_names = ["head_clf"]
            elif model_name == "malconv":
                if ensemble:
                    raise NotImplementedError()
                model = MalConvForSequenceClassification.from_pretrained(model_name_or_path, config=config)
                _config = MalConvConfig.from_pretrained(model_name_or_path)
                _head_names = ["head_clf"]
            elif model_name == "malconv2":
                if ensemble:
                    model = MalConv2EnsembleForSequenceClassification.from_pretrained(model_name_or_path, config=config)
                    _config = MalConv2Config.from_pretrained(next(iter(model_name_or_path.values())))
                else:
                    model = MalConv2ForSequenceClassification.from_pretrained(model_name_or_path, config=config)
                    _config = MalConv2Config.from_pretrained(model_name_or_path)
                _head_names = ["head_clf"]
            elif get_model_type(model_name_or_path) == "HF":
                if ensemble:
                    raise NotImplementedError()
                model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=config)
                _config = AutoConfig.from_pretrained(model_name_or_path)
                _head_names = []
                raise NotImplementedError(f"The head name for {model_name=} is unknown and cannot be checked.")
            else:
                raise ValueError(f"Invalid model name: {model_name}")

            # If we're loading a non-classification model for a classification task, we may need
            # to initialize the head weights. This is a bizarre bug that I can't figure out the origin
            # of, but whatever. When loading the config of a language model, the config automatically
            # get populated with some placeholder id2label values and stuff. We can use some crude
            # heuristics to determine if we're loading a classification model from a pretrained LM.
            # Obviously, we don't need to check a classification head for having unusual weights as
            # these have presumably been updated during training.
            _config: PretrainedConfig
            is_not_for_classification = all([
                _config.num_labels == 2,
                _config.label2id == {'LABEL_0': 0, 'LABEL_1': 1},
                _config.id2label=={0: 'LABEL_0', 1: 'LABEL_1'},
                not any("ForClassification" in a for a in _config.architectures),
            ])
            if is_not_for_classification:
                # for h in _head_names:
                #     check_for_anomalous_weights(getattr_recursively(model, h), errors="warn")

                # Wow. I am so done with this stupid bug. Let's not even bother dealing with it.
                # In all fairness, its literally not even consistent. It seems to randomly pop up.
                # We'll just manually seed the head weights every time we load something from disk.
                # This is only going to work if the model is using my custom classification head,
                # so using anything without this head is going to break (for now).
                # Since we're hard-coding this, we can comment out the warnings above.
                print("Initializing head weights...")
                model.head_clf.init_weights_(config.initializer_range)
                check_for_anomalous_weights(model.head_clf, errors="raise", std_tolerance=2 * config.initializer_range)

            return model

        if task == Task.MLM:
            if model_name == "hrrformer":
                return HRRForMaskedLM.from_pretrained(model_name_or_path, **kwds)
            if model_name == "mamba":
                return MambaForMaskedLM.from_pretrained(model_name_or_path, **kwds)
            return AutoModelForMaskedLM.from_pretrained(model_name_or_path, **kwds)

        if task == Task.CLM:
            if model_name == "hrrformer":
                return HRRForCausalLM.from_pretrained(model_name_or_path, **kwds)
            if model_name == "mamba":
                return MambaForCausalLM.from_pretrained(model_name_or_path, **kwds)
            return AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwds)

    # Get model from config.
    elif should_load_from_conf:
        print("Creating new model.")
        if task in (Task.DET, Task.FAM, Task.BEH):
            if isinstance(config, MalConvConfig):
                if ensemble:
                    raise NotImplementedError()
                return MalConvForSequenceClassification(config)
            if isinstance(config, MalConv2Config):
                if ensemble:
                    return MalConv2EnsembleForSequenceClassification(config)
                return MalConv2ForSequenceClassification(config)
            if isinstance(config, HRRConfig):
                if ensemble:
                    return HRREnsembleForSequenceClassification(config)
                return HRRForSequenceClassification(config)
            if isinstance(config, RwkvConfig):
                if ensemble:
                    raise NotImplementedError()
                return RwkvForSequenceClassification(config)
            if isinstance(config, MambaConfig):
                if ensemble:
                    return MambaEnsembleForSequenceClassification(config)
                return MambaForSequenceClassification(config)
            if isinstance(config, PretrainedConfig):
                if ensemble:
                    raise NotImplementedError()
                return AutoModelForSequenceClassification.from_config(config)
        if task == Task.MLM:
            if ensemble:
                raise NotImplementedError()
            if isinstance(config, HRRConfig):
                return HRRForMaskedLM(config)
            if isinstance(config, RwkvConfig):
                raise NotImplementedError()
            if isinstance(config, MambaConfig):
                return MambaForMaskedLM(config)
            if isinstance(config, PretrainedConfig):
                return AutoModelForMaskedLM.from_config(config)
        if task == Task.CLM:
            if ensemble:
                raise NotImplementedError()
            if isinstance(config, HRRConfig):
                return HRRForCausalLM(config)
            if isinstance(config, RwkvConfig):
                raise NotImplementedError()
            if isinstance(config, MambaConfig):
                return MambaForCausalLM(config)
            if isinstance(config, PretrainedConfig):
                return AutoModelForCausalLM.from_config(config)

    raise RuntimeError()


def get_map_kwds_for_hf_datasets(
    function: Callable,
    dataset: Dataset | DatasetDict | IterableDataset | IterableDatasetDict,
    **kwds,
) -> dict[str, Any]:
    either_style_kwds = {
        "batch_size": None,
    }
    map_style_kwds = {
        "keep_in_memory": KEEP_IN_MEMORY,
        "num_proc": NUM_PROC,
        "cache_file_names": CACHE_FILE_NAME,
        "writer_batch_size": WRITER_BATCH_SIZE,
        "features": None,  # IterableDatasetDict does not take features in its map(), but IterableDataset does.
        "desc": None,
    }
    iterable_style_kwds = {

    }

    kwds = either_style_kwds | map_style_kwds | iterable_style_kwds | kwds
    if isinstance(dataset, (Dataset, DatasetDict)):
        for k in iterable_style_kwds:
            kwds.pop(k)
        kwds["batch_size"] = BATCH_SIZE_MAP
    if isinstance(dataset, (IterableDataset, IterableDatasetDict)):
        for k in map_style_kwds:
            kwds.pop(k)
        kwds["batch_size"] = BATCH_SIZE_ITR

    kwds.update({
        "function": function,
        "batched": True,
    })

    return kwds


def get_processed_dataset_hf(
    materials: Materials,
    lift_level: LiftLevel,
    args: Args,
    num_shards: Optional[int] = None,
    tokenizer: Optional[PreTrainedTokenizerFast] = None,
    remove_columns: Optional[tuple[str]] = ("name", "bytes"),
) -> DatasetDict | IterableDatasetDict:

    dataset = get_dataset_hf(
        materials,
        args.streaming,
        num_shards,
        max_length=args.data_read_bytes,
    )

    # TODO: remove after verifying that the labels are correct for DET and BEH
    # from itertools import chain
    # truth = {}
    # with open("./tmp/avclass_family_cache.txt") as fp:
    #     for line in fp:
    #         parts = line.strip().split()
    #         sha = parts[0]
    #         label = " ".join(parts[1:])
    #         truth[sha] = label

    # for archived_file, label in zip(chain(materials.files["tr"], materials.files["vl"]), chain(materials.labels["tr"], materials.labels["vl"])):
    #     sha = archived_file.name.split(".")[0]
    #     label = materials.id2label[label]
    #     if truth[sha] != label:
    #         print(f"Error (materials): {sha} {label} != {truth[sha]}")

    # for d in chain(dataset["tr"], dataset["vl"]):
    #     sha = d["name"]
    #     label = materials.id2label[d["labels"]]
    #     if truth[sha] != label:
    #         print(f"Error (dataset): {sha} {label} != {truth[sha]}")

    if materials.problem_type == "multi_label_classification":
        func = partial(hf_multilabel_encode, num_classes=materials.num_classes)
        # The one-hot encoding requires floating point one-hot labels.
        features = datasets.Features({
            "labels": datasets.Sequence(datasets.Value("float32")),
            "name": datasets.Value("string"),
            "bytes": datasets.Value("binary"),
        })
        desc = "One-hot encoding labels..."
        kwds = get_map_kwds_for_hf_datasets(func, dataset, features=features, desc=desc)
        dataset = dataset.map(**kwds)

    if args.compression_algorithm is not None:
        func = partial(hf_compress_bytes, compression_type=args.compression_algorithm, compression_level=args.compression_level)
        desc = "Compressing bytes..."
        kwds = get_map_kwds_for_hf_datasets(func, dataset, desc=desc)
        dataset = dataset.map(**kwds)
    if args.encryption_algorithm is not None:
        func = partial(hf_encrypt_bytes, encryption_type=args.encryption_algorithm, key=None)
        desc = "Encrypting bytes..."
        kwds = get_map_kwds_for_hf_datasets(func, dataset, desc=desc)
        dataset = dataset.map(**kwds)

    # If we're mapping bytes to integers, we can do it in a more efficient way.
    # Otherwise, we need to use the tokenizer.
    if lift_level in (LiftLevel.RAW, LiftLevel.NOP) and args.tokenization_algorithm == TokenizationAlgorithm.WORDLEVEL:
        func = partial(
            hf_bytes_to_input_ids,
            bits_in_byte=args.bits_in_byte,
            num_special_ids=len(tokenizer.all_special_ids),
            max_length=args.max_length,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        desc = "Mapping bytes..."
        kwds = get_map_kwds_for_hf_datasets(func, dataset, desc=desc, remove_columns=remove_columns, num_proc=NUM_PROC)
        dataset = dataset.map(**kwds)
    else:
        # The RAW level contains raw-bytes whereas DIS and DEC contains encoded strings.
        hf_tokenize = hf_tokenize_bytes if lift_level in (LiftLevel.RAW, LiftLevel.NOP) else hf_tokenize_str
        func = partial(hf_tokenize, tokenizer=tokenizer, max_length=args.max_length)
        desc = "Tokenizing bytes..."
        kwds = get_map_kwds_for_hf_datasets(function=func, dataset=dataset, desc=desc, remove_columns=remove_columns, num_proc=None)
        dataset = dataset.map(**kwds)

    return dataset


def get_processed_dataset_pt(
    materials: Materials,
    args: Args,
    num_shards: Optional[int] = None,  # pylint: disable=unused-argument
    tokenizer: Optional[PreTrainedTokenizerFast] = None,
):

    if not isinstance(materials.files["tr"][0], (str, Path)):
        raise NotImplementedError()

    if materials.problem_type == "multi_label_classification":
        raise NotImplementedError()

    preprocess_fns = []

    if args.compression_algorithm is not None:
        func = partial(compress, compression_type=args.compression_algorithm, compression_level=args.compression_level)
        preprocess_fns.append(func)
    if args.encryption_algorithm is not None:
        func = partial(encrypt, encryption_type=args.encryption_algorithm, key=None)
        preprocess_fns.append(func)
    if args.lift_level == LiftLevel.RAW and args.tokenization_algorithm == TokenizationAlgorithm.WORDLEVEL:
        func = partial(
            bytes_to_input_ids,
            bits_in_byte=args.bits_in_byte,
            num_special_ids=len(tokenizer.all_special_ids),
            max_length=args.max_length,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        preprocess_fns.append(func)
    else:
        func = partial(tokenize_bytes, tokenizer=tokenizer, max_length=args.max_length)
        preprocess_fns.append(func)

    dataset = get_dataset_pt(
        materials,
        args.streaming,
        max_length=args.data_read_bytes,
        preprocess_fn=compose_functions(*preprocess_fns),
    )
    return dataset


def compute_unigram_probabilities(
    dataset: Dataset | IterableDataset,
    tokenizer: PreTrainedTokenizerFast,
    num_samples: Optional[int] = None,
    batch_size: int = 1024,
) -> dict[str, float]:
    counts = {i: 0 for i in range(0, len(tokenizer))}
    total = 0
    num_iterations = None if num_samples is None else num_samples // batch_size + 1
    for data in tqdm(dataset.iter(batch_size), desc="Computing unigram probabilities...", total=num_iterations):
        ids = np.concatenate(data["input_ids"])
        val, cnt = np.unique(ids, return_counts=True)
        for v, c in zip(val, cnt):
            counts[int(v)] += c
        total += int(np.sum(cnt))

    # Remove the special tokens from the counts and remove them from the total.
    for i in tokenizer.all_special_ids:
        total -= counts.pop(i)

    probs = {i: v / total for i, v in counts.items()}
    return {tokenizer.convert_ids_to_tokens(i): p for i, p in probs.items()}


def main(args: Args, training_arguments: TrainingArguments) -> None:

    random.seed(training_arguments.seed)
    np.random.seed(training_arguments.seed)
    torch.random.manual_seed(training_arguments.seed)

    # FIXME: come up with a better solution for this problem!
    if args.task == Task.DET and args.streaming and os.environ.get("SORT_WHEN_MAKING_ARCHIVES_CONTIGUOUS", "1") == "1":
        raise RuntimeError(
            "To synchronize the materials across lift_levels, we implemented sorting in the archive reader. "
            "Unfortunately, since the benign and malicious samples are obviously not uniformly distributed within the archives, "
            "this results in a dataset whose malware and benigware is not evenly distributed among batches. "
            "When not streaming, the dataset is shuffled before use, so this doesn't matter. "
            "When streaming, the original file-order is used, and the approximate shuffling technique of IterableDataset is not sufficient. "
            "If you want to use streaming for the detection task, you need to set `SORT_WHEN_MAKING_ARCHIVES_CONTIGUOUS=0`."
        )
    if args.task == Task.DET and args.streaming and os.environ.get("SORT_WHEN_MAKING_ARCHIVES_CONTIGUOUS", "1") == "0":
        warnings.warn(
            "You've set `SORT_WHEN_MAKING_ARCHIVES_CONTIGUOUS=0`. This will shuffle the archives prior to reading data. "
            "There will still be clumpiness in the data distribution proporitional to the number of files within each archive. "
            "By itself, this could be problematic, but there is also IterableDataset approximate shuffling which uses a pool. "
        )

        # Example of iterated label distribution prior to IterableDataset's pooling technique:
        # 1 7
        # 0 6
        # 1 32
        # 0 35
        # 1 108
        # 0 9
        # 1 55
        # 0 12
        # 1 40
        # 0 31
        # 1 23
        # 0 11
        # 1 1
        # 0 47

    print(f"args={pformat(args)}")
    print(BR, flush=True)

    print(f"training_arguments={pformat(training_arguments)}")
    print(BR, flush=True)

    MODEL_NAME = object_to_model_name(args.model_name_or_path)
    USING_EPOCHS = training_arguments.max_steps is None or training_arguments.max_steps <= 0

    if training_arguments.world_size > 1 and MODEL_NAME == "mamba":
        if int(torch.__version__.split(".")[0]) < 2 or int(torch.__version__.split(".")[1]) < 2:
            raise RuntimeError("Distributed mamba requires torch>=2.2.0")

    kwds = {
        "root": args.root,
        "packing_protocol": args.packing_protocol,
        "bits_in_byte": args.bits_in_byte,
        "lift_level": args.lift_level,
        "lift_level_ddp": args.lift_level_ddp,
        "tokenization_algorithm": args.tokenization_algorithm,
        "vocab_size": args.vocab_size,
        "max_length": args.max_length,
        "model_name_or_path": args.model_name_or_path,
        "task": args.task,
        "arch_config": args.arch_config,
        "split_mode": args.split_mode,
        "weighted_loss": args.weighted_loss,
        "trainer_config": training_arguments.__dict__ | {"world_size": training_arguments.world_size},
    }
    # NOTE: this is quite bad, but args.model_name_or_path might change from a str representing the
    # model's name, e.g., "mamba", to a path to the model's checkpoint, e.g., "output/mamba/checkpoint-1234",
    # or a dictionary of paths to the model's checkpoints, e.g., {"raw": "output/mamba/checkpoint-1234", ...}.
    if args.pretraining_task is not None and not Path(args.model_name_or_path).exists():
        pretrain_kwds = deepcopy(kwds)
        # It would be nice to remove non-critical pretraining kwds, but its just going to be too difficult.
        # ARCH_KEYS_FROM_CHECKPOINT_TO_IGNORE_FOR_FINETUNING = (
        #     "hidden_dropout_prob",
        #     "attention_probs_dropout_prob",
        #     "head_dropout",
        #     "head_hidden_size",
        #     "head_num_hidden_layers"
        # )
        # for k in ARCH_KEYS_FROM_CHECKPOINT_TO_IGNORE_FOR_FINETUNING:
        #     if k in pretrain_kwds["arch_config"]:
        #         pretrain_kwds["arch_config"].pop(k)
        #         print(f"Removed {k} from the architecture configuration to look for a pretrained model.")
        args.model_name_or_path = OutputHelper.get_finetuning_model_name_or_path(
            args.pretraining_task, args.pretraining_checkpoint, **pretrain_kwds,
        )
        # NOTE: we're going to put the finetuned ensembles beneath the RAW models.
        # Should we add symlinks to the output path beneath the DIS and DEC models?
        # This might be useful for documentation, but could prove annoying when trying to navigate the directory.
        if isinstance(args.model_name_or_path, dict):
            kwds["model_name_or_path"] = args.model_name_or_path[LiftLevel.RAW]
    oh = OutputHelper(**kwds)
    print(f"Output Helper:\n{str(oh)}")
    print(f"Output Path: {oh.path}")
    print(BR)

    # The logits for the CLM/MLM tasks have shape N x T x V, so we need to either
    # skip the evaluate entirely, or metrics need to computed every batch.
    if args.task in (Task.CLM, Task.MLM):
        # training_arguments = replace(training_arguments, prediction_loss_only=True)
        if not (training_arguments.prediction_loss_only or training_arguments.batch_eval_metrics):
            warnings.warn(
                "Cannot have `prediction_loss_only` and `batch_eval_metrics` both set to False for CLM/MLM task. "
                "Setting `batch_eval_metrics` to True to avoid CUDA OOM errors."
            )
            training_arguments = replace(training_arguments, batch_eval_metrics=True)

    tokenizer: PreTrainedTokenizerFast
    multitokenizer: dict[LiftLevel, PreTrainedTokenizerFast]
    if args.lift_level == LiftLevel.ALL:
        multitokenizer = {
            lift_level: get_fast_tokenizer(
                lift_level=lift_level,
                algorithm=args.tokenization_algorithm,
                bits_in_byte=args.bits_in_byte,
                vocab_size=args.vocab_size,
                model_max_length=args.max_length,
                add_cls_token=False,
                add_bos_token=True,
                add_eos_token=True,
                add_sep_token=False,
            )
            for lift_level in (LiftLevel.RAW, LiftLevel.DIS, LiftLevel.DEC)
        }
        for t in multitokenizer.values():
            t.model_input_names = ["input_ids"]
        tokenizer = multitokenizer[LiftLevel.RAW]
    else:
        tokenizer = get_fast_tokenizer(
            lift_level=args.lift_level,
            algorithm=args.tokenization_algorithm,
            bits_in_byte=args.bits_in_byte,
            vocab_size=args.vocab_size,
            model_max_length=args.max_length,
            add_cls_token=False,
            add_bos_token=True,
            add_eos_token=True,
            add_sep_token=False,
        )
        tokenizer.model_input_names = ["input_ids"]
        multitokenizer = {LiftLevel.RAW: tokenizer}

    # TODO: should we add masks after mapping or before?
    if MODEL_NAME in REQ_ATTENTION_MASK:
        for t in multitokenizer.values():
            t.model_input_names.append("attention_mask")
    if MODEL_NAME in REQ_TOKEN_TYPE_IDS:
        for t in multitokenizer.values():
            t.model_input_names.append("token_type_ids")

    print_tokenizer(tokenizer)
    print(BR, flush=True)

    # Get the raw materials for the dataset, i.e., the files, labels, etc.
    if args.task == Task.CLM:
        get_materials = get_materials_esp_clm
    elif args.task == Task.MLM:
        get_materials = get_materials_esp_mlm
    elif args.task == Task.DET:
        get_materials = get_materials_esp_det
    elif args.task == Task.FAM:
        get_materials = get_materials_esp_fam
    elif args.task == Task.BEH:
        get_materials = get_materials_esp_beh

    materials: Materials
    multimaterials: dict[LiftLevel, Materials]
    if args.lift_level == LiftLevel.ALL:
        multimaterials = {
            lift_level: get_materials(lift_level, lift_level_ddp=args.lift_level_ddp)
            for lift_level in (LiftLevel.RAW, LiftLevel.DIS, LiftLevel.DEC)
        }
        materials = multimaterials[LiftLevel.RAW]
    else:
        materials = get_materials(args.lift_level, lift_level_ddp=args.lift_level_ddp)
        multimaterials = {args.lift_level: materials}

    print(f"Dataset Multimaterials:\n{list(multimaterials.keys())}")
    print(f"Dataset Materials:\n{materials}")
    print(BR, flush=True)

    # If we know the length of the dataset, we can compute the number of steps from training epochs.
    # This lets us use epochs for training in streaming mode and eval/save multiple times per epoch.
    kwds = {}
    max_steps, save_steps, eval_steps = training_arguments.epochs_to_steps(len(materials.files["tr"]))
    if args.streaming and USING_EPOCHS:
        kwds.update({"max_steps": max_steps})
    if training_arguments.saves_per_epoch is not None:
        kwds.update({"save_steps": save_steps, "save_strategy": "steps"})
    if training_arguments.evals_per_epoch is not None:
        kwds.update({"eval_steps": eval_steps, "eval_strategy": "steps"})
    training_arguments = replace(training_arguments, **kwds)


    if args.dataset_backend == "HF":
        num_shards = max(training_arguments.world_size, 1) * max(training_arguments.dataloader_num_workers, 1)
        if args.lift_level == LiftLevel.ALL:
            # On armitage, we don't have all the files, so we need to sync the data across representations.
            if SYSTEM == System.ARMITAGE:
                for split in ["tr", "vl"]:
                    intersection = None
                    for m in multimaterials.values():
                        names = set(af.name.split(".")[0] for af in m.files[split])
                        intersection = names if intersection is None else intersection & names
                    for m in multimaterials.values():
                        remove = [i for i, af in enumerate(m.files[split]) if af.name.split(".")[0] not in intersection]
                        m.files[split] = [af for i, af in enumerate(m.files[split]) if i not in remove]
                        m.labels[split] = [l for i, l in enumerate(m.labels[split]) if i not in remove]

            multidataset = {
                lift_level: get_processed_dataset_hf(
                    multimaterials[lift_level],
                    lift_level,
                    args,
                    num_shards,
                    multitokenizer[lift_level],
                    remove_columns=("bytes",),
                )
                for lift_level in (LiftLevel.RAW, LiftLevel.DIS, LiftLevel.DEC)
            }

            dataset = merge_raw_dis_dec_datasets(
                multidataset[LiftLevel.RAW],
                multidataset[LiftLevel.DIS],
                multidataset[LiftLevel.DEC],
            )
            del multidataset
        else:
            dataset = get_processed_dataset_hf(materials, args.lift_level, args, num_shards, tokenizer)
        if os.environ.get("TR_SIZE") is not None:
            dataset["tr"] = dataset["tr"].take(int(os.environ["TR_SIZE"]))
        if os.environ.get("VL_SIZE") is not None:
            dataset["vl"] = dataset["vl"].take(int(os.environ["VL_SIZE"]))
        print_dataset_hf(dataset)
        print(BR)
    else:
        dataset = get_processed_dataset_pt(materials, args, None, tokenizer)
        print_dataset_pt(dataset)
        print(BR)

    # TODO: should we add masks after mapping or before?
    # if MODEL_NAME in REQ_ATTENTION_MASK:
    #     tokenizer.model_input_names.append("attention_mask")
    # if MODEL_NAME in REQ_TOKEN_TYPE_IDS:
    #     tokenizer.model_input_names.append("token_type_ids")

    if args.exit_after_map:
        print("Exiting after map.")
        sys.exit(0)

    if args.do_compute_unigram_probabilities:
        print("Computing unigram probabilities.")
        unigrams = compute_unigram_probabilities(dataset["tr"], tokenizer, len(materials.files["tr"]), 4096)
        print(f"Unigrams:\n{pformat(unigrams)}")
        save_unigrams(unigrams, args.lift_level, args.tokenization_algorithm, args.bits_in_byte, args.vocab_size)
        print("Exiting after unigram computation.")
        sys.exit(0)


    config = get_config(
        args.model_name_or_path,
        tokenizer,
        args.max_length,
        arch_config=args.arch_config,
        num_labels=materials.num_classes,
        id2label=materials.id2label,
        label2id=materials.label2id,
        problem_type=materials.problem_type,
    )
    print_config(config)
    print(BR, flush=True)

    pad_to_multiple_of = PAD_TO
    if isinstance(config, transformers.ReformerConfig):
        pad_to_multiple_of = _get_least_common_mult_chunk_len(config)

    # TODO: should we add masks after mapping or before?
    # Change the tokenizer's attributes for the data_collator to use correctly.
    # This let's us use the previously generated cache files then drop the
    # attention_mask before passing the inputs to the model.
    # if MODEL_NAME not in REQ_ATTENTION_MASK:
    #     if "attention_mask" in tokenizer.model_input_names:
    #         tokenizer.model_input_names.remove("attention_mask")

    if args.task == Task.CLM:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=pad_to_multiple_of,
        )
    elif args.task == Task.MLM:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=0.25,
            pad_to_multiple_of=pad_to_multiple_of,
        )
    elif args.lift_level == LiftLevel.ALL:
        data_collator = EnsembleDataCollatorWithPadding(
            raw_tokenizer=multitokenizer[LiftLevel.RAW],
            dis_tokenizer=multitokenizer[LiftLevel.DIS],
            dec_tokenizer=multitokenizer[LiftLevel.DEC],
            padding="longest",
            pad_to_multiple_of=pad_to_multiple_of,
            max_length=args.max_length,
        )
    else:
        data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding="longest",
            pad_to_multiple_of=pad_to_multiple_of,
            max_length=args.max_length,
        )
    print_data_collator(data_collator)
    print(BR)

    unigrams = None
    if args.task in (Task.CLM, Task.MLM):
        # Need an array that matches the orientation of the output layer for MLM tasks.
        # NOTE: This sets the unigram probabilities of the special tokens to NaN.
        unigrams_map = load_unigrams(args.lift_level, args.tokenization_algorithm, args.bits_in_byte, args.vocab_size)
        unigrams_map = {tokenizer.convert_tokens_to_ids(k): v for k, v in unigrams_map.items()}
        unigrams = []
        for i in range(len(tokenizer)):
            if i in unigrams_map:
                unigrams.append(unigrams_map[i])
            elif i in tokenizer.all_special_ids:
                unigrams.append(float("nan"))
            else:
                unigrams.append(0.0)
        unigrams = np.array(unigrams, dtype=np.float32)

    # TODO: maybe we should just be doing the really fast perplexity normalization during training?
    # Then we wouldn't need the logits during evaluation at all and could theoretically compute
    # the evaluation on the entire eval dataset at once instead of in batches.
    if args.task == Task.DET:
        compute_metrics = CLFComputeMetricsBinary()
    if args.task == Task.FAM:
        compute_metrics = CLFComputeMetricsSingleLabel()
    if args.task == Task.BEH:
        compute_metrics = CLFComputeMetricsMultiLabel()
    if args.task == Task.CLM:
        compute_metrics = CLMComputeMetrics(unigrams, True, False, False, tokenizer.all_special_ids + [-100], True)
    if args.task == Task.MLM:
        compute_metrics = MLMComputeMetrics(unigrams, True, False, False, tokenizer.all_special_ids + [-100], True)

    training_arguments = replace(training_arguments, include_for_metrics=compute_metrics.include_for_metrics)

    print(f"{compute_metrics=}")
    print(BR)

    # We need to add some special callbacks if we're implementing new evaluation frequencies.
    callbacks = []
    if training_arguments.save_strategy == IntervalStrategy.STEPS or training_arguments.eval_strategy == IntervalStrategy.STEPS:  # pylint: disable=consider-using-in
        callbacks.append(RobustEpochCallback())
    if args.early_stopping:
        callbacks.append(EarlyStoppingCallback(args.early_stopping_patience, args.early_stopping_threshold))
    print(f"{callbacks=}")
    print(BR)

    compute_loss_func = None
    if args.weighted_loss is not None:
        if materials.problem_type not in ("single_label_classification", "multi_label_classification"):
            raise NotImplementedError(f"Weighted loss not implemented for {materials.problem_type=}")

        if args.weighted_loss == WeightedLossAlgorithm.FOCAL_LOSS:
            loss_fn = FocalLoss()
            compute_loss_func = CustomComputeLossFunction(config.num_labels, materials.problem_type, loss_fn)
        else:
            if args.weighted_loss == WeightedLossAlgorithm.SAMPLE_REWEIGHTING:
                weight = sample_reweighting(materials.dist_tr, beta=args.beta)
            elif args.weighted_loss == WeightedLossAlgorithm.INVERSE_CLASS_FREQUENCY:
                weight = inverse_class_frequency(materials.dist_tr)
            weight = tensor([weight[materials.id2label[i]] for i in sorted(materials.id2label.keys())])
            loss_fn = CrossEntropyLoss(weight=weight)
            compute_loss_func = CustomComputeLossFunction(config.num_labels, materials.problem_type, loss_fn)

        if compute_loss_func is None:
            raise RuntimeError(f"compute_loss_func was not assigned: {compute_loss_func=}")

    print(f"{compute_loss_func=}")
    print(BR)

    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
    if not args.streaming:  # dataset has been processed, so we disable thread-based parallelism
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if training_arguments.do_train:
        print(f"training_arguments={pformat(training_arguments)}")
        print(BR, flush=True)

        model = get_model(
            args.task,
            args.model_name_or_path,
            config,
            args.lift_level==LiftLevel.ALL,
            num_labels=materials.num_classes,
            id2label=materials.id2label,
            label2id=materials.label2id,
        )
        print_model(model)
        print(BR, flush=True)

        if len((all_issues := check_model_parameters(model, -0.999, 0.999))) > 0:
            for name, issues in all_issues:
                if "LayerNorm" in name:  # LayerNorm is has parameters initialized with 1s
                    continue
                print(f"Warning: Issue(s) detected in model: {name=} {issues=}")


        # Initial evaluation of the model on the validation set to detect OOM and CudaOOM errors.
        # This will also reduce the eval_batch size in the training_arguments variable.
        @find_executable_batch_size(starting_batch_size=training_arguments.per_device_eval_batch_size)
        def _eval(batch_size: int) -> tuple[dict[str, float], int]:
            nonlocal training_arguments  # access variables outside of this function.
            training_arguments = replace(training_arguments, per_device_eval_batch_size=batch_size)
            print(f"Evaluating with {batch_size=}...", flush=True)
            trainer = Trainer(
                model=model,
                args=training_arguments,
                train_dataset=dataset["tr"],
                eval_dataset=dataset["vl"],
                data_collator=data_collator,
                tokenizer=tokenizer if args.dataset_backend == "HF" else None,  # TODO: only pass in the tokenizer if a tokenization algorithm is required (needs to be tested).
                callbacks=callbacks,
                compute_metrics=compute_metrics,
                compute_loss_func=compute_loss_func,
            )
            return trainer.evaluate(dataset["vl"]), batch_size

        initial_metrics: dict[str, float]  = {}
        max_per_device_eval_batch_size: int = None
        if not args.skip_eval_check:
            print("Initial Evaluation...", flush=True)

            if args.auto_find_batch_size_and_gradient_accumulation_steps:
                initial_metrics, max_per_device_eval_batch_size = _eval()  # pylint: disable=no-value-for-parameter
            else:
                trainer = Trainer(
                    model=model,
                    args=training_arguments,
                    train_dataset=dataset["tr"],
                    eval_dataset=dataset["vl"],
                    data_collator=data_collator,
                    tokenizer=tokenizer if args.dataset_backend == "HF" else None,  # TODO: only pass in the tokenizer if a tokenization algorithm is required (needs to be tested).
                    callbacks=callbacks,
                    compute_metrics=compute_metrics,
                    compute_loss_func=compute_loss_func,
                )
                initial_metrics = trainer.evaluate(dataset["vl"])
                max_per_device_eval_batch_size = training_arguments.per_device_eval_batch_size

            model = model.to(torch.float32).to("cpu")
            clear_cuda_caches()

            if args.sync_batch_size:
                # Set the train batch size to the same size as the eval one and adjust gradient accumulation to keep same logical batch size.
                training_arguments = replace(training_arguments, gradient_accumulation_steps=(
                    training_arguments.per_device_train_batch_size
                    * training_arguments.gradient_accumulation_steps
                    // max_per_device_eval_batch_size)
                )
                training_arguments = replace(training_arguments, per_device_train_batch_size=max_per_device_eval_batch_size)

            print(f"{initial_metrics=}", flush=True)


        @find_executable_batch_size_and_gradient_accumulation_steps(
            starting_batch_size=training_arguments.per_device_train_batch_size,
            starting_gradient_accumulation_steps=training_arguments.gradient_accumulation_steps,
        )
        def _train(batch_size: int, gradient_accumulation_steps: int) -> TrainOutput:
            nonlocal training_arguments, oh
            print(f"Training with {batch_size=} and {gradient_accumulation_steps=}...", flush=True)
            try:  # Try to remove a created, but empty directory from a previous attempt.
                oh.lock_file.unlink(missing_ok=True)
                remove_empty_directories(oh.task_path.as_posix(), missing_ok=True)
            except OSError:
                pass

            # If the OOM error arises during the evaluation loop while training, this could
            # cause unessecary reduction of the training batch size when we really
            # should be decrementing the evaluation batch size. However, after several months
            # of using this code, this never really seemed to happen, so I think we can just ignore it.
            # Anyway, considering the GPU fragmentation issues that randomly arise if we let
            # the train and validation batch sizes differ, I don't think its all that relevant.
            per_device_eval_batch_size = max_per_device_eval_batch_size
            if args.sync_batch_size or max_per_device_eval_batch_size is None:
                per_device_eval_batch_size = batch_size

            training_arguments = replace(
                training_arguments,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=per_device_eval_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
            )
            oh.trainer_config = training_arguments.__dict__ | {
                "world_size": training_arguments.world_size,
                "max_steps": -1 if USING_EPOCHS else training_arguments.max_steps
            }
            oh.mkdir()

            training_arguments = replace(training_arguments, output_dir=oh.checkpoints_dir.as_posix())
            print(f"{training_arguments.per_device_train_batch_size=} {training_arguments.per_device_eval_batch_size=}")
            trainer = Trainer(
                model=model,
                args=training_arguments,
                train_dataset=dataset["tr"],
                eval_dataset=dataset["vl"],
                data_collator=data_collator,
                tokenizer=tokenizer if args.dataset_backend == "HF" else None,  # TODO: only pass in the tokenizer if a tokenization algorithm is required (needs to be tested).
                callbacks=callbacks,
                compute_metrics=compute_metrics,
                compute_loss_func=compute_loss_func,
            )
            try:
                return trainer.train(training_arguments.resume_from_checkpoint)
            except Exception:
                # Cleaning up references to the optimizer states might help memory fragmentation issues.
                if hasattr(trainer, "optimizer"):
                    optimizer_to_(trainer.optimizer, "cpu")
                    trainer.optimizer = None
                trainer = None
                del trainer
                gc.collect()
                raise


        print("Training...", flush=True)
        if args.auto_find_batch_size_and_gradient_accumulation_steps:
            trainer_output: TrainOutput = _train()  # pylint: disable=no-value-for-parameter
        else:
            oh.trainer_config = training_arguments.__dict__ | {
                "world_size": training_arguments.world_size,
                "max_steps": -1 if USING_EPOCHS else training_arguments.max_steps
            }
            oh.mkdir()
            training_arguments = replace(training_arguments, output_dir=oh.checkpoints_dir.as_posix())
            trainer = Trainer(
                model=model,
                args=training_arguments,
                train_dataset=dataset["tr"],
                eval_dataset=dataset["vl"],
                data_collator=data_collator,
                tokenizer=tokenizer if args.dataset_backend == "HF" else None,
                callbacks=callbacks,
                compute_metrics=compute_metrics,
                compute_loss_func=compute_loss_func,
            )
            trainer_output: TrainOutput = trainer.train(training_arguments.resume_from_checkpoint)

        # In case the output path has chnaged, we wait until the very end to save outputs.
        with open(oh.initial_validation_results_file, "w") as fp:
            json.dump(initial_metrics, fp, indent=4)
        with open(oh.trainer_output_file, "w") as fp:
            json.dump(trainer_output.metrics, fp, indent=4)


    if training_arguments.do_eval:

        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        d = Dataset.from_list(list(tqdm(dataset["vl"], desc="Processing validation set...", total=len(materials.files["vl"]))))
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

        for checkpoint in tqdm(args.eval_checkpoints, desc="Evaluating saved models..."):
            model = get_model(
                args.task,
                checkpoint,
                config,
                args.lift_level==LiftLevel.ALL,
                num_labels=materials.num_classes,
                id2label=materials.id2label,
                label2id=materials.label2id,
            )
            output: dict = Trainer(
                model=model,
                args=training_arguments,
                train_dataset=dataset["tr"],
                eval_dataset=dataset["vl"],
                data_collator=data_collator,
                tokenizer=tokenizer,
                callbacks=callbacks,
                compute_metrics=compute_metrics,
                compute_loss_func=compute_loss_func,
            ).evaluate(d)
            print(output)
            outfile = Path(checkpoint) / "vl_metrics.json"
            print(f"Saving metrics to {outfile}...")
            with open(outfile, "w") as fp:
                json.dump(output, fp, indent=4)

        sys.exit(0)

        # Clean up any residual references to model and optimizer from training.
        if training_arguments.do_train:
            model = model.to("cpu")
            model = None
            try:
                optimizer_to_(trainer.optimizer, "cpu")
                trainer.optimizer = None
                trainer = None
            except AttributeError:
                pass
            clear_cuda_caches()

        # Get a fresh model from disk. Here we pass the config although I've forgotten why.
        model = get_model(
            args.task,
            oh.best_model_dir,
            config,
            args.lift_level==LiftLevel.ALL,
            num_labels=materials.num_classes,
            id2label=materials.id2label,
            label2id=materials.label2id,
        )
        print_model(model)
        print(BR, flush=True)

        # Update the compute_metrics ComputeMetrics's settings (optional)
        # compute_metrics.update({})

        split = "vl" if is_dataset_empty(dataset.get("ts")) else "ts"
        print(f"Evaluating {split}...")
        output: PredictionOutput = Trainer(
            model=model,
            args=training_arguments,
            data_collator=data_collator,
            tokenizer=tokenizer,
            callbacks=callbacks,
            compute_metrics=compute_metrics,
            compute_loss_func=compute_loss_func,
        ).predict(dataset[split])
        oh.test_results_dir.mkdir(exist_ok=True, parents=False)
        with open(oh.test_results_file, "w") as fp:
            json.dump(output.metrics, fp, indent=4)


    if args.do_tune:
        training_arguments = replace(
            training_arguments,
            do_eval=True,
            eval_strategy="steps",
            fp16_full_eval=False,  # Due to bug in transformers, this must be False.
            load_best_model_at_end=False,  # Greater flexibility with eval_steps.
            disable_tqdm=torch.cuda.device_count() > 1,  # Unreadable when using multiple GPUs.
        )
        model_init = partial(
            hp_model_init,
            task=args.task,
            model_name_or_path=args.model_name_or_path,
            tokenizer=tokenizer,
            max_length=args.max_length,
            num_labels=materials.num_classes,
            id2label=materials.id2label,
            label2id=materials.label2id,
        )
        trainer = Trainer(
            model_init=model_init,
            args=training_arguments.hf_training_arguments_object(),
            train_dataset=dataset["tr"],
            eval_dataset=dataset["vl"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            callbacks=None,
            compute_metrics=compute_metrics,
            compute_loss_func=compute_loss_func,
        )

        search_alg = HyperOptSearch(
            space=None,
            metric="eval_loss",
            mode="min",
            n_initial_points=N_INITIAL_POINTS,
        )
        scheduler = None  # ASHAScheduler(metric="eval_loss", mode="min")
        print(f"{search_alg=}")
        print(f"{scheduler=}")
        print(f"{TUNE_RESOURCES_PER_TRIAL=}")
        print(f"{RAISE_ON_FAILED_TRIAL=}")
        print(BR, flush=True)

        oh.mkdir()
        oh.tuning_results_dir.mkdir(exist_ok=True, parents=False)
        print("Tuning...")

        if isinstance(config, (LongformerConfig,)):
            hp_space = hp_space_longformer
        elif isinstance(config, (HRRConfig,)):
            hp_space = hp_space_hrrformer
        elif isinstance(config, (MalConvConfig,)):
            hp_space = hp_space_malconv
        else:
            raise ValueError(f"No hyperparameter space defined for this model: {type(config)=}")

        try:
            best_trial: BestRun = trainer.hyperparameter_search(
                hp_space=hp_space,
                compute_objective=hp_compute_objective,
                n_trials=N_TRIALS,
                direction="minimize",
                backend="ray",
                hp_name=None,
                search_alg=search_alg,
                resources_per_trial=TUNE_RESOURCES_PER_TRIAL,
                raise_on_failed_trial=RAISE_ON_FAILED_TRIAL,
            )
        except TuneError as err:
            print(BR, flush=True)
            print(
                f"Encountered a TuneError with {RAISE_ON_FAILED_TRIAL=}."
                "If RAISE_ON_FAILED_TRIAL=True or RAISE_ON_FAILED_TRIAL=None, then this indicates"
                "that at least one trial failed. To avoid this, set RAISE_ON_FAILED_TRIAL=False."
            )
            print(BR, flush=True)
            raise err
        except AttributeError as err:
            print(BR, flush=True)
            print(
                f"Encountered an AttributeError with {RAISE_ON_FAILED_TRIAL=}."
                "If RAISE_ON_FAILED_TRIAL=False, then this most likely indicates"
                "that every trial failed."
            )
            print(BR, flush=True)
            raise err

        analysis: tune.ExperimentAnalysis = best_trial.run_summary
        analysis.dataframe().to_csv(oh.tuning_results_dir / "dataframe.csv")


def cli():
    parser = HfArgumentParser((Args, TrainingArguments))
    # pylint: disable=unbalanced-tuple-unpacking
    args, training_arguments = parser.parse_args_into_dataclasses()
    # pylint: enable=unbalanced-tuple-unpacking
    main(args, training_arguments)
    print(f"ENDING @{datetime.now()}\n{BR}", flush=True)


if __name__ == "__main__":
    print(f"STARTING @{datetime.now()}\n{BR}", flush=True)
    print(f"{torch.backends.cudnn.enabled=}")
    cli()
    print(f"ENDING @{datetime.now()}\n{BR}", flush=True)
