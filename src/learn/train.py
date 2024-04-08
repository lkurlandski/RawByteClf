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

# pylint: disable=wrong-import-position
print(f"Entered {__file__=}")

from collections import Counter
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
from typing import Any, Callable, Literal, Optional
import os
import sys
import warnings

if __name__ == "__main__":
    print(f"STARTING @{datetime.now()}\n{'-' * 88}", flush=True)
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# pylint: enable=wrong-import-position

from datasets import (
    DatasetDict,
    Dataset,
    IterableDataset,
    IterableDatasetDict,
)
import numpy as np
import torch
from torch import tensor, Tensor
from torch.nn import CrossEntropyLoss, Embedding
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
    ReformerConfig,
    NystromformerConfig,
    FNetConfig,
    RwkvPreTrainedModel,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    EarlyStoppingCallback,
)
from transformers import TrainingArguments as HfTrainingArguments
from transformers.trainer_utils import (
    BestRun,
    PredictionOutput,
    TrainOutput,
    PREFIX_CHECKPOINT_DIR,
)
from transformers.models.reformer.modeling_reformer import _get_least_common_mult_chunk_len
from transformers.utils import CONFIG_NAME, is_ninja_available
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune import TuneError

from src.cfg import BR
from src.utils import (
    count_parameters,
    get_highest_path,
    object_from_superset_of_constructor_kwds,
    to_long_tensor,
    compress,
    COMPRESSION_TYPES,
    encrypt,
    ENCRYPTION_TYPES,
    compose_functions,
)
from src.architectures.malconv import (
    AutoMalConvForSequenceClassification,
    MalConvConfig,
    MalConvGCTConfig,
    MyMalConvConfig,
    MalConv,
    MalConvGCT,
    MyMalConv,
)
from src.architectures.hrrformer import (
    HRRConfig,
    HRRForSequenceClassification,
    HRRForMaskedLM,
)
from src.architectures.mamba import (
    MambaConfig,
    MambaForSequenceClassification,
    MambaLMHeadModel as MambaForCausalLM,
    MambaPreTrainedModel,
)
from src.architectures.rwkv import (
    RwkvConfig,
    RwkvForSequenceClassification,
)
from src.data.loaders_core import (
    get_materials_clf_bodmas,
    get_materials_clf_sorel,
    get_materials_clf_bodmas_balanced_slice,
    get_materials_clf_bodmas_with_k_samples_per_class_in_train_set,
    get_materials_pretrain_sorel,
    get_materials_clf_sorel_length_extrapolation,
)
from src.data.loaders_hf import get_dataset_hf, print_dataset_hf
from src.data.loaders_pt import get_dataset_pt, print_dataset_pt, MapBinaryDatasetDict, IterableBinaryDatasetDict
from src.learn.helpers import Args, OutputHelper
from src.learn.evaluation import clf_compute_metrics
from src.learn.preprocessing import (
    hf_bytes_to_input_ids,
    hf_tokenize_bytes,
    bytes_to_input_ids,
    tokenize_bytes,
    hf_compress_bytes,
    hf_encrypt_bytes,
)
from src.learn.tuning import (
    hp_space_mymalconv,
    hp_space_malconv,
    hp_space_malconvgct,
    hp_space_longformer,
    hp_space_hrrformer,
)
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
)
from src.learn.tokenization import get_tokenizer


random.seed(0)
np.random.seed(0)
torch.random.manual_seed(0)


PAD_TO = 8

# Some random temporary flags.
MOVE_IN_MEMORY = False

# Default variables for the datasets.Dataset.map() and datasets.IterableDataset.map()
BATCH_SIZE: Optional[int] = 1000
WRITER_BATCH_SIZE: Optional[int] = 1000
CACHE_FILE_NAME: Optional[str] = None
NUM_PROC: Optional[int] = len(os.sched_getaffinity(0)) if len(os.sched_getaffinity(0)) != 40 else 20
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
    "malconvgct",
    "mymalconv",
    "hrrformer",
    "rwkv",
    "mamba",
]

RETURN_ATTENTION_MASK = [
    "longformer",
    "reformer",
    "nystromformer",
    "hrrformer",
]

REQ_CLS_TOKEN = [
    "longformer",
    "reformer",
    "nystromformer",
    "fnet",
    "hrrformer",
]

REQ_SEP_TOKEN = [
    "longformer",
    "reformer",
    "nystromformer",
    "fnet",
    "hrrformer",
]

REQ_BOS_TOKEN = [
    "rwkv",
    "mamba",
]

REQ_EOS_TOKEN = [
    "rwkv",
    "mamba",
]

MODEL_NAME_TO_CONFIG_CLASS = {
    "longformer": LongformerConfig,
    "reformer": ReformerConfig,
    "nystromformer": NystromformerConfig,
    "fnet": FNetConfig,
    "malconv": MalConvConfig,
    "malconvgct": MalConvGCTConfig,
    "mymalconv": MyMalConvConfig,
    "hrrformer": HRRConfig,
    "rwkv": RwkvConfig,
    "mamba": MambaConfig,
}


class TrainingArguments(HfTrainingArguments):

    def __init__(self, **kwds):
        # If do_eval flag is not passed, as would be the case when passing do_tune, it is set to
        # True by the transformers.TrainingArguments. When we pass do_tune, we do not want the
        # do_eval flag to be set to True, so we adjust the value as needed at the end of __init__.
        do_eval = kwds.get("do_eval", False)

        # If resume_from_checkpoint is passed as a flag without an argument, it is not set to True
        # by the transformers.TrainingArguments. This let's us pass in "true" from the command line,
        # and have the flag be set to True, i.e., resume training from the last checkpoint.
        if kwds.get("resume_from_checkpoint", None):
            try:
                kwds["resume_from_checkpoint"] = str_or_bool_to_str(kwds["resume_from_checkpoint"])
            except ValueError:
                pass

        kwds["metric_for_best_model"] = kwds.pop("metric_for_best_model", "eval_loss")

        super().__init__(**kwds)
        self.do_eval = do_eval

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

    def __init__(self, do_print: bool = False, *args, **kwds) -> None:
        super().__init__(*args, **kwds)
        self.do_print = do_print
        self._time_step_start = -1.0
        self._time_step_end = -1.0
        self._time_step_deltas: list[float] = []

    def on_log(self, args: HfTrainingArguments, state: TrainerState, control: TrainerControl, **kwds) -> None:
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

    def on_step_begin(self, args: HfTrainingArguments, state: TrainerState, control: TrainerControl, **kwds) -> None:
        self._time_step_start = time.time()

    def on_step_end(self, args: HfTrainingArguments, state: TrainerState, control: TrainerControl, **kwds) -> None:
        self._time_step_end = time.time()
        self._time_step_deltas.append(self._time_step_end - self._time_step_start)


class ImbalancedClassificationTrainer(Trainer):
    def __init__(self, weight: Optional[Tensor] = None, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = CrossEntropyLoss(weight=weight)
        self.num_labels = self.model.config.num_labels

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.label_smoother is not None or self.args.past_index >= 0:
            raise NotImplementedError()

        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits

        device = logits.device
        if self.loss_fn.weight is not None and self.loss_fn.weight.device != device:
            self.loss_fn.weight = self.loss_fn.weight.to(device)

        # num_labels = unwrap_model(model).config.num_labels
        loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def hp_model_init(
    trial: Optional[dict[str, Any]],
    task: str,
    model_name_or_path: str,
    tokenizer: Optional[PreTrainedTokenizerFast] = None,
    max_length: Optional[int] = None,
    num_labels: Optional[int] = None,
    id2label: Optional[dict[int, str]] = None,
    label2id: Optional[dict[str, int]] = None,
) -> PreTrainedModel | MalConv | MalConvGCT:
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
        tensor_log_path=None,
        arch_config=None,
        **(hparams | kwds),
    )
    model = get_model(task, None, config, **kwds)
    if model is None:
        raise RuntimeError("Model is None for some reason.")
    return model


def hp_compute_objective(metrics: dict[str, float]) -> float:
    return metrics["eval_loss"]


# TODO: add support for passing in a PreTrainedModel object.
def object_to_model_name(obj: PretrainedConfig | str | Path) -> str:
    if obj in MODEL_NAMES:
        return obj
    if isinstance(obj, (FNetConfig,)):
        return "fnet"
    if isinstance(obj, (NystromformerConfig,)):
        return "nystromformer"
    if isinstance(obj, (ReformerConfig,)):
        return "reformer"
    if isinstance(obj, (LongformerConfig,)):
        return "longformer"
    if isinstance(obj, (HRRConfig,)):
        return "hrrformer"
    if isinstance(obj, (RwkvConfig,)):
        return "rwkv"
    if isinstance(obj, (MambaConfig,)):
        return "mamba"
    if isinstance(obj, (MalConvConfig,)):
        return "malconv"
    if isinstance(obj, (MalConvGCTConfig,)):
        return "malconvgct"
    if isinstance(obj, (MyMalConvConfig,)):
        return "mymalconv"
    if isinstance(obj, (str, Path)) and Path(obj).exists():
        try:
            return object_to_model_name(AutoConfig.from_pretrained(str(obj)))
        except ValueError:
            pass

    possible_model_names = []
    for model_name in MODEL_NAMES:
        if model_name in str(obj).lower():
            possible_model_names.append(model_name)
    if len(possible_model_names) == 1:
        return possible_model_names[0]
    if len(possible_model_names) > 1:
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
        MalConv,
        MalConvGCT,
        MyMalConv,
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
    model_name_or_path: str,
    tokenizer: Optional[PreTrainedTokenizerFast] = None,
    max_length: Optional[int] = None,
    tensor_log_path: Optional[Path] = None,
    arch_config: Optional[dict[str, Any]] = None,
    **kwds,
) -> PretrainedConfig:
    """
    Get the configuration for the model.

    Precedence (highest to lowest):
        - config from the tokenizer, max_length, and tensor_log_path args
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
    for k in ["num_labels", "id2label", "label2id"]:
        if k in kwds and kwds[k] is None:
            kwds.pop(k)

    if Path(model_name_or_path).exists():
        return get_config_from_path(model_name_or_path, **kwds)

    # Handle float values when hyperparameter tuning.
    # TODO: this probably got broken at some point.
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
            tensor_log_path=tensor_log_path,
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
            out_size=kwds["num_labels"],
            pad_idx=tokenizer.pad_token_id,
            num_embd=vocab_size,
        )
        return MalConvConfig(**kwds)

    if model_name_or_path.lower() == "malconvgct":
        kwds = kwds | dict(
            out_size=kwds["num_labels"],
            pad_idx=tokenizer.pad_token_id,
            num_embd=vocab_size,
        )
        return MalConvGCTConfig(**kwds)

    if model_name_or_path.lower() == "mymalconv":
        kwds = kwds | dict(
            out_size=kwds["num_labels"],
            pad_idx=tokenizer.pad_token_id,
            num_embd=vocab_size,
            max_length=max_posititional_embeddings,
        )
        return MyMalConvConfig(**kwds)

    # pylint: enable=use-dict-literal

    raise ValueError(f"Invalid model name or path: {model_name_or_path}")


def get_model(
    task: str,
    model_name_or_path: Optional[str] = None,
    config: Optional[PretrainedConfig] = None,
    **kwds,
) -> PreTrainedModel | MalConv | MalConvGCT:
    if model_name_or_path is None and config is None:
        raise ValueError("Must specify `model_name_or_path` or `config`.")
    if model_name_or_path is None == config is None:
        warnings.warn(
            f"Specified both `model_name_or_path` or `config`. {model_name_or_path=} will take "
            f"precidence over {type(config)=} if its a path that exists."
        )

    # PreTrainedModel doesn't like None values for num_labels id2label and label2id.
    for k in ["num_labels", "id2label", "label2id"]:
        if k in kwds and kwds[k] is None:
            kwds.pop(k)

    # Get model from disk
    if model_name_or_path is not None and Path(model_name_or_path).exists():
        model_name = object_to_model_name(model_name_or_path)
        if task == "clf":
            if model_name == "hrrformer":
                return HRRForSequenceClassification.from_pretrained(model_name_or_path, **kwds)
            if model_name == "rwkv":
                return RwkvForSequenceClassification.from_pretrained(model_name_or_path, **kwds)
            if model_name == "mamba":
                return MambaForSequenceClassification.from_pretrained(model_name_or_path, **kwds)
            if get_model_type(model_name_or_path) == "HF":
                return AutoModelForSequenceClassification.from_pretrained(model_name_or_path, **kwds)
            if get_model_type(model_name_or_path) == "MC":
                return AutoMalConvForSequenceClassification.from_pretrained(model_name_or_path, **kwds)
        if task == "mlm":
            if model_name == "hrrformer":
                return HRRForMaskedLM.from_pretrained(model_name_or_path, **kwds)
            return AutoModelForMaskedLM.from_pretrained(model_name_or_path, **kwds)
        if task == "clm":
            if model_name == "mamba":
                return MambaForCausalLM.from_pretrained(model_name_or_path, **kwds)
            return AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwds)

    # Get model from config
    if config:
        if task == "clf":
            if isinstance(config, MalConvConfig):
                return MalConv(config)
            if isinstance(config, MalConvGCTConfig):
                print(config)
                return MalConvGCT(config)
            if isinstance(config, MyMalConvConfig):
                return MyMalConv(config)
            if isinstance(config, HRRConfig):
                return HRRForSequenceClassification(config)
            if isinstance(config, RwkvConfig):
                return RwkvForSequenceClassification(config)
            if isinstance(config, MambaConfig):
                return MambaForSequenceClassification(config)
            if isinstance(config, PretrainedConfig):
                return AutoModelForSequenceClassification.from_config(config)
        if task == "mlm":
            if isinstance(config, HRRConfig):
                return HRRForMaskedLM(config)
            if isinstance(config, RwkvConfig):
                raise NotImplementedError()
            if isinstance(config, MambaConfig):
                raise NotImplementedError()
                # return MambaForMaskedLM(config)
            if isinstance(config, PretrainedConfig):
                return AutoModelForMaskedLM.from_config(config)
        if task == "clm":
            if isinstance(config, HRRConfig):
                raise NotImplementedError()
            if isinstance(config, RwkvConfig):
                pass
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
    map_kwds = {
        "function": function,
        "batched": True,
        "batch_size": BATCH_SIZE,
    }
    if isinstance(dataset, (DatasetDict, Dataset)):
        map_kwds.update(
            {
                "keep_in_memory": KEEP_IN_MEMORY,
                "num_proc": NUM_PROC,
                "cache_file_names": CACHE_FILE_NAME,
                "writer_batch_size": WRITER_BATCH_SIZE,
            }
        )
    map_kwds.update(kwds)
    return map_kwds


def main(args: Args, training_arguments: TrainingArguments) -> None:
    m = get_mem(unit="MB")
    print(f"MEMORY: mem_used={m[2]}, mem_avail={m[1]}, mem_total={m[0]}", flush=True)
    print(f"{args=}", flush=True)

    MODEL_TYPE: Literal["HF", "MC"] = get_model_type(args.model_name_or_path)
    MODEL_NAME = object_to_model_name(args.model_name_or_path)

    oh = OutputHelper(
        args.model_name_or_path,
        args.representation,
        args.algorithm,
        args.vocab_size,
        args.max_length,
        args.task,
        args.tr_size,
        args.depth,
        args.min_freq,
        args.top_k,
        args.enforce_cutoff,
        args.tr_length_cutoff,
        args.ft_freeze_positional_embeddings,
        args.ft_duplicate_positional_embeddings,
        args.ft_initialize_positional_embeddings,
        args.root,
        args.arch_config,
        training_arguments.__dict__,
    )
    print(f"{oh=}")
    print(BR, flush=True)

    training_arguments = replace(
        training_arguments,
        prediction_loss_only=args.task in ("mlm", "clm"),  # NOTE 0
    )
    print(f"{training_arguments=}")
    print(BR, flush=True)

    tokenizer = get_tokenizer(
        representation=args.representation,
        algorithm=args.algorithm,
        vocab_size=args.vocab_size,
        model_max_length=args.max_length,
        add_cls_token=False,
        add_bos_token=True,
        add_eos_token=True,
        add_sep_token=False,
    )
    print(f"{tokenizer=}")
    pprint({k: v for k, v in zip(tokenizer.all_special_tokens, tokenizer.all_special_ids)})
    print(f"{tokenizer.model_input_names=}")
    print(BR, flush=True)


    # Get the raw materials for the dataset, i.e., the files, labels, etc.
    id2label: Optional[dict[int, str]] = None
    label2id: Optional[dict[str, int]] = None
    dist: Optional[Counter] = None
    num_classes: Optional[int] = None
    weight: Optional[Tensor] = None
    if args.task in ("mlm", "clm"):
        materials = get_materials_pretrain_sorel(args.tr_size, args.vl_size, args.ts_size)
    elif args.task == "clf":
        materials = get_materials_clf_bodmas(
            args.tr_size,
            args.vl_size,
            args.ts_size,
            top_k=args.top_k,
            min_freq=args.min_freq,
        )
        id2label = materials.id2label
        label2id = materials.label2id
        dist = materials.dist
        num_classes = len(id2label)
        weight = tensor([1 / freq for freq in [dist[c] for c in label2id.keys()]])
    print(f"{dist=}")
    print(BR, flush=True)


    # If we have apriori knowledge of the length of the dataset, we can compute the number of steps
    # from the number of training epochs.
    if args.streaming and training_arguments.max_steps == -1:
        max_steps = compute_total_steps(
            len(materials.tr_vl_ts_files_and_labels["tr"][0]),
            training_arguments.num_train_epochs,
            training_arguments.per_device_train_batch_size,
            training_arguments.gradient_accumulation_steps,
        )
        assert isinstance(max_steps, int)
        training_arguments = replace(training_arguments, max_steps=max_steps)


    dataset: DatasetDict | IterableDatasetDict | MapBinaryDatasetDict | IterableBinaryDatasetDict
    if args.dataset_backend == "HF":
        dataset = get_dataset_hf(
            materials,
            args.streaming,
            training_arguments.dataloader_num_workers,
            max_length=args.data_read_bytes,
        )
        print_dataset_hf(dataset)

        if args.algorithm.lower() == "raw" or args.algorithm in COMPRESSION_TYPES + ENCRYPTION_TYPES:
            preprocess_fns = []
            if args.algorithm in COMPRESSION_TYPES:
                preprocess_fns.append(partial(hf_compress_bytes, compression_type=args.algorithm, compression_level=args.compression_level))
            elif args.algorithm in ENCRYPTION_TYPES:
                preprocess_fns.append(partial(hf_encrypt_bytes, encryption_type=args.algorithm, key=None))
            preprocess_fns.append(partial(
                hf_bytes_to_input_ids,
                bits_in_byte=args.representation,
                num_special_ids=len(tokenizer.all_special_ids),
                max_length=args.max_length,
                cls_token_id=tokenizer.cls_token_id if MODEL_NAME in REQ_CLS_TOKEN else None,
                bos_token_id=tokenizer.bos_token_id if MODEL_NAME in REQ_BOS_TOKEN else None,
                eos_token_id=tokenizer.eos_token_id if MODEL_NAME in REQ_EOS_TOKEN else None,
            ))
            preprocess_fn = compose_functions(*preprocess_fns)
            num_proc = NUM_PROC
        else:
            preprocess_fn = partial(
                hf_tokenize_bytes,
                tokenizer=tokenizer,
                max_length=args.max_length,
            )
            num_proc = None
        dataset = dataset.map(**get_map_kwds_for_hf_datasets(
            function=preprocess_fn,
            dataset=dataset,
            remove_columns=["name", "bytes"] if args.task == "clf" else ["name", "bytes", "labels"],
            num_proc=num_proc,
        ))
        print_dataset_hf(dataset)

        if args.exit_after_map:
            sys.exit(0)
        if MOVE_IN_MEMORY:
            for s in dataset:
                dataset[s] = dataset[s].select(range(len(dataset[s])), keep_in_memory=True)

    else:
        if args.algorithm.lower() == "raw" or args.algorithm in COMPRESSION_TYPES + ENCRYPTION_TYPES:
            preprocess_fns = []
            if args.algorithm in COMPRESSION_TYPES:
                preprocess_fns.append(partial(compress, compression_type=args.algorithm, compression_level=args.compression_level))
            elif args.algorithm in ENCRYPTION_TYPES:
                preprocess_fns.append(partial(encrypt, encryption_type=args.algorithm, key=None))
            preprocess_fns.append(partial(
                bytes_to_input_ids,
                bits_in_byte=args.representation,
                num_special_ids=len(tokenizer.all_special_ids),
                max_length=args.max_length,
                cls_token_id=tokenizer.cls_token_id if MODEL_NAME in REQ_CLS_TOKEN else None,
                bos_token_id=tokenizer.bos_token_id if MODEL_NAME in REQ_BOS_TOKEN else None,
                eos_token_id=tokenizer.eos_token_id if MODEL_NAME in REQ_EOS_TOKEN else None,
            ))
            preprocess_fn = compose_functions(*preprocess_fns)
        else:
            preprocess_fn = partial(
                tokenize_bytes,
                tokenizer=tokenizer,
                max_length=args.max_length,
            )

        dataset = get_dataset_pt(
            materials,
            args.streaming,
            max_length=args.data_read_bytes,
            preprocess_fn=preprocess_fn,
        )
        print_dataset_pt(dataset)

    config = get_config(
        args.model_name_or_path,
        tokenizer,
        args.max_length,
        tensor_log_path=oh.tensor_log_path,
        arch_config=args.arch_config,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
    )
    print(f"{config=}")
    print(BR, flush=True)


    pad_to_multiple_of = PAD_TO
    if isinstance(config, transformers.ReformerConfig):
        pad_to_multiple_of = _get_least_common_mult_chunk_len(config)

    # Change the tokenizer's attributes for the data_collator to use correctly.
    # This let's us use the previously generated cache files then drop the
    # attention_mask before passing the inputs to the model.
    if MODEL_NAME not in RETURN_ATTENTION_MASK:
        tokenizer.model_input_names.remove("attention_mask")

    if args.task == "clf":
        data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding="max_length" if MODEL_TYPE == "MC" else "longest",  # malconv needs padding to its max_length
            pad_to_multiple_of=pad_to_multiple_of,
            max_length=args.max_length,  # hopefully, the sequences were truncated before hand...
        )
        compute_metrics = clf_compute_metrics
    elif args.task == "mlm":
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            pad_to_multiple_of=pad_to_multiple_of,
        )
        compute_metrics = None
    elif args.task == "clm":
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=pad_to_multiple_of,
        )
        compute_metrics = None

    print(f"{data_collator=}")
    print(f"{compute_metrics=}")
    print(BR, flush=True)

    callbacks = [UtilCallback(False)]
    if args.early_stopping:
        callbacks.append(EarlyStoppingCallback(args.early_stopping_patience, args.early_stopping_threshold))
    if MODEL_TYPE == "MC":
        callbacks.append(SaveConfigToCheckpointCallback())
    print(f"{callbacks=}")

    if args.task == "clf":
        ModelTrainer = partial(ImbalancedClassificationTrainer, weight=weight)
    elif args.task in ("mlm", "clm"):
        ModelTrainer = Trainer

    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
    if not args.streaming:  # dataset has been processed, so we disable thread-based parallelism
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if training_arguments.do_train:
        model = get_model(
            args.task,
            args.model_name_or_path,
            config,
            num_labels=num_classes,
            id2label=id2label,
            label2id=label2id,
        )
        print(f"{model=}")
        print(f"{count_parameters(model, requires_grad=False)=}")
        print(f"{count_parameters(model, requires_grad=True)=}")
        print(BR, flush=True)

        # Resize the embeddings if necessary
        if modify_positional_embeddings_allowed(model):
            add_positional_embeddings = args.max_length > model.config.max_position_embeddings

            if add_positional_embeddings and isinstance(config, LongformerConfig):
                max_position_embeddings = longformer_max_position_embeddings(
                    args.max_length, config.attention_window[0]
                )
            elif add_positional_embeddings:
                max_position_embeddings = args.max_length
            else:
                max_position_embeddings = model.config.max_position_embeddings

            config.max_position_embeddings = max_position_embeddings

            model = modify_positional_embeddings(
                model,
                max_position_embeddings=max_position_embeddings,
                duplicate=args.ft_duplicate_positional_embeddings and add_positional_embeddings,
                initialize=args.ft_initialize_positional_embeddings and add_positional_embeddings,
                freeze=args.ft_freeze_positional_embeddings,
                initalizer_range=config.initializer_range,
            )
            print(f"{model=}")
            print(f"{count_parameters(model, requires_grad=False)=}")
            print(f"{count_parameters(model, requires_grad=True)=}")
            print(BR, flush=True)


        # Initial evaluation of the model on the validation set to detect OOM and CudaOOM errors.
        # This will also reduce the eval_batch size in the training_arguments variable.
        @find_executable_batch_size(starting_batch_size=training_arguments.per_device_eval_batch_size)
        def _eval(batch_size: int) -> tuple[PredictionOutput, int]:
            nonlocal training_arguments  # access variables outside of this function.
            training_arguments = replace(training_arguments, per_device_eval_batch_size=batch_size)
            print(f"Evaluating with {batch_size=}...", flush=True)
            trainer = ModelTrainer(
                model=model,
                args=training_arguments,
                train_dataset=dataset["tr"],
                eval_dataset=dataset["vl"],
                data_collator=data_collator,
                tokenizer=tokenizer if args.dataset_backend == "HF" else None,  # TODO: args.algorithm.lower() != "raw"
                callbacks=callbacks,
                compute_metrics=compute_metrics,
            )
            return trainer.predict(dataset["vl"]), batch_size

        print("Initial Evaluation...", flush=True)
        initial_output: PredictionOutput = None
        max_per_device_eval_batch_size: int = None
        if not args.skip_eval_check:
            initial_output, max_per_device_eval_batch_size = _eval()  # pylint: disable=no-value-for-parameter
            model = model.to(torch.float32).to("cpu")
            torch.cuda.empty_cache()
            gc.collect()
            print(f"{initial_output.metrics=}", flush=True)


        @find_executable_batch_size_and_gradient_accumulation_steps(
            starting_batch_size=training_arguments.per_device_train_batch_size,
            starting_gradient_accumulation_steps=training_arguments.gradient_accumulation_steps,
        )
        def _train(batch_size: int, gradient_accumulation_steps: int) -> TrainOutput:
            nonlocal training_arguments, oh
            print(f"Training with {batch_size=} and {gradient_accumulation_steps=}...", flush=True)
            try:  # Try to remove a created, but empty directory from a previous attempt.
                oh.rmdir(ignore_config=True)
            except OSError:
                pass

            # TODO: if the OOM error arises during the evaluation loop while training, this could
            # cause irresponsible and unessecary reduction of the training batch size when we really
            # should be decrementing the evaluation batch size.
            per_device_eval_batch_size = batch_size
            if max_per_device_eval_batch_size is not None:
                per_device_eval_batch_size = max_per_device_eval_batch_size

            training_arguments = replace(
                training_arguments,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=per_device_eval_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
            )
            oh.trainer_config = training_arguments.__dict__
            oh.mkdir()

            training_arguments = replace(training_arguments, output_dir=oh.checkpoints_dir.as_posix())
            print(f"{training_arguments.per_device_train_batch_size=} {training_arguments.per_device_eval_batch_size=}")
            trainer = ModelTrainer(
                model=model,
                args=training_arguments,
                train_dataset=dataset["tr"],
                eval_dataset=dataset["vl"],
                data_collator=data_collator,
                tokenizer=tokenizer if args.dataset_backend == "HF" else None,
                callbacks=callbacks,
                compute_metrics=compute_metrics,
            )
            return trainer.train(training_arguments.resume_from_checkpoint)

        print("Training...", flush=True)
        if args.auto_find_batch_size_and_gradient_accumulation_steps:
            trainer_output: TrainOutput = _train()  # pylint: disable=no-value-for-parameter
        else:
            oh.mkdir()
            training_arguments = replace(training_arguments, output_dir=oh.checkpoints_dir.as_posix())
            trainer = ModelTrainer(
                model=model,
                args=training_arguments,
                train_dataset=dataset["tr"],
                eval_dataset=dataset["vl"],
                data_collator=data_collator,
                tokenizer=tokenizer if args.dataset_backend == "HF" else None,
                callbacks=callbacks,
                compute_metrics=compute_metrics,
            )
            trainer_output: TrainOutput = trainer.train(training_arguments.resume_from_checkpoint)

        if not args.skip_eval_check:
            with open(oh.initial_validation_results_file, "w") as fp:
                json.dump(initial_output.metrics, fp, indent=4)
        with open(oh.trainer_output_file, "w") as fp:
            json.dump(trainer_output.metrics, fp, indent=4)


    if training_arguments.do_eval:
        if not (training_arguments.do_train and training_arguments.load_best_model_at_end):
            print("Getting model from disk for evaluation.")
            if training_arguments.do_train:
                print("Deleting current model from memory.")
                model.to("cpu")
                del model
                torch.cuda.empty_cache()
                gc.collect()
            model = get_model(
                args.task,
                oh.best_model_dir,
                None,
                num_labels=num_classes,
                id2label=id2label,
                label2id=label2id,
            )
        print(f"{model=}")
        print(f"{count_parameters(model, requires_grad=False)=}")
        print(f"{count_parameters(model, requires_grad=True)=}")

        if args.task == "clf":
            single_shot_classes = [label2id[l] for l in dist if dist[l] == 3]
            compute_metrics = partial(clf_compute_metrics, single_shot_classes=single_shot_classes)
            print("single_shot_classes=")
            pprint(single_shot_classes)

        trainer = ModelTrainer(
            model=model,
            args=training_arguments,
            data_collator=data_collator,
            tokenizer=tokenizer,
            callbacks=callbacks,
            compute_metrics=compute_metrics,
        )

        oh.test_results_dir.mkdir(exist_ok=True, parents=False)
        print("Evaluating...")
        output: PredictionOutput = trainer.predict(dataset["ts"])

        results = output.metrics
        with open(oh.test_results_file, "w") as fp:
            json.dump(results, fp, indent=4)

    if args.do_tune:
        training_arguments = replace(
            training_arguments,
            do_eval=True,
            evaluation_strategy="steps",
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
            num_labels=num_classes,
            id2label=id2label,
            label2id=label2id,
        )
        trainer = ModelTrainer(
            model_init=model_init,
            args=training_arguments.hf_training_arguments_object(),
            train_dataset=dataset["tr"],
            eval_dataset=dataset["vl"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            callbacks=None,
            compute_metrics=compute_metrics,
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
        elif isinstance(config, (MalConvGCTConfig,)):
            hp_space = hp_space_malconvgct
        elif isinstance(config, (MyMalConvConfig,)):
            hp_space = hp_space_mymalconv
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
