"""
Train and evaluate the models for malware family classification.

# FIXME: determine if the tokenizer needs to add the [CLS] token for sequence
# classification.

# TODO: investigate increasing the batch size and number of processes for the
# Dataset.map() calls. Experiment with malconv, as the impact of disk access is
# more pronounced there than with the heavy transformers models.

# TODO: does the attention mask need to be handled by the tokenizer? Can it be
# handled by the dataloader instead? If so, we could skip the entire second map 
# process, which would greatly enhance the speed of the entire process...
"""

# pylint: disable=wrong-import-position
print(f"Entered {__file__=}")

from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import datetime
from functools import partial
import gc
import inspect
import json
import math
from pathlib import Path
from pprint import pformat, pprint
import random
from typing import Any, Callable, Literal, Optional
import os
import sys

if __name__ == "__main__":
    print(f"STARTING @{datetime.now()}\n{'-' * 88}", flush=True)
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# pylint: enable=wrong-import-position

from datasets import (
    DatasetDict,
    Dataset,
    IterableDataset,
    IterableDatasetDict,
    concatenate_datasets,
)
import evaluate
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import torch
from torch import tensor, Tensor
from torch.nn import CrossEntropyLoss, Embedding
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    DefaultDataCollator,
    EarlyStoppingCallback,
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
)
from transformers import TrainingArguments as HfTrainingArguments
from transformers.trainer_utils import (
    BestRun,
    EvalPrediction,
    PredictionOutput,
    PREFIX_CHECKPOINT_DIR,
)
from transformers.models.reformer.modeling_reformer import _get_least_common_mult_chunk_len
from transformers.utils import CONFIG_NAME, is_ninja_available
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune import TuneError

from src.cfg import BR, OUTPUT_PATH
from src.utils import object_from_superset_of_constructor_kwds
from src.malconv import (
    AutoMalConvForSequenceClassification,
    MalConvConfig,
    MalConvGCTConfig,
    MyMalConvConfig,
    MalConv,
    MalConvGCT,
    MyMalConv,
)
from src.hrrformer import (
    HRRConfig,
    HRRForSequenceClassification,
    HRRForMaskedLM,
)
from src.rwkv import (
    RwkvConfig,
    RwkvForSequenceClassification,
)
from src.utils import get_highest_path
from src.data.loaders import get_sorel_dataset, get_bodmas_dataset
from src.learn.tuning import (
    TunedConfigs,
    hp_space_mymalconv,
    hp_space_malconv,
    hp_space_malconvgct,
    hp_space_longformer,
    hp_space_hrrformer,
)
from src.learn.utils import (
    count_parameters,
    pad_to_multiple_of_fn,
    find_two_largest_factors,
    get_tokenizer_object,
    get_fast_tokenizer,
    examples_to_text,
    examples_to_input_ids,
    tokenize_fn,
    float_to_int,
    compute_total_steps,
    str_or_bool_to_str,
)


random.seed(0)
np.random.seed(0)
torch.random.manual_seed(0)


PAD_TO = 8

SUBSET = None # 80000 # tune_hrrformer
STREAMING = True
KEEP_IN_MEMORY = False
EXIT_AFTER_MAP = False
BODMAS_TOP_K = None
BODMAS_MIN_FREQ = None
PREPROCESS_AS_TEXT = True
PREPROCESS_AS_INPUT_IDS = False
PREPROCESS_AS_INPUT_IDS_DO_PAD = True
CACHE_FILE_NAME: Optional[str] = None  # "/home/lk3591/Documents/code/RawByteClf/INPUT_IDS_1048576"
NUM_PROC: Optional[int] = None

TUNE_TR_N_SAMPLES = 8000
TUNE_VL_N_SAMPLES = 8000
TUNE_TS_N_SAMPLES = 0

N_INITIAL_POINTS = 8
N_TRIALS = 32  # including the initial points
TUNE_RESOURCES_PER_TRIAL = {
    "cpu": 1,
    "gpu": 1,
}


ACCURACY = evaluate.load("accuracy")
F1 = evaluate.load("f1")


# FIXME: fix the entire compute metrics pipeline....
def COMPUTE_METRICS(
    eval_pred: EvalPrediction, single_shot_classes: Optional[list[int]] = None
) -> dict[str, float]:
    predictions, labels = eval_pred.predictions, eval_pred.label_ids
    predictions = np.argmax(predictions, axis=1)
    metrics = {
        "accuracy": ACCURACY.compute(predictions=predictions, references=labels)["accuracy"],
        "f1-macro": F1.compute(predictions=predictions, references=labels, average="macro")["f1"],
        "f1-micro": F1.compute(predictions=predictions, references=labels, average="micro")["f1"],
    }
    if single_shot_classes is None:
        return metrics

    include = np.array([i for i, l in enumerate(labels) if l in single_shot_classes])
    predictions = predictions[include]
    labels = labels[include]
    metrics.update(
        {
            "ss_accuracy": ACCURACY.compute(predictions=predictions, references=labels)["accuracy"],
            "ss_f1-macro": F1.compute(predictions=predictions, references=labels, average="macro")[
                "f1"
            ],
            "ss_f1-micro": F1.compute(predictions=predictions, references=labels, average="micro")[
                "f1"
            ],
        }
    )
    return metrics


class TrainingArguments(HfTrainingArguments):
    def __init__(self, **kwds):
        do_eval = kwds.get("do_eval", False)

        if kwds.get("resume_from_checkpoint", None):  # lets us pass in "true" from command line
            try:
                kwds["resume_from_checkpoint"] = str_or_bool_to_str(kwds["resume_from_checkpoint"])
            except ValueError:
                pass

        super().__init__(**kwds)
        self.do_eval = do_eval

    def hf_training_arguments_object(self) -> HfTrainingArguments:
        return object_from_superset_of_constructor_kwds(HfTrainingArguments, **self.__dict__)


class SaveConfigToCheckpointCallback(TrainerCallback):
    def on_save(
        self, args: HfTrainingArguments, state: TrainerState, control: TrainerControl, **kwds
    ):
        checkpoint_folder = f"{args.output_dir}/{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        kwds["model"].config.save_pretrained(checkpoint_folder)
        with open(f"{checkpoint_folder}/{CONFIG_NAME}", "r") as fp:
            config = json.load(fp)
        config["architectures"] = type(kwds["model"]).__name__
        with open(f"{checkpoint_folder}/{CONFIG_NAME}", "w") as fp:
            json.dump(config, fp, indent=4, sort_keys=True)


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

    config = get_config(model_name_or_path, tokenizer, max_length, **(hparams | kwds))
    model = get_model(task, None, config, **kwds)
    if model is None:
        raise RuntimeError("Model is None for some reason.")
    return model


def hp_compute_objective(metrics: dict[str, float]) -> float:
    return metrics["eval_loss"]


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
]


RETURN_ATTENTION_MASK = {
    "longformer": True,
    "reformer": True,
    "nystromformer": True,
    "fnet": False,
    "malconv": False,
    "malconvgct": False,
    "mymalconv": False,
    "hrrformer": True,
    "rwkv": False,
}


def object_to_model_name_or_path(obj) -> str:
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
    if isinstance(obj, (MalConvConfig,)):
        return "malconv"
    if isinstance(obj, (MalConvGCTConfig,)):
        return "malconvgct"
    if isinstance(obj, (MyMalConvConfig,)):
        return "mymalconv"
    if isinstance(obj, (str, Path)) and Path(obj).exists():
        return object_to_model_name_or_path(AutoConfig.from_pretrained(str(obj)))
    raise RuntimeError()


def get_model_type(model_name_or_path: str | Path) -> Literal["HF", "MC"]:
    model_name_or_path = str(model_name_or_path)
    if "malconv" in model_name_or_path:
        return "MC"
    return "HF"


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
        if self.loss_fn.weight.device != device:
            self.loss_fn.weight = self.loss_fn.weight.to(device)

        # num_labels = unwrap_model(model).config.num_labels
        loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


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


@dataclass
class Args:
    model_name_or_path: str = field()
    max_length: int = field()
    task: str = field()
    depth: int = field(default=1)
    ft_freeze_positional_embeddings: bool | str = field(default=False)
    ft_duplicate_positional_embeddings: bool | str = field(default=False)
    ft_initialize_positional_embeddings: bool | str = field(default=False)
    root: Path = field(default=OUTPUT_PATH)
    do_tune: bool = field(default=False)


class OutputHelper:
    def __init__(
        self,
        model_name_or_path: str,
        max_length: int,
        task: str,
        depth: int,
        ft_freeze_positional_embeddings: bool | str,
        ft_duplicate_positional_embeddings: bool | str,
        ft_initialize_positional_embeddings: bool | str,
        root: Path,
    ) -> None:
        self.root = Path(root)
        args = [
            model_name_or_path,
            str(max_length),
            task,
            str(depth),
        ]
        if task == "clf":
            args.extend(
                [
                    str(str_or_bool_to_str(ft_freeze_positional_embeddings)),
                    str(str_or_bool_to_str(ft_duplicate_positional_embeddings)),
                    str(str_or_bool_to_str(ft_initialize_positional_embeddings)),
                ]
            )
        self.path = self.root.joinpath(*args)

    def __repr__(self) -> str:
        return self.path.as_posix()

    def __str__(self) -> str:
        return self.path.as_posix()

    @property
    def best_model_dir(self) -> Path:
        with open(self.last_checkpoint / "trainer_state.json") as fp:
            state = json.load(fp)
        best_model_checkpoint = state["best_model_checkpoint"]
        return Path(best_model_checkpoint)

    @property
    def checkpoints_dir(self) -> Path:
        return self.path / "checkpoints"

    @property
    def config_file(self) -> Path:
        return self.path / "config.json"

    @property
    def test_results_dir(self) -> Path:
        return self.path / "test_results"

    @property
    def test_results_file(self) -> Path:
        return self.test_results_dir / "results.json"

    @property
    def test_predictions_file(self) -> Path:
        return self.test_results_dir / "predictions.txt"

    @property
    def test_probas_file(self) -> Path:
        return self.test_results_dir / "probas.txt"

    @property
    def test_labels_file(self) -> Path:
        return self.test_results_dir / "labels.txt"

    @property
    def test_confusion_matrix_file(self) -> Path:
        return self.test_results_dir / "confusion_matrix.png"

    @property
    def tuning_results_dir(self) -> Path:
        return self.path / "tuning_results"

    @property
    def last_checkpoint(self) -> Path:
        return get_highest_path(self.checkpoints_dir, lstrip="checkpoint-")

    def mkdir(self) -> None:
        self.path.mkdir(exist_ok=True, parents=True)


class ComputeMetrics:
    def __init__(self, detailed: bool = False) -> None:
        self.detailed = detailed

    def set_detailed(self, detailed: bool) -> None:
        self.detailed = detailed

    def return_report(self, report: dict[str, float | dict]) -> dict[str, float | dict]:
        if self.detailed:
            return report
        return {
            "accuracy": report["accuracy"],
            "f1_macro": report["macro avg"]["f1-score"],
            "f1_weighted": report["weighted avg"]["f1-score"],
        }


class CLFComputeMetrics(ComputeMetrics):
    def __call__(self, eval_pred: EvalPrediction) -> dict[str, float | dict]:
        # predictions (B, M)
        # label_ids (B,)
        y_true, y_pred = self.get_y_true_y_pred(eval_pred.predictions, eval_pred.label_ids)
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=np.nan)
        return super().return_report(report)

    @staticmethod
    def get_y_true_y_pred(
        predictions: np.ndarray, label_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return CLFComputeMetrics.get_y_true(label_ids), CLFComputeMetrics.get_y_pred(predictions)

    @staticmethod
    def get_y_pred(predictions: np.ndarray) -> np.ndarray:
        predictions = tensor(predictions, dtype=torch.float32)
        probas = torch.softmax(predictions, dim=1).numpy()
        y_pred = np.argmax(probas, axis=1)
        return y_pred

    @staticmethod
    def get_y_true(label_ids: np.ndarray) -> np.ndarray:
        return label_ids.astype(np.int64)


# FIXME: using this seems to cause OOM errors during the evaluation loop.
# It seems reasonable to suspect that the CLFComputeMetrics has similar issues.
class MLMComputeMetrics(ComputeMetrics):
    def __call__(self, eval_pred: EvalPrediction) -> dict[str, float | dict]:
        # predictions (B, L, M)
        # label_ids (B, M)
        y_true, y_pred = self.get_y_true_y_pred(eval_pred.predictions, eval_pred.label_ids)
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=np.nan)
        return super().return_report(report)

    @staticmethod
    def get_y_true_y_pred(
        predictions: np.ndarray, label_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        y_pred = MLMComputeMetrics.get_y_pred(predictions)
        y_true = MLMComputeMetrics.get_y_true(label_ids)
        mask = y_true == -100
        y_pred = y_pred[~mask]
        y_true = y_true[~mask]
        return y_true, y_pred

    @staticmethod
    def get_y_pred(predictions: np.ndarray) -> np.ndarray:
        predictions = tensor(predictions, dtype=torch.float32)
        predictions = predictions.view(-1, predictions.shape[2])  # (B * L, M)
        probas = torch.softmax(predictions, dim=1).numpy()
        y_pred = np.argmax(probas, axis=1)
        return y_pred

    @staticmethod
    def get_y_true(label_ids: np.ndarray) -> np.ndarray:
        y_true = tensor(label_ids, dtype=torch.float32).view(-1)  # (B * L,)
        return y_true.numpy().astype(np.int64)


def get_config(
    model_name_or_path: str,
    tokenizer: Optional[PreTrainedTokenizerFast] = None,
    max_length: Optional[int] = None,
    **kwds,
) -> PretrainedConfig:
    """
    kwds
    ----
        num_labels (int): use for classification
    """
    if Path(model_name_or_path).exists():
        return AutoConfig.from_pretrained(model_name_or_path, **kwds)

    # Handle float values when hyperparameter tuning.
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
    max_posititional_embeddings = pad_to_multiple_of_fn(max_length, 8)

    # kwds overrides the tuned_kwds
    kwds = TunedConfigs[model_name_or_path.lower()][max_length] | kwds

    if model_name_or_path.lower() == "longformer":
        attention_window = kwds.pop("attention_window", 512)
        return LongformerConfig(
            vocab_size=vocab_size,
            max_position_embeddings=longformer_max_position_embeddings(max_length, attention_window),
            attention_window=attention_window,
            sep_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **kwds,
        )
    if model_name_or_path.lower() == "reformer":
        num_hidden_layers = kwds.pop("num_hidden_layers", 6)
        hidden_size = kwds.pop("hidden_size", 256)
        return ReformerConfig(
            vocab_size=vocab_size,
            max_position_embeddings=max_posititional_embeddings,
            hidden_size=hidden_size,
            feed_forward_size=kwds.pop("feed_forward_size", kwds.pop("intermediate_size", 512)),
            attn_layers=["local" if i % 2 == 0 else "lsh" for i in range(num_hidden_layers)],
            axial_pos_shape=list(find_two_largest_factors(max_length)),
            axial_pos_embds_dim=[hidden_size // 2, hidden_size // 2],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **kwds,
        )
    if model_name_or_path.lower() == "nystromformer":
        return NystromformerConfig(
            vocab_size=vocab_size,
            max_position_embeddings=max_posititional_embeddings,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **kwds,
        )
    if model_name_or_path.lower() == "fnet":
        return FNetConfig(
            vocab_size=vocab_size,
            max_position_embeddings=max_posititional_embeddings,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **kwds,
        )
    if model_name_or_path.lower() == "hrrformer":
        return HRRConfig(
            vocab_size=vocab_size,
            max_position_embeddings=max_posititional_embeddings,
            pad_token_id=tokenizer.pad_token_id,
            **kwds,
        )
    if model_name_or_path.lower() == "rwkv":
        if not is_ninja_available():
            raise RuntimeError("Ninja is required to use RWKV. Without it, the model is too slow.")
        return RwkvConfig(
            vocab_size=vocab_size,
            context_length=max_posititional_embeddings,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **kwds,
        )
    if model_name_or_path.lower() == "malconv":
        return MalConvConfig(
            out_size=kwds["num_labels"],
            pad_idx=tokenizer.pad_token_id,
            num_embd=vocab_size,
            **kwds,
        )
    if model_name_or_path.lower() == "malconvgct":
        return MalConvGCTConfig(
            out_size=kwds["num_labels"],
            pad_idx=tokenizer.pad_token_id,
            num_embd=vocab_size,
            **kwds,
        )
    if model_name_or_path.lower() == "mymalconv":
        return MyMalConvConfig(
            out_size=kwds["num_labels"],
            pad_idx=tokenizer.pad_token_id,
            num_embd=vocab_size,
            max_length=max_posititional_embeddings,
            **kwds,
        )

    raise ValueError(f"Invalid model name or path: {model_name_or_path}")


def get_model(
    task: str,
    model_name_or_path: Optional[str] = None,
    config: Optional[PretrainedConfig] = None,
    **kwds,
) -> PreTrainedModel | MalConv | MalConvGCT:
    if model_name_or_path is None == config is None:
        raise ValueError("Must specify exactly one of `model_name_or_path` or `config`.")
    if model_name_or_path is not None and not Path(model_name_or_path).exists():
        raise FileNotFoundError(f"Invalid model name or path: {model_name_or_path}.")

    # Get model from disk
    if model_name_or_path:
        if task == "clf":
            if get_model_type(model_name_or_path) == "HF":
                return AutoModelForSequenceClassification.from_pretrained(
                    model_name_or_path, **kwds
                )
            if get_model_type(model_name_or_path) == "MC":
                return AutoMalConvForSequenceClassification.from_pretrained(
                    model_name_or_path,
                    **kwds,
                )
        if task == "mlm":
            return AutoModelForMaskedLM.from_pretrained(model_name_or_path, **kwds)
        if task == "clm":
            return AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwds)

    # Get model from config
    if config:
        if task == "clf":
            if isinstance(config, MalConvConfig):
                return MalConv(config)
            if isinstance(config, MalConvGCTConfig):
                return MalConvGCT(config)
            if isinstance(config, MyMalConvConfig):
                return MyMalConv(config)
            if isinstance(config, HRRConfig):
                return HRRForSequenceClassification(config)
            if isinstance(config, RwkvConfig):
                return RwkvForSequenceClassification(config)
            if isinstance(config, PretrainedConfig):
                return AutoModelForSequenceClassification.from_config(config)
        if task == "mlm":
            if isinstance(config, HRRConfig):
                return HRRForMaskedLM(config)
            if isinstance(config, PretrainedConfig):
                return AutoModelForMaskedLM.from_config(config)
        if task == "clm":
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
    }
    map_kwds.update(kwds)
    if isinstance(dataset, (DatasetDict, Dataset)):
        map_kwds.update(
            {
                "keep_in_memory": KEEP_IN_MEMORY,
                "num_proc": NUM_PROC,
                "cache_file_names": CACHE_FILE_NAME,
            }
        )
    return map_kwds


def main(args: Args, training_arguments: TrainingArguments) -> None:
    print(f"{args=}")

    TYPE = get_model_type(args.model_name_or_path)

    oh = OutputHelper(
        args.model_name_or_path,
        args.max_length,
        args.task,
        args.depth,
        args.ft_freeze_positional_embeddings,
        args.ft_duplicate_positional_embeddings,
        args.ft_initialize_positional_embeddings,
        args.root,
    )
    print(f"{oh=}")
    print(BR, flush=True)

    training_arguments = replace(
        training_arguments,
        output_dir=oh.checkpoints_dir.as_posix(),
        load_best_model_at_end=True,
        resume_from_checkpoint=(
            True
            if training_arguments.resume_from_checkpoint == "last-checkpoint"
            else training_arguments.resume_from_checkpoint
        ),
    )
    print(f"{training_arguments=}")
    print(BR, flush=True)

    tokenizer = get_tokenizer_object()
    tokenizer = get_fast_tokenizer(tokenizer, model_max_length=args.max_length)
    print(f"{tokenizer=}")
    print(BR, flush=True)

    dataset: DatasetDict
    dist: Counter = None

    if args.task in ("mlm", "clm"):
        dataset: DatasetDict = get_sorel_dataset(SUBSET)
    elif args.task == "clf":
        dataset, dist = get_bodmas_dataset(SUBSET, BODMAS_TOP_K, BODMAS_MIN_FREQ)

    if args.do_tune:
        if isinstance(TUNE_TR_N_SAMPLES, int):
            if TUNE_TR_N_SAMPLES == 0:
                dataset.pop("tr")
            else:
                dataset["tr"] = dataset["tr"].select(range(TUNE_TR_N_SAMPLES))
        if isinstance(TUNE_VL_N_SAMPLES, int):
            if TUNE_VL_N_SAMPLES == 0:
                dataset.pop("vl")
            else:
                dataset["vl"] = dataset["vl"].select(range(TUNE_VL_N_SAMPLES))
        if isinstance(TUNE_TS_N_SAMPLES, int):
            if TUNE_TS_N_SAMPLES == 0:
                dataset.pop("ts")
            else:
                dataset["ts"] = dataset["ts"].select(range(TUNE_TS_N_SAMPLES))

    print(f"{dataset=}")
    print(f"{dist=}")
    print(BR, flush=True)

    if STREAMING:
        if training_arguments.max_steps == -1:
            max_steps = compute_total_steps(
                len(dataset["tr"]),
                training_arguments.num_train_epochs,
                training_arguments.per_device_train_batch_size,
                training_arguments.gradient_accumulation_steps,
            )
            assert isinstance(max_steps, int)
            training_arguments = replace(training_arguments, max_steps=max_steps)
        # For some reason, the intuitive way of creating a IterableDatasetDict causes issues.
        ds = IterableDatasetDict()
        num_shards = training_arguments.dataloader_num_workers
        if not training_arguments.dataloader_num_workers:
            num_shards = 1
        for split in dataset:
            ds[split] = dataset[split].to_iterable_dataset(num_shards)
        dataset = ds
        del ds, num_shards

    # CLM/MLM heads ignore classification-specific arugments.
    if args.task in ("mlm", "clm"):
        num_classes, id2label, label2id = None, {}, {}
    elif args.task == "clf":
        num_classes = dataset["tr"].info.features["labels"].num_classes
        id2label = {i: l for i, l in enumerate(dataset["tr"].info.features["labels"].names)}
        label2id = {l: i for i, l in enumerate(id2label.values())}
        dataset = dataset.rename_column("labels", "label")
        weight = tensor([1 / freq for freq in [dist[c] for c in label2id.keys()]])

    config = get_config(
        args.model_name_or_path,
        tokenizer,
        args.max_length,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
    )
    print(f"{config=}")
    print(BR, flush=True)

    if not RETURN_ATTENTION_MASK[object_to_model_name_or_path(config)]:
        tokenizer.model_input_names.remove("attention_mask")

    if PREPROCESS_AS_TEXT:
        function = partial(
            examples_to_text,
            max_length=args.max_length if args.task == "clf" else args.max_length * args.depth,
        )
    elif PREPROCESS_AS_INPUT_IDS:
        function = partial(
            examples_to_input_ids,
            max_length=args.max_length,
            do_pad=PREPROCESS_AS_INPUT_IDS_DO_PAD,
            pad_idx=tokenizer.pad_token_id,
            pad_to_length=args.max_length,
        )
    else:
        raise RuntimeError("Specify either `PREPROCESS_AS_TEXT` or `PREPROCESS_AS_INPUT_IDS`.")
    dataset = dataset.map(**get_map_kwds_for_hf_datasets(function, dataset))

    if PREPROCESS_AS_TEXT:
        remove_columns = ["name", "bytes", "size", "length", "text"]
        remove_columns = remove_columns + ["labels"] if args.task != "clf" else remove_columns
        function = partial(  # The partial function here is picky (`tokenizer` must be arg not kwd)
            tokenize_fn,
            tokenizer,
            truncation=True,
            max_length=args.max_length,
            return_overflowing_tokens=args.task in ("mlm", "clm"),
        )
        dataset = dataset.map(**get_map_kwds_for_hf_datasets(function, dataset, remove_columns=remove_columns))

    if EXIT_AFTER_MAP:
        sys.exit(0)

    pad_to_multiple_of = PAD_TO
    if isinstance(config, transformers.ReformerConfig):
        pad_to_multiple_of = _get_least_common_mult_chunk_len(config)

    if args.task == "clf":
        if PREPROCESS_AS_INPUT_IDS and PREPROCESS_AS_INPUT_IDS_DO_PAD:
            data_collator = DefaultDataCollator()
        else:
            data_collator = DataCollatorWithPadding(
                tokenizer=tokenizer, padding=True, pad_to_multiple_of=pad_to_multiple_of
            )
        compute_metrics = COMPUTE_METRICS
    elif args.task == "mlm":
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, pad_to_multiple_of=pad_to_multiple_of
        )
        compute_metrics = None
    elif args.task == "clm":
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False, pad_to_multiple_of=pad_to_multiple_of
        )
        compute_metrics = None

    print(f"{data_collator=}")
    print(f"{compute_metrics=}")
    print(BR, flush=True)

    callbacks = []
    if TYPE == "MC":
        callbacks.append(SaveConfigToCheckpointCallback())
    print(f"{callbacks=}")

    if args.task == "clf":
        ModelTrainer = partial(ImbalancedClassificationTrainer, weight=weight)
    elif args.task in ("mlm", "clm"):
        ModelTrainer = Trainer

    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
    if not STREAMING:  # dataset has already been processed, so we disable thread-based parallelism
        os.environ["TOKENIZERS_PARALLELISM"] = "false"


    if training_arguments.do_train:
        model = get_model(
            args.task,
            None,
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
        if isinstance(model, PreTrainedModel) and not isinstance(model, (MalConv, MalConvGCT, MyMalConv, RwkvPreTrainedModel)):
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

        oh.mkdir()
        trainer = ModelTrainer(
            model=model,
            args=training_arguments,
            train_dataset=dataset["tr"],
            eval_dataset=dataset["vl"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            callbacks=callbacks,
            compute_metrics=compute_metrics,
        )
        gc.collect()

        print("Training...")
        print(BR, flush=True)
        trainer.train(training_arguments.resume_from_checkpoint)

    if training_arguments.do_eval:
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

        if isinstance(compute_metrics, ComputeMetrics):
            compute_metrics.set_detailed(True)

        single_shot_classes = [label2id[l] for l in dist if dist[l] == 3]
        compute_metrics = partial(COMPUTE_METRICS, single_shot_classes=single_shot_classes)
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

        # compute_metrics = CLFComputeMetrics()
        # y_true, y_pred = compute_metrics.get_y_true_y_pred(output.predictions, output.label_ids)
        # np.savetxt(oh.test_predictions_file, y_pred, "%i")
        # np.savetxt(oh.test_labels_file, y_true, "%i")
        # if args.task == "clf":
        #     cf_matrix = confusion_matrix(y_true, y_pred)
        #     ConfusionMatrixDisplay(cf_matrix).plot()
        #     plt.savefig(oh.test_confusion_matrix_file)

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
        raise_on_failed_trial = False
        print(f"{search_alg=}")
        print(f"{scheduler=}")
        print(f"{TUNE_RESOURCES_PER_TRIAL=}")
        print(f"{raise_on_failed_trial=}")
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
                raise_on_failed_trial=raise_on_failed_trial,
            )
        except TuneError as err:
            print(BR, flush=True)
            print(
                f"Encountered a TuneError with {raise_on_failed_trial=}."
                "If raise_on_failed_trial=True or raise_on_failed_trial=None, then this indicates"
                "that at least one trial failed. To avoid this, set raise_on_failed_trial=False."
            )
            print(BR, flush=True)
            raise err
        except AttributeError as err:
            print(BR, flush=True)
            print(
                f"Encountered an AttributeError with {raise_on_failed_trial=}."
                "If raise_on_failed_trial=False, then this most likely indicates"
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
