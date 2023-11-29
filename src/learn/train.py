"""
Train and evaluate the models for malware family classification.
"""

from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import datetime
from functools import partial
import gc
import json
import math
from pathlib import Path
from pprint import pformat, pprint
from typing import Any, Literal, NewType, Optional, TypeAlias
import os
import sys

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# pylint: disable=wrong-import-position

from datasets import (
    DatasetDict,
    Dataset,
    IterableDataset,
    IterableDatasetDict,
    concatenate_datasets,
)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import torch
from torch import tensor, Tensor
from torch.nn import CrossEntropyLoss
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
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
)
from transformers import TrainingArguments as HfTrainingArguments
from transformers.trainer_utils import BestRun, EvalPrediction, PredictionOutput
from transformers.models.reformer.modeling_reformer import _get_least_common_mult_chunk_len
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler

from src.cfg import BR, INPUT_PATH, OUTPUT_PATH
from src.data.loaders import get_sorel_dataset, get_bodmas_dataset
from src.data.utils import print_dataset
from src.learn.utils import (
    count_parameters,
    pad_to_multiple_of_fn,
    find_two_largest_factors,
    get_tokenizer_object,
    get_fast_tokenizer,
    preprocess_a,
    tokenize_fn,
    float_to_int,
    compute_total_steps,
)

# from src.malconv import MalConvModel, MalConvConfig, MalConvTrainer

# Absolutely bizarre, but the below causes and error when hyperparmaeter tuning with ray.
# `AttributeError: Can't get attribute 'MalConvConfig' on <module '__main__'`
# MalConvModel = NewType("MalConvModel", object)
# MalConvConfig = NewType("MalConvConfig", object)
# MalConvTrainer = NewType("MalConvTrainer", object)

# Pseduo type aliases for the MalConv classes.
MalConvModel: TypeAlias = PreTrainedModel
MalConvConfig: TypeAlias = PretrainedConfig
MalConvTrainer: TypeAlias = Trainer


PAD_TO = 8
HIDDEN_SIZE = 512
INTERMEDIATE_SIZE = 1024
NUM_HIDDEN_LAYERS = 4
NUM_ATTENTION_HEADS = 8
ATTENTION_WINDOW = 512
SUBSET = None
STREAMING = False
BODMAS_TOP_K = None
BODMAS_MIN_FREQ = None
DEPTH = 4
N_INITIAL_POINTS = 1
N_TRIALS = 2  # including the initial points


class TrainingArguments(HfTrainingArguments):
    def __init__(self, **kwds):
        do_eval = kwds.get("do_eval", False)
        super().__init__(**kwds)
        self.do_eval = do_eval


def hp_ray_space(trial: Any) -> dict[str, float | int]:  # pylint: disable=unused-argument
    """
    - hidden_size % num_attention_heads == 0
    """
    # L_LEARNING_RATE = 1e-5
    # U_LEARNING_RATE = 1e-3
    # L_WEIGHT_DECAY = 1e-5
    # U_WEIGHT_DECAY = 1e-2
    # L_INTERMEDIATE_SIZE = 512
    # U_INTERMEDIATE_SIZE = 2048
    # Q_INTERMEDIATE_SIZE = PAD_TO
    # L_NUM_HIDDEN_LAYERS = 1
    # U_NUM_HIDDEN_LAYERS = 5
    # Q_NUM_HIDDEN_LAYERS = 1
    # L_NUM_ATTENTION_HEADS = 2
    # U_NUM_ATTENTION_HEADS = 8
    # Q_NUM_ATTENTION_HEADS = 2
    # L_HIDDEN_SIZE = 240
    # U_HIDDEN_SIZE = 1032
    # Q_HIDDEN_SIZE = math.lcm(PAD_TO, *list(range(L_NUM_ATTENTION_HEADS, U_NUM_ATTENTION_HEADS + 1, Q_NUM_ATTENTION_HEADS)))
    # L_ATTENTION_WINDOW = 128
    # U_ATTENTION_WINDOW = 4096
    # Q_ATTENTION_WINDOW = 128
    # B_ATTENTION_WINDOW = 8

    # return {
    #     "learning_rate": tune.uniform(L_LEARNING_RATE, U_LEARNING_RATE),
    #     "weight_decay": tune.uniform(L_WEIGHT_DECAY, U_WEIGHT_DECAY),
    #     "hidden_size": tune.quniform(L_HIDDEN_SIZE, U_HIDDEN_SIZE, q=Q_HIDDEN_SIZE),
    #     "intermediate_size": tune.quniform(L_INTERMEDIATE_SIZE, U_INTERMEDIATE_SIZE, q=Q_INTERMEDIATE_SIZE),
    #     "num_hidden_layers": tune.qrandint(L_NUM_HIDDEN_LAYERS, U_NUM_HIDDEN_LAYERS, q=Q_NUM_HIDDEN_LAYERS),
    #     "num_attention_heads": tune.qrandint(L_NUM_ATTENTION_HEADS, U_NUM_ATTENTION_HEADS, q=Q_NUM_ATTENTION_HEADS),
    #     "attention_window": tune.qloguniform(L_ATTENTION_WINDOW, U_ATTENTION_WINDOW, q=Q_ATTENTION_WINDOW, base=B_ATTENTION_WINDOW),
    # }

    return {
        "learning_rate": tune.uniform(1e-5, 1e-3),
        "weight_decay": tune.uniform(1e-5, 1e-2),
        "warmup_steps": tune.choice([250, 500, 750, 1000]),
        "hidden_size": tune.choice([256, 512, 768, 1024]),
        "intermediate_size": tune.choice([512, 1024, 1536, 2048]),
        "num_hidden_layers": tune.choice([1, 2, 3, 4]),
        "num_attention_heads": tune.choice([1, 2, 4, 8]),
        "attention_window": tune.choice([128, 256, 512, 1024, 2048, 4096]),
    }


def hp_model_init(
    trial: Optional[dict[str, Any]],
    task: str,
    model_name_or_path: str,
    tokenizer: Optional[PreTrainedTokenizerFast] = None,
    max_length: Optional[int] = None,
    num_labels: Optional[int] = None,
    id2label: Optional[dict[int, str]] = None,
    label2id: Optional[dict[str, int]] = None,
) -> PreTrainedModel | MalConvModel:
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
    model = get_model(task, model_name_or_path, config, **kwds)
    if model is None:
        raise RuntimeError("Model is None for some reason.")
    return model


def hp_compute_objective(metrics: dict[str, float]) -> float:
    return metrics["eval_loss"]


RETURN_ATTENTION_MASK = {
    "longformer": True,
    "reformer": True,
    "nystromformer": True,
    "fnet": False,
}


def object_to_model_name_or_path(obj) -> str:
    if obj in ("longformer", "reformer", "nystromformer", "fnet"):
        return obj
    if isinstance(obj, (FNetConfig,)):
        return "fnet"
    if isinstance(obj, (NystromformerConfig,)):
        return "nystromformer"
    if isinstance(obj, (ReformerConfig,)):
        return "reformer"
    if isinstance(obj, (LongformerConfig,)):
        return "longformer"
    if isinstance(obj, (str | Path)) and Path(obj).exists():
        return object_to_model_name_or_path(AutoConfig.from_pretrained(str(obj)))
    raise RuntimeError()


def get_model_type(model: str) -> Literal["HF", "MC"]:
    if model in ("malconv", "malconv2", "malconvGCG"):
        return "MC"
    return "HF"


class ImbalancedClassificationTrainer(Trainer):
    def __init__(self, weight: Optional[Tensor] = None, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = CrossEntropyLoss(weight=weight)

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.label_smoother is not None or self.args.past_index >= 0:
            raise NotImplementedError()

        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits

        loss = self.loss_fn(logits.to("cpu"), labels.to("cpu")).to(logits.device)
        return (loss, outputs) if return_outputs else loss


@dataclass
class Args:
    model_name_or_path: str = field()
    max_length: int = field()
    task: str = field()
    root: Path = field(default=OUTPUT_PATH)
    do_tune: bool = field(default=False)


class OutputHelper:
    def __init__(
        self,
        model_name_or_path: str,
        max_length: int,
        task: str,
        root: Path,
    ) -> None:
        self.root = Path(root)
        self.path = self.root.joinpath(model_name_or_path, str(max_length), task)

    @property
    def best_model_dir(self) -> Path:
        return self.path / "best_model"

    @property
    def checkpoints_dir(self) -> Path:
        return self.path / "checkpoints"

    @property
    def log_history_file(self) -> Path:
        return self.path / "log_history.json"

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
    num_hidden_layers: int = NUM_HIDDEN_LAYERS,
    num_attention_heads: int = NUM_ATTENTION_HEADS,
    hidden_size: int = HIDDEN_SIZE,
    intermediate_size: int = INTERMEDIATE_SIZE,
    attention_window: int = ATTENTION_WINDOW,
    **kwds,
) -> PretrainedConfig:
    """
    kwds
    ----
        num_labels (int): use for classification
    """
    if Path(model_name_or_path).exists():
        return AutoConfig.from_pretrained(model_name_or_path, **kwds)

    num_hidden_layers = float_to_int(num_hidden_layers)
    num_attention_heads = float_to_int(num_attention_heads)
    hidden_size = float_to_int(hidden_size)
    intermediate_size = float_to_int(intermediate_size)
    attention_window = float_to_int(attention_window)

    vocab_size = pad_to_multiple_of_fn(len(tokenizer), PAD_TO)
    max_posititional_embeddings = pad_to_multiple_of_fn(max_length, 8)

    if model_name_or_path.lower() == "longformer":
        return LongformerConfig(
            vocab_size=vocab_size,
            max_position_embeddings=pad_to_multiple_of_fn(attention_window + max_length, PAD_TO),
            num_attention_heads=num_attention_heads,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            sep_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            attention_window=attention_window,
            **kwds,
        )
    if model_name_or_path.lower() == "reformer":
        return ReformerConfig(
            vocab_size=vocab_size,
            max_position_embeddings=max_length,
            num_attention_heads=num_attention_heads,
            hidden_size=hidden_size,
            feed_forward_size=intermediate_size,
            attn_layers=["local" if i % 2 == 0 else "lsh" for i in range(NUM_HIDDEN_LAYERS)],
            attention_head_size=64,
            axial_norm_std=1.0,
            axial_pos_embds=True,
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
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **kwds,
        )
    if model_name_or_path.lower() == "fnet":
        return FNetConfig(
            vocab_size=vocab_size,
            max_position_embeddings=max_posititional_embeddings,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **kwds,
        )
    if model_name_or_path.lower() == "malconv":
        return MalConvConfig(
            num_embd=vocab_size,
            embed_size=8,
            max_length=max_posititional_embeddings,
            window_size=512,
            hidden_size=512,
            pad_idx=tokenizer.pad_token_id,
            **kwds,
        )
    if model_name_or_path.lower() == "malconv2":
        raise NotImplementedError()
    if model_name_or_path.lower() == "malconvgcg":
        raise NotImplementedError()

    raise ValueError(f"Invalid model name or path: {model_name_or_path}")


def get_model(
    task: str,
    model_name_or_path: str,
    config: PretrainedConfig | MalConvConfig,
    **kwds,
) -> PreTrainedModel | MalConvModel:
    # Get model from disk
    if Path(model_name_or_path).exists():
        if task == "clf":
            if get_model_type(model_name_or_path) == "HF":
                return AutoModelForSequenceClassification.from_pretrained(
                    model_name_or_path, **kwds
                )
            if get_model_type(model_name_or_path) == "MC":
                # TODO: match design pattern as from_pretrained()
                raise NotImplementedError()
                # model = MalConvModel(config)
                # model.load_state_dict(MalConvModel.get_state_dict(model_name_or_path))
                # return model
        if task == "mlm":
            return AutoModelForMaskedLM.from_pretrained(model_name_or_path, **kwds)
        if task == "clm":
            return AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwds)
        raise RuntimeError()

    # Get model from config
    if task == "clf":
        if isinstance(config, PretrainedConfig):
            return AutoModelForSequenceClassification.from_config(config)
        if isinstance(config, MalConvConfig):
            return MalConvModel(config)
    if task == "mlm":
        return AutoModelForMaskedLM.from_config(config)
    if task == "clm":
        return AutoModelForCausalLM.from_config(config)
    raise RuntimeError()


def main(args: Args, training_arguments: TrainingArguments) -> None:
    TYPE = get_model_type(args.model_name_or_path)

    oh = OutputHelper(args.model_name_or_path, args.max_length, args.task, args.root)
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
        if args.do_tune:
            subset = SUBSET + 1
            vl_size = subset // 2
            ts_size = 1
        else:
            subset = SUBSET
            vl_size = None
            ts_size = None
        dataset: DatasetDict = get_sorel_dataset(subset=subset, vl_size=vl_size, ts_size=ts_size)
        del subset, vl_size, ts_size
    elif args.task == "clf":
        dataset, dist = get_bodmas_dataset(
            subset=SUBSET, top_k=BODMAS_TOP_K, min_freq=BODMAS_MIN_FREQ
        )
    print(f"{dataset=}")
    print(BR, flush=True)

    if STREAMING:
        if training_arguments.max_steps == -1:
            max_steps = compute_total_steps(
                len(dataset["tr"]),
                training_arguments.num_train_epochs,
                training_arguments.per_device_train_batch_size,
                training_arguments.gradient_accumulation_steps,
            )
            training_arguments = replace(training_arguments, max_steps=max_steps)
        # For some reason, the intuitive way of creating a IterableDatasetDict causes issues.
        ds = IterableDatasetDict()
        num_shards = training_arguments.dataloader_num_workers
        if not training_arguments.dataloader_num_workers:
            num_shards = 1
        ds["tr"] = dataset["tr"].to_iterable_dataset(num_shards)
        ds["ts"] = dataset["ts"].to_iterable_dataset(num_shards)
        ds["vl"] = dataset["vl"].to_iterable_dataset(num_shards)
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

    # Converts the first `max_length` "bytes" into a UTF-8 "text" column.
    # We trim the bytes initially to reduce the memory footprint when tokenizing.
    # Additional bytes are left (more than `args.mask_length`) for language modeling.
    dataset = dataset.map(
        partial(
            preprocess_a,
            max_length=args.max_length * DEPTH if args.task in ("mlm", "clm") else args.max_length,
        ),
        batched=True,
    )
    # Converts the "text" column into a "input_ids" column.
    # Additional rows are added for language modeling.
    dataset = dataset.map(
        partial(
            tokenize_fn,  # The partial function here is picky (`tokenizer` must be arg not kwd).
            tokenizer,
            truncation=True,
            max_length=args.max_length,
            return_overflowing_tokens=args.task in ("mlm", "clm"),
        ),
        batched=True,
        remove_columns=["name", "bytes", "labels", "size", "length", "text"],
    )

    pad_to_multiple_of = PAD_TO
    if isinstance(config, transformers.ReformerConfig):
        pad_to_multiple_of = _get_least_common_mult_chunk_len(config)

    if args.task == "clf":
        data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer, padding=True, pad_to_multiple_of=pad_to_multiple_of
        )
        compute_metrics = CLFComputeMetrics()
    elif args.task == "mlm":
        if TYPE != "HF":
            raise ValueError("Langauge modeling not supported for this model.")
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, pad_to_multiple_of=pad_to_multiple_of
        )
        compute_metrics = MLMComputeMetrics()
    elif args.task == "clm":
        if TYPE != "HF":
            raise ValueError("Langauge modeling not supported for this model.")
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False, pad_to_multiple_of=pad_to_multiple_of
        )
        compute_metrics = None

    # FIXME: figure out the OOM issues with the compute_metrics.
    compute_metrics = None

    print(f"{data_collator=}")
    print(BR, flush=True)

    callbacks = []  # [EarlyStoppingCallback(early_stopping_patience=5)]
    print(f"{callbacks=}")

    if TYPE == "HF":
        if args.task == "clf":
            ModelTrainer = partial(ImbalancedClassificationTrainer, weight=weight)
        elif args.task in ("mlm", "clm"):
            ModelTrainer = Trainer
    elif TYPE == "MC":
        ModelTrainer = MalConvTrainer

    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
    if not STREAMING:  # dataset has already been processed, so we disable thread-based parallelism
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
        print(f"{count_parameters(model)=}")

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
        gc.collect()  # TODO: determine if this is necessary
        print("Training...")
        trainer.train(training_arguments.resume_from_checkpoint)
        if training_arguments.load_best_model_at_end:
            model.save_pretrained(oh.best_model_dir.as_posix())
        with open(oh.log_history_file, "w") as fp:
            json.dump(trainer.state.log_history, fp, indent=4)

    if training_arguments.do_eval:
        if training_arguments.do_train and training_arguments.load_best_model_at_end:
            if TYPE == "MC":
                raise NotImplementedError()
            pass
        elif training_arguments.do_train:
            # TODO: added the num_labels, id2label, and label2id kwds here for consistency
            # with other parts of the code, but they may not be necessary or even cause errors.
            model = get_model(
                args.task,
                oh.best_model_dir,
                config,
                num_labels=num_classes,
                id2label=id2label,
                label2id=label2id,
            )
        else:
            # TODO: added the num_labels, id2label, and label2id kwds here for consistency
            # with other parts of the code, but they may not be necessary or even cause errors.
            model = get_model(
                args.task,
                args.model_name_or_path,
                config,
                num_labels=num_classes,
                id2label=id2label,
                label2id=label2id,
            )
        print(f"{model=}")
        print(f"{count_parameters(model)=}")

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
        compute_metrics.set_detailed(True)
        output: PredictionOutput = trainer.predict(dataset["ts"])

        results = output.metrics
        with open(oh.test_results_file, "w") as fp:
            json.dump(results, fp, indent=4)

        y_true, y_pred = compute_metrics.get_y_true_y_pred(output.predictions, output.label_ids)
        np.savetxt(oh.test_predictions_file, y_pred, "%i")
        np.savetxt(oh.test_labels_file, y_true, "%i")
        if args.task == "clf":
            cf_matrix = confusion_matrix(y_true, y_pred)
            ConfusionMatrixDisplay(cf_matrix).plot()
            plt.savefig(oh.test_confusion_matrix_file)

    if args.do_tune:
        training_arguments = replace(training_arguments, do_eval=True, evaluation_strategy="steps")
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
            args=training_arguments,
            train_dataset=dataset["tr"],
            eval_dataset=dataset["vl"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
            compute_metrics=compute_metrics,
        )

        search_alg = HyperOptSearch(
            space=None,
            metric="eval_loss",
            mode="min",
            n_initial_points=N_INITIAL_POINTS,
        )
        scheduler = ASHAScheduler(metric="eval_loss", mode="min")
        print(f"{search_alg=}")
        print(f"{scheduler=}")
        print(BR, flush=True)

        oh.mkdir()
        oh.tuning_results_dir.mkdir(exist_ok=True, parents=False)
        print("Tuning...")
        best_trial: BestRun = trainer.hyperparameter_search(
            hp_space=hp_ray_space,
            compute_objective=hp_compute_objective,
            n_trials=N_TRIALS,
            direction="minimize",
            backend="ray",
            hp_name=None,
            scheduler=scheduler,
            search_alg=search_alg,
            resources_per_trial={
                "cpu": len(os.sched_getaffinity(0)),
                "gpu": torch.cuda.device_count(),
            },
        )

        analysis: tune.ExperimentAnalysis = best_trial.run_summary
        analysis.dataframe().to_csv(oh.tuning_results_dir / "dataframe.csv")


def cli():
    parser = HfArgumentParser((Args, TrainingArguments))
    args, training_arguments = parser.parse_args_into_dataclasses()  # pylint: disable=unbalanced-tuple-unpacking
    main(args, training_arguments)
    print(f"ENDING @{datetime.now()}\n{BR}", flush=True)


if __name__ == "__main__":
    print(f"STARTING @{datetime.now()}\n{BR}", flush=True)
    print(f"{torch.backends.cudnn.enabled=}")
    cli()
    print(f"ENDING @{datetime.now()}\n{BR}", flush=True)
