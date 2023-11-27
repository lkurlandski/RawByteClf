"""
Train and evaluate the models for malware family classification.
"""

from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import datetime
from functools import partial
import gc
import json
from pathlib import Path
from pprint import pformat, pprint
from typing import Literal, NewType, Optional
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
    TrainingArguments,
    LongformerConfig,
    ReformerConfig,
    NystromformerConfig,
    FNetConfig,
)
from transformers.trainer_utils import EvalPrediction, PredictionOutput
from transformers.models.reformer.modeling_reformer import _get_least_common_mult_chunk_len

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
)

# from src.malconv import MalConvModel, MalConvConfig, MalConvTrainer
MalConvModel = NewType("MalConvModel", object)
MalConvConfig = NewType("MalConvConfig", object)
MalConvTrainer = NewType("MalConvTrainer", object)


PAD_TO = 8
HIDDEN_SIZE = 512
INTERMEDIATE_SIZE = 1024
NUM_HIDDEN_LAYERS = 4
NUM_ATTENTION_HEADS = 8
SUBSET = None
STREAMING = True
BODMAS_TOP_K = None
BODMAS_MIN_FREQ = None
DEPTH = 4


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


class ImbalancedClassificationTrainer(Trainer):
    def __init__(self, weight: Optional[Tensor] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
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


def get_model_type(model: str) -> Literal["HF", "MC"]:
    if model in ("malconv", "malconv2", "malconvGCG"):
        return "MC"
    return "HF"


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

    vocab_size = pad_to_multiple_of_fn(len(tokenizer), PAD_TO)
    max_posititional_embeddings = pad_to_multiple_of_fn(max_length, 8)

    if model_name_or_path.lower() == "longformer":
        attention_window = 512
        return LongformerConfig(
            attention_window=attention_window,
            sep_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_hidden_layers=NUM_HIDDEN_LAYERS,
            num_attention_heads=NUM_ATTENTION_HEADS,
            vocab_size=vocab_size,
            hidden_size=HIDDEN_SIZE,
            intermediate_size=INTERMEDIATE_SIZE,
            max_position_embeddings=pad_to_multiple_of_fn(attention_window + max_length, PAD_TO),
            **kwds,
        )
    if model_name_or_path.lower() == "reformer":
        return ReformerConfig(
            vocab_size=vocab_size,
            max_position_embeddings=max_length,
            num_attention_heads=NUM_ATTENTION_HEADS,
            hidden_size=HIDDEN_SIZE,
            feed_forward_size=INTERMEDIATE_SIZE,
            attention_head_size=64,
            attn_layers=["local", "lsh"],
            axial_norm_std=1.0,
            axial_pos_embds=True,
            axial_pos_shape=list(find_two_largest_factors(max_length)),
            axial_pos_embds_dim=[HIDDEN_SIZE // 2, HIDDEN_SIZE // 2],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **kwds,
        )
    if model_name_or_path.lower() == "nystromformer":
        return NystromformerConfig(
            vocab_size=vocab_size,
            max_position_embeddings=max_posititional_embeddings,
            num_hidden_layers=NUM_HIDDEN_LAYERS,
            num_attention_heads=NUM_ATTENTION_HEADS,
            hidden_size=HIDDEN_SIZE,
            intermediate_size=INTERMEDIATE_SIZE,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **kwds,
        )
    if model_name_or_path.lower() == "fnet":
        return FNetConfig(
            vocab_size=vocab_size,
            max_position_embeddings=max_posititional_embeddings,
            num_hidden_layers=NUM_HIDDEN_LAYERS,
            num_attention_heads=NUM_ATTENTION_HEADS,
            hidden_size=HIDDEN_SIZE,
            intermediate_size=INTERMEDIATE_SIZE,
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


def get_model_from_disk(
    task: str,
    model_name_or_path: str,
    **kwds,
) -> PreTrainedModel | MalConvModel:
    if task == "clf":
        if get_model_type(model_name_or_path) == "HF":
            return AutoModelForSequenceClassification.from_pretrained(model_name_or_path, **kwds)
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


def get_model_from_config(
    task: str, config: PretrainedConfig | MalConvConfig
) -> PreTrainedModel | MalConvModel:
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


def get_model(
    task: str, model_name_or_path: str, config: PretrainedConfig | MalConvConfig, **kwds
) -> PreTrainedModel | MalConvModel:
    if Path(model_name_or_path).exists():
        return get_model_from_disk(task, model_name_or_path, **kwds)
    return get_model_from_config(task, config)


def compute_total_steps(n_samples: int, n_epochs: int, batch_size: int, n_accumulation: int) -> int:
    q, r = divmod(n_samples * n_epochs, batch_size * n_accumulation)
    if r == 0:
        return q
    return q + 1


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
        dataset: DatasetDict = get_sorel_dataset(subset=SUBSET)
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
        del ds

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
            #compute_metrics=compute_metrics,
        )
        gc.collect()
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
            model = get_model(args.task, oh.best_model_dir, config)
        else:
            model = get_model(args.task, args.model_name_or_path, config)
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


def cli():
    parser = HfArgumentParser((Args, TrainingArguments))
    args, training_arguments = parser.parse_args_into_dataclasses()
    main(args, training_arguments)
    print(f"ENDING @{datetime.now()}\n{BR}", flush=True)


if __name__ == "__main__":
    print(f"STARTING @{datetime.now()}\n{BR}", flush=True)
    print(f"{torch.backends.cudnn.enabled=}")
    cli()
    print(f"ENDING @{datetime.now()}\n{BR}", flush=True)
