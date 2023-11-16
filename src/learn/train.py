"""
Train and evaluate the models for malware family classification.
"""

from dataclasses import dataclass, field, replace
from datetime import datetime
from functools import partial
import json
from pathlib import Path
from pprint import pformat, pprint
from typing import Optional
import os
import sys

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# pylint: disable=wrong-import-position

from datasets import DatasetDict, Dataset, concatenate_datasets
import evaluate
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
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
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)

from src.cfg import BR, INPUT_PATH, OUTPUT_PATH

# from src.malconv import MalConvModel, MalConvConfig, MalConvTrainer
from src.learn.utils import (
    count_parameters,
    pad_to_multiple_of_fn,
    find_two_largest_factors,
    get_tokenizer_object,
    get_fast_tokenizer,
    preprocess_a,
    tokenize_fn,
)


PAD_TO = 8
HIDDEN_SIZE = 512
INTERMEDIATE_SIZE = 1024
NUM_HIDDEN_LAYERS = 2
NUM_ATTENTION_HEADS = 8
SUBSET = 64


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


class ClfComputeMetrics:
    def __init__(self, multiclass: bool) -> None:
        self.accuracy = evaluate.load("accuracy")
        self.f1 = evaluate.load("f1")
        self.precision = evaluate.load("precision")
        self.recall = evaluate.load("recall")
        self.roc_auc_score = evaluate.load("roc_auc", "multiclass" if multiclass else None)

    def __call__(self, eval_pred) -> dict[str, float]:
        probas, labels = eval_pred
        probas: np.ndarray = probas.astype(np.float32)
        labels: np.ndarray = labels.astype(np.int64)
        probas = torch.softmax(torch.tensor(probas, dtype=torch.float32), dim=1).numpy()
        predictions = np.argmax(probas, axis=1)
        # fmt: off
        return {
            "accuracy": self.accuracy.compute(predictions=predictions, references=labels),
            "precision": self.precision.compute(predictions=predictions, references=labels),
            "recall": self.recall.compute(predictions=predictions, references=labels),
            "f1-weighted": self.f1.compute(predictions=predictions, references=labels, average="weighted"),
            "f1-macro": self.f1.compute(predictions=predictions, references=labels, average="macro"),
            "f1-micro": self.f1.compute(predictions=predictions, references=labels, average="micro"),
            "roc-auc-ovr": self.roc_auc_score.compute(prediction_scores=probas, references=labels, multi_class='ovr'),
            "roc-auc-ovo": self.roc_auc_score.compute(prediction_scores=probas, references=labels, multi_class='ovo'),
        }
        # fmt: on


def get_model_type(model: str) -> str:
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

    if model_name_or_path.lower() == "longformer":
        attention_window = 512
        return transformers.LongformerConfig(
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
        return transformers.ReformerConfig(
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
        )
    if model_name_or_path.lower() == "nystromformer":
        transformers.NystromformerConfig(
            vocab_size=vocab_size,
            max_position_embeddings=max_length,
            num_hidden_layers=NUM_HIDDEN_LAYERS,
            num_attention_heads=NUM_ATTENTION_HEADS,
            hidden_size=HIDDEN_SIZE,
            intermediate_size=INTERMEDIATE_SIZE,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    if model_name_or_path == "malconv":
        return MalConvConfig(
            num_embd=pad_to_multiple_of_fn(len(tokenizer), PAD_TO),
            embed_size=8,
            max_length=max_length,
            window_size=512,
            hidden_size=512,
            pad_idx=tokenizer.pad_token_id,
            **kwds,
        )
    if model_name_or_path == "malconv2":
        raise NotImplementedError()
    if model_name_or_path == "malconvGCG":
        raise NotImplementedError()

    raise ValueError(f"Invalid model name or path: {model_name_or_path}")


def main(args: Args, training_arguments: TrainingArguments) -> None:
    TYPE = get_model_type(args.model_name_or_path)

    oh = OutputHelper(args.model_name_or_path, args.max_length, args.task, args.root)
    print(f"{oh=}")
    print(BR, flush=True)

    training_arguments = replace(
        training_arguments,
        output_dir=oh.checkpoints_dir.as_posix(),
        load_best_model_at_end=True,
    )
    print(f"{training_arguments=}")
    print(BR, flush=True)

    tokenizer = get_tokenizer_object()
    tokenizer = get_fast_tokenizer(tokenizer, max_length=args.max_length, truncate=True)
    print(f"{tokenizer=}")
    print(BR, flush=True)

    dataset = concatenate_datasets([Dataset.load_from_disk(p) for p in INPUT_PATH.glob("sorel_*")])
    if SUBSET:
        dataset = dataset.select(range(SUBSET))

    dataset = dataset.train_test_split()
    dataset["ts"] = dataset.pop("test")
    _dataset = dataset.pop("train").train_test_split()
    dataset["tr"] = _dataset.pop("train")
    dataset["vl"] = _dataset.pop("test")
    del _dataset
    try:
        num_classes = dataset["tr"].info.features["label"].num_classes
    except KeyError:
        num_classes = 0

    if not SUBSET:
        for k in dataset:
            dataset[k] = dataset[k].to_iterable_dataset()

    dataset = dataset.map(preprocess_a, batched=True, remove_columns=["bytes"]).map(
        partial(tokenize_fn, tokenizer, truncation=True, max_length=args.max_length),
        batched=True,
        remove_columns=["text", "labels"],
    )
    print(f"{dataset=}")
    print(BR, flush=True)

    config = get_config(
        args.model_name_or_path,
        tokenizer,
        args.max_length,
        num_labels=num_classes,
    )
    print(f"{config=}")
    print(BR, flush=True)

    if args.task == "clf":
        data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer, padding=True, pad_to_multiple_of=PAD_TO
        )
        compute_metrics = ClfComputeMetrics(num_classes > 2)
    elif args.task == "mlm":
        if TYPE != "HF":
            raise ValueError("Langauge modeling not supported for this model.")
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, pad_to_multiple_of=PAD_TO
        )
        compute_metrics = None
    elif args.task == "clm":
        if TYPE != "HF":
            raise ValueError("Langauge modeling not supported for this model.")
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False, pad_to_multiple_of=PAD_TO
        )
        compute_metrics = None

    print(f"{data_collator=}")
    print(BR, flush=True)

    callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]
    print(f"{callbacks=}")

    if TYPE == "HF":
        ModelTrainer = Trainer
    elif TYPE == "MC":
        ModelTrainer = MalConvTrainer

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

    if training_arguments.do_train:
        if args.task == "clf":
            if TYPE == "HF":
                model = AutoModelForSequenceClassification.from_config(config)
            if TYPE == "MC":
                model = MalConvModel(config)
        elif args.task == "mlm":
            model = AutoModelForMaskedLM.from_config(config)
        elif args.task == "clm":
            model = AutoModelForCausalLM.from_config(config)
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

        print("Training...")
        trainer.train(training_arguments.resume_from_checkpoint)
        if training_arguments.load_best_model_at_end:
            model.save_pretrained(oh.best_model_dir.as_posix())
        with open(oh.log_history_file, "w") as fp:
            json.dump(trainer.state.log_history, fp, indent=4)

    if training_arguments.do_eval:
        if args.task == "clf":
            if TYPE == "HF":
                model = AutoModelForSequenceClassification.from_pretrained(
                    oh.best_model_dir.as_posix()
                )
            if TYPE == "MC":  # TODO: match design pattern as from_pretrained()
                model = MalConvModel(config)
                model.load_state_dict(MalConvModel.get_state_dict(oh.best_model_dir))
        elif args.task == "mlm":
            model = AutoModelForMaskedLM.from_pretrained(oh.best_model_dir.as_posix())
        elif args.task == "clm":
            model = AutoModelForCausalLM.from_pretrained(oh.best_model_dir.as_posix())
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
        probas, labels, results = trainer.predict(dataset["ts"])
        predictions = probas.argmax(axis=1)

        # FIXME: remove
        print(f"{probas.shape=} {probas=}")
        print(f"{labels.shape=} {labels=}")
        print(f"{results=}")
        sys.exit(0)
        np.savetxt(oh.test_probas_file, probas, "%f")
        np.savetxt(oh.test_predictions_file, predictions, "%i")
        np.savetxt(oh.test_labels_file, labels, "%i")

        cf_matrix = confusion_matrix(labels, predictions)
        ConfusionMatrixDisplay(cf_matrix).plot()
        plt.savefig(oh.test_confusion_matrix_file)
        with open(oh.test_results_file, "w") as fp:
            json.dump(results, fp, indent=4)


def cli():
    parser = HfArgumentParser((Args, TrainingArguments))
    args, training_arguments = parser.parse_args_into_dataclasses()
    main(args, training_arguments)
    print(f"ENDING @{datetime.now()}\n{BR}", flush=True)


def debug() -> None:
    pass


if __name__ == "__main__":
    print(f"STARTING @{datetime.now()}\n{BR}", flush=True)
    print(f"{torch.backends.cudnn.enabled=}")
    if len(sys.argv) == 1 or sys.argv[1] == "--debug":
        debug()
    else:
        cli()
    print(f"ENDING @{datetime.now()}\n{BR}", flush=True)
