"""
Train and evaluate the models for malware family classification.
"""

from dataclasses import dataclass, field, replace
from datetime import datetime
import json
import math
from pathlib import Path
from pprint import pformat, pprint
from typing import Callable, Optional, Protocol
import os
import sys
import warnings

from datasets import DatasetDict
import evaluate
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    BertConfig,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    HfArgumentParser,
    LongformerConfig,
    PretrainedConfig,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)

from cfg import BR
from helpers import OutputHelper, OutputArgs
from malconv import MalConvModel, MalConvConfig, MalConvTrainer
from tokenization import get_fast_tokenizer
from utils import count_parameters, get_scale_fn, pad_to_multiple_of_fn


PAD_TO = 8


@dataclass
class TrainArgs:
    scale: Optional[float] = field(default=None)
    scale_numerator: float = field(default=1.0)
    scale_denominator: float = field(default=1.0)
    early_stopping: bool = field(default=False)
    early_stopping_patience: int = field(default=10)
    early_stopping_threshold: float = field(default=0.0)

    def __post_init__(self) -> None:
        if self.scale is not None:
            self.scale = self.scale_numerator / self.scale_denominator


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
    scale: int = 1,
    **kwds,
) -> PretrainedConfig:
    """
    kwds
    ----
        num_labels (int): use for classification
    """
    if Path(model_name_or_path).exists():
        return AutoConfig.from_pretrained(model_name_or_path, **kwds)

    scale_fn = get_scale_fn(scale)

    if model_name_or_path == "longformer":
        attention_window = math.ceil(scale_fn(512) / 2.0) * 2
        return LongformerConfig(
            attention_window=attention_window,
            sep_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_hidden_layers=scale_fn(12),
            num_attention_heads=scale_fn(12),
            vocab_size=pad_to_multiple_of_fn(len(tokenizer), PAD_TO),
            hidden_size=pad_to_multiple_of_fn(scale_fn(768), PAD_TO),
            intermediate_size=pad_to_multiple_of_fn(scale_fn(3072), PAD_TO),
            max_position_embeddings=pad_to_multiple_of_fn(attention_window + max_length, PAD_TO),
            **kwds,
        )
    if model_name_or_path == "bert":
        return BertConfig(
            num_hidden_layers=scale_fn(12),
            num_attention_heads=scale_fn(12),
            vocab_size=pad_to_multiple_of_fn(len(tokenizer), PAD_TO),
            hidden_size=pad_to_multiple_of_fn(scale_fn(768), PAD_TO),
            intermediate_size=pad_to_multiple_of_fn(scale_fn(3072), PAD_TO),
            max_position_embeddings=max_length,
            pad_token_id=tokenizer.pad_token_id,
            classifier_dropout=None,
            **kwds,
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


def main(margs: OutputArgs, targs: TrainArgs, training_arguments: TrainingArguments) -> None:
    TYPE = get_model_type(margs.model)

    oh = OutputHelper(margs)
    print(f"{oh=}")
    print(BR, flush=True)

    training_arguments = replace(
        training_arguments,
        output_dir=oh.checkpoints_dir.as_posix(),
        load_best_model_at_end=True,
    )
    print(f"{training_arguments=}")
    print(BR, flush=True)

    model_name_or_path = margs.model
    if margs.pretrain_task is not None:
        model_name_or_path = OutputHelper(
            replace(margs, task=margs.pretrain_task)
        ).best_model_dir.as_posix()
    print(f"{model_name_or_path=}")
    print(BR, flush=True)

    tokenizer = get_fast_tokenizer(oh.tokenizer_file, margs.max_length)
    print(f"{tokenizer=}")
    print(BR, flush=True)

    dataset = DatasetDict.load_from_disk(oh.dataset_dir)
    num_classes = dataset["tr"].info.features["label"].num_classes
    print(f"{dataset=}")
    print(BR, flush=True)

    config = get_config(
        model_name_or_path,
        tokenizer,
        margs.max_length,
        margs.scale,
        num_labels=num_classes,
    )

    if margs.task == "clf":
        data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer, padding=True, pad_to_multiple_of=PAD_TO
        )
        compute_metrics = ClfComputeMetrics(num_classes > 2)
    elif margs.task == "mlm":
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, pad_to_multiple_of=PAD_TO
        )
        compute_metrics = None
    elif margs.task == "clm":
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False, pad_to_multiple_of=PAD_TO
        )
        compute_metrics = None

    callbacks = []
    if targs.early_stopping:
        callbacks.append(
            EarlyStoppingCallback(targs.early_stopping_patience, targs.early_stopping_threshold)
        )

    if TYPE == "HF":
        ModelTrainer = Trainer
    if TYPE == "MC":
        ModelTrainer = MalConvTrainer

    print(f"{config=}")
    print(f"{data_collator=}")
    print(f"{callbacks=}")
    print(BR, flush=True)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

    if training_arguments.do_train:
        oh.mkdir(exist_ok=True)

        if TYPE == "HF":
            model = AutoModelForSequenceClassification.from_config(config)
        if TYPE == "MC":
            model = MalConvModel(config)
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

        print(f"{model=}")
        print(f"{count_parameters(model)=}")
        print(BR, flush=True)

        trainer.train(training_arguments.resume_from_checkpoint)
        if training_arguments.load_best_model_at_end:
            model.save_pretrained(oh.best_model_dir.as_posix())
        with open(oh.log_history_file, "w") as fp:
            json.dump(trainer.state.log_history, fp, indent=4)

    if training_arguments.do_eval:
        if TYPE == "HF":
            model = AutoModelForSequenceClassification.from_pretrained(oh.best_model_dir.as_posix())
        if TYPE == "MC":
            model = MalConvModel(config)  # TODO: match design pattern as from_pretrained()
            model.load_state_dict(MalConvModel.get_state_dict(oh.best_model_dir))
        trainer = ModelTrainer(
            model=model,
            args=training_arguments,
            data_collator=data_collator,
            tokenizer=tokenizer,
            callbacks=callbacks,
            compute_metrics=compute_metrics,
        )

        print(f"{model=}")
        print(f"{count_parameters(model)=}")
        print(BR, flush=True)

        probas, labels, results = trainer.predict(dataset["ts"])
        predictions = probas.argmax(axis=1)
        cf_matrix = confusion_matrix(labels, predictions)
    
        np.savetxt(oh.test_probas_file, probas, "%f")
        np.savetxt(oh.test_predictions_file, predictions, "%i")
        np.savetxt(oh.test_labels_file, labels, "%i")
        ConfusionMatrixDisplay(cf_matrix).plot()
        plt.savefig(oh.test_confusion_matrix_file)
        with open(oh.test_results_file, "w") as fp:
            json.dump(results, fp, indent=4)


def cli():
    parser = HfArgumentParser((OutputArgs, TrainArgs, TrainingArguments))
    margs, targs, training_arguments = parser.parse_args_into_dataclasses()
    main(margs, targs, training_arguments)
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
