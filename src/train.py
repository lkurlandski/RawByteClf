"""
Train and evaluate the models for malware family classification.
"""

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
from pprint import pformat, pprint
from typing import Optional
import os
import sys
import warnings

from datasets import concatenate_datasets, Dataset, DatasetDict
import evaluate
import numpy as np
import torch
from transformers import (
    AutoConfig,
    BertConfig,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    LongformerConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)

from cfg import *
from helpers import OutputHelper
from malconv import MalConvModel, MalConvConfig, MalConvTrainer
from tokenization import get_fast_tokenizer
from utils import count_parameters


accuracy = evaluate.load("accuracy")


@dataclass
class ModelArgs:
    max_length: int = field(metadata={"help": ""})
    model: str = field(metadata={"help": "One of `longformer`, `bert`, `malconv`."})
    algorithm: str = field(metadata={"help": ""})
    vocab_size: Optional[int] = field(default=256, metadata={"help": ""})
    scale: float = field(default=1.0, metadata={"help": ""})
    num_tok: int = field(default=1000, metadata={"help": ""})
    num: float = field(default=1.0, metadata={"help": ""})
    task: Optional[str] = field(
        default="None",
        metadata={"help": ("One of `clm` or `mlm`. " "Not used if training for classification.")},
    )
    pretrain_task: str = field(
        default="clf",
        metadata={
            "help": (
                "One of `clf`, `clm`, or `mlm`. "
                "If not `clf`, will be used to search for pretrained model."
            )
        },
    )


@dataclass
class CallbackArgs:
    early_stopping: bool = field(default=True, metadata={"help": ""})
    early_stopping_patience: Optional[int] = 5
    early_stopping_threshold: Optional[float] = 0.0


def compute_metrics(eval_pred):
    probas, labels = eval_pred
    probas: np.ndarray = probas.astype(np.float32)
    labels: np.ndarray = labels.astype(np.int64)
    probas = torch.softmax(torch.tensor(probas, dtype=torch.float32), dim=1).numpy()
    predictions = np.argmax(probas, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def get_config_malconv(
    _,
    tokenizer: Optional[PreTrainedTokenizerFast],
    max_length: Optional[int] = None,
    scale: int = 1,
    num_labels: Optional[int] = None,
) -> MalConvConfig:
    scale_fn = lambda x: int(round(x * scale))

    return MalConvConfig(
        num_embd=len(tokenizer),
        embed_size=scale_fn(8),
        max_length=max_length,
        window_size=scale_fn(512),
        hidden_size=scale_fn(128),
        num_classes=num_labels,
        pad_idx=tokenizer.pad_token_id,
    )


def get_config_hf(
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

    scale_fn = lambda x: int(round(x * scale))

    if model_name_or_path == "longformer":
        attention_window = scale_fn(512)
        return LongformerConfig(
            attention_window=attention_window + attention_window % 2,
            sep_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            vocab_size=len(tokenizer),
            hidden_size=scale_fn(768),
            num_hidden_layers=scale_fn(12),
            num_attention_heads=scale_fn(12),
            intermediate_size=scale_fn(3072),
            max_position_embeddings=attention_window + max_length,
            **kwds,
        )
    if model_name_or_path == "bert":
        return BertConfig(
            vocab_size=len(tokenizer),
            hidden_size=scale_fn(768),
            num_hidden_layers=scale_fn(12),
            num_attention_heads=scale_fn(12),
            intermediate_size=scale_fn(3072),
            max_position_embeddings=max_length,
            pad_token_id=tokenizer.pad_token_id,
            classifier_dropout=None,
            **kwds,
        )
    raise ValueError(f"Invalid model name or path: {model_name_or_path}")


def main(model_args: ModelArgs, callback_args: CallbackArgs, training_args: TrainingArguments):
    oh = OutputHelper(
        algorithm=model_args.algorithm,
        vocab_size=model_args.vocab_size,
        num_tok=model_args.num_tok,
        max_length=model_args.max_length,
        task="clf",
        num=model_args.num,
        model=model_args.model,
        pretrain_task=model_args.pretrain_task,
    )

    if model_args.pretrain_task is not None and model_args.pretrain_task != "clf":
        model_name_or_path = OutputHelper(
            algorithm=model_args.algorithm,
            vocab_size=model_args.vocab_size,
            num_tok=model_args.num_tok,
            max_length=model_args.max_length,
            task=model_args.pretrain_task,
            num=model_args.num,
            model=model_args.model,
            pretrain_task=model_args.pretrain_task,
        ).best_model_dir
        if not Path(model_name_or_path).exists():
            raise FileNotFoundError(model_name_or_path)
    else:
        model_name_or_path = model_args.model

    tokenizer = get_fast_tokenizer(oh.tokenizer_file, model_args.max_length)
    print(f"{tokenizer=}")
    print(BR, flush=True)

    dataset = DatasetDict.load_from_disk(oh.dataset_dir)
    dataset["tr"].info.features["label"].num_classes = 10
    dataset["ts"].info.features["label"].num_classes = 10
    dataset["vl"].info.features["label"].num_classes = 10
    print(f"{dataset=}")
    print(BR, flush=True)

    if model_args.model == "malconv":
        config = get_config_malconv(
            model_args.model,
            tokenizer,
            model_args.max_length,
            model_args.scale,
            num_labels=dataset["tr"].info.features["label"].num_classes,
        )
    else:
        config = get_config_hf(
            model_name_or_path,
            tokenizer,
            model_args.max_length,
            model_args.scale,
            num_labels=dataset["tr"].info.features["label"].num_classes,
        )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
    )
    callbacks = []
    if callback_args.early_stopping:
        early_stopping_callback = EarlyStoppingCallback(
            callback_args.early_stopping_patience,
            callback_args.early_stopping_threshold,
        )
        callbacks.append(early_stopping_callback)

    if isinstance(config, PretrainedConfig):
        model = AutoModelForSequenceClassification.from_config(config)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["tr"],
            eval_dataset=dataset["vl"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            callbacks=callbacks,
            compute_metrics=compute_metrics,
        )
    else:
        model = MalConvModel(config)
        trainer = MalConvTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["tr"],
            eval_dataset=dataset["vl"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            callbacks=callbacks,
            compute_metrics=compute_metrics,
        )

    print(f"{config=}")
    print(f"{model=}")
    print(f"{data_collator=}")
    print(f"{callbacks=}")
    print(f"{count_parameters(model)=}")
    print(BR, flush=True)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

    oh.clf_model_dir.mkdir(exist_ok=True)
    oh.model_dir.mkdir(exist_ok=True)
    oh.checkpoints_dir.mkdir(exist_ok=True)
    oh.best_model_dir.mkdir(exist_ok=True)

    if training_args.do_train:
        training_args.output_dir = oh.checkpoints_dir.as_posix()
        trainer.train(training_args.resume_from_checkpoint)
        if training_args.load_best_model_at_end:
            model.save_pretrained(oh.best_model_dir.as_posix())
        with open(oh.log_history_path, "w") as fp:
            json.dump(trainer.state.log_history, fp, indent=4)

    if training_args.do_eval:
        if isinstance(model, PreTrainedModel):
            model = AutoModelForSequenceClassification.from_pretrained(oh.best_model_dir.as_posix())
            trainer = Trainer(
                model=model,
                args=training_args,
                data_collator=data_collator,
                tokenizer=tokenizer,
                callbacks=callbacks,
                compute_metrics=compute_metrics,
            )
        else:  # TODO: match design pattern as from_pretrained(), ie, return new model
            model.load_state_dict(MalConvModel.get_state_dict(oh.best_model_dir))
            trainer = MalConvTrainer(
                model=model,
                args=training_args,
                data_collator=data_collator,
                tokenizer=tokenizer,
                callbacks=callbacks,
                compute_metrics=compute_metrics,
            )

        results = trainer.evaluate(dataset["ts"])
        with open(oh.test_results_path, "w") as fp:
            json.dump(results, fp, indent=4)


def cli():
    parser = HfArgumentParser((ModelArgs, CallbackArgs, TrainingArguments))
    model_args, callback_args, training_args = parser.parse_args_into_dataclasses()
    if training_args.dataloader_num_workers and training_args.dataloader_num_workers < 0:
        training_args.dataloader_num_workers = int(
            len(os.sched_getaffinity(0)) // abs(training_args.dataloader_num_workers)
        )
    assert training_args.load_best_model_at_end

    print(f"model_args={pformat(model_args)}")
    print(f"callback_args={pformat(callback_args)}")
    print(f"training_args={pformat(training_args)}")
    print(BR, flush=True)
    main(model_args, callback_args, training_args)
    print(f"ENDING @{datetime.now()}\n{BR}", flush=True)


def debug() -> None:
    pass


if __name__ == "__main__":
    print(f"STARTING @{datetime.now()}\n{BR}", flush=True)
    if len(sys.argv) == 1 or sys.argv[1] == "--debug":
        debug()
    else:
        cli()
    print(f"ENDING @{datetime.now()}\n{BR}", flush=True)
