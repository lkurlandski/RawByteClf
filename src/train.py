"""
Main.
"""

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
import gc
from itertools import chain
from pathlib import Path
from pprint import pformat, pprint
import random
import time
from typing import Optional
import os
import sys

from datasets import concatenate_datasets, Dataset, DatasetDict
import evaluate
import numpy as np
import torch
from transformers import (
    BertConfig,
    PreTrainedTokenizerFast,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    LongformerConfig,
    ReformerConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    PretrainedConfig,
)
from tqdm import tqdm

from cfg import *
from helpers import OutputHelper
from malconv import MalConvModel, MalConvConfig, MalConvTrainer
from tokenization import get_fast_tokenizer
from utils import count_parameters


accuracy = evaluate.load("accuracy")


@dataclass
class ModelArgs:
    max_length: int = field(metadata={"help": ""})
    model: str = field(metadata={"help": "One of `longformer`, `reformer`, ..., "})
    algorithm: str = field(metadata={"help": ""})
    vocab_size: Optional[int] = field(default=256, metadata={"help": ""})
    scale: float = field(default=1.0, metadata={"help": ""})
    num_tok: int = field(default=1000, metadata={"help": ""})
    num: float = field(default=1.0, metadata={"help": ""})


@dataclass
class CallbackArgs:
    early_stopping: bool = field(default=True, metadata={"help": ""})
    early_stopping_patience: Optional[int] = 5
    early_stopping_threshold: Optional[float] = 0.0


def compute_metrics(eval_pred):
    probas, labels = eval_pred
    probas = torch.softmax(torch.tensor(probas), dim=1).numpy()
    predictions = np.argmax(probas, axis=1)
    print(
        "\n",
        {
            "confs": np.mean(np.max(probas, axis=1)),
            "preds": Counter(predictions),
            "labels": Counter(labels),
        },
        sep="",
    )
    return accuracy.compute(predictions=predictions, references=labels)


def main(model_args: ModelArgs, callback_args: CallbackArgs, training_args: TrainingArguments):

    scale_fn = lambda x: int(round(x * model_args.scale))

    oh = OutputHelper(
        algorithm=model_args.algorithm,
        vocab_size=model_args.vocab_size,
        num_tok=model_args.num_tok,
        max_length=model_args.max_length,
        num=model_args.num,
        model=model_args.model,
    )

    tokenizer = get_fast_tokenizer(oh.tokenizer_file, model_args.max_length)
    print(f"{tokenizer=}")
    print(BR, flush=True)

    dataset = DatasetDict.load_from_disk(oh.dataset_dir)
    dataset["tr"].info.features["label"].num_classes = 10
    dataset["ts"].info.features["label"].num_classes = 10
    dataset["vl"].info.features["label"].num_classes = 10
    print(f"{dataset=}")
    print(BR, flush=True)

    ############################################################
    # c = Counter()
    # for s in dataset:
    #     for i in tqdm(range(dataset[s].num_rows)):
    #         c.update([dataset[s][i]["label"]])
    # print(f"{c=}")

    # def fn(example: dict):
    #     example = dict(example)
    #     example["label"] = example["label"] % 2
    #     return example
    # dataset = dataset.map(fn, batched=False)
    # dataset["tr"].info.features["label"].num_classes = 2
    # dataset["ts"].info.features["label"].num_classes = 2
    # dataset["vl"].info.features["label"].num_classes = 2
    # l = -1
    # for s in dataset:
    #     for i in tqdm(range(dataset[s].num_rows)):
    #         l = max(l, len(dataset[s][i]["input_ids"]))
    # print(f"{l=}")

    # c = Counter()
    # for s in dataset:
    #     for i in tqdm(range(dataset[s].num_rows)):
    #         c.update([dataset[s][i]["label"]])
    # print(f"{c=}")
    ############################################################

    if model_args.model == "longformer":
        attention_window = scale_fn(512)
        config = LongformerConfig(
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
            max_position_embeddings=attention_window + model_args.max_length,
            num_labels=dataset["tr"].info.features["label"].num_classes,
        )
    elif model_args.model == "bert":
        config = BertConfig(
            vocab_size=len(tokenizer),
            hidden_size=scale_fn(768),
            num_hidden_layers=scale_fn(12),
            num_attention_heads=scale_fn(12),
            intermediate_size=scale_fn(3072),
            max_position_embeddings=model_args.max_length,
            pad_token_id=tokenizer.pad_token_id,
            classifier_dropout=None,
            num_labels=dataset["tr"].info.features["label"].num_classes,
        )
    elif model_args.model == "malconv":
        config = MalConvConfig(
            num_embd=len(tokenizer),
            num_classes=dataset["tr"].info.features["label"].num_classes,
            pad_idx=tokenizer.pad_token_id,
            max_length=model_args.max_length,
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

    if training_args.do_train:
        oh.model_dir.mkdir(parents=True, exist_ok=True)
        training_args.output_dir = oh.model_dir.as_posix()
        trainer.train(training_args.resume_from_checkpoint)

    if training_args.do_eval:
        trainer.evaluate(dataset["ts"])


def debug():

    model_args = ModelArgs(
        vocab_size=1030,
        num=0.1,
        algorithm="SentencePieceBPE",
        model="longformer",
        max_length=1000000,
        scale=4,
    )
    callback_args = CallbackArgs()
    training_args = TrainingArguments(
        output_dir="tmp",
        overwrite_output_dir=True,
        load_best_model_at_end=True,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        dataloader_num_workers=1,
        num_train_epochs=2,
        per_device_eval_batch_size=8,
        per_device_train_batch_size=8,
        no_cuda=True,
        do_train=True,
    )
    main(model_args, callback_args, training_args)


def cli():
    parser = HfArgumentParser((ModelArgs, CallbackArgs, TrainingArguments))
    model_args, callback_args, training_args = parser.parse_args_into_dataclasses()
    print(f"model_args={pformat(model_args)}")
    print(f"callback_args={pformat(callback_args)}")
    print(f"training_args={pformat(training_args)}")
    print(BR, flush=True)
    main(model_args, callback_args, training_args)
    print(f"ENDING @{datetime.now()}\n{BR}", flush=True)


if __name__ == "__main__":
    print(f"STARTING @{datetime.now()}\n{BR}", flush=True)
    if len(sys.argv) == 1 or sys.argv[1] == "--debug":
        debug()
    else:
        cli()
    print(f"ENDING @{datetime.now()}\n{BR}", flush=True)
