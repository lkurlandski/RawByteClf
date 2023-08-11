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
    BertConfig,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    LongformerConfig,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)

from cfg import *
from helpers import OutputHelper
from malconv import MalConvModel, MalConvConfig, MalConvTrainer
from train import compute_metrics, get_config_hf, ModelArgs, CallbackArgs
from tokenization import get_fast_tokenizer
from utils import count_parameters


MLM: bool = True
AutoModelForLM: type = AutoModelForMaskedLM if MLM else AutoModelForCausalLM


def main(model_args: ModelArgs, callback_args: CallbackArgs, training_args: TrainingArguments):

    oh = OutputHelper(
        algorithm=model_args.algorithm,
        vocab_size=model_args.vocab_size,
        num_tok=model_args.num_tok,
        max_length=model_args.max_length,
        num=model_args.num,
        task="mlm" if MLM else "clm",
        model=model_args.model,
    )
    print(f"{oh=}")

    tokenizer = get_fast_tokenizer(oh.tokenizer_file, model_args.max_length)
    print(f"{tokenizer=}")
    print(BR, flush=True)

    dataset = DatasetDict.load_from_disk(oh.dataset_dir)
    dataset = dataset.remove_columns("label")
    print(f"{dataset=}")
    print(BR, flush=True)

    config = get_config_hf(
        model_args.model,
        tokenizer,
        model_args.max_length,
        model_args.scale,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=MLM)
    callbacks = []
    if callback_args.early_stopping:
        early_stopping_callback = EarlyStoppingCallback(
            callback_args.early_stopping_patience,
            callback_args.early_stopping_threshold,
        )
        callbacks.append(early_stopping_callback)

    model = AutoModelForLM.from_config(config)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["tr"],
        eval_dataset=dataset["vl"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=callbacks,
        # compute_metrics=compute_metrics,
    )

    print(f"{config=}")
    print(f"{model=}")
    print(f"{data_collator=}")
    print(f"{callbacks=}")
    print(f"{count_parameters(model)=}")
    print(BR, flush=True)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

    oh.model_dir.mkdir(exist_ok=True)
    oh.checkpoints_dir.mkdir(exist_ok=True)
    oh.best_model_dir.mkdir(exist_ok=True)

    if training_args.do_train:
        training_args.output_dir = oh.checkpoints_dir.as_posix()
        trainer.train(training_args.resume_from_checkpoint)
        if training_args.load_best_model_at_end:
            if isinstance(model, MalConvModel):
                warnings.warn("MalConvModel does not support load_best_model_at_end.")
            model.save_pretrained(oh.best_model_dir.as_posix())
        with open(oh.log_history_path, "w") as fp:
            json.dump(trainer.state.log_history, fp, indent=4)

    if training_args.do_eval:
        model = AutoModelForLM.from_pretrained(oh.best_model_dir.as_posix())
        trainer = Trainer(
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
