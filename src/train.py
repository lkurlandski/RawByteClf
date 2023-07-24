"""
Main.
"""

from dataclasses import dataclass, field
from datetime import datetime
import gc
from itertools import chain
from pathlib import Path
from pprint import pformat, pprint
import time
from typing import Optional
import os
import sys

from datasets import Dataset, DatasetDict
import torch
from transformers import (
    PreTrainedTokenizerFast,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    LongformerConfig,
    ReformerConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from tqdm import tqdm

from cfg import *
from preprocessing import datasets_path
from tokenization import get_fast_tokenizer, tokenizer_path
from utils import count_parameters


@dataclass
class ModelArgs:
    max_length: int = field(metadata={"help": ""})
    model: str = field(metadata={"help": "One of `longformer`, `reformer`, ..., "})
    algorithm: str = field(metadata={"help": ""})
    vocab_size: Optional[int] = field(default=256, metadata={"help": ""})
    scale: int = field(default=1, metadata={"help": ""})
    num_tok: int = field(default=1000, metadata={"help": ""})
    num: float = field(default=None, metadata={"help": ""})


@dataclass
class CallbackArgs:
    early_stopping_patience: Optional[int] = 5
    early_stopping_threshold: Optional[float] = 0.0


def model_path(model: str, algorithm: str, vocab_size: int, num: Optional[float]) -> Path:
    return MODELS / model / algorithm / str(vocab_size) / (str(num) if num is not None else "full")


def main(model_args: ModelArgs, callback_args: CallbackArgs, training_args: TrainingArguments):
    training_args.output_dir = model_path(
        model_args.model, model_args.algorithm, model_args.vocab_size, model_args.num
    )
    tokenizer = get_fast_tokenizer(
        tokenizer_path(model_args.algorithm, model_args.vocab_size, model_args.num_tok),
        model_args.max_length,
    )
    print(f"{tokenizer=}")
    print(BR, flush=True)

    dataset = DatasetDict.load_from_disk(
        datasets_path(model_args.algorithm, model_args.vocab_size, model_args.num)
    )
    print(f"{dataset=}")
    print(BR, flush=True)

    for i, d in tqdm(enumerate(dataset["tr"]), total=int(dataset["tr"].num_rows)):
        x = d["input_ids"]
        if max(x) >= model_args.vocab_size:
            raise ValueError(f"{i=} {max(x)=}")
        if min(x) < 0:
            raise ValueError(f"{i=} {min(x)=}")
        if len(x) > 10**6:
            raise ValueError(f"{i=} {len(x)=}")

    if model_args.model == "longformer":
        config = LongformerConfig(
            attention_window=int(512 // model_args.scale),
            sep_token_id=tokenizer.sep_token_id,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            vocab_size=len(tokenizer),
            hidden_size=int(768 // model_args.scale),
            num_hidden_layers=int(12 // model_args.scale),
            num_attention_heads=int(12 // model_args.scale),
            intermediate_size=int(3072 // model_args.scale),
            max_position_embeddings=model_args.max_length,
            num_labels=10,
        )
    elif model_args.model == "reformer":
        raise NotImplementedError()
        config = ReformerConfig(
            attention_head_size=64,
            attn_layers=["local", "lsh", "local", "lsh", "local", "lsh"],
            axial_norm_std=1.0,
            axial_pos_embds=True,
            axial_pos_shape=[2**8, 2**8],
            axial_pos_embds_dim=[64, 192],
            chunk_size_lm_head=0,
            eos_token_id=2,
            feed_forward_size=512,
            hash_seed=None,
            hidden_act="relu",
            hidden_dropout_prob=0.05,
            hidden_size=256,
            initializer_range=0.02,
            is_decoder=False,
            layer_norm_eps=1e-12,
            local_num_chunks_before=1,
            local_num_chunks_after=0,
            local_attention_probs_dropout_prob=0.05,
            local_attn_chunk_length=64,
            lsh_attn_chunk_length=64,
            lsh_attention_probs_dropout_prob=0.0,
            lsh_num_chunks_before=1,
            lsh_num_chunks_after=0,
            max_position_embeddings=4096,
            num_attention_heads=12,
            num_buckets=None,
            num_hashes=1,
            pad_token_id=0,
            vocab_size=320,
            tie_word_embeddings=False,
            use_cache=True,
            classifier_dropout=None,
        )

    model = AutoModelForSequenceClassification.from_config(config)
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
        max_length=model_args.max_length,
    )
    callbacks = [
        EarlyStoppingCallback(
            callback_args.early_stopping_patience, callback_args.early_stopping_threshold
        )
    ]
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["tr"],
        eval_dataset=dataset["vl"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=callbacks,
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
        trainer.train(training_args.resume_from_checkpoint)


def cli():
    print(f"STARTING @{datetime.now()}\n{BR}", flush=True)
    parser = HfArgumentParser((ModelArgs, CallbackArgs, TrainingArguments))
    model_args, callback_args, training_args = parser.parse_args_into_dataclasses()
    print(f"model_args={pformat(model_args)}")
    print(f"callback_args={pformat(callback_args)}")
    print(f"training_args={pformat(training_args)}")
    print(BR, flush=True)
    main(model_args, callback_args, training_args)
    print(f"ENDING @{datetime.now()}\n{BR}", flush=True)


if __name__ == "__main__":
    cli()
