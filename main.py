"""
Main.

2 ** 8 == 256
2 ** 9 == 512
2 ** 10 == 1024
2 ** 11 == 2048
2 ** 12 == 4096
2 ** 13 == 8192
2 ** 14 == 16384
2 ** 15 == 32768
2 ** 16 == 65536
2 ** 17 == 131072
2 ** 18 == 262144
2 ** 19 == 524288
2 ** 20 == 1048576
2 ** 21 == 2097152
2 ** 22 == 4194304
2 ** 23 == 8388608
2 ** 24 == 16777216
2 ** 25 == 33554432
2 ** 26 == 67108864
2 ** 27 == 134217728
2 ** 28 == 268435456
2 ** 29 == 536870912
2 ** 30 == 1073741824
2 ** 31 == 2147483648
2 ** 32 == 4294967296
"""

from dataclasses import dataclass, field
from datetime import datetime
from itertools import chain
from pprint import pformat
import time
from typing import Optional
import os
import sys

from datasets import Dataset
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

import data
import preprocessing
import tokenization
import utils


NUM_PROC: Optional[int] = 1
WRITER_BATCH_SIZE: Optional[int] = 1000

SPECIALS = {
    "unk_token": "<unk>",
    "pad_token": "<pad>",
    "mask_token": "<mask>",
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "cls_token": "<cls>",
}

BR = "|" + "-" * 88 + "|"


@dataclass
class TokenizerArguments:
    algorithm: str = field(metadata={"help": ""})
    vocab_size: Optional[int] = field(default=256, metadata={"help": ""})
    n_tok: Optional[int] = field(default=None, metadata={"help": ""})
    n_tok_examples: Optional[int] = field(default=None, metadata={"help": ""})
    use_saved_tokenizer: Optional[bool] = field(default=True, metadata={"help": ""})
    overwrite_saved_tokenizer: Optional[bool] = field(default=False, metadata={"help": ""})
    block_size: int = field(default=2**12, metadata={"help": ""})
    tok_batch_size: int = field(default=2**10, metadata={"help": ""})
    max_token_length: int = field(default=None, metadata={"help": ""})

    def __post_init__(self):
        self.vocab_size += len(SPECIALS)


@dataclass
class DatasetArguments:
    max_size: int = field(metadata={"help": ""})
    do_preprocess: bool = field(default=True, metadata={"help": ""})
    n_dat: int = field(default=None, metadata={"help": ""})
    min_bytes: int = field(default=1000, metadata={"help": ""})
    max_bytes: int = field(default=10**6, metadata={"help": ""})
    train_test_split: float = field(default=0.1, metadata={"help": ""})
    load_from_cache_file: bool = field(default=True, metadata={"help": ""})


@dataclass
class ModelArguments:
    model: str = field(metadata={"help": "One of `longformer`, `reformer`, ..., "})
    downscale: int = field(default=1, metadata={"help": ""})


@dataclass
class CallbackArguments:
    early_stopping_patience: Optional[int] = 5
    early_stopping_threshold: Optional[float] = 0.0


print(f"STARTING @{datetime.now()}\n{BR}", flush=True)


dataclasses = (
    TokenizerArguments,
    DatasetArguments,
    ModelArguments,
    CallbackArguments,
    TrainingArguments,
)
parser = HfArgumentParser(dataclasses)
args = parser.parse_args_into_dataclasses()
tokenizer_args = args[0]
dataset_args = args[1]
model_args = args[2]
callback_args = args[3]
training_args = args[4]

print(f"{dataset_args=}")
print(f"{tokenizer_args=}")
print(f"{model_args=}")
print(f"{training_args=}")
print(f"RAM={utils.process_mem('G')}")
print(BR, flush=True)

tokenization_dataset = Dataset.from_generator(
    data.MicrosoftDatasetGen(tokenizer_args.n_tok),
    num_proc=NUM_PROC,
    writer_batch_size=WRITER_BATCH_SIZE,
)
tokenization_dataset = tokenization_dataset.remove_columns(["label", "file"])
tokenization_dataset = tokenization_dataset.map(
    preprocessing.get_group_texts_fn(tokenizer_args.block_size),
    batched=True,
    num_proc=NUM_PROC,
    writer_batch_size=WRITER_BATCH_SIZE,
)

print(f"{tokenization_dataset=}")
print(f"{tokenization_dataset[0]['text'][0:16]=}")
print(f"RAM={utils.process_mem('G')}")
print(BR, flush=True)

print(f"GETTING TOKENIZER @{datetime.now()}\n{BR}", flush=True)
tokenizer = tokenization.get_tokenizer(
    tokenizer_args.vocab_size,
    tokenizer_args.algorithm,
    special_tokens=SPECIALS,
    dataset=tokenization_dataset,
    use_cache=tokenizer_args.use_saved_tokenizer,
    overwrite_cache=tokenizer_args.overwrite_saved_tokenizer,
    n_examples=tokenizer_args.n_tok_examples,
    batch_size=tokenizer_args.tok_batch_size,
    max_token_length=tokenizer_args.max_token_length,
)
fast_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    model_max_length=dataset_args.max_size,
)
fast_tokenizer.add_special_tokens(SPECIALS)

print(f"{tokenizer=}")
print(f"{fast_tokenizer=}")
print(f"RAM={utils.process_mem('G')}")
print(BR, flush=True)

if not dataset_args.do_preprocess:
    print(f"FINISHING @{datetime.now()}\n{BR}", flush=True)
    sys.exit(0)
print(f"TOKENIZING @{datetime.now()}\n{BR}", flush=True)

dataset = Dataset.from_generator(data.MicrosoftDatasetGen(dataset_args.n_dat))
tokenize_fn = preprocessing.get_tokenize_fn(fast_tokenizer, dataset_args.max_size, truncation=False)
dataset = dataset.map(
    tokenize_fn, batched=True, load_from_cache_file=dataset_args.load_from_cache_file
)
print(f"{dataset=}\n{BR}", flush=True)
dataset = dataset.filter(lambda example: len(example["input_ids"]) < dataset_args.max_size)
dataset_info = data.process_info(data.info(dataset))
print(f"{dataset=}\n{dataset_info=}\n{BR}", flush=True)

split_dataset = dataset.train_test_split(dataset_args.train_test_split)

if not training_args.do_train:
    print(f"FINISHING @{datetime.now()}\n{BR}", flush=True)
    sys.exit(0)
print(f"TRAINING @{datetime.now()}\n{BR}", flush=True)

if model_args.model == "longformer":
    config = LongformerConfig(
        attention_window=512 // model_args.downscale,
        sep_token_id=fast_tokenizer.sep_token_id,
        pad_token_id=fast_tokenizer.pad_token_id,
        bos_token_id=fast_tokenizer.bos_token_id,
        eos_token_id=fast_tokenizer.eos_token_id,
        vocab_size=len(fast_tokenizer),
        hidden_size=768 // model_args.downscale,
        num_hidden_layers=12 // model_args.downscale,
        num_attention_heads=12 // model_args.downscale,
        intermediate_size=3072 // model_args.downscale,
        max_position_embeddings=dataset_args.max_size,
    )
elif model_args.model == "reformer":
    config = ReformerConfig()
print(f"{config=}\n{'-' * 80}", flush=True)

model = AutoModelForSequenceClassification.from_config(config)
data_collator = DataCollatorWithPadding(tokenizer=fast_tokenizer)
callbacks = [
    EarlyStoppingCallback(
        callback_args.early_stopping_patience, callback_args.early_stopping_threshold
    )
]
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=split_dataset["train"],
    eval_dataset=split_dataset["test"],
    data_collator=data_collator,
    tokenizer=fast_tokenizer,
    callbacks=callbacks,
)

print(f"{model=}\n{'-' * 80}", flush=True)
print(f"{data_collator=}\n{'-' * 80}", flush=True)
print(f"{callbacks=}\n{'-' * 80}", flush=True)
print(f"{trainer=}\n{'-' * 80}", flush=True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

trainer.train()
