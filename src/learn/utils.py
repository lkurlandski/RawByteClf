"""
Utility functions for training and evaluation.
"""

from collections import OrderedDict
import math
from typing import Any

from datasets import Dataset, IterableDataset, IterableDatasetDict, concatenate_datasets
from datasets.formatting.formatting import LazyBatch, LazyRow
import numpy as np
from tokenizers import models, pre_tokenizers, Tokenizer, Regex
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    BertConfig,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    HfArgumentParser,
    LongformerConfig,
    PretrainedConfig,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)

from src.cfg import INPUT_PATH, OUTPUT_PATH
from src.utils import count_parameters


SPECIALS = OrderedDict(
    {
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "mask_token": "<msk>",
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "cls_token": "<cls>",
        "sep_token": "<sep>",
    }
)


def get_tokenizer_object() -> Tokenizer:
    alphabet = [bytes([i]).decode("latin1") for i in range(256)]
    vocab = {v: i for i, v in enumerate(SPECIALS.values())} | {
        v: i for i, v in enumerate(alphabet, start=len(SPECIALS))
    }

    model = models.WordLevel(
        vocab=vocab,
        unk_token=SPECIALS["unk_token"],
    )
    tokenizer = Tokenizer(model)
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex("."), behavior="isolated"),
        ]
    )
    return tokenizer


def get_fast_tokenizer(tokenizer_object: Tokenizer, **kwds) -> PreTrainedTokenizerFast:
    """Suggested kwds are max_length."""
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_object, **(kwds | SPECIALS))
    tokenizer.add_special_tokens(SPECIALS)
    return tokenizer


def preprocess_a(examples: Any) -> dict:
    """This is about half the speed of preprocess_b, but lets us use the vast HF ecosystem."""
    return {
        "text": [b.decode("latin1") for b in examples["bytes"]],
    }


def preprocess_b(examples: Any) -> dict:
    return {
        "input_ids": [np.frombuffer(b, dtype=np.uint8) for b in examples["bytes"]],
    }


def tokenize_fn(tokenizer: PreTrainedTokenizerFast, examples: Any, **kwds) -> dict:
    """
    Suggested kwds include truncation=True, max_length=max_length, etc.
    """
    return tokenizer(examples["text"], **kwds)


def find_two_largest_factors(number: int) -> tuple[int, int]:
    if (s := math.sqrt(number)).is_integer():
        return int(s), int(s)

    factors = []
    for i in range(int(s), 0, -1):
        if number % i == 0:
            factors.append(i)
            if number // i != i:
                factors.append(number // i)
            if len(factors) >= 2:
                return factors


def pad_to_multiple_of_fn(val: int, pad_to_multiple_of: int = 1) -> int:
    q, r = divmod(val, pad_to_multiple_of)
    if r == 0:
        return val
    return (q + 1) * pad_to_multiple_of
