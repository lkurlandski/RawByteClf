"""
Utility functions for training and evaluation.
"""

from collections import OrderedDict
import functools
import gc
import inspect
import math
import sys
from typing import Any, Optional

from accelerate.utils.memory import should_reduce_batch_size
from accelerate.utils import is_xpu_available
from datasets import Dataset, IterableDataset, IterableDatasetDict, concatenate_datasets
from datasets.formatting.formatting import LazyBatch, LazyRow
import numpy as np
from tokenizers import models, pre_tokenizers, Tokenizer, Regex
import torch
from transformers import PreTrainedTokenizerFast

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


def preprocess_a(examples: Any, max_length: Optional[int] = None) -> dict:
    """This is about half the speed of preprocess_b, but lets us use the vast HF ecosystem."""
    return {
        "text": [b[0:max_length].decode("latin1") for b in examples["bytes"]],
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


def find_executable_batch_size_sub(function: callable = None, starting_batch_size: int = 128, subtract: int = 8):
    """
    Monkey patch for accelerate.utils.memory.find_executable_batch_size to subtract from
    the batch size rather than dividing by 2.

    >>> from accelerate.utils.memory import find_executable_batch_size
    >>> from src.learn.utils import find_executable_batch_size_sub
    >>> find_executable_batch_size_sub_8 = functools.partial(f, subtract=8)
    >>> accelerate.utils.memory.find_executable_batch_size = find_executable_batch_size_sub_8
    """

    if function is None:
        return functools.partial(
            find_executable_batch_size_sub,
            starting_batch_size=starting_batch_size,
            subtract=subtract,
        )

    batch_size = starting_batch_size

    def decorator(*args, **kwargs):
        nonlocal batch_size
        gc.collect()
        if not is_xpu_available():
            torch.cuda.empty_cache()
        else:
            torch.xpu.empty_cache()
        params = list(inspect.signature(function).parameters.keys())
        # Guard against user error
        if len(params) < (len(args) + 1):
            arg_str = ", ".join([f"{arg}={value}" for arg, value in zip(params[1:], args[1:])])
            raise TypeError(
                f"Batch size was passed into `{function.__name__}` as the first argument when called."
                f"Remove this as the decorator already does so: `{function.__name__}({arg_str})`"
            )
        while True:
            if batch_size == 0:
                raise RuntimeError("No executable batch size found, reached zero.")
            try:
                return function(batch_size, *args, **kwargs)
            except Exception as e:
                if should_reduce_batch_size(e):
                    gc.collect()
                    if not is_xpu_available():
                        torch.cuda.empty_cache()
                    else:
                        torch.xpu.empty_cache()
                    print(f"Failed with {batch_size=}. Trying batch_size={batch_size - subtract}.")
                    batch_size -= subtract
                else:
                    raise

    return decorator
