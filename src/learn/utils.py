"""
Utility functions for training and evaluation.
"""

# pylint: disable=wrong-import-position
print(f"Entered {__file__=}")
# pylint: enable=wrong-import-position

from collections import Counter
import functools
import gc
import inspect
import math
import os
from pprint import pprint
import random
import sys
from typing import Any, Optional

from accelerate.utils.memory import should_reduce_batch_size
from accelerate.utils import is_xpu_available
import numpy as np
import torch
from transformers import PreTrainedTokenizerFast


def examples_to_text(
    examples: dict[str, list], max_length: Optional[int] = None
) -> dict[str, list]:
    return {"text": [b[0:max_length].decode("latin1") for b in examples["bytes"]]}


def examples_to_input_ids(
    examples: dict[str, list],
    max_length: int = sys.maxsize,
    do_pad: bool = True,
    pad_idx: int = -1,
    pad_to_length: Optional[int] = None,
) -> dict[str, list]:
    if do_pad and pad_to_length is None:
        pad_to_length = max(len(b) for b in examples["bytes"])
        pad_to_length = min(pad_to_length, max_length)

    r = {"input_ids": []}
    for b in examples["bytes"]:
        b = b[0:max_length]
        x = np.frombuffer(b, dtype=np.uint8)
        if do_pad:
            p = np.full(pad_to_length - len(b), pad_idx, dtype=np.uint8)
            x = np.concatenate((x, p))
        r["input_ids"].append(x)
    return r


def oversample_based_on_label(
    examples: dict[str, list[Any]], probabilities: list[float]
) -> dict[str, list[Any]]:
    """Over sampling to approximately match the target class distribution.

    Args:
        examples: A dictionary of lists of examples.
        probabilities: The class distribution to match.
    """

    total_samples = len(examples["label"])
    oversampled_examples = {key: [] for key in examples.keys()}
    label_counter = Counter(examples["label"])

    for label in label_counter.keys():
        p = probabilities[label]

        label_count = label_counter[label]
        oversample_count = int((p * total_samples - label_count) / (1 - p))

        label_indices = [i for i, value in enumerate(examples["label"]) if value == label]
        oversampled_indices = random.choices(label_indices, k=oversample_count)

        for key, values in examples.items():
            oversampled_examples[key].extend([values[i] for i in oversampled_indices])

    examples = {
        k: (examples[k] + oversampled_examples[k])
        if isinstance(examples[k], list)
        else np.concatenate((examples[k], oversampled_examples[k]))
        for k in examples
    }

    return examples


def tokenize_fn(tokenizer: PreTrainedTokenizerFast, examples: Any, **kwds) -> dict[str, list]:
    """
    Suggested kwds include truncation=True, max_length=max_length, return_overflowing_tokens=True...
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


def find_executable_batch_size_sub(
    function: callable = None, starting_batch_size: int = 128, subtract: int = 8
):
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


def float_to_int(x: float | int) -> int:
    if isinstance(x, int):
        return x
    if x.is_integer():
        return int(x)
    raise TypeError(f"Tried to convert {x=} to int, but it is not an integer.")


def compute_total_steps(n_samples: int, n_epochs: int, batch_size: int, n_accumulation: int) -> int:
    n_samples = float_to_int(n_samples)
    n_epochs = float_to_int(n_epochs)
    batch_size = float_to_int(batch_size)
    n_accumulation = float_to_int(n_accumulation)

    q, r = divmod(n_samples * n_epochs, batch_size * n_accumulation)
    if r == 0:
        return q
    return q + 1


def str_or_bool_to_str(s: str | bool) -> bool:
    if isinstance(s, bool):
        return s
    if not isinstance(s, str):
        raise TypeError(f"Expected str or bool, got {type(s)}")

    if s.lower() in ("true", "yes"):
        return True
    if s.lower() in ("false", "no"):
        return False

    raise ValueError(f"Expected 'true', 'false', 'yes' or 'no', got {s}")


def prob_norm(x: np.ndarray | list) -> np.ndarray | list:
    if isinstance(x, np.ndarray):
        return x / np.sum(x)
    x = list(x)
    s = sum(x)
    return [i / s for i in x]


def print_float_list(x: list[float], digits: int = 2) -> str:
    return "[" + ", ".join([str(round(i, digits)) for i in x]) + "]"


def round_list(x: list[float], digits: int = 2) -> list[float]:
    return [round(i, digits) for i in x]


def test_oversample_based_on_label():
    C = 10
    class_probabilities = prob_norm(np.random.rand(C))

    for n in [10, 100, 1000, 10000, 100000, 1000000]:
        examples = {"label": np.random.randint(0, 10, 100)}
        oversampled_examples = oversample_based_on_label(examples, class_probabilities)

        dist_old = Counter(examples["label"])
        dist_new = Counter(oversampled_examples["label"])

        dist_old = [x[1] for x in sorted(dist_old.items(), key=lambda x: x[0])]
        dist_new = [x[1] for x in sorted(dist_new.items(), key=lambda x: x[0])]

        p_old = prob_norm(dist_old)
        p_new = prob_norm(dist_new)

        distance_old = np.linalg.norm(class_probabilities - p_old)
        distance_new = np.linalg.norm(class_probabilities - p_new)
        delta = distance_old - distance_new

        print(f"{n=}")
        print(f"\t{distance_old=}")
        print(f"\t{distance_new=}")
        print(f"\t{delta=}")
        print("-" * 80)

        # print("class_probabilities:", round_list(class_probabilities), "\n")
        # dist = sorted(Counter(examples["labels"]).items(), key=lambda x: x[0])
        # print("original distribution:", dist, "\n")
        # print("original probabilities:", round_list(prob_norm([d[1] for d in dist])), "\n")
        # dist = sorted(Counter(oversampled_examples["labels"]).items(), key=lambda x: x[0])
        # print("new distribution:", dist, "\n")
        # print("new probabilities:", round_list(prob_norm([d[1] for d in dist])), "\n")
