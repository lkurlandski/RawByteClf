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
import time
from typing import Any, Optional, Literal

from accelerate.utils import is_xpu_available, is_npu_available
import numpy as np
import psutil
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

    for label in label_counter.keys():  # pylint: disable=consider-using-dict-items
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

    raise RuntimeError()


def pad_to_multiple_of_fn(val: int, pad_to_multiple_of: int = 1) -> int:
    q, r = divmod(val, pad_to_multiple_of)
    if r == 0:
        return val
    return (q + 1) * pad_to_multiple_of


def should_reduce_batch_size(exception: Exception) -> bool:
    """
    Checks if `exception` relates to CUDA out-of-memory, CUDNN not supported, or CPU out-of-memory

    Args:
        exception (`Exception`):
            An exception

    00 - when CUDA_LAUNCH_BLOCKING=0, this can sometimes be raised
    01 - when CUDA_LAUNCH_BLOCKING=1, the above error manifests as this. I have no idea
        wtf this means, but lets give it a shot
    """
    _statements = [
        "CUDA out of memory.",  # CUDA OOM
        "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED.",  # CUDNN SNAFU
        "DefaultCPUAllocator: can't allocate memory",  # CPU OOM
        "CUDA error: an illegal memory access was encountered",  # 00
        "Triton Error [CUDA]: an illegal memory access was encountered",  # 01
        "an illegal memory access was encountered",
    ]
    if isinstance(exception, RuntimeError) and len(exception.args) == 1:
        return any(err in exception.args[0] for err in _statements)
    return False


def find_executable_batch_size(function: callable = None, starting_batch_size: int = 128):
    if function is None:
        return functools.partial(find_executable_batch_size, starting_batch_size=starting_batch_size)

    batch_size = starting_batch_size

    def decorator(*args, **kwargs):
        nonlocal batch_size
        gc.collect()
        if is_xpu_available():
            torch.xpu.empty_cache()
        elif is_npu_available():
            torch.npu.empty_cache()
        else:
            torch.cuda.empty_cache()
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
            except Exception as e:  # pylint: disable=broad-exception-caught
                if should_reduce_batch_size(e):
                    print(f"HANDLING --- {e}", flush=True)
                    gc.collect()
                    if is_xpu_available():
                        torch.xpu.empty_cache()
                    elif is_npu_available():
                        torch.npu.empty_cache()
                    else:
                        torch.cuda.empty_cache()
                    batch_size //= 2
                else:
                    print(f"RAISING --- {e}", flush=True)
                    raise

    return decorator


def find_executable_batch_size_sub(
    function: callable = None,
    starting_batch_size: int = 128,
    subtract: int = 8,
):

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
            except Exception as e:  # pylint: disable=broad-exception-caught
                if should_reduce_batch_size(e):
                    print(f"HANDLING --- {e}", flush=True)
                    gc.collect()
                    if not is_xpu_available():
                        torch.cuda.empty_cache()
                    else:
                        torch.xpu.empty_cache()
                    batch_size -= subtract
                else:
                    print(f"RAISING --- {e}", flush=True)
                    raise

    return decorator


def find_executable_batch_size_and_gradient_accumulation_steps(
    function: callable = None,
    starting_batch_size: int = 128,
    starting_gradient_accumulation_steps: Optional[int] = 1,
) -> None:
    if function is None:
        return functools.partial(
            find_executable_batch_size_and_gradient_accumulation_steps,
            starting_batch_size=starting_batch_size,
            starting_gradient_accumulation_steps=starting_gradient_accumulation_steps,
        )

    batch_size = starting_batch_size
    gradient_accumulation_steps = starting_gradient_accumulation_steps if starting_gradient_accumulation_steps else 1

    def decorator(*args, **kwargs):
        nonlocal batch_size, gradient_accumulation_steps
        gc.collect()
        torch.cuda.empty_cache()
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
                return function(batch_size, gradient_accumulation_steps, *args, **kwargs)
            except Exception as e:  # pylint: disable=broad-exception-caught
                if should_reduce_batch_size(e):
                    print(f"HANDLING --- {e}", flush=True)
                    gc.collect()
                    torch.cuda.empty_cache()
                    batch_size //= 2
                    gradient_accumulation_steps *= 2
                else:
                    print(f"RAISING --- {e}", flush=True)
                    raise

    return decorator


def float_to_int(x: float | int) -> int:
    if isinstance(x, int):
        return x
    if x.is_integer():
        return int(x)
    raise TypeError(f"Tried to convert {x=} to int, but it is not an integer.")


def compute_total_steps(
    n_samples: int,
    n_epochs: int,
    batch_size: Optional[int] = None,
    per_device_batch_size: Optional[int] = None,
    n_accumulation_steps: Optional[int] = None,
    n_devices: Optional[int] = None,
) -> int:

    # Ray tune seems to fuck up the data types? WTF. Anyway, cast them all to int.
    n_samples = float_to_int(n_samples)
    n_epochs = float_to_int(n_epochs)
    batch_size = float_to_int(batch_size) if batch_size is not None else None
    per_device_batch_size = float_to_int(per_device_batch_size) if per_device_batch_size is not None else None
    n_accumulation_steps = float_to_int(n_accumulation_steps) if n_accumulation_steps is not None else None
    n_devices = float_to_int(n_devices) if n_devices is not None else None

    if batch_size is None:
        if any(x is None for x in (per_device_batch_size, n_accumulation_steps, n_devices)):
            raise ValueError()
        batch_size = per_device_batch_size * n_accumulation_steps * n_devices
    else:
        if not all(x is None for x in (per_device_batch_size, n_accumulation_steps, n_devices)):
            raise ValueError()

    q, r = divmod(n_samples * n_epochs, batch_size)
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


def get_mem(unit: Literal["KB", "MB", "GB"] = "MB", f: Optional[int] = None, p: Optional[str] = None):
    """
    Single call takes ~6e-05 seconds.
    """
    if f is None:
        unit = unit.lower()
        if "kb" in unit:
            f = 1024 ** 1
            p = "KB"
        elif "mb" in unit:
            f = 1024 ** 2
            p = "MB"
        elif "gb" in unit:
            f = 1024 ** 3
            p = "GB"
        elif "b" in unit:
            f = 1024 ** 0
            p = "B"
        else:
            f = 1
            p = "B"
    else:
        p = "??" if p is None else p
    mem = psutil.virtual_memory()

    return mem.total / f, mem.available / f, mem.used / f


def time_get_mem():
    start = time.time()
    for _ in range(1000):
        print(get_mem("MB"))
    end = time.time()
    print(f"{end - start=}")
    print(f"{(end - start) / 1000=}")


def is_power_of_two(n: int):
    return n != 0 and (n & (n - 1)) == 0


def interpret_bytes_as_integers(b: bytes, bits_in_byte: int = 8) -> np.ndarray:

    dtype = np.uint8
    if bits_in_byte > 8:
        dtype = np.uint16
    if bits_in_byte > 16:
        dtype = np.uint32
    if bits_in_byte > 32:
        dtype = np.uint64
    if bits_in_byte > 64:
        dtype = np.uint128
    if bits_in_byte > 128:
        dtype = np.uint256
    if bits_in_byte > 256:
        raise ValueError()

    length_must_be_divisible_by = int(math.lcm(8, bits_in_byte) / 8)
    if len(b) % length_must_be_divisible_by != 0:
        pad = length_must_be_divisible_by - (len(b) % length_must_be_divisible_by)
        b = b + (bytes([0]) * pad)

    if is_power_of_two(bits_in_byte):
        return np.frombuffer(b, dtype=dtype)

    if bits_in_byte != 12:
        raise NotImplementedError("The computations are only for the 12-bit interpretation.")

    # Get the 8-bit array representation
    arr = np.frombuffer(b, dtype=np.uint8)
    arr = arr.astype(dtype)
    arr = arr.reshape(-1, 3)  # take three eight-bit numbers and represent them as two 12-bit ones

    out = np.zeros((arr.shape[0], 2), dtype=np.uint16)
    out[:,0] = (arr[:,0] << 4) + (arr[:,1] >> 4)
    # shift left by 12 to clear the most signfificant bits,
    # then shift right by four to put in the correct place.
    out[:,1] = ((arr[:,1] << 12) >> 4) + (arr[:,2])
    out = out.flatten()
    # assert out.max() < 2 ** bits_in_byte, out.max()
    return out


def test_interpret_bytes_as_integers():

    b = bytes([random.randint(0, 255) for _ in range(53877)])
    # b = b'\x01\x02\x03\x04\x05\x06\x07\x08\x09'


    arr = interpret_bytes_as_integers(b, 12)
    print(arr)

    b = b'\x01\x02\x03\x04\x05\x06\x07\x08\x09'
    # arr = interpret_bytes_as_integers(b, 12)
    # print(arr)
    # sys.exit(0)
    # time_get_mem()


if __name__ == "__main__":
    test_interpret_bytes_as_integers()
