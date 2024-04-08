"""
Functions to process bytes into a representation suitable for learning.
"""

from functools import reduce, partial
import re
from typing import Callable, Optional

import torch
from torch import LongTensor
from transformers import PreTrainedTokenizerFast

from src.utils import to_long_tensor, compress, encrypt
from src.learn.bytes_to_str_utf8 import bytes_to_str_utf8
from src.learn.utils import interpret_bytes_as_integers


BYTE_TO_UTF8 = tuple(chr(i + 10752) for i in range(256))


# 0.0015 of a seconds to convert 2 ** 20 bytes to a str.
__all__ = ["bytes_to_str_utf8"]


# 0.052 of a seconds to convert 2 ** 20 bytes to a str.
# def bytes_to_str_utf8(b: bytes) -> str:
#     return "".join(BYTE_TO_UTF8[i] for i in b)


# 0.00031 of a seconds to convert 2 ** 20 bytes to a str.
def bytes_to_str_ascii(b: bytes) -> str:
    return b.decode("latin1")


def replace_consecutive_bytes_with_singular_byte(bs: bytes, num_consecutive: int):
    pattern = rb'(.)\1{' + str(num_consecutive - 1).encode() + rb',}'
    replacement = rb'\1'
    return re.sub(pattern, replacement, bs)


def preprocess_fn_add_cls_token(x: LongTensor, cls_token_id: int) -> LongTensor:
    return torch.cat([torch.tensor([cls_token_id], dtype=torch.long), x])


def preprocess_fn_add_bos_token(x: LongTensor, bos_token_id: int) -> LongTensor:
    return torch.cat([torch.tensor([bos_token_id], dtype=torch.long), x])


def preprocess_fn_add_eos_token(x: LongTensor, eos_token_id: int) -> LongTensor:
    return torch.cat([x, torch.tensor([eos_token_id], dtype=torch.long)])


def preprocess_fn_shift_token_idx(x: LongTensor, shift: int) -> LongTensor:
    return x + shift


def bytes_to_input_ids(
    b: bytes,
    bits_in_byte: int = 8,
    num_special_ids: int = 0,
    max_length: Optional[int] = None,
    cls_token_id: Optional[int] = None,
    bos_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
) -> LongTensor:
    if cls_token_id is not None and bos_token_id is not None:
        raise ValueError(f"Cannot have both {cls_token_id=} and {bos_token_id=}.")

    x = interpret_bytes_as_integers(b, bits_in_byte=bits_in_byte)
    x = to_long_tensor(x)
    x = preprocess_fn_shift_token_idx(x, shift=num_special_ids)
    if cls_token_id is not None:
        x = preprocess_fn_add_cls_token(x, cls_token_id=cls_token_id)
    if bos_token_id is not None:
        x = preprocess_fn_add_bos_token(x, bos_token_id=bos_token_id)
    x = x[:max_length]
    if eos_token_id is not None:
        x = x[:max_length - 1] if isinstance(max_length, int) else x
        x = preprocess_fn_add_eos_token(x, eos_token_id=eos_token_id)

    return x


def tokenize_bytes(
    b: bytes | list[bytes],
    tokenizer: PreTrainedTokenizerFast,
    bytes_to_str: Callable[[bytes], str] = bytes_to_str_utf8,
    truncation: bool = True,
    max_length: Optional[int] = None,
    return_overflowing_tokens: bool = False,
    add_special_tokens: bool = True,
    **kwds,
) -> LongTensor | list[LongTensor]:
    if isinstance(b, bytes):
        b = [b]

    text = [bytes_to_str(i) for i in b]
    batch_encoding = tokenizer(
        text,
        truncation=truncation,
        max_length=max_length,
        return_overflowing_tokens=return_overflowing_tokens,
        add_special_tokens=add_special_tokens,
        **kwds,
    )
    if len(b) == 1:
        return to_long_tensor(batch_encoding.data["input_ids"][0])
    return [to_long_tensor(i) for i in batch_encoding.data["input_ids"]]


def hf_compress_bytes(examples: dict[str, list], compression_type: str, compression_level: int = 9) -> dict[str, list]:
    return {"bytes": [compress(bs, compression_type, compression_level) for bs in examples["bytes"]]}


def hf_encrypt_bytes(examples: dict[str, list], encryption_type: str, key: Optional[bytes] = None) -> dict[str, list]:
    return {"bytes": [encrypt(bs, encryption_type, key) for bs in examples["bytes"]]}


def hf_tokenize_bytes(
    examples: dict[str, list],
    tokenizer: PreTrainedTokenizerFast,
    bytes_to_str: Callable[[bytes], str] = bytes_to_str_utf8,
    truncation: bool = True,
    max_length: Optional[int] = None,
    return_overflowing_tokens: bool = False,
    add_special_tokens: bool = True,
    **kwds,
) -> dict[str, list]:
    text = [bytes_to_str(b) for b in examples["bytes"]]
    return tokenizer(
        text,
        truncation=truncation,
        max_length=max_length,
        return_overflowing_tokens=return_overflowing_tokens,
        add_special_tokens=add_special_tokens,
        **kwds,
    )


def hf_bytes_to_input_ids(
    examples: dict[str, list],
    bits_in_byte: int = 8,
    num_special_ids: int = 0,
    max_length: Optional[int] = None,
    cls_token_id: Optional[int] = None,
    bos_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
) -> dict[str, list]:
    return {
        "input_ids": [
            bytes_to_input_ids(
                b,
                bits_in_byte=bits_in_byte,
                num_special_ids=num_special_ids,
                max_length=max_length,
                cls_token_id=cls_token_id,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
            )
            for b in examples["bytes"]
        ]
    }
