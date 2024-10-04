"""
Tokenization of raw-bytes.
"""

import warnings

from tokenizers import Tokenizer, Regex
from tokenizers import models
from tokenizers import normalizers
from tokenizers.normalizers import Normalizer
from tokenizers import pre_tokenizers
from tokenizers.pre_tokenizers import PreTokenizer
from tokenizers import processors

from src.enums import TokenizationAlgorithm
from src.tokenization import SPECIALS
from src.tokenization.core import SENTINAL_NORMALIZER, SENTINAL_PRETOKENIZER


def get_raw_raw_tokenizer_08() -> Tokenizer:
    alphabet = [bytes([i]).decode("latin1") for i in range(256)]
    vocab = {v: i for i, v in enumerate(SPECIALS.values())} | {
        v: i for i, v in enumerate(alphabet, start=len(SPECIALS))
    }

    model = models.WordLevel(vocab=vocab, unk_token=SPECIALS["unk_token"])
    tokenizer = Tokenizer(model)
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            # For some reason, when \n\n is encountered, the Regex(".") fails to
            # split, so we need to split on \n first then split on . (matches all).
            pre_tokenizers.Split(Regex("\n"), behavior="isolated"),
            pre_tokenizers.Split(Regex("."), behavior="isolated"),
        ]
    )

    return tokenizer


def get_raw_raw_tokenizer_12() -> Tokenizer:
    warnings.warn("Warning: the full tokenizer functionality is not implemented for 12-bits!")
    alphabet = [
        bytes([i]).decode("latin1") + bytes([j]).decode("latin1")
        for i in range(16)
        for j in range(256)
    ]
    vocab = {v: i for i, v in enumerate(SPECIALS.values())} | {
        v: i for i, v in enumerate(alphabet, start=len(SPECIALS))
    }

    model = models.WordLevel(vocab=vocab, unk_token=SPECIALS["unk_token"])
    tokenizer = Tokenizer(model)
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(r"[\s\S]{1,2}"), behavior="isolated"),
        ]
    )

    return tokenizer


def get_raw_raw_tokenizer_16() -> Tokenizer:
    alphabet = [
        bytes([i]).decode("latin1") + bytes([j]).decode("latin1")
        for i in range(256)
        for j in range(256)
    ]
    vocab = {v: i for i, v in enumerate(SPECIALS.values())} | {
        v: i for i, v in enumerate(alphabet, start=len(SPECIALS))
    }

    model = models.WordLevel(vocab=vocab, unk_token=SPECIALS["unk_token"])
    tokenizer = Tokenizer(model)
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(r"[\s\S]{1,2}"), behavior="isolated"),
        ]
    )

    return tokenizer


def get_raw_raw_tokenizer(representation: int) -> Tokenizer:
    if representation == 8:
        return get_raw_raw_tokenizer_08()
    if representation == 12:
        return get_raw_raw_tokenizer_12()
    if representation == 16:
        return get_raw_raw_tokenizer_16()
    raise ValueError(f"{representation=}")


def get_raw_normalizer(algorithm: TokenizationAlgorithm) -> Normalizer:  # pylint: disable=unused-argument
    return SENTINAL_NORMALIZER


def get_raw_pretokenizer(algorithm: TokenizationAlgorithm) -> PreTokenizer:  # pylint: disable=unused-argument
    return SENTINAL_PRETOKENIZER
