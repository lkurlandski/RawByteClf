"""
Outward facing API.
"""

from typing import Optional

from transformers import PreTrainedTokenizerFast

from src.enums import TokenizationAlgorithm, LiftLevel, BitsInByte
from src.tokenization import SPECIALS
from src.tokenization.core import get_postprocessor, get_character_tokenizer
from src.tokenization.helpers import TokenizerIOHelper, DirectoryIsEmptyError
from src.tokenization.raw import get_raw_raw_tokenizer


def load_unigrams(
    lift_level: LiftLevel,
    algorithm: TokenizationAlgorithm,
    bits_in_byte: Optional[BitsInByte] = None,
    vocab_size: Optional[int] = None,
) -> dict[str, int]:
    if bits_in_byte != BitsInByte.EIGHT:
        raise NotImplementedError(f"{bits_in_byte=}")

    io_helper = TokenizerIOHelper.fromdisk(lift_level, algorithm, vocab_size, None)
    unigrams = io_helper.load_unigrams()
    if len(unigrams) != vocab_size:
        raise ValueError(f"Unigram length mismatch: {len(unigrams)=} {vocab_size=}")
    return unigrams


def save_unigrams(
    unigrams: dict[str, int],
    lift_level: LiftLevel,
    algorithm: TokenizationAlgorithm,
    bits_in_byte: Optional[BitsInByte] = None,
    vocab_size: Optional[int] = None,
) -> None:
    if bits_in_byte != BitsInByte.EIGHT:
        raise NotImplementedError(f"{bits_in_byte=}")

    if len(unigrams) != vocab_size:
        raise ValueError(f"Unigram length mismatch: {len(unigrams)=} {vocab_size=}")

    try:
        io_helper = TokenizerIOHelper.fromdisk(lift_level, algorithm, vocab_size, None)
    except DirectoryIsEmptyError:
        if not(algorithm == TokenizationAlgorithm.WORDLEVEL and lift_level == LiftLevel.RAW):
            raise
        io_helper = TokenizerIOHelper(lift_level, algorithm, vocab_size, 0)
        io_helper.path.mkdir(parents=True, exist_ok=True)

    io_helper.save_unigrams(unigrams)


def get_fast_tokenizer(
    lift_level: LiftLevel,
    algorithm: TokenizationAlgorithm,
    bits_in_byte: Optional[BitsInByte] = None,
    vocab_size: Optional[int] = None,
    add_cls_token: bool = False,
    add_bos_token: bool = False,
    add_eos_token: bool = False,
    add_sep_token: bool = False,
    **kwds,
) -> PreTrainedTokenizerFast:
    """
    Get a fast tokenizer for learning.

    kwds
       model_max_length: will caused the tokenizer to trim the tokenized input.
    """
    if bits_in_byte != BitsInByte.EIGHT:
        raise NotImplementedError(f"{bits_in_byte=}")

    lift_level = LiftLevel(lift_level)
    algorithm = TokenizationAlgorithm(algorithm)

    if algorithm == TokenizationAlgorithm.WORDLEVEL and lift_level == LiftLevel.RAW:
        tokenizer = get_raw_raw_tokenizer(bits_in_byte)
    elif algorithm == TokenizationAlgorithm.WORDLEVEL and lift_level != LiftLevel.RAW and vocab_size in (128, 256):
        tokenizer = get_character_tokenizer()
    else:
        tokenizer = TokenizerIOHelper.fromdisk(lift_level, algorithm, vocab_size, None).load()

    tokenizer.post_processor = get_postprocessor(add_cls_token, add_bos_token, add_eos_token, add_sep_token)
    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, **(kwds | SPECIALS))
    return fast_tokenizer
