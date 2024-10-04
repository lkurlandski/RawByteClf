"""
Outward facing API.
"""

from typing import Optional

from transformers import PreTrainedTokenizerFast

from src.enums import TokenizationAlgorithm, LiftLevel
from src.tokenization import SPECIALS
from src.tokenization.core import get_postprocessor
from src.tokenization.helpers import TokenizerIOHelper
from src.tokenization.raw import get_raw_raw_tokenizer


def get_fast_tokenizer(
    lift_level: LiftLevel,
    algorithm: TokenizationAlgorithm,
    representation: Optional[int] = None,
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

    lift_level = LiftLevel(lift_level)
    algorithm = TokenizationAlgorithm(algorithm)

    if algorithm == TokenizationAlgorithm.WORDLEVEL and lift_level == LiftLevel.RAW:
        tokenizer = get_raw_raw_tokenizer(representation)
    else:
        tokenizer = TokenizerIOHelper.fromdisk(lift_level, algorithm, vocab_size, None).load()

    tokenizer.post_processor = get_postprocessor(add_cls_token, add_bos_token, add_eos_token, add_sep_token)
    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, **(kwds | SPECIALS))
    return fast_tokenizer
