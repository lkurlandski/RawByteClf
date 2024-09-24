"""
Tokenization for raw, disassembled, and/or decompiled code.
"""

from tokenizers import Regex
from tokenizers import normalizers
from tokenizers import pre_tokenizers
from tokenizers import processors
from tokenizers.processors import PostProcessor

from src.tokenization import SPECIALS


SENTINAL_PATTERN = Regex(r"a^")
SENTINAL_NORMALIZER = normalizers.Replace(SENTINAL_PATTERN, "")
SENTINAL_PRETOKENIZER = pre_tokenizers.Split(SENTINAL_PATTERN, "isolated")
SENTINAL_POSTPROCESSOR = processors.ByteLevel(trim_offsets=False)  # TODO: verify that this does nothing.


def get_postprocessor(
    add_cls_token: bool,
    add_bos_token: bool,
    add_eos_token: bool,
    add_sep_token: bool,
) -> PostProcessor:
    if not any((add_cls_token, add_bos_token, add_eos_token, add_sep_token)):
        return SENTINAL_POSTPROCESSOR

    if add_bos_token:
        start = SPECIALS["bos_token"]
    elif add_cls_token:
        start = SPECIALS["cls_token"]
    else:
        start = None

    if add_eos_token:
        end = SPECIALS["eos_token"]
    elif add_sep_token:
        end = SPECIALS["sep_token"]
    else:
        end = None

    # TODO: what does the `pair` do?
    # TODO: what if `start`` or `end` are None?
    return processors.TemplateProcessing(
        single=f"{start} $0 {end}",
        pair=f"{start} $A {end} {SPECIALS['sep_token']} {start} $B:1 {end}",
        special_tokens=tuple((s, i) for i, s in enumerate(SPECIALS.values())),
    )
