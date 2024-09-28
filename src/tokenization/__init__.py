"""
Tokenization of malware.
"""

from collections import OrderedDict
from enum import Enum
from typing import Literal


class LiftLevel(Enum):
    RAW = "raw"
    DIS = "dis"
    DEC = "dec"


class TokenizerAlgorithm(Enum):
    BPE       = "bpe"
    UNIGRAM   = "uni"
    WORDPIECE = "wdp"
    WORDLEVEL = "wdl"


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
SPECIALS_IDS = {k: i for i, k in enumerate(SPECIALS)}
