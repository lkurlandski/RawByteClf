"""
Custom options.
"""

from enum import Enum


class LiftLevel(Enum):
    RAW = "raw"
    DIS = "dis"
    DEC = "dec"


class TokenizationAlgorithm(Enum):
    BPE       = "bpe"
    UNIGRAM   = "uni"
    WORDPIECE = "wdp"
    WORDLEVEL = "wdl"


class System(Enum):
    ARM = "arm"
    WIN = "win"
    RC  = "rc"
    LAB = "lab"


class EncryptionAlgorithm(Enum):
    AES = "aes"


class CompressionAlgorithm(Enum):
    GZIP = "gzip"
    BZ2  = "bz2"
    LZMA = "lzma"
    ZLIB = "zlib"
    S7Z  = "s7z"
