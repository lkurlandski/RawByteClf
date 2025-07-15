"""
Custom options.
"""

from enum import Enum, IntEnum


class LiftLevel(Enum):
    RAW = "raw"
    DIS = "dis"
    DEC = "dec"
    NOP = "nop"
    ALL = "all"


class TokenizationAlgorithm(Enum):
    BPE       = "bpe"
    UNIGRAM   = "uni"
    WORDPIECE = "wdp"
    WORDLEVEL = "wdl"


class System(Enum):
    ARMITAGE   = "arm"
    WINTERMUTE = "win"
    SPORC      = "rc"
    GCCIS      = "lab"
    DEFAULT    = "def"


class EncryptionAlgorithm(Enum):
    AES = "aes"


class CompressionAlgorithm(Enum):
    GZIP = "gzip"
    BZ2  = "bz2"
    LZMA = "lzma"
    ZLIB = "zlib"
    S7Z  = "s7z"


class Task(Enum):
    MLM = "mlm"
    CLM = "clm"
    DET = "det"
    FAM = "fam"
    BEH = "beh"


class PackingProtocol(Enum):
    ANY = "any"
    YES = "yes"
    NO  = "no"
    UNK = "unk"


class BitsInByte(IntEnum):
    EIGHT   = 8
    TWELVE  = 12
    SIXTEEN = 16


class SplitMode(Enum):
    RANDOM             = "random"
    TEMPORAL_CLASSWISE = "tmpclf"
    TEMPORAL_ABSOLUTE  = "tmpabs"


class WeightedLossAlgorithm(Enum):
    SAMPLE_REWEIGHTING      = "srw"
    INVERSE_CLASS_FREQUENCY = "icf"
    FOCAL_LOSS              = "foc"


class DatasetName(Enum):
    ASSEMBLAGE = "ass"
    BODMAS     = "bod"
    SOREL      = "sor"
    WINDOWS    = "win"
