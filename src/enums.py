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


class ExplanationAlgorithm(Enum):
    """
    Notes:
        - We do not include FeaturePermutation because its permutes an input
          based on other examples in the batch, which makes it less appealing.
        - We do not include Occlusion because the captum implementation does not
          support per-sample control over the perturbation behavior. Its possible
          to convert a `feature_mask` to arguments for `sliding_window_shapes` and
          `strides` to attain similar behavior for one sample, but it cannot be done
          for a batch of samples.
    """

    LIME = "lime"  # LIME                   # Surrogate
    KSHP = "kshp"  # Kernel SHAP            # Surrogate
    # ANCH = "anch"  # Anchors                # Surrogate
    IGRD = "igrd"  # Integrated Gradients   # Gradient
    GSHP = "gshp"  # Gradient SHAP          # Gradient
    DLFT = "dlft"  # DeepLIFT               # Gradient
    FABL = "fabl"  # Feature Ablation       # Perturbation
    SSHP = "sshp"  # Sampling SHAP          # Perturbation
    # FPRM = "fprm"  # Feature Permutation    # Perturbation


class ExplanationMethod(Enum):
    TOK = "tok"  # Token-level.
    CHK = "chk"  # Fixed-size chunks.
    NUM = "num"  # Fixed-size chunks; number of chunks set to one above the number of functions.
    LEN = "len"  # Fixed-size chunks; size of chunks set to the median function length.
    NML = "nml"  # Fixed-size chunks; combination of NUM and LEN over the region of functions.
    FUN = "fun"  # Function-level.


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
