"""
Tokenization for decompiled code.
"""

from tokenizers import Regex, NormalizedString
from tokenizers import normalizers
from tokenizers.normalizers import Normalizer
from tokenizers import pre_tokenizers
from tokenizers.pre_tokenizers import PreTokenizer

from src.tokenization import TokenizerAlgorithm


def get_dec_normalizer(algorithm: TokenizerAlgorithm) -> Normalizer:
    ...


def get_dec_pretokenizer(algorithm: TokenizerAlgorithm) -> PreTokenizer:
    ...
