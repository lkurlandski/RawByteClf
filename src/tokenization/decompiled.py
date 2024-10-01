"""
Tokenization for decompiled code.
"""

from typing import Optional

from tokenizers import Regex, NormalizedString
from tokenizers import normalizers
from tokenizers.normalizers import Normalizer
from tokenizers import pre_tokenizers
from tokenizers.pre_tokenizers import PreTokenizer

from src.enums import TokenizerAlgorithm


class CommentRemovalNormalizer:

    def normalize(self, normalized: NormalizedString):
        text = []
        parts = normalized.split(Regex(r"\/\*|\*\/"), "isolated")
        inside_comment = False
        for part in parts:
            print(f"{inside_comment=} {str(part)=}")
            if str(part) == "/*":
                inside_comment = True

            if not inside_comment:
                text.append(str(part))

            if str(part) == "*/":
                inside_comment = False

        text = "\n".join(text)
        replace = str(normalized.slice((0, len(normalized.normalized))))
        normalized.replace(replace, text)


def get_dec_normalizer(algorithm: TokenizerAlgorithm) -> Optional[Normalizer]:  # pylint: disable=unused-argument
    return None


def get_dec_pretokenizer(algorithm: TokenizerAlgorithm) -> Optional[PreTokenizer]:
    if algorithm != TokenizerAlgorithm.WORDLEVEL:
        return pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(r"\n"), behavior="removed"),
        ])
    raise NotImplementedError()
    # return pre_tokenizers.Sequence([
    #     pre_tokenizers.Split(Regex(r"\n"), behavior="removed"),
    #     pre_tokenizers.Split(Regex(r"[^a-zA-Z0-9_]"), behavior="isolated"),
    #     pre_tokenizers.Split(Regex(r"\s"), behavior="removed"),
    #     pre_tokenizers.Split(Regex(r"(0x)|[0-9A-F]"), behavior="isolated"),
    # ])
