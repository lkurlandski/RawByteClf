"""
Tokenization for decompiled code.
"""

from tokenizers import Regex, NormalizedString
from tokenizers import normalizers
from tokenizers.normalizers import Normalizer
from tokenizers import pre_tokenizers
from tokenizers.pre_tokenizers import PreTokenizer

from src.tokenization import TokenizerAlgorithm


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


def _get_dec_normalizer(remove_comments: bool = True) -> Normalizer:
    l = [
        normalizers.Replace(Regex(r"^(.+?\t)(.+?\t)(.+?\t)(.+?\t)"), ""),
        normalizers.NFD(),
        normalizers.StripAccents(),
        normalizers.Lowercase(),
    ]
    if remove_comments:
        l.append(normalizers.Normalizer.custom(CommentRemovalNormalizer()))
    return normalizers.Sequence(l)


def _get_dec_pretokenizer() -> PreTokenizer:
    l = [
        pre_tokenizers.Split(Regex(r"\n"), behavior="removed"),
    ]
    return pre_tokenizers.Sequence(l)


def get_dec_normalizer(algorithm: TokenizerAlgorithm) -> Normalizer:
    return _get_dec_normalizer()


def get_dec_pretokenizer(algorithm: TokenizerAlgorithm) -> PreTokenizer:
    return _get_dec_pretokenizer()
