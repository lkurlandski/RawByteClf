"""
Tokenization for disassembly code.
"""

from typing import Optional

from tokenizers import Regex, NormalizedString
from tokenizers import normalizers
from tokenizers.normalizers import Normalizer
from tokenizers import pre_tokenizers
from tokenizers.pre_tokenizers import PreTokenizer

from src.tokenization import TokenizerAlgorithm


def replace_normalized_string(org: NormalizedString, new: str) -> NormalizedString:

    SIZE = 2 ** 16

    for i in range(0, len(org.normalized), SIZE):
        print(f"{i=}")
        o = org.normalized[i : i + SIZE]
        n = new[i: i + SIZE]
        print(f"{len(o)=}")
        print(f"{len(n)=}")
        org.replace(o, n)


class StripAllExceptInstructionsNormalizer:
    """
    The regex-based approach:
        normalizers.Replace(Regex(r"^(.+?\t)(.+?\t)(.+?\t)(.+?\t)"), ""),
      does not work for very large files due to
        Exception: Exception: Compiled regex exceeds size limit of 10485760 bytes.
    
    This approach splits each line based on the tab character and keeps the last part.
    """

    def normalize(self, normalized: NormalizedString):

        text = []
        lines = normalized.split("\n", "isolated")
        for line in lines:
            parts = line.split("\t", "removed")
            text.append(str(parts[-1]))

        text = "\n".join(text)

        replace_normalized_string(normalized, text)


class CapitalizeHexCharactersFromCharacterStream:

    def __init__(self) -> None:
        self.capitalize_in_hex = False
        self.capitalize_in_fun = False
        self.history = ""

    def __call__(self, char: str) -> str:
        self.history += char
        self.history = self.history[-4:]

        if self.history[-2:] == "0x":
            self.capitalize_in_hex = True
            return char

        if self.history[-4:] == "fun_":
            self.capitalize_in_fun = True
            return char

        if self.capitalize_in_hex:
            if not char.isalnum():
                self.capitalize_in_hex = False
                return char
            return char.upper()

        if self.capitalize_in_fun:
            if char == "(":
                self.capitalize_in_fun = False
                return char
            return char.upper()

        return char


class HexCapitalizationNormalizer:

    def normalize(self, normalized: NormalizedString):
        func = CapitalizeHexCharactersFromCharacterStream()
        normalized.map(func)


class SignatureRemovalNormalizer:

    def normalize(self, normalized: NormalizedString):
        text = []
        functions = normalized.split("\n\n", "removed")
        for function in functions:
            lines = function.split("\n", "removed")
            text.append("\n")
            for line in lines[1:]:
                text.append(str(line))

        text = "\n".join(text)
        replace = str(normalized.slice((0, len(normalized.normalized))))
        normalized.replace(replace, text)


def get_dis_normalizer(algorithm: TokenizerAlgorithm) -> Optional[Normalizer]:  # pylint: disable=unused-argument
    return None


def get_dis_pretokenizer(algorithm: TokenizerAlgorithm) -> Optional[PreTokenizer]:
    if algorithm != TokenizerAlgorithm.WORDLEVEL:
        return pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(r"\n"), behavior="removed"),
        ])
    return pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(r"\n"), behavior="removed"),
        pre_tokenizers.Split(Regex(r"[^a-zA-Z0-9_]"), behavior="isolated"),
        pre_tokenizers.Split(Regex(r"\s"), behavior="removed"),
        pre_tokenizers.Split(Regex(r"(0x)|[0-9A-F]"), behavior="isolated"),
    ])
