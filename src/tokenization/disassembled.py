"""
Tokenization for disassembly code.
"""

from tokenizers import Regex, NormalizedString
from tokenizers import normalizers
from tokenizers.normalizers import Normalizer
from tokenizers import pre_tokenizers
from tokenizers.pre_tokenizers import PreTokenizer

from src.tokenization import TokenizerAlgorithm


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


def _get_dis_normalizer(
    capitalize_hex: bool = False,
    remove_signatures: bool = True,
) -> Normalizer:

    l = [
        normalizers.Replace(Regex(r"^(.+?\t)(.+?\t)(.+?\t)(.+?\t)"), ""),
        normalizers.NFD(),
        normalizers.StripAccents(),
        normalizers.Lowercase(),
    ]

    if capitalize_hex:
        l.append(normalizers.Normalizer.custom(HexCapitalizationNormalizer()))
    if remove_signatures:
        l.append(normalizers.Normalizer.custom(SignatureRemovalNormalizer()))

    return normalizers.Sequence(l)


def _get_dis_pretokenizer(
    split_nonalphanumeric: bool = False,
    split_spaces: bool = False,
    split_hex: bool = False,
) -> PreTokenizer:

    l = [
        pre_tokenizers.Split(Regex(r"\n"), behavior="removed"),
    ]

    if split_nonalphanumeric:
        l.append(pre_tokenizers.Split(Regex(r"[^a-zA-Z0-9_]"), behavior="isolated"))
    if split_spaces:
        l.append(pre_tokenizers.Split(Regex(r"\s"), behavior="removed"))
    if split_hex:
        l.append(pre_tokenizers.Split(Regex(r"(0x)|[0-9A-F]"), behavior="isolated"))

    return pre_tokenizers.Sequence(l)


def get_dis_normalizer(algorithm: TokenizerAlgorithm) -> Normalizer:  # pylint: disable=unused-argument
    return _get_dis_normalizer()


def get_dis_pretokenizer(algorithm: TokenizerAlgorithm) -> PreTokenizer:
    if algorithm == TokenizerAlgorithm.WORDLEVEL:
        return _get_dis_pretokenizer(split_nonalphanumeric=True, split_spaces=True, split_hex=True)
    return _get_dis_pretokenizer()
