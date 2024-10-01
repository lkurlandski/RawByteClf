"""
Helpers for the module.
"""

from __future__ import annotations
import os
from pathlib import Path
import shutil
import sys
from typing import Optional

from tokenizers import Tokenizer

# pylint: disable=wrong-import-position
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# pylint: enable=wrong-import-position

from src.cfg import TOKENIZERS_OUTPUT_PATH
from src.enums import TokenizationAlgorithm, LiftLevel
from src.tokenization.core import SENTINAL_NORMALIZER, SENTINAL_PRETOKENIZER
from src.tokenization.disassembled import get_dis_normalizer, get_dis_pretokenizer
from src.tokenization.decompiled import get_dec_normalizer, get_dec_pretokenizer


class DirectoryIsEmptyError(FileNotFoundError):
    ...


class TokenizerIOHelper:

    def __init__(self, lift_level: LiftLevel, algorithm: TokenizationAlgorithm, vocab_size: int, num_files: int) -> None:
        self.lift_level = LiftLevel(lift_level)
        self.algorithm = TokenizationAlgorithm(algorithm)
        self.vocab_size = vocab_size
        self.num_files = num_files

    @property
    def path(self) -> Path:
        return TOKENIZERS_OUTPUT_PATH \
            / f"{self.lift_level.value}" \
            / f"{self.algorithm.value}" \
            / f"{self.vocab_size}" \
            / f"{self.num_files}"

    @property
    def outfile(self) -> Path:
        return self.path / "tokenizer.json"

    def save(self, tokenizer: Tokenizer) -> None:

        shutil.rmtree(self.path, ignore_errors=True)
        self.path.mkdir(parents=True)

        # Handle custom normalizers and pretokenizers.
        if self.lift_level == LiftLevel.RAW:
            pass
        elif self.lift_level == LiftLevel.DIS:
            pass
        elif self.lift_level == LiftLevel.DEC:
            pass
        else:
            raise ValueError(f"{self.lift_level=}")

        tokenizer.save(self.outfile.as_posix())

    def load(self) -> Tokenizer:

        tokenizer = Tokenizer.from_file(self.outfile.as_posix())

        # Handle custom normalizers and pretokenizers.
        if self.lift_level == LiftLevel.RAW:
            pass
        elif self.lift_level == LiftLevel.DIS:
            pass
        elif self.lift_level == LiftLevel.DEC:
            pass
        else:
            raise ValueError(f"{self.lift_level=}")

        return tokenizer

    @classmethod
    def fromdisk(
        cls,
        lift_level: LiftLevel,
        algorithm: Optional[TokenizationAlgorithm],
        vocab_size: Optional[int],
        num_files: Optional[int],
    ) -> TokenizerIOHelper:
        p = TOKENIZERS_OUTPUT_PATH
        if lift_level is None:
            raise ValueError("You must specify a lift_level.")
        p /= lift_level

        if algorithm is None:
            if not (ps := sorted(p.iterdir())):
                raise DirectoryIsEmptyError(p)
            algorithm = ps[0].name
        p /= algorithm

        if vocab_size is None:
            if not (ps := sorted(p.iterdir())):
                raise DirectoryIsEmptyError(p)
            vocab_size = int(ps[0].name)
        p /= str(vocab_size)

        if num_files is None:
            if not (ps := sorted(p.iterdir())):
                raise DirectoryIsEmptyError(p)
            num_files = int(ps[0].name)
        p /= str(num_files)

        if not (p := p / "tokenizer.json").exists():
            raise FileNotFoundError(f"{p=}")

        return cls(lift_level, algorithm, vocab_size, num_files)
