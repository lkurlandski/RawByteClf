"""
Train the tokenization algorithms.
"""

from __future__ import annotations
from argparse import ArgumentParser
import asyncio
from collections.abc import Generator
from datetime import datetime
from itertools import chain, islice
import os
from pathlib import Path
from pprint import pformat, pprint
import sys
from typing import Callable, Optional

# pylint: disable=wrong-import-position
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# pylint: enable=wrong-import-position

from tokenizers import Tokenizer
from tokenizers import models
from tokenizers.normalizers import Normalizer
from tokenizers.pre_tokenizers import PreTokenizer
from tokenizers import trainers
from tqdm import tqdm

from src.learn.bytes_to_str_utf8 import bytes_to_str_utf8  # pylint: disable=no-name-in-module
from src.data.utils import get_processed_data
from src.tokenization import SPECIALS, TokenizerAlgorithm, LiftLevel
from src.tokenization.disassembled import get_dis_normalizer, get_dis_pretokenizer
from src.tokenization.decompiled import get_dec_normalizer, get_dec_pretokenizer
from src.tokenization.raw import get_raw_normalizer, get_raw_pretokenizer
from src.tokenization.helpers import TokenizerIOHelper
from src.utils import batched


RAW_WORD_SIZE = 256


class TokenizationTrainingIterator:

    def __init__(
        self,
        lift_level: LiftLevel,
        batch_size: int,
        block_size: int,
        num_files: Optional[int] = None,
    ) -> None:
        """
        Yield samples to train the tokenizer.

        Args:
            lift_level: The level at which the data is lifted.
            batch_size: The number of "documents" in each batch.
            block_size: The number of "words" in each document.
            num_files: The number of files to process.
        """
        self.lift_level = LiftLevel(lift_level)
        self.batch_size = batch_size
        self.block_size = block_size
        self.num_files = num_files
        self.stream: list[bytes] = []
        self.idx: Optional[int] = None

    def __call__(self) -> TokenizationTrainingIterator:
        documents = islice((b for n, b in get_processed_data(self.lift_level.value, "sorel_pe")), self.num_files)

        batch = []
        for document in documents:
            for word in self.decompose_document(document):
                batch.append(word)
                if len(batch) == self.batch_size:
                    self.stream.append(tuple(batch))
                    batch.clear()

        if batch:
            self.stream.append(tuple(batch))

        self.idx = 0

        return self

    def __iter__(self) -> TokenizationTrainingIterator:
        if self.idx is None:
            raise RuntimeError("TokenizationTrainingIterator.__call__ must be called first.")
        return self

    def __len__(self) -> int:
        if self.idx is None:
            raise RuntimeError("TokenizationTrainingIterator.__call__ must be called first.")
        return len(self.stream)

    def __next__(self) -> list[str]:
        if self.idx is None:
            raise RuntimeError("TokenizationTrainingIterator.__call__ must be called first.")
        if self.idx == len(self.stream):
            raise StopIteration()

        batch = self.stream[self.idx]
        batch = [self.bytes_to_str(word) for word in batch]

        self.idx += 1
        return batch

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return "TokenizationTrainingIterator(" \
            f"{self.lift_level=}, " \
            f"{self.batch_size=}, " \
            f"{self.block_size=}, " \
            f"{self.num_files=}, " \
            f"{len(self.stream)=})" \
            ")"

    @property
    def bytes_to_str(self) -> Callable:
        if self.lift_level == LiftLevel.RAW:
            return bytes_to_str_utf8
        if self.lift_level == LiftLevel.DIS:
            return bytes.decode
        if self.lift_level == LiftLevel.DEC:
            return bytes.decode
        raise ValueError(f"{self.lift_level=}")

    def decompose_document(self, document: bytes) -> list[bytes]:
        if self.lift_level == LiftLevel.RAW:
            return list(batched(document, RAW_WORD_SIZE))
        if self.lift_level == LiftLevel.DIS:
            return document.split(b"\n")
        if self.lift_level == LiftLevel.DEC:
            return document.split(b"\n")
        raise ValueError(f"{self.lift_level=}")


class TrainTokenizer:

    def __init__(
        self,
        iterator: TokenizationTrainingIterator,
        lift_level: LiftLevel,
        algorithm: TokenizerAlgorithm,
        vocab_size: int,
        max_token_length: Optional[int] = None,
    ) -> None:
        self.iterator = iterator
        self.lift_level = LiftLevel(lift_level)
        self.algorithm = TokenizerAlgorithm(algorithm)
        self.vocab_size = vocab_size
        self.max_token_length = max_token_length

        if self.lift_level == LiftLevel.RAW and self.algorithm == TokenizerAlgorithm.WORDLEVEL:
            raise ValueError("WordLevel tokenization at the raw-byte level does not need training.")

    def __call__(self) -> Tokenizer:
        model = self.get_model()
        tokenizer = Tokenizer(model)
        if (normalizer := self.get_normalizer()) is not None:
            tokenizer.normalizer = normalizer
        if (pretokenizer := self.get_pretokenizer()) is not None:
            tokenizer.pre_tokenizer = pretokenizer
        trainer = self.get_trainer()
        tokenizer.train_from_iterator(self.iterator, trainer, length=len(self.iterator) * self.iterator.batch_size)
        return tokenizer

    def get_normalizer(self) -> Optional[Normalizer]:
        if self.lift_level == LiftLevel.RAW:
            return get_raw_normalizer(self.algorithm)
        if self.lift_level == LiftLevel.DIS:
            return get_dis_normalizer(self.algorithm)
        if self.lift_level == LiftLevel.DEC:
            return get_dec_normalizer(self.algorithm)
        raise ValueError(f"{self.lift_level=}")

    def get_pretokenizer(self) -> Optional[PreTokenizer]:
        if self.lift_level == LiftLevel.RAW:
            return get_raw_pretokenizer(self.algorithm)
        if self.lift_level == LiftLevel.DIS:
            return get_dis_pretokenizer(self.algorithm)
        if self.lift_level == LiftLevel.DEC:
            return get_dec_pretokenizer(self.algorithm)
        raise ValueError(f"{self.lift_level=}")

    def get_model(self) -> models.Model:
        if self.algorithm == TokenizerAlgorithm.BPE:
            return models.BPE()
        if self.algorithm == TokenizerAlgorithm.UNIGRAM:
            return models.Unigram()
        if self.algorithm == TokenizerAlgorithm.WORDPIECE:
            return models.WordPiece()
        if self.algorithm == TokenizerAlgorithm.WORDLEVEL:
            return models.WordLevel()
        raise ValueError(f"{self.algorithm=}")

    def get_trainer(self) -> trainers.Trainer:
        special_tokens = list(SPECIALS.values())
        vocab_size = self.vocab_size + len(special_tokens)
        if self.algorithm == TokenizerAlgorithm.BPE:
            return trainers.BpeTrainer(
                vocab_size=vocab_size,
                special_tokens=special_tokens,
                max_token_length=self.max_token_length,
            )
        if self.algorithm == TokenizerAlgorithm.UNIGRAM:
            return trainers.UnigramTrainer(
                vocab_size=vocab_size,
                special_tokens=special_tokens,
                unk_token=SPECIALS["unk_token"],
                max_piece_length=self.max_token_length,
            )
        if self.algorithm == TokenizerAlgorithm.WORDPIECE:
            return trainers.WordPieceTrainer(
                vocab_size=vocab_size,
                special_tokens=special_tokens,
            )
        if self.algorithm == TokenizerAlgorithm.WORDLEVEL:
            return trainers.WordLevelTrainer(
                vocab_size=vocab_size,
                special_tokens=special_tokens,
            )
        raise ValueError(f"{self.algorithm=}")


def main():
    print(f"START @{datetime.now()}")

    parser = ArgumentParser()
    parser.add_argument("--lift_level", type=LiftLevel, required=True)
    parser.add_argument("--algorithm", type=TokenizerAlgorithm, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--num_files", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2**10)
    parser.add_argument("--block_size", type=int, default=2**10)
    parser.add_argument("--max_token_length", type=int, default=64)
    args = parser.parse_args()

    print(f"args={pformat(args.__dict__)}")

    iterator = TokenizationTrainingIterator(args.lift_level, args.batch_size, args.num_files)()
    tokenizer = TrainTokenizer(iterator, args.lift_level, args.algorithm, args.vocab_size, args.max_token_length)()
    io_helper = TokenizerIOHelper(args.lift_level, args.algorithm, args.vocab_size, args.num_files)
    io_helper.save(tokenizer)
    tokenizer = io_helper.load()

    print(f"FINISH @{datetime.now()}")


if __name__ == "__main__":
    main()
