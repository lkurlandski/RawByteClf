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
from src.data.cfg import DATASET_TO_FILES
from src.data.utils import read_binary_files_asynch
from src.tokenization import SPECIALS, TokenizerAlgorithm, LiftLevel
from src.tokenization.disassembled import get_dis_normalizer, get_dis_pretokenizer
from src.tokenization.decompiled import get_dec_normalizer, get_dec_pretokenizer
from src.tokenization.raw import get_raw_normalizer, get_raw_pretokenizer
from src.tokenization.helpers import TokenizerIOHelper
from src.utils import batched


class TokenizationTrainingIterator:

    def __init__(
        self,
        lift_level: LiftLevel,
        batch_size: int,
        num_files: Optional[int] = None,
    ) -> None:
        self.lift_level = lift_level
        self.batch_size = batch_size
        self.num_files = num_files
        self.documents: list[bytes] = []
        self.stream: list[bytes] = []

    def __call__(self) -> TokenizationTrainingIterator:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        future = read_binary_files_asynch(
            self.files,
            max_length=None,
            in_memory_dtype="bytes",
            disable_tqdm=False,
        )
        self.documents = loop.run_until_complete(future)
        self.stream = batched(self.documents, self.batch_size)
        return self

    def __iter__(self) -> TokenizationTrainingIterator:
        if not self.documents:
            raise ValueError()
        return self

    def __len__(self) -> int:
        l = len(self.files) / self.batch_size
        return int(l) if l.is_integer() else int(l) + 1

    def __next__(self) -> list[str]:
        batch = next(self.stream)
        batch = [self.bytes_to_str(item) for item in batch]
        print(f"TokenizationTrainingIterator.__next__: {len(batch)=}")
        print(f"TokenizationTrainingIterator.__next__: {[len(b) for b in batch]=}")
        return batch

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"TokenizationTrainingIterator(len={len(self)})"

    @property
    def files(self) -> list[str]:
        if self.lift_level == LiftLevel.RAW:
            key = "binaries"
        elif self.lift_level == LiftLevel.DIS:
            key = "disassembled"
        elif self.lift_level == LiftLevel.DEC:
            key = "decompiled"
        else:
            raise ValueError(f"{self.algorithm=}")
        files = DATASET_TO_FILES[key]["sorel_pe"]()
        files = sorted(map(str, islice(files, self.num_files)))
        return files

    @property
    def bytes_to_str(self) -> Callable:
        if self.lift_level == LiftLevel.RAW:
            return byte_to_str_utf8
        if self.lift_level == LiftLevel.DIS:
            return bytes.decode
        if self.lift_level == LiftLevel.DEC:
            return bytes.decode
        raise ValueError(f"{self.algorithm=}")


class TrainTokenizer:

    def __init__(
        self,
        lift_level: LiftLevel,
        algorithm: TokenizerAlgorithm,
        vocab_size: int,
        batch_size: int,
        num_files: int,
        max_token_length: Optional[int] = None,
    ) -> None:
        self.lift_level = lift_level
        self.algorithm = algorithm
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.num_files = num_files
        self.max_token_length = max_token_length

        if self.lift_level == LiftLevel.RAW and self.algorithm == TokenizerAlgorithm.WORDLEVEL:
            raise ValueError(f"WordLevel tokenization at the raw-byte level does not need training.")

    def __call__(self) -> Tokenizer:
        iterator = TokenizationTrainingIterator(self.lift_level, self.batch_size, self.num_files)()
        normalizer = self.get_normalizer()
        pretokenizer = self.get_pretokenizer()
        model = self.get_model()
        trainer = self.get_trainer()
        tokenizer = Tokenizer(model)
        tokenizer.normalizer = normalizer
        tokenizer.pre_tokenizer = pretokenizer
        tokenizer.train_from_iterator(iterator, trainer)

        return tokenizer


    def get_normalizer(self) -> Normalizer:
        if self.lift_level == LiftLevel.RAW:
            return get_raw_normalizer(self.algorithm)
        if self.lift_level == LiftLevel.DIS:
            return get_dis_normalizer(self.algorithm)
        if self.lift_level == LiftLevel.DEC:
            return get_dec_normalizer(self.algorithm)
        raise ValueError(f"{self.lift_level=}")

    def get_pretokenizer(self) -> PreTokenizer:
        if self.lift_level == LiftLevel.RAW:
            return get_raw_pretokenizer(self.algorithm)
        if self.lift_level == LiftLevel.DIS:
            return get_dis_pretokenizer(self.algorithm)
        if self.lift_level == LiftLevel.DEC:
            return get_dec_pretokenizer(self.algorithm)
        raise ValueError(f"{self.lift_level=}")

    def get_model(self) -> models.Model:
        # TODO: should the Model recieve the unk_token and/or unk_token_id?
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
    parser.add_argument("--max_token_length", type=int, default=64)
    args = parser.parse_args()

    print(f"args={pformat(args)}")

    tokenizer = TrainTokenizer(
        args.lift_level,
        args.algorithm,
        args.vocab_size,
        args.batch_size,
        args.num_files,
        args.max_token_length,
    )()

    io_helper = TokenizerIOHelper(args.lift_level, args.algorithm, args.vocab_size, args.num_files)
    io_helper.save(tokenizer)
    tokenizer = io_helper.load()

    print(f"FINISH @{datetime.now()}")


if __name__ == "__main__":
    main()
