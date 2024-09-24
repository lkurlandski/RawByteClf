"""
Train the tokenization algorithms.
"""

from argparse import ArgumentParser
import asyncio
from collections.abc import Generator
from datetime import datetime
from itertools import chain, islice
import os
from pathlib import Path
from pprint import pprint
import sys
from typing import Callable, Optional

# pylint: disable=wrong-import-position
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# pylint: enable=wrong-import-position

import psutil
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


def process_mem(fmt: str = "G") -> str:
    if fmt == "B":
        d = 1
    elif fmt == "M":
        d = 2
    elif fmt == "G":
        d = 3
    else:
        raise ValueError()
    m = psutil.Process(os.getpid()).memory_info().rss / 1024**d
    return f"{round(m, 2)}{fmt}"


def tokenization_gen(
    files: list[Path],
    batch_size: int,
    block_size: int,
    bytes_to_str: Callable[[bytes], str],
    total: Optional[int] = None,
) -> Generator[list[str], None, None]:

    def return_batch(lbs: list[bytes | list[int]]) -> list[str]:
        return [bytes_to_str(bytes(bs)) for bs in lbs]

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    future = read_binary_files_asynch(
        files,
        max_length=None,
        in_memory_dtype="bytes",
        disable_tqdm=False,
    )
    lbs = loop.run_until_complete(future)

    byte_stream = chain.from_iterable(lbs)

    pbar = tqdm(batched(byte_stream, block_size), total=total, dynamic_ncols=True)

    batch = []
    for block in pbar:
        pbar.set_postfix({"rss": process_mem("G")})
        batch.append(block)
        if len(batch) == batch_size:
            yield return_batch(batch)
            batch = []
    if batch:
        yield return_batch(batch)


class TrainTokenizer:

    def __init__(
        self,
        lift_level: LiftLevel,
        algorithm: TokenizerAlgorithm,
        vocab_size: int,
        batch_size: int,
        block_size: int,
        num_files: int,
        max_token_length: Optional[int] = None,
    ) -> None:
        self.lift_level = lift_level
        self.algorithm = algorithm
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.block_size = block_size
        self.num_files = num_files
        self.max_token_length = max_token_length

        if self.lift_level == LiftLevel.RAW and self.algorithm == TokenizerAlgorithm.WORDLEVEL:
            raise ValueError(f"WordLevel tokenization at the raw-byte level does not need training.")

    def __call__(self) -> Tokenizer:
        print("Gathering data...")
        files = self.get_files()
        length = self.compute_iterator_length(files)
        bytes_to_str = self.get_bytes_to_str()
        iterator = tokenization_gen(files, self.batch_size, self.block_size, bytes_to_str, length)

        print("Training tokenizer...")
        normalizer = self.get_normalizer()
        pretokenizer = self.get_pre_tokenizer()
        model = self.get_model()
        trainer = self.get_trainer()
        tokenizer = Tokenizer(model)
        tokenizer.normalizer = normalizer
        tokenizer.pre_tokenizer = pretokenizer
        tokenizer.train_from_iterator(iterator, trainer, length)

        return tokenizer

    def get_files(self) -> list[str]:
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

    def compute_iterator_length(self, files: list[str]) -> Optional[int]:
        if self.lift_level == LiftLevel.RAW:
            size = sum(os.stat(f).st_size for f in files)
            return size // self.block_size + 1
        if self.lift_level == LiftLevel.DIS:
            return None  # TODO
        if self.lift_level == LiftLevel.DEC:
            return None  # TODO
        raise ValueError(f"{self.lift_level=}")

    def get_bytes_to_str(self) -> Callable[[bytes], str]:
        if self.lift_level == LiftLevel.RAW:
            return bytes_to_str_utf8
        if self.lift_level == LiftLevel.DIS:
            return bytes.decode
        if self.lift_level == LiftLevel.DEC:
            return bytes.decode
        raise ValueError(f"{self.lift_level=}")

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
        if self.algorithm.lower() == TokenizerAlgorithm.BPE:
            return models.BPE()
        if self.algorithm.lower() == TokenizerAlgorithm.UNIGRAM:
            return models.Unigram()
        if self.algorithm.lower() == TokenizerAlgorithm.WORDPIECE:
            return models.WordPiece()
        if self.algorithm == TokenizerAlgorithm.WORDLEVEL:
            return models.WordLevel()
        raise ValueError(f"{self.algorithm=}")

    def get_trainer(self) -> trainers.Trainer:
        special_tokens = list(SPECIALS.values())
        vocab_size = self.vocab_size + len(special_tokens)
        if self.algorithm.lower() == TokenizerAlgorithm.BPE:
            return trainers.BpeTrainer(
                vocab_size=vocab_size,
                special_tokens=special_tokens,
                max_token_length=self.max_token_length,
            )
        if self.algorithm.lower() == TokenizerAlgorithm.UNIGRAM:
            return trainers.UnigramTrainer(
                vocab_size=vocab_size,
                special_tokens=special_tokens,
                unk_token=SPECIALS["unk_token"],
                max_piece_length=self.max_token_length,
            )
        if self.algorithm.lower() == TokenizerAlgorithm.WORDPIECE:
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
    parser.add_argument("--block_size", type=int, default=2**12)
    parser.add_argument("--batch_size", type=int, default=2**10)
    parser.add_argument("--max_token_length", type=int, default=64)
    args = parser.parse_args()

    pprint(f"args={pprint(dict(args))}")

    tokenizer = TrainTokenizer(
        args.lift_level,
        args.algorithm,
        args.vocab_size,
        args.batch_size,
        args.block_size,
        args.num_files,
        args.max_token_length,
    )()

    io_helper = TokenizerIOHelper(args.lift_level, args.algorithm, args.vocab_size, args.num_files)
    io_helper.save(tokenizer)
    tokenizer = io_helper.load()

    print(f"FINISH @{datetime.now()}")


if __name__ == "__main__":
    main()
