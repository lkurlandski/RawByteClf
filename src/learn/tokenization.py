"""
Handles preprocessing of malware bytes.

# FIXME: there are some issues with trained tokenizers, e.g., Added Token 16391
# FIXME: the SPECIALS dict is duplicated; it is also in the cfg module
"""

# pylint: disable=wrong-import-position
print(f"Entered {__file__=}")

import asyncio
from collections.abc import Iterator, Generator
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product, chain, islice
import inspect
import math
import os
from pathlib import Path
import pickle
from pprint import pformat, pprint
import re
import shutil
import sys
from typing import Any, Callable, Literal, Optional
import warnings

if __name__ == "__main__":
    print(f"STARTING @{datetime.now()}\n{'-' * 88}", flush=True)
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# pylint: enable=wrong-import-position

import psutil
from tokenizers import Regex, Tokenizer, NormalizedString
from tokenizers import models
from tokenizers.models import Model
from tokenizers import normalizers
from tokenizers.normalizers import Normalizer
from tokenizers import pre_tokenizers
from tokenizers.pre_tokenizers import PreTokenizer
from tokenizers import processors
from tokenizers.processors import PostProcessor
from tokenizers import trainers
from tokenizers.trainers import Trainer
from transformers import HfArgumentParser, PreTrainedTokenizerFast
from tqdm import tqdm

from src.cfg import BR, SPECIALS, TOKENIZERS_OUTPUT_PATH
from src.learn.preprocessing import bytes_to_str_utf8
from src.data.cfg import DATASET_TO_FILES
from src.data.utils import read_binary_files_asynch
from src.utils import batched, get_highest_path, COMPRESSION_TYPES, ENCRYPTION_TYPES


LiftLevel = Literal["raw", "dis", "dec"]

TokenizerAlgorithm = Literal[
    "Raw",
    "BPE",
    "Unigram",
    "WordPiece",
    "WordLevel",
]

SPECIALS = OrderedDict(
    {
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "mask_token": "<msk>",
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "cls_token": "<cls>",
        "sep_token": "<sep>",
    }
)
SPECIALS_IDS = {k: i for i, k in enumerate(SPECIALS)}


@dataclass
class TokenizationArgs:
    lift_level: LiftLevel = field()
    algorithm: TokenizerAlgorithm = field()
    vocab_size: Optional[int] = field(default=256, metadata={"help": "EXCLUDING SPECIAL TOKENS"})
    num_files: Optional[int] = field(default=None, metadata={"help": ""})
    block_size: int = field(default=2**12, metadata={"help": ""})
    batch_size: int = field(default=2**10, metadata={"help": ""})
    max_token_length: int = field(default=None, metadata={"help": ""})


################################################################################
# Utilities for streaming the data for training the tokenizers. Note that the
# typical manner of using asyncio to read files asynchronously results in error:
    # RuntimeError: There is no current event loop in thread 'Dummy-1'
# so we use a more convoluted way to read the files quickly.
################################################################################


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
    bytes_to_str: Callable[[bytes], str] = bytes_to_str_utf8,
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


################################################################################
# "Tokenzing" a stream of raw-bytes into integers using huggingface's tokenizers
# library is actually nontrivial. These functions return Tokenizer objects
# configured with the appropriate pre-tokenization helpers to do just that.
################################################################################


def get_tokenizer_object_8bit() -> Tokenizer:
    alphabet = [bytes([i]).decode("latin1") for i in range(256)]
    vocab = {v: i for i, v in enumerate(SPECIALS.values())} | {
        v: i for i, v in enumerate(alphabet, start=len(SPECIALS))
    }

    model = models.WordLevel(vocab=vocab, unk_token=SPECIALS["unk_token"])
    tokenizer = Tokenizer(model)
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            # For some reason, when \n\n is encountered, the Regex(".") fails to
            # split, so we need to split on \n first then split on . (matches all).
            pre_tokenizers.Split(Regex("\n"), behavior="isolated"),
            pre_tokenizers.Split(Regex("."), behavior="isolated"),
        ]
    )

    return tokenizer


def get_tokenizer_object_12bit() -> Tokenizer:
    warnings.warn("Warning: the full tokenizer functionality is not implemented for 12-bits!")
    alphabet = [
        bytes([i]).decode("latin1") + bytes([j]).decode("latin1")
        for i in range(16)
        for j in range(256)
    ]
    vocab = {v: i for i, v in enumerate(SPECIALS.values())} | {
        v: i for i, v in enumerate(alphabet, start=len(SPECIALS))
    }

    model = models.WordLevel(vocab=vocab, unk_token=SPECIALS["unk_token"])
    tokenizer = Tokenizer(model)
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(r"[\s\S]{1,2}"), behavior="isolated"),
        ]
    )

    return tokenizer


def get_tokenizer_object_16bit() -> Tokenizer:
    alphabet = [
        bytes([i]).decode("latin1") + bytes([j]).decode("latin1")
        for i in range(256)
        for j in range(256)
    ]
    vocab = {v: i for i, v in enumerate(SPECIALS.values())} | {
        v: i for i, v in enumerate(alphabet, start=len(SPECIALS))
    }

    model = models.WordLevel(vocab=vocab, unk_token=SPECIALS["unk_token"])
    tokenizer = Tokenizer(model)
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(r"[\s\S]{1,2}"), behavior="isolated"),
        ]
    )

    return tokenizer


def tokenizer_path_read(
    lift_level: LiftLevel,
    algorithm: TokenizerAlgorithm,
    vocab_size: int,
    num_files: Optional[int] = None,
) -> Path:
    if num_files is not None:
        return tokenizer_path(lift_level, algorithm, vocab_size, num_files)
    return get_highest_path(
        list(TOKENIZERS_OUTPUT_PATH.glob(f"{lift_level}_{algorithm}_{vocab_size}_*.json")),
        lstrip=f"{lift_level}_{algorithm}_{vocab_size}_",
        rstrip=".json",
    )


def tokenizer_path(
    lift_level: LiftLevel,
    algorithm: TokenizerAlgorithm,
    vocab_size: int,
    num_files: int,
) -> Path:
    return TOKENIZERS_OUTPUT_PATH / f"{lift_level}_{algorithm}_{vocab_size}_{num_files}.json"


def get_fast_tokenizer(
    tokenizer: Tokenizer | Path | str,
    model_max_length: int,
) -> PreTrainedTokenizerFast:
    if isinstance(tokenizer, (Path, str)):
        tokenizer = Path(tokenizer)
        if not tokenizer.exists():
            raise FileNotFoundError(f"{tokenizer=}")
        tokenizer = Tokenizer.from_file(tokenizer.as_posix())
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        model_max_length=model_max_length,
    )
    fast_tokenizer.add_special_tokens(SPECIALS)
    return fast_tokenizer


def get_tokenizer(
    representation: int = 8,
    lift_level: LiftLevel = "raw",
    algorithm: TokenizerAlgorithm = "Raw",
    vocab_size: Optional[int] = None,
    add_cls_token: bool = False,
    add_bos_token: bool = False,
    add_eos_token: bool = False,
    add_sep_token: bool = False,
    **kwds,
) -> PreTrainedTokenizerFast:
    """
    kwds
       model_max_length: will caused the tokenizer to trim the tokenized input.
    """

    tokenizer: Tokenizer

    if algorithm.lower() == "raw" or algorithm in COMPRESSION_TYPES + ENCRYPTION_TYPES:
        if lift_level.lower() != "raw":
            raise ValueError(f"Must be working with raw bytes. {lift_level=}")

        if representation == 8:
            tokenizer = get_tokenizer_object_8bit()
        elif representation == 12:
            tokenizer = get_tokenizer_object_12bit()
        elif representation == 16:
            tokenizer = get_tokenizer_object_16bit()
        else:
            raise ValueError(f"Representation not supported: {representation}")

    else:
        path: Path = tokenizer_path_read(lift_level, algorithm, vocab_size)
        if not path.exists():
            raise FileNotFoundError(f"Could not locate {path.as_posix()=}")
        tokenizer = Tokenizer.from_file(path.as_posix())

    if add_bos_token:
        start = SPECIALS["bos_token"]
    elif add_cls_token:
        start = SPECIALS["cls_token"]
    else:
        start = None

    if add_eos_token:
        end = SPECIALS["eos_token"]
    elif add_sep_token:
        end = SPECIALS["sep_token"]
    else:
        end = None

    if start or end:  # FYI, I have idea what this does for the `pair`
        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{start} $0 {end}",
            pair=f"{start} $A {end} {SPECIALS['sep_token']} {start} $B:1 {end}",
            special_tokens=tuple((s, i) for i, s in enumerate(SPECIALS.values())),
        )

    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, **(kwds | SPECIALS))
    return tokenizer


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


class RemoveCommentsFromCharacterStream:

    def __init__(self) -> None:
        ...

    def __call__(self, char: str) -> str:
        ...


class HexCapitalizationNormalizer:

    def normalize(self, normalized: NormalizedString):
        func = CapitalizeHexCharactersFromCharacterStream()
        normalized.map(func)


SENTINAL_PATTERN = Regex(r"a^")
SENTINAL_NORMALIZER = normalizers.Replace(SENTINAL_PATTERN, "")
SENTINAL_PRETOKENIZER = pre_tokenizers.Split(SENTINAL_PATTERN, "isolated")

RAW_NORMALIZER = None
RAW_PRETOKENIZER = None

DIS_NORMALIZER = normalizers.Sequence([
    # Removes the text between four tabs.
    normalizers.Replace(Regex(r"^(.+?\t)(.+?\t)(.+?\t)(.+?\t)"), ""),
    # Standardizes text.
    normalizers.NFD(),
    normalizers.StripAccents(),
    normalizers.Lowercase(),
    # Capitalize hexidecimal digits.
    normalizers.Normalizer.custom(HexCapitalizationNormalizer()),
])
DIS_PRETOKENIZER = pre_tokenizers.Sequence([
    # Split on any non-alphanumeric character, except "_".
    pre_tokenizers.Split(Regex(r"[^a-zA-Z0-9_]"), behavior="isolated"),
    # Split on whitespace.
    pre_tokenizers.Split(Regex(r"\s"), behavior="removed"),
    # Split hexadecimal strings into "0x" and the constituent digits.
    pre_tokenizers.Split(Regex(r"(0x)|[0-9A-F]"), behavior="isolated"),
])

DEC_NORMALIZER = None
DEC_PRETOKENIZER = None


class TokenizerIOHelper:

    def __init__(self, lift_level: str, algorithm: str, vocab_size: int, num_files: int) -> None:
        self.lift_level = lift_level
        self.algorithm = algorithm
        self.vocab_size = vocab_size
        self.num_files = num_files

    @property
    def path(self) -> Path:
        return TOKENIZERS_OUTPUT_PATH \
            / f"{self.lift_level}" \
            / f"{self.algorithm}" \
            / f"{self.vocab_size}" \
            / f"{self.num_files}"

    @property
    def outfile(self) -> Path:
        return self.path / "tokenizer.json"

    def save(self, tokenizer: Tokenizer) -> None:

        shutil.rmtree(self.path, ignore_errors=True)
        self.path.mkdir(parents=True)

        if self.lift_level == "raw":
            raise NotImplementedError()
        elif self.lift_level == "dis":
            tokenizer.normalizer = SENTINAL_NORMALIZER
        elif self.lift_level == "dec":
            raise NotImplementedError()
        else:
            raise ValueError(f"{self.lift_level=}")

        tokenizer.save(self.outfile.as_posix())

    def load(self) -> Tokenizer:

        tokenizer = Tokenizer.from_file(self.outfile.as_posix())

        if self.lift_level == "raw":
            raise NotImplementedError()
        elif self.lift_level == "dis":
            tokenizer.normalizer = DIS_NORMALIZER
        elif self.lift_level == "dec":
            raise NotImplementedError()
        else:
            raise ValueError(f"{self.lift_level=}")

        return tokenizer

    @staticmethod
    def get_variable_name(var: Any) -> list[str]:
        callers_local_vars = inspect.currentframe().f_back.f_locals.items()
        return [var_name for var_name, var_val in callers_local_vars if var_val is var]


class TrainTokenizer:

    def __init__(
        self,
        lift_level: str,
        algorithm: str,
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

    def __call__(self) -> Tokenizer:
        print("Gathering data...")
        files = self.get_files()
        length = self.compute_iterator_length(files)
        bytes_to_str = self.get_bytes_to_str()
        iterator = tokenization_gen(files, self.batch_size, self.block_size, bytes_to_str, length)

        print("Training tokenizer...")
        normalizer = self.get_normalizer()
        pre_tokenizer = self.get_pre_tokenizer()
        model = self.get_model()
        trainer = self.get_trainer()
        tokenizer = Tokenizer(model)
        tokenizer.normalizer = normalizer
        tokenizer.pre_tokenizer = pre_tokenizer
        tokenizer.train_from_iterator(iterator, trainer, length)

        return tokenizer

    def get_files(self) -> list[str]:
        if self.lift_level == "raw":
            key = "binaries"
        elif self.lift_level == "dis":
            key = "disassembled"
        elif self.lift_level == "dec":
            key = "decompiled"
        else:
            raise ValueError(f"{self.algorithm=}")
        files = DATASET_TO_FILES[key]["sorel_pe"]()
        files = sorted(map(str, islice(files, self.num_files)))
        return files

    def compute_iterator_length(self, files: list[str]) -> Optional[int]:
        if self.lift_level == "raw":
            size = sum(os.stat(f).st_size for f in files)
            return size // self.block_size + 1
        if self.lift_level == "dis":
            return None
        if self.lift_level == "dec":
            return None
        raise ValueError(f"{self.lift_level=}")

    def get_bytes_to_str(self) -> Callable[[bytes], str]:
        if self.lift_level == "raw":
            return bytes_to_str_utf8
        if self.lift_level == "dis":
            return bytes.decode
        if self.lift_level == "dec":
            return bytes.decode
        raise ValueError(f"{self.lift_level=}")

    def get_normalizer(self) -> normalizers.Normalizer:
        if self.lift_level == "raw":
            return RAW_NORMALIZER
        if self.lift_level == "dis":
            return DIS_NORMALIZER
        if self.lift_level == "dec":
            return DEC_NORMALIZER
        raise ValueError(f"{self.lift_level=}")

    def get_pre_tokenizer(self) -> pre_tokenizers.PreTokenizer:
        if self.lift_level == "raw":
            return RAW_PRETOKENIZER
        if self.lift_level == "dis":
            return DIS_PRETOKENIZER
        if self.lift_level == "dec":
            return DEC_PRETOKENIZER
        raise ValueError(f"{self.lift_level=}")

    def get_model(self) -> models.Model:
        # TODO: should the Model recieve the unk_token and/or unk_token_id?
        if self.algorithm.lower() == "bpe":
            return models.BPE()
        if self.algorithm.lower() == "unigram":
            return models.Unigram()
        if self.algorithm.lower() == "wordpiece":
            return models.WordPiece()
        if self.algorithm == "WordLevel":
            return models.WordLevel()
        raise ValueError(f"{self.algorithm=}")

    def get_trainer(self) -> trainers.Trainer:
        special_tokens = list(SPECIALS.values())
        vocab_size = self.vocab_size + len(special_tokens)
        if self.algorithm.lower() == "bpe":
            return trainers.BpeTrainer(
                vocab_size=vocab_size,
                special_tokens=special_tokens,
                max_token_length=self.max_token_length,
            )
        if self.algorithm.lower() == "unigram":
            return trainers.UnigramTrainer(
                vocab_size=vocab_size,
                special_tokens=special_tokens,
                unk_token=SPECIALS["unk_token"],
                max_piece_length=self.max_token_length,
            )
        if self.algorithm.lower() == "wordpiece":
            return trainers.WordPieceTrainer(
                vocab_size=vocab_size,
                special_tokens=special_tokens,
            )
        if self.algorithm == "WordLevel":
            return trainers.WordLevelTrainer(
                vocab_size=vocab_size,
                special_tokens=special_tokens,
            )
        raise ValueError(f"{self.algorithm=}")


def main():
    print(f"START @{datetime.now()}")

    parser = HfArgumentParser((TokenizationArgs,))
    args = parser.parse_args_into_dataclasses()[0]

    trainer = TrainTokenizer(
        args.lift_level,
        args.algorithm,
        args.vocab_size,
        args.batch_size,
        args.block_size,
        args.num_files,
        args.max_token_length,
    )
    tokenizer = trainer()

    io_helper = TokenizerIOHelper(args.lift_level, args.algorithm, args.vocab_size, args.num_files)
    io_helper.save(tokenizer)
    tokenizer = io_helper.load()

    print(f"FINISH @{datetime.now()}")


if __name__ == "__main__":
    main()
