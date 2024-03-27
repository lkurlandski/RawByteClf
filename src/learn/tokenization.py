"""
Handles preprocessing of malware bytes.

# FIXME: there are some issues with trained tokenizers, e.g., Added Token 16391
# FIXME: the SPECIALS dict is duplicated; it is also in the cfg module
"""

# pylint: disable=wrong-import-position
print(f"Entered {__file__=}")

from collections.abc import Iterator, Generator
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product, chain, islice
import math
import os
from pathlib import Path
from pprint import pformat, pprint
import sys
from typing import Callable, Literal, Optional, Union
import warnings

if __name__ == "__main__":
    print(f"STARTING @{datetime.now()}\n{'-' * 88}", flush=True)
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# pylint: enable=wrong-import-position

import psutil
from tokenizers import Regex, Tokenizer
from tokenizers import SentencePieceBPETokenizer as _SentencePieceBPETokenizer
from tokenizers import SentencePieceUnigramTokenizer as _SentencePieceUnigramTokenizer
from tokenizers.implementations.base_tokenizer import BaseTokenizer
from tokenizers import models
from tokenizers import pre_tokenizers
from tokenizers import processors
from tokenizers import trainers
from transformers import HfArgumentParser, PreTrainedTokenizerFast
from tqdm import tqdm

from src.cfg import BR, SPECIALS, TOKENIZERS_OUTPUT_PATH
from src.learn.preprocessing import bytes_to_str_ascii, bytes_to_str_utf8
from src.data.cfg import DATASET_TO_FILES
from src.utils import batched


TokenizerAlgorithm = Literal["Raw", "BPE", "Unigram", "WordPiece", "WordLevel", "SentencePieceBPE", "SentencePieceUnigram"]


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
    algorithm: TokenizerAlgorithm = field(
        metadata={
            "help":
                "One of `Raw`, `BPE`, `Unigram`, `WordPiece`, `WordLevel`, "
                "`SentencePieceBPE`, `SentencePieceUnigram`"
        }
    )
    vocab_size: Optional[int] = field(default=256, metadata={"help": "EXCLUDING SPECIAL TOKENS"})
    num_files: Optional[int] = field(default=None, metadata={"help": ""})
    block_size: int = field(default=2**12, metadata={"help": ""})
    batch_size: int = field(default=2**10, metadata={"help": ""})
    max_token_length: int = field(default=None, metadata={"help": ""})


class SentencePieceUnigramTokenizer(_SentencePieceUnigramTokenizer):
    def train_from_iterator(
        self,
        iterator: Union[Iterator[str], Iterator[Iterator[str]]],
        vocab_size: int = 8000,
        show_progress: bool = True,
        special_tokens=None,
        initial_alphabet: Optional[list[str]] = None,
        unk_token: Optional[str] = None,
        length: Optional[int] = None,
        max_token_length: int = 16,
    ):
        max_token_length = 16 if max_token_length is None else max_token_length

        if special_tokens is None:
            special_tokens = []

        if initial_alphabet is None:
            initial_alphabet = []

        trainer = trainers.UnigramTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            show_progress=show_progress,
            initial_alphabet=initial_alphabet,
            unk_token=unk_token,
            max_piece_length=max_token_length,
            shrinking_factor=0.95,
        )

        self._tokenizer.train_from_iterator(
            iterator,
            trainer=trainer,
            length=length,
        )


class SentencePieceBPETokenizer(_SentencePieceBPETokenizer):
    def train_from_iterator(
        self,
        iterator: Union[Iterator[str], Iterator[Iterator[str]]],
        vocab_size: int = 30000,
        min_frequency: int = 2,
        special_tokens=None,
        limit_alphabet: int = 256 + len(SPECIALS),
        initial_alphabet: list[str] = None,
        show_progress: bool = True,
        length: Optional[int] = None,
    ):
        if special_tokens is None:
            special_tokens = []

        if initial_alphabet is None:
            initial_alphabet = []

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            limit_alphabet=limit_alphabet,
            initial_alphabet=initial_alphabet,
            show_progress=show_progress,
        )
        self._tokenizer.train_from_iterator(
            iterator,
            trainer=trainer,
            length=length,
        )


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
    bytes_to_str: Callable[[bytes], str] = bytes_to_str_ascii,
    total: Optional[int] = None,
) -> Generator[list[str], None, None]:

    def return_batch(lbs: list[bytes | list[int]]) -> list[str]:
        return [bytes_to_str(bytes(bs)) for bs in lbs]

    byte_stream = chain.from_iterable((open(f, "rb").read() for f in files))

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


class FastTokenizerForModelsThatRequireCLSToken(PreTrainedTokenizerFast):
    def build_inputs_with_special_tokens(self,
        token_ids_0: list[int],
        token_ids_1: Optional[list[int]] = None,
    ) -> list:
        output = [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        if token_ids_1 is not None:
            output += token_ids_1 + [self.sep_token_id]
        return output


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


def tokenizer_path(algorithm: TokenizerAlgorithm, vocab_size: int):
    return TOKENIZERS_OUTPUT_PATH / f"{algorithm}_{vocab_size}.json"


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
    model_requires_cls_token: bool = False,
    representation: int = 8,
    algorithm: TokenizerAlgorithm = "Raw",
    vocab_size: Optional[int] = None,
    **kwds,
) -> PreTrainedTokenizerFast:
    """
    kwds
       model_max_length: will caused the tokenizer to trim the tokenized input.
    """

    tokenizer: Tokenizer

    if algorithm.lower() == "raw":
        if representation == 8:
            tokenizer = get_tokenizer_object_8bit()
        elif representation == 12:
            tokenizer = get_tokenizer_object_12bit()
        elif representation == 16:
            tokenizer = get_tokenizer_object_16bit()
        else:
            raise ValueError(f"Representation not supported: {representation}")
        if model_requires_cls_token:
            tokenizer.post_processor = processors.BertProcessing(
                sep=(SPECIALS["sep_token"], SPECIALS_IDS["sep_token"]),
                cls=(SPECIALS["cls_token"], SPECIALS_IDS["cls_token"]),
            )

    else:
        path: Path = tokenizer_path(algorithm, vocab_size)
        if not path.exists():
            raise FileNotFoundError(f"For {representation=}, could not locate {path.as_posix()=}")
        tokenizer = Tokenizer.from_file(path.as_posix())

    if model_requires_cls_token:
        tokenizer = FastTokenizerForModelsThatRequireCLSToken(tokenizer_object=tokenizer, **(kwds | SPECIALS))
    else:
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, **(kwds | SPECIALS))

    tokenizer.add_special_tokens(SPECIALS)
    return tokenizer


def train_tokenizer(
    algorithm: str,
    vocab_size: int,
    batch_size: int,
    block_size: int,
    num_files: int = None,
    max_token_length: int = None,
    save_to_file: bool = True,
) -> BaseTokenizer:

    vocab_size_with_specials = vocab_size + len(SPECIALS)

    if max_token_length and (256**max_token_length < vocab_size):
        raise ValueError(f"{vocab_size=} too big for {max_token_length=}")

    files = list(islice(DATASET_TO_FILES["binaries"]["sorel_pe"](), num_files))
    length = sum(f.stat().st_size for f in files) // block_size + 1
    iterator = tokenization_gen(list(map(str, files)), batch_size, block_size, total=length)
    unk_token = SPECIALS["unk_token"]
    special_tokens = list(SPECIALS.values())

    print("Tokenizing...")
    if algorithm == "Raw":
        raise NotImplementedError("No Idea what this is doing...")
        # num_bits = math.log2(vocab_size - len(special_tokens)) / 8
        # if not num_bits.is_integer():
        #     raise ValueError(
        #         f"{vocab_size=} is invalid for {algorithm=}. Requires power of 2 divisible by 8."
        #     )
        # num_bits = int(num_bits)
        # alphabet = list(BYTE_TO_UTF8.values())
        # alphabet = ("".join(i) for i in product(alphabet, repeat=num_bits))
        # vocab = {v: i for i, v in enumerate(special_tokens)}
        # vocab.update({v: i for i, v in enumerate(alphabet, start=len(special_tokens))})
        # model = models.WordLevel(
        #     vocab=vocab,
        #     unk_token=unk_token,
        # )
        # tokenizer = Tokenizer(model)
        # tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        #     [
        #         pre_tokenizers.Split(Regex("."), behavior="isolated"),
        #     ]
        # )
    elif algorithm == "SentencePieceBPE":
        tokenizer = SentencePieceBPETokenizer()
        tokenizer.train_from_iterator(
            iterator,
            vocab_size=vocab_size_with_specials,
            special_tokens=special_tokens,
            length=length,
        )
    elif algorithm == "SentencePieceUnigram":
        tokenizer = SentencePieceUnigramTokenizer()
        tokenizer.train_from_iterator(
            iterator,
            vocab_size=vocab_size_with_specials,
            show_progress=False,
            special_tokens=special_tokens,
            length=length,
            unk_token=unk_token,
            max_token_length=max_token_length,
        )
    else:
        if algorithm == "BPE":
            model = models.BPE()
            trainer = trainers.BpeTrainer(
                vocab_size=vocab_size_with_specials,
                special_tokens=special_tokens,
                max_token_length=max_token_length,
            )
        elif algorithm == "Unigram":
            model = models.Unigram()
            trainer = trainers.UnigramTrainer(
                vocab_size=vocab_size_with_specials,
                special_tokens=special_tokens,
                unk_token=unk_token,
                max_piece_length=max_token_length,
            )
        elif algorithm == "WordPiece":
            model = models.WordPiece()
            trainer = trainers.WordPieceTrainer(
                vocab_size=vocab_size_with_specials,
                special_tokens=special_tokens,
            )
        elif algorithm == "WordLevel":
            model = models.WordLevel()
            trainer = trainers.WordLevelTrainer(
                vocab_size=vocab_size_with_specials,
                special_tokens=special_tokens,
            )
        else:
            raise ValueError(f"{algorithm} is invalid.")

        tokenizer = Tokenizer(model)
        tokenizer.train_from_iterator(iterator, trainer, length=length)

    print("Training complete!", flush=True)
    if save_to_file:
        path = tokenizer_path(algorithm, vocab_size)
        path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(path.as_posix())
    return tokenizer


def cli():
    parser = HfArgumentParser((TokenizationArgs,))
    args = parser.parse_args_into_dataclasses()[0]
    print(f"args={pformat(args)}")
    print(BR, flush=True)
    train_tokenizer(
        args.algorithm,
        args.vocab_size,
        args.batch_size,
        args.block_size,
        args.num_files,
        args.max_token_length,
    )


if __name__ == "__main__":
    print(f"START @{datetime.now()}")
    print(BR, flush=True)
    cli()
    print(f"FINISH @{datetime.now()}")
    print(BR, flush=True)
