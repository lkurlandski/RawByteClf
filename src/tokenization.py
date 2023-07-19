"""Tokenization.

Notes
-----
  - consider using the byte-level bpe tokenizer from huggingface.
  - consider max_token_length and max_piece length
    - should be same for both bpe and unigram
    - 256 ** l >= vocab_size
    - l >= log_256(vocab_size)

 - need to perform a thorough investigation of tokenization algorithms
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pformat, pprint
from typing import Optional, Union

from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers import SentencePieceBPETokenizer as _SentencePieceBPETokenizer
from tokenizers import SentencePieceUnigramTokenizer as _SentencePieceUnigramTokenizer
from tokenizers.implementations.base_tokenizer import BaseTokenizer
from tokenizers import models
from tokenizers import trainers
from transformers import HfArgumentParser, PreTrainedTokenizerFast
from tqdm import tqdm

from cfg import *
from data import MicrosoftDatasetGen
from utils import process_mem


@dataclass
class TokenizationArgs:
    algorithm: str = field(metadata={"help": ""})
    vocab_size: Optional[int] = field(default=256, metadata={"help": ""})
    num_files: Optional[int] = field(default=None, metadata={"help": ""})
    block_size: int = field(default=2**12, metadata={"help": ""})
    batch_size: int = field(default=2**10, metadata={"help": ""})
    max_token_length: int = field(default=None, metadata={"help": ""})

    def __post_init__(self):
        self.vocab_size += len(SPECIALS)


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
        limit_alphabet: int = 1000,
        initial_alphabet: list[str] = None,
        show_progress: bool = True,
        length: Optional[int] = None,
        max_token_length: int = None,
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
            max_token_length=max_token_length,
        )
        self._tokenizer.train_from_iterator(
            iterator,
            trainer=trainer,
            length=length,
        )


def tokenization_gen(dataset: Dataset, batch_size: int = 512):
    pbar = tqdm(dataset.iter(batch_size), total=len(dataset) // batch_size, dynamic_ncols=True)
    for batch in pbar:
        pbar.set_postfix({"rss": process_mem("G")})
        yield batch["text"]


def tokenizer_path(algorithm: str, vocab_size: int) -> Path:
    return TOKENIZERS / algorithm / str(vocab_size) / "vocab.json"


def get_fast_tokenizer(
    tokenizer: Tokenizer | Path | str,
    model_max_length: int,
) -> PreTrainedTokenizerFast:
    if isinstance(tokenizer, Path):
        tokenizer = tokenizer.as_posix()
    if isinstance(tokenizer, str):
        tokenizer = Tokenizer.from_file(tokenizer)
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        model_max_length=model_max_length,
    )
    fast_tokenizer.add_special_tokens(SPECIALS)
    return fast_tokenizer


def main(
    vocab_size: int,
    algorithm: str,
    batch_size: int,
    num_files: int = None,
    max_token_length: int = None,
) -> BaseTokenizer:

    if max_token_length:
        assert 256**max_token_length >= vocab_size

    dataset = Dataset.from_generator(MicrosoftDatasetGen(num_files))
    iterator = tokenization_gen(dataset, batch_size)
    length = len(dataset) // batch_size
    unk_token = SPECIALS["unk_token"]
    special_tokens = list(SPECIALS.values())

    if algorithm == "SentencePieceBPE":
        tokenizer = SentencePieceBPETokenizer()
        tokenizer.train_from_iterator(
            iterator,
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            length=length,
            max_token_length=max_token_length,
        )
    elif algorithm == "SentencePieceUnigram":
        tokenizer = SentencePieceUnigramTokenizer()
        tokenizer.train_from_iterator(
            iterator,
            vocab_size=vocab_size,
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
                vocab_size=vocab_size,
                special_tokens=special_tokens,
                max_token_length=max_token_length,
            )
        elif algorithm == "Unigram":
            model = models.Unigram()
            trainer = trainers.UnigramTrainer(
                vocab_size=vocab_size,
                special_tokens=special_tokens,
                unk_token=unk_token,
                max_piece_length=max_token_length,
            )
        elif algorithm == "WordPiece":
            model = models.WordPiece()
            trainer = trainers.WordPieceTrainer(
                vocab_size=vocab_size,
                special_tokens=special_tokens,
            )
        elif algorithm == "WordLevel":
            model = models.WordLevel()
            trainer = trainers.WordLevelTrainer(
                vocab_size=vocab_size,
                special_tokens=special_tokens,
            )
        else:
            raise ValueError(f"{algorithm} is invalid.")

        tokenizer = Tokenizer(model)
        tokenizer.train_from_iterator(iterator, trainer, length=length)

    path = tokenizer_path(algorithm, vocab_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(path.as_posix())

    return tokenizer


def cli():
    parser = HfArgumentParser((TokenizationArgs,))
    args = parser.parse_args_into_dataclasses()[0]
    print(f"args={pformat(args)}")
    print(BR, flush=True)
    main(
        args.vocab_size,
        args.algorithm,
        args.batch_size,
        args.num_files,
        args.max_token_length,
    )


if __name__ == "__main__":
    cli()
