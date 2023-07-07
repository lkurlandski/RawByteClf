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
import gc
from itertools import chain
from pathlib import Path
from pprint import pprint
import sys
from typing import Optional, Union
import warnings

import datasets
from datasets import Dataset
import evaluate
import numpy as np
import torch
from tokenizers import (
    Tokenizer,
    SentencePieceBPETokenizer,
    SentencePieceUnigramTokenizer,
)
from tokenizers.implementations.base_tokenizer import BaseTokenizer
from tokenizers import models
from tokenizers import pre_tokenizers
from tokenizers import trainers
import transformers
from tqdm import tqdm

import utils
from utils import print_gpu_utilization


class SentencePieceUnigramTokenizer_(SentencePieceUnigramTokenizer):
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


class SentencePieceBPETokenizer_(SentencePieceBPETokenizer):
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
        pbar.set_postfix({"rss": utils.process_mem("G")})
        yield batch["text"]


def get_tokenizer(
    vocab_size: int,
    algorithm: str,
    special_tokens: dict[str, str] = None,
    dataset: Dataset = None,
    batch_size: int = 512,
    use_cache: bool = True,
    overwrite_cache: bool = False,
    n_examples: int = None,
    max_token_length: int = None,
) -> BaseTokenizer:
    path = Path(f"tokenizers/{algorithm}/{vocab_size}/vocab.json")
    if path.exists():
        if use_cache:
            return Tokenizer.from_file(path.as_posix())

    print("Training a new tokenizer.", flush=True)
    assert all((special_tokens, dataset, batch_size))
    assert 256**max_token_length >= vocab_size
    if path.exists():
        if overwrite_cache:
            print("Will overwrite existing tokenizer", flush=True)
        else:
            print("Will not overwrite existing tokenizer", flush=True)

    if n_examples and n_examples < dataset.num_rows:
        dataset = dataset.select(range(n_examples))

    iterator = tokenization_gen(dataset, batch_size)
    length = len(dataset) // batch_size
    unk_token = special_tokens["unk_token"]
    special_tokens = list(special_tokens.values())

    if algorithm == "SentencePieceBPE":
        tokenizer = SentencePieceBPETokenizer_()
        tokenizer.train_from_iterator(
            iterator,
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            length=length,
            max_token_length=max_token_length,
        )
    elif algorithm == "SentencePieceUnigram":
        tokenizer = SentencePieceUnigramTokenizer_()
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

    if not path.exists() or overwrite_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(path.as_posix())

    return tokenizer
