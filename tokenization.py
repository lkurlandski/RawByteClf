"""
"""

from itertools import chain
from pathlib import Path
from pprint import pprint
import sys
import warnings

import datasets
from datasets import Dataset
import evaluate
from itertools import chain
import numpy as np
import torch
import tokenizers
from tokenizers import models
from tokenizers import pre_tokenizers
from tokenizers import trainers
import transformers

from utils import print_gpu_utilization


def tokenization_gen(grouped_dataset: Dataset, batch_size: int = 512):
    for i in range(0, len(grouped_dataset), batch_size):
        yield grouped_dataset[i : i + batch_size]["text"]


def get_tokenizer(
    vocab_size: int,
    special_tokens: dict[str, str] = None,
    dataset: Dataset = None,
    batch_size: int = 1024,
    use_cache: bool = True,
    overwrite_cache: bool = False,
):
    path = Path(f"tokenizers/BPE/{vocab_size}/vocab.json")
    if path.exists():
        if use_cache:
            return tokenizers.Tokenizer.from_file(path.as_posix())
        elif overwrite_cache:
            warnings.warn("Overwriting existing tokenizer.")
        else:
            warnings.warn("Training new tokenizer, but not overwriting existing.")

    model = models.BPE(unk_token=special_tokens["unk_token"])
    tokenizer = tokenizers.Tokenizer(model)
    tokenizer.pre_tokenizer = pre_tokenizers.Split(" ", behavior="removed")
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=list(special_tokens.values()))
    tokenizer.train_from_iterator(tokenization_gen(dataset, batch_size), trainer, length=len(dataset))
    
    if not path.exists() or overwrite_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(path.as_posix())
    
    return tokenizer
