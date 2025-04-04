"""
Compute statistics about the tokenizers.
"""

from argparse import ArgumentParser
from collections import Counter, defaultdict
from itertools import chain
import json
import os
import sys
from typing import Optional

from datasets import Dataset
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

# pylint: disable=wrong-import-position
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# pylint: enable=wrong-import-position

from src.enums import LiftLevel, TokenizationAlgorithm, BitsInByte, Task
from src.data.loaders_core import _get_materials_esp_lm, Materials
from src.data.loaders_hf import get_dataset_hf
from src.tokenization.api import get_fast_tokenizer
from src.tokenization.helpers import TokenizerIOHelper
from src.learn.helpers import Args
from src.learn.train import get_processed_dataset_hf


NUM_FILES = 10000


def compute_sequence_lengths(dataset: Dataset, field: str, batch_size: int = 250) -> list[int]:
    lengths = []
    for d in tqdm(dataset.iter(batch_size)):
        lengths.extend([len(i) for i in d[field]])
    return lengths


def get_args(lift_level: LiftLevel, algorithm: TokenizationAlgorithm, vocab_size: int) -> Args:
    return Args(
        lift_level=lift_level,
        tokenization_algorithm=algorithm,
        vocab_size=vocab_size,
        task=Task.CLM,
        streaming=True,
        max_length=sys.maxsize,
        bits_in_byte=BitsInByte.EIGHT,
    )


def run(lift_level: LiftLevel, algorithm: TokenizationAlgorithm, vocab_size: int, idx: Optional[int] = None, materials: Optional[Materials] = None) -> None:

    print(f"run: lift_level={lift_level.value}")
    print(f"run: algorithm={algorithm.value}")
    print(f"run: {vocab_size=}")
    print(f"run: {idx=}")

    if idx is not None:
        if not idx >= 0 or not idx <= 9:
            raise RuntimeError("idx must be between 0 and 9, inclusive.")
        siz = int(NUM_FILES / 10)
        low = siz * idx
        upp = low + siz
        print(f"run: {siz=}")
        print(f"run: {low=}")
        print(f"run: {upp=}")

    if materials is None:
        materials = _get_materials_esp_lm(lift_level=lift_level)
        materials.files["tr"] = sorted(materials.files["tr"], key=lambda af: af.name)[0:NUM_FILES][low:upp]
        materials.files["vl"] = []

    if algorithm == TokenizationAlgorithm.WORDLEVEL and vocab_size == 256:
        dataset = get_dataset_hf(materials, streaming=True, max_length=sys.maxsize)["tr"]
        print("run: compute_sequence_lengths")
        lengths = compute_sequence_lengths(dataset, "bytes")
        num_files = 0
    else:
        tokenizer = get_fast_tokenizer(lift_level, algorithm, BitsInByte.EIGHT, vocab_size)
        tokenizer.model_input_names = ["input_ids"]
        args = get_args(lift_level, algorithm, vocab_size)
        dataset = get_processed_dataset_hf(materials, lift_level, args, None, tokenizer)["tr"]
        print("run: compute_sequence_lengths")
        lengths = compute_sequence_lengths(dataset, "input_ids")
        num_files = 4096

    io_helper = TokenizerIOHelper(lift_level, algorithm, vocab_size, num_files)
    io_helper.path.mkdir(parents=True, exist_ok=True)
    io_helper.save_sequence_lengths(lengths, idx)


def main():

    for lift_level in [LiftLevel.NOP, LiftLevel.RAW, LiftLevel.DIS, LiftLevel.DEC]:

        materials = _get_materials_esp_lm(lift_level=lift_level)
        materials.files["tr"] = sorted(materials.files["tr"], key=lambda af: af.name)[0:NUM_FILES]
        materials.files["vl"] = []

        algorithm  = TokenizationAlgorithm.WORDLEVEL
        vocab_size = 256
        run(lift_level, algorithm, vocab_size, materials)

        for algorithm in [TokenizationAlgorithm.BPE, TokenizationAlgorithm.UNIGRAM]:
            for vocab_size in [1024, 4096, 16384]:
                run(lift_level, algorithm, vocab_size, materials)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--lift_level", type=LiftLevel, required=True)
    parser.add_argument("--algorithm", type=TokenizationAlgorithm, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--idx", type=int, required=False)
    args = parser.parse_args()
    run(args.lift_level, args.algorithm, args.vocab_size, args.idx, None)
    # main()
