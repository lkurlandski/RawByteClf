"""
"""

from collections import Counter, defaultdict
from itertools import chain
import json
import os
import sys

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src
from src.enums import LiftLevel, TokenizationAlgorithm, BitsInByte, Task
from src.data.loaders_core import get_materials_esp_clm
from src.data.loaders_hf import get_dataset_hf
from src.tokenization.api import get_fast_tokenizer
from src.tokenization.helpers import TokenizerIOHelper
from src.learn.helpers import Args
from src.learn.train import get_processed_dataset_hf


N_TR = 10000
N_VL = 0

bits_in_byte       = BitsInByte.EIGHT
task               = Task.CLM
num_shards         = 1
batch_size         = 64
src.BATCH_SIZE_ITR = batch_size


# for lift_level in [LiftLevel.RAW, LiftLevel.DISASSEMBLED, LiftLevel.DECOMPILED]:
for lift_level in [LiftLevel.RAW]:
# for lift_level in [LiftLevel.DISASSEMBLED]:
# for lift_level in [LiftLevel.DECOMPILED]:

    materials = get_materials_esp_clm(lift_level=lift_level, rm_finetuning_files=False)
    total = (len(materials.files["tr"]) + len(materials.files["vl"]))
    if N_TR is not None:
        total = N_TR
    if N_VL is not None:
        total += N_VL
    total = total // batch_size

    args = Args(
        task=task,
        streaming=True,
        max_length=sys.maxsize,
        lift_level=lift_level,
        bits_in_byte=bits_in_byte,
    )

    dataset = get_dataset_hf(materials, args.streaming, num_shards, max_length=sys.maxsize)
    dataset["tr"] = dataset["tr"].take(N_TR)
    dataset["vl"] = dataset["vl"].take(N_VL)
    stream  = chain(dataset["tr"].iter(batch_size), dataset["vl"].iter(batch_size))
    lengths = []
    for d in tqdm(stream, desc=f"Computing baseline for {lift_level}...", total=total):
        lengths.extend([len(b) for b in d["bytes"]])
    io_helper = TokenizerIOHelper(lift_level, TokenizationAlgorithm.WORDLEVEL, 256, 0)
    io_helper.path.mkdir(parents=True, exist_ok=True)
    io_helper.save_sequence_lengths(lengths)

    for algorithm in TokenizationAlgorithm.BPE, TokenizationAlgorithm.UNIGRAM:
        # for vocab_size in [1024, 4096, 16384]:
        for vocab_size in [16384]:
            tokenizer = get_fast_tokenizer(lift_level, algorithm, bits_in_byte, vocab_size)
            tokenizer.model_input_names = ["input_ids"]
            args = Args(
                task=task,
                streaming=True,
                max_length=sys.maxsize,
                lift_level=lift_level,
                bits_in_byte=bits_in_byte,
                tokenization_algorithm=algorithm,
                vocab_size=vocab_size,
            )

            dataset = get_processed_dataset_hf(materials, args, num_shards, tokenizer)
            dataset["tr"] = dataset["tr"].take(N_TR)
            dataset["vl"] = dataset["vl"].take(N_VL)
            stream  = chain(dataset["tr"].iter(batch_size), dataset["vl"].iter(batch_size))
            lengths = []
            for d in tqdm(stream, desc=f"Computing tokenized for {lift_level}-{algorithm}-{vocab_size}...", total=total):
                lengths.extend([len(i) for i in d["input_ids"]])

            io_helper = TokenizerIOHelper.fromdisk(lift_level, algorithm, vocab_size, None)
            io_helper.save_sequence_lengths(lengths)
