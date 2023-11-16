"""
Determine which architecture scales for long sequence modeling.
"""

from argparse import ArgumentParser
from collections import OrderedDict
from functools import partial
import gc
import inspect
import os
from pathlib import Path
from pprint import pprint
import sys
import time
from typing import Any

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# pylint: disable=wrong-import-position

from datasets import Dataset, IterableDataset, IterableDatasetDict, concatenate_datasets
from datasets.utils.logging import disable_progress_bar
from datasets.formatting.formatting import LazyBatch, LazyRow
import numpy as np
import pandas as pd
from tokenizers import models, pre_tokenizers, Tokenizer, Regex
import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
)

from src.cfg import INPUT_PATH, OUTPUT_PATH
from src.learn.utils import (
    count_parameters,
    get_fast_tokenizer,
    preprocess_a,
    tokenize_fn,
    get_tokenizer_object,
    find_two_largest_factors,
)


parser = ArgumentParser()
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--pad_to_multiple", type=int, default=2**10)
parser.add_argument("--clm", action="store_true")
parser.add_argument("--low", type=int, default=2**10)
parser.add_argument("--high", type=int, default=2**20)
parser.add_argument(
    "--configs",
    nargs="+",
    default=None,
    choices=[
        "Longformer",
        "BigBird",
        "Rwkv",
        "Yoso",
        "SqueezeBERT",
        "FNet",
        "Nystromformer",
        "Reformer",
    ],
)
args = parser.parse_args()


disable_progress_bar()


dataset = concatenate_datasets(
    [
        Dataset.load_from_disk(INPUT_PATH / "virus_total_dll"),
    ]
)
dataset = dataset.filter(lambda x: x["length"] == 2**20)
dataset = dataset.select(range(args.batch_size))
dataset.cleanup_cache_files()
print(f"{dataset=}")

tokenizer_object = get_tokenizer_object()
print(f"{tokenizer_object=}")


training_arguments = TrainingArguments(
    output_dir=(OUTPUT_PATH / "tmp").as_posix(),
    overwrite_output_dir=True,
    per_device_train_batch_size=args.batch_size,
    num_train_epochs=1,
    fp16=True,
    disable_tqdm=True,
)

configs = {
    "Longformer": (
        lambda max_length: transformers.LongformerConfig(
            vocab_size=264,
            max_position_embeddings=512 + max_length,
            num_hidden_layers=1,
            num_attention_heads=12,
            hidden_size=768,
            intermediate_size=3072,
            attention_window=512,
        ),
        None,
    ),
    "BigBird": (
        lambda max_length: transformers.BigBirdConfig(
            vocab_size=264,
            max_position_embeddings=max_length,
            num_hidden_layers=1,
            num_attention_heads=12,
            hidden_size=768,
            intermediate_size=3072,
            attention_type="original_full" if max_length <= 1024 else "block_sparse",
            block_size=128,
            num_random_blocks=4,
        ),
        None,
    ),
    "FNet": (
        lambda max_length: transformers.FNetConfig(
            vocab_size=264,
            max_position_embeddings=max_length,
            num_hidden_layers=1,
            hidden_size=768,
            intermediate_size=3072,
            hidden_act="gelu_new",
            hidden_dropout_prob=0.1,
        ),
        ["input_ids", "token_type_ids", "position_ids"],
    ),
    "Nystromformer": (
        lambda max_length: transformers.NystromformerConfig(
            vocab_size=264,
            max_position_embeddings=max_length,
            num_hidden_layers=1,
            num_attention_heads=12,
            hidden_size=768,
            intermediate_size=3072,
            segment_means_seq_len=64,
            num_landmarks=64,
            conv_kernel_size=65,
            inv_coeff_init_option=False,
        ),
        None,
    ),
    "Reformer": (
        lambda max_length: transformers.ReformerConfig(
            vocab_size=264,
            max_position_embeddings=max_length,
            num_attention_heads=12,
            hidden_size=768,
            feed_forward_size=3072,
            attention_head_size=64,
            attn_layers=[
                "local",
                "lsh",
            ],  # , "local", "lsh", "local", "lsh"],
            axial_norm_std=1.0,
            axial_pos_embds=True,
            axial_pos_shape=list(find_two_largest_factors(max_length)),
            axial_pos_embds_dim=[768 // 2, 768 // 2],
            chunk_size_lm_head=0,
            hash_seed=None,
            local_num_chunks_before=1,
            local_num_chunks_after=0,
            local_attention_probs_dropout_prob=0.05,
            local_attn_chunk_length=64,
            lsh_attn_chunk_length=64,
            lsh_attention_probs_dropout_prob=0.0,
            lsh_num_chunks_before=1,
            lsh_num_chunks_after=0,
            num_buckets=max_length // 64,
            num_hashes=1,
            pad_token_id=0,
        ),
        None,
    ),
    "SqueezeBert": (
        lambda max_length: transformers.SqueezeBertConfig(
            vocab_size=264,
            max_position_embeddings=max_length,
            hidden_size=768,
            num_hidden_layers=1,
            num_attention_heads=12,
            intermediate_size=3072,
            q_groups=4,
            k_groups=4,
            v_groups=4,
            post_attention_groups=1,
            intermediate_groups=4,
            output_groups=4,
        ),
        None,
    ),
    "Yoso": (
        lambda max_length: transformers.YosoConfig(
            vocab_size=264,
            max_position_embeddings=max_length,
            hidden_size=768,
            num_hidden_layers=1,
            num_attention_heads=12,
            intermediate_size=3072,
            use_expectation=True,
            hash_code_len=9,
            num_hash=64,
            conv_window=None,
            use_fast_hash=True,
            lsh_backward=True,
        ),
        None,
    ),
    "Rwkv": (
        lambda max_length: transformers.RwkvConfig(
            vocab_size=264,
            context_length=max_length,
            hidden_size=768,
            num_hidden_layers=32,
        ),
        None,
    ),
}
assert all(k in configs for k in args.configs)
configs = {k: v for k, v in configs.items() if k in args.configs}


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"


def test(config: tuple, max_length: int) -> tuple[str, float, int]:
    c = config[0](max_length)
    if args.clm:
        model = AutoModelForCausalLM.from_config(c)
    else:
        model = AutoModelForMaskedLM.from_config(c)
    print(
        f"{type(model).__name__} {round(count_parameters(model) / 1e6, 2)}M parameters. @{max_length=}"
    )

    kwds = {}
    if config[1]:
        kwds["model_input_names"] = config[1]
    tokenizer = get_fast_tokenizer(tokenizer_object, max_length=max_length, **kwds)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=not args.clm,
        pad_to_multiple_of=8,
    )

    _dataset = dataset.map(preprocess_a, batched=True, remove_columns=["bytes"]).map(
        partial(tokenize_fn, tokenizer, truncation=True, max_length=max_length),
        batched=True,
        remove_columns=["text", "labels"],
    )

    trainer = transformers.Trainer(
        model=model,
        args=training_arguments,
        train_dataset=_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()

    return count_parameters(model), max_length


log = []

for name, config in configs.items():
    low, high = args.low, args.high

    while low < high - args.pad_to_multiple:
        mid = ((low + high) // (2 * args.pad_to_multiple)) * args.pad_to_multiple
        try:
            params, max_length = test(config, mid)
            low = mid
        except torch.cuda.OutOfMemoryError:
            high = mid

        if high - low == args.pad_to_multiple:
            break

    log.append((name, params, max_length))
