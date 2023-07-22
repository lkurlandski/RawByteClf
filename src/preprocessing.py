"""
Preprocess data by strictly tokenizing it, i.e., no padding or tokenization.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from pprint import pformat, pprint
import shutil
import sys
from typing import Optional

from datasets import Dataset, DatasetDict
from transformers import HfArgumentParser, PreTrainedTokenizer

from cfg import *
from data import microsoft_dataset_callable
from tokenization import get_fast_tokenizer, tokenizer_path


@dataclass
class DatasetArgs:
    algorithm: str = field(metadata={"help": ""})
    vocab_size: Optional[int] = field(default=256, metadata={"help": ""})
    num_tok: int = field(default=1000, metadata={"help": ""})
    num: float = field(default=None, metadata={"help": ""})
    num_proc: int = field(default=1, metadata={"help": ""})
    batch_size: int = field(default=1000, metadata={"help": ""})
    writer_batch_size: int = field(default=1000, metadata={"help": ""})
    datasets_root_path: str = field(default=None, metadata={"help": ""})

    def __post_init__(self) -> None:
        if self.datasets_root_path is None:
            self.datasets_root_path = DATASETS / self.algorithm / str(self.vocab_size)


def get_tokenize_fn(tokenizer: PreTrainedTokenizer):
    def fn(examples):
        return tokenizer(examples["text"], truncation=False, padding=False)

    return fn


def datasets_path(root: Path = DATASETS, num: int = None) -> Path:
    return root / (str(num) if num is not None else "full")


def main(
    algorithm: str = "SentencePieceBPE",
    vocab_size: int = 256,
    num_tok: int = 1000,
    num: float = None,
    num_proc: int = 1,
    batch_size: int = 1000,
    writer_batch_size: int = 1000,
    datasets_root_path: str = None,
) -> Dataset:
    print("Fetching raw datasets...")

    tr = Dataset.from_generator(microsoft_dataset_callable(splits=["tr"]))
    vl = Dataset.from_generator(microsoft_dataset_callable(splits=["vl"]))
    ts = Dataset.from_generator(microsoft_dataset_callable(splits=["ts"]))
    if num:
        tr = tr.select(range(int(num * tr.num_rows)))
        vl = vl.select(range(int(num * vl.num_rows)))
        ts = tr.select(range(int(num * ts.num_rows)))
    dataset = DatasetDict({"tr": tr, "ts": ts, "vl": vl}).remove_columns(["file"])
    print(f"{dataset=}")
    print(BR, flush=True)

    tokenizer_file = tokenizer_path(algorithm, vocab_size, num_tok)
    tokenizer = get_fast_tokenizer(tokenizer_file, None)
    print(f"{tokenizer=}")
    print("Tokenizing...")
    print(BR, flush=True)
    dataset = dataset.map(
        get_tokenize_fn(tokenizer),
        batched=True,
        num_proc=num_proc,
        batch_size=batch_size,
        writer_batch_size=writer_batch_size,
    )
    path = datasets_path(datasets_root_path, num)
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(exist_ok=True, parents=True)
    dataset.save_to_disk(path.as_posix())
    return dataset


def cli():
    parser = HfArgumentParser((DatasetArgs,))
    args = parser.parse_args_into_dataclasses()[0]
    print(f"args={pformat(args)}")
    print(BR, flush=True)
    main(
        args.algorithm,
        args.vocab_size,
        args.num_tok,
        args.num,
        args.num_proc,
        args.batch_size,
        args.writer_batch_size,
        args.datasets_root_path,
    )


if __name__ == "__main__":
    print(f"START @{datetime.now()}")
    print(BR, flush=True)
    cli()
    print(f"FINISH @{datetime.now()}")
    print(BR, flush=True)
