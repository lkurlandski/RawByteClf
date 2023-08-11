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

from datasets import concatenate_datasets, Dataset, DatasetDict
from transformers import HfArgumentParser, PreTrainedTokenizer
from tqdm import tqdm

from cfg import *
from data import microsoft_dataset_callable
from helpers import OutputHelper
from tokenization import get_fast_tokenizer
from utils import get_highest_path


@dataclass
class DatasetArgs:
    algorithm: str = field(metadata={"help": ""})
    max_length: int = field(default=None, metadata={"help": ""})
    vocab_size: Optional[int] = field(default=256, metadata={"help": ""})
    num_tok: int = field(default=1000, metadata={"help": ""})
    num: float = field(default=1.0, metadata={"help": ""})
    num_proc: int = field(default=1, metadata={"help": ""})
    batch_size: int = field(default=1000, metadata={"help": ""})
    writer_batch_size: int = field(default=1000, metadata={"help": ""})
    shardsize: Optional[int] = field(default=None, metadata={"help": ""})


def get_tokenize_fn(tokenizer: PreTrainedTokenizer):
    def fn(examples):
        return tokenizer(examples["text"], truncation=True)

    return fn


def is_dataset_path(path: Path) -> bool:
    REQUIRED = ("dataset_info.json", "state.json")
    ALLOWED = (".arrow",)

    paths = [p.name for p in path.iterdir()]
    if not all(p in paths for p in REQUIRED):
        return False

    for p in paths:
        if p in REQUIRED:
            continue
        if Path(p).suffix not in ALLOWED:
            return False
    return True


def completed(path: Path) -> bool:
    tr_path = path / "tr"
    vl_path = path / "vl"
    ts_path = path / "ts"
    return all(p.exists() for p in (tr_path, vl_path, ts_path)) and all(
        is_dataset_path(p) for p in (tr_path, vl_path, ts_path)
    )


def main(
    algorithm: str = "SentencePieceBPE",
    max_length: int = 10**6,
    vocab_size: int = 256,
    num_tok: int = 1000,
    num: float = None,
    num_proc: int = 1,
    batch_size: int = 1000,
    writer_batch_size: int = 1000,
    shardsize: Optional[int] = None,
) -> DatasetDict:
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

    oh = OutputHelper(
        algorithm=algorithm,
        vocab_size=vocab_size,
        num_tok=num_tok,
        max_length=max_length,
        num=num,
    )
    tokenizer = get_fast_tokenizer(oh.tokenizer_file, max_length)
    print(f"{tokenizer=}")
    print("Tokenizing...")
    print(BR, flush=True)

    if completed(oh.dataset_dir):
        raise FileExistsError(oh.dataset_dir)

    if shardsize is None:
        dataset = dataset.map(
            get_tokenize_fn(tokenizer),
            batched=True,
            num_proc=num_proc,
            batch_size=batch_size,
            writer_batch_size=writer_batch_size,
        )
    else:
        # Assumes a continuation from an OOMd processing run.
        # Will initially save shards of dataset into a temporary directory
        # with the number of examples in the dataset. Then, if all shards are
        # present, will merge them, deleting the temporary directories, and
        # finally saving the entire dataset as would be done normally.
        for split in reversed(list(dataset.keys())):
            oh.dataset_dir.mkdir(exist_ok=True, parents=True)
            path: Path = oh.dataset_dir / split
            if path.exists():
                start = int(get_highest_path(path).stem) + shardsize
            else:
                start = 0

            starts = list(range(start, dataset[split].num_rows, shardsize))
            for i in tqdm(starts, initial=(start // shardsize), total=len(starts)):
                idx = list(range(i, min(i + shardsize, dataset[split].num_rows)))
                d = dataset[split].select(idx)
                d = d.map(
                    get_tokenize_fn(tokenizer),
                    batched=True,
                    num_proc=num_proc,
                    batch_size=batch_size,
                    writer_batch_size=writer_batch_size,
                )
                path.mkdir(exist_ok=True)
                p = path / f"{idx[-1] + 1}"
                p.mkdir()
                d.save_to_disk(p.as_posix())
            ds = [
                Dataset.load_from_disk(p.as_posix(), keep_in_memory=False) for p in path.iterdir()
            ]
            dataset[split] = concatenate_datasets(ds)

    path = oh.dataset_dir
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
        args.max_length,
        args.vocab_size,
        args.num_tok,
        args.num,
        args.num_proc,
        args.batch_size,
        args.writer_batch_size,
        args.shardsize,
    )


if __name__ == "__main__":
    print(f"START @{datetime.now()}")
    print(BR, flush=True)
    cli()
    print(f"FINISH @{datetime.now()}")
    print(BR, flush=True)
