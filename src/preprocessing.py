"""
Preprocess data by strictly tokenizing it, i.e., no padding or tokenization.
"""

from dataclasses import dataclass, field
from pathlib import Path
from pprint import pformat, pprint
import shutil
import sys

from datasets import Dataset
from transformers import HfArgumentParser, PreTrainedTokenizer

from cfg import *
from data import MicrosoftDatasetGen
from tokenization import get_fast_tokenizer


@dataclass
class DatasetArgs:
    tokenizer_file: str = field(metadata={"help": ""})
    num_files: int = field(default=None, metadata={"help": ""})
    min_length: int = field(default=10**3, metadata={"help": ""})
    max_length: int = field(default=10**6, metadata={"help": ""})
    num_proc: int = field(default=1, metadata={"help": ""})
    batch_size: int = field(default=1000, metadata={"help": ""})
    writer_batch_size: int = field(default=1000, metadata={"help": ""})
    datasets_root_path: str = field(default=None, metadata={"help": ""})

    def __post_init__(self) -> None:
        if self.datasets_root_path is None:
            tok = Path(self.tokenizer_file)
            for i, part in enumerate(tok.parts, 1):
               if part == "tokenizers":
                   break
            self.datasets_root_path = DATASETS / Path(*tok.parts[i:-1])
            print(self.datasets_root_path)


def get_group_texts_fn(block_size=2**12):
    def fn(examples):
        concatenated = "".join(examples["text"])
        total_length = len(concatenated)
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        chopped = [concatenated[i : i + block_size] for i in range(0, total_length, block_size)]
        examples["text"] = chopped
        return examples

    return fn


def get_tokenize_fn(tokenizer: PreTrainedTokenizer):
    def fn(examples):
        return tokenizer(examples["text"], truncation=False, padding=False)

    return fn


def datasets_path(root: Path = DATASETS, n_dat: int = None) -> Path:
    return root / (str(n_dat) if n_dat is not None else "full")


def main(args: DatasetArgs) -> Dataset:
    print("Fetching raw dataset...")
    dataset = Dataset.from_generator(
        MicrosoftDatasetGen(args.num_files, args.min_length, args.max_length),
    ).remove_columns(["file"])
    print(f"{dataset=}")
    print(BR, flush=True)

    tokenizer = get_fast_tokenizer(args.tokenizer_file, None)
    print(f"{tokenizer=}")
    print("Tokenizing...")
    print(BR, flush=True)
    dataset = dataset.map(
        get_tokenize_fn(tokenizer),
        batched=True,
        num_proc=args.num_proc,
        batch_size=args.batch_size,
        writer_batch_size=args.writer_batch_size,
    )
    path = datasets_path(args.datasets_root_path, dataset.num_rows)
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(exist_ok=True, parents=True)
    dataset.save_to_disk(path.as_posix())
    return dataset


def cli():
    parser = HfArgumentParser((DatasetArgs,))
    args = parser.parse_args_into_dataclasses()[0]
    print(f"args={pformat(args)}")
    print(BR, flush=True)
    main(args)


if __name__ == "__main__":
    cli()
