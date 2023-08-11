"""
"""

from collections import namedtuple
from itertools import chain, islice
from pathlib import Path
from pprint import pprint
import random
import subprocess
import sys
from typing import Callable, Literal, Optional

from datasets import Dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from cfg import *


random.seed(0)


byte_to_utf8 = {i: chr(i + 10752) for i in range(256)}


def byte_string(p: Path) -> str:
    return "".join([byte_to_utf8[b] for b in p.read_bytes()])


def microsoft_byte_file_to_str(p: Path) -> str:
    ignore = {"??", "NaN", "nan"}
    df = pd.read_csv(p, header=None, sep=" ", dtype=str, index_col=0)
    try:
        data = [int(str(s), 16) for _, row in df.iterrows() for s in row if str(s) not in ignore]
    except TypeError as e:
        print(df)
        raise e
    return "".join([byte_to_utf8[d] for d in data])


class MicrosoftDatasetGen:
    def __init__(
        self,
        min_size: int = 1,
        max_size: int = sys.maxsize,
        tr_vl_ts: tuple[float] = (0.90, 0.05, 0.05),
        split: Optional[Literal["tr", "ts", "vl"]] = None,
        microsoft_subset: Optional[Literal["train", "test"]] = "train",
    ) -> None:
        self.min_size = min_size
        self.max_size = max_size
        self.microsoft_subset = microsoft_subset
        self.keys = pd.read_csv("data/trainLabels.csv", index_col=0).to_dict()["Class"]
        self.iteration = 0

        files = (
            p for p in (MICROSOFT_ROOT / self.microsoft_subset).iterdir() if p.suffix == ".txt"
        )
        files = filter(lambda p: min_size <= p.stat().st_size < max_size, files)
        self.files = sorted(list(files))
        sizes = [int(s * len(self.files)) for s in tr_vl_ts]
        while sum(sizes) != len(self.files):
            sizes[0] += 1
        idx = list(range(len(self.files)))
        tr_idx, self.ts_idx = train_test_split(idx, test_size=tr_vl_ts[2], random_state=42)
        self.tr_idx, self.vl_idx = train_test_split(tr_idx, test_size=tr_vl_ts[1], random_state=42)

        if split == "tr":
            self.idx = self.tr_idx
        elif split == "vl":
            self.idx = self.vl_idx
        elif split == "ts":
            self.idx = self.ts_idx
        else:
            self.idx = idx

    def __call__(self):
        return iter(self)

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.idx)

    def __next__(self):
        if self.iteration >= len(self):
            raise StopIteration

        f = self.files[self.idx[self.iteration]]
        s = f.read_text(encoding="utf-8")
        l = self.keys.get(f.stem, None)
        self.iteration += 1
        return {"text": s, "label": l, "file": f.as_posix()}


def info(dataset: Dataset) -> list[dict[str, float]]:
    stats = []
    for d in dataset:
        stat = {}
        if "text" in d:
            stat["text"] = len(d["text"])
        if "file" in d:
            stat["st_size"] = Path(d["file"]).stat().st_size
        if "input_ids" in d:
            stat["input_ids"] = len(d["input_ids"])
        stats.append(stat)
    return stats


def process_info(stats: list[dict[str, float]]) -> dict[str, float]:
    if not stats:
        return {}

    Stat = namedtuple("Stat", ["mean", "median", "std", "max", "min"])

    def fn(nums):
        return Stat(
            round(np.mean(nums)), round(np.median(nums)), round(np.std(nums)), max(nums), min(nums)
        )

    keys = tuple(stats[0].keys())
    summary = {}
    if "text" in keys:
        summary["text"] = fn([s["text"] for s in stats])
    if "st_size" in keys:
        summary["st_size"] = fn([s["st_size"] for s in stats])
    if "input_ids" in keys:
        summary["input_ids"] = fn([s["input_ids"] for s in stats])
    return summary


def microsoft_dataset_callable(
    min_size: int = 1,
    max_size: int = sys.maxsize,
    tr_vl_ts: tuple[float] = (0.90, 0.05, 0.05),
    splits: Optional[list[Literal["tr", "ts", "vl"]]] = None,
    microsoft_subset: Optional[Literal["train", "test"]] = "train",
) -> Callable:
    if not splits:
        return MicrosoftDatasetGen(min_size, max_size, tr_vl_ts, None)

    datasets = [
        MicrosoftDatasetGen(min_size, max_size, tr_vl_ts, s, microsoft_subset) for s in splits
    ]
    assert all(d.tr_idx == datasets[0].tr_idx for d in datasets), "Dumb piece of shit."
    return lambda: chain(*[d() for d in datasets])


def convert_microsoft_to_utf_bytes(microsoft_subset: Literal["train", "test"]):
    files = set(p for p in (MICROSOFT_ROOT / microsoft_subset).iterdir())
    for f in tqdm(files):
        if f.stat().st_size == 0:
            print(f"{f.name}.stat().st_size == 0")
            continue
        if f.suffix == ".txt":
            continue
        if f.suffix == ".bytes" and f.with_suffix(".txt") in files:
            if f.with_suffix(".txt").stat().st_size > 0:
                f.unlink()
            continue
        try:
            s = microsoft_byte_file_to_str(f)
        except Exception as e:
            print(f)
            raise e
        with open(f.with_suffix(".txt"), "w", encoding="utf-8") as handle:
            handle.write(s)
        f.unlink()


if __name__ == "__main__":
    gen = microsoft_dataset_callable(10, splits=["tr", "vl"])
    for i, d in enumerate(gen()):
        print(i, d["file"])

    from datasets import Dataset

    num_files = 10
    dataset = Dataset.from_generator(microsoft_dataset_callable(num_files, splits=["tr", "vl"]))
    print(dataset)

    # convert_microsoft_to_utf_bytes("train")

    # files = [
    #     "58kxhXouHzFd4g3rmInB.bytes",
    #     "6tfw0xSL2FNHOCJBdlaA.bytes",
    #     "a9oIzfw03ED4lTBCt52Y.bytes",
    #     "cf4nzsoCmudt1kwleOTI.bytes",
    #     "d0iHC6ANYGon7myPFzBe.bytes",
    #     "da3XhOZzQEbKVtLgMYWv.bytes",
    #     "fRLS3aKkijp4GH0Ds6Pv.bytes",
    #     "IidxQvXrlBkWPZAfcqKT.bytes",
    # ]
    # for f in files:
    #     f = MICROSOFT_ROOT / "train" / f
    #     s = microsoft_byte_file_to_str(f)
    #     if s == "":
    #         print(f)
    #         continue
    #     with open(f.with_suffix(".txt"), "w", encoding="utf-8") as handle:
    #         handle.write(s)
    #     if f.with_suffix(".txt").stat().st_size == 0:
    #         print(f)
    #         continue
    #     f.unlink()
