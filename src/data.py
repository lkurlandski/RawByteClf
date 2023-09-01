"""
Lowest level of dataprocessing utilities.
"""

from __future__ import annotations
from collections import namedtuple
from collections.abc import Callable, Iterable
from itertools import chain
import json
import multiprocessing as mp
from pathlib import Path
from pprint import pprint
import random
import sys
from typing import Any, Literal, Optional

from datasets import Dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from cfg import MICROSOFT_ROOT, ANDROZOO_ROOT


random.seed(0)


BYTE_TO_UTF8 = {i: chr(i + 10752) for i in range(256)}


class DatasetGen:
    def __init__(
        self,
        files: Path | Iterable[Path],
        labels: Optional[Path] = None,
        min_size: int = 1,
        max_size: int = sys.maxsize,
        tr_vl_ts: tuple[float] = (0.90, 0.05, 0.05),
        split: Optional[Literal["tr", "ts", "vl"]] = None,
    ) -> None:

        # Acquire, filter, then randomize the files.
        if isinstance(files, Path):
            files = (p for p in files.iterdir() if p.suffix == ".txt")
        files = filter(lambda p: min_size <= p.stat().st_size < max_size, files)
        self.files = sorted(list(files))

        # Determine the train, test, and validation indices.
        sizes = [int(s * len(self.files)) for s in tr_vl_ts]
        while sum(sizes) != len(self.files):
            sizes[0] += 1
        idx = list(range(len(self.files)))
        tr_idx, self.ts_idx = train_test_split(idx, test_size=tr_vl_ts[2], random_state=42)
        self.tr_idx, self.vl_idx = train_test_split(tr_idx, test_size=tr_vl_ts[1], random_state=42)

        # Determine which split to use based on user preference.
        if split == "tr":
            self.idx = self.tr_idx
        elif split == "vl":
            self.idx = self.vl_idx
        elif split == "ts":
            self.idx = self.ts_idx
        else:
            self.idx = idx

        # Determine the labels, if the dataset is labeled.
        self.keys = lambda _: None
        if labels is not None:
            self.keys = pd.read_csv(labels, index_col=0).to_dict()["Class"]

        # Iterable starts at iteration 0.
        self.iteration = 0

    def __call__(self) -> Iterable:
        return iter(self)

    def __iter__(self) -> DatasetGen:
        return self

    def __len__(self) -> int:
        return len(self.idx)

    def __next__(self) -> dict[str, Optional[str]]:
        if self.iteration >= len(self):
            raise StopIteration

        f: Path = self.files[self.idx[self.iteration]]
        s: Optional[str] = f.read_text(encoding="utf-8")
        l: Optional[str] = self.keys.get(f.stem, None)
        self.iteration += 1
        return {"text": s, "label": l, "file": f.as_posix()}


def microsoft_dataset_callable(
    min_size: int = 1,
    max_size: int = sys.maxsize,
    tr_vl_ts: tuple[float] = (0.90, 0.05, 0.05),
    splits: tuple[Literal["tr", "ts", "vl"]] = ("tr", "ts", "vl"),
    microsoft_subset: Literal["train", "test"] | tuple[Literal["train", "test"]] = "train",
) -> Callable:
    subsets = (microsoft_subset,) if isinstance(microsoft_subset, str) else microsoft_subset

    datasets = []
    for _subset in subsets:
        for _split in splits:
            d = DatasetGen(
                MICROSOFT_ROOT / _subset,
                MICROSOFT_ROOT / "trainLabels.csv" if _subset == "train" else None,
                min_size,
                max_size,
                tr_vl_ts,
                _split,
            )
            datasets.append(d)

    assert all(d.tr_idx == datasets[0].tr_idx for d in datasets), "Dumb piece of shit."
    return lambda: chain(*[d() for d in datasets])


def androzoo_dataset_callable(
    min_size: int = 1,
    max_size: int = sys.maxsize,
    tr_vl_ts: tuple[float] = (0.90, 0.05, 0.05),
    splits: tuple[Literal["tr", "ts", "vl"]] = ("tr", "ts", "vl"),
    _: Any = None,
) -> Callable:

    datasets = []
    for _split in splits:
        d = DatasetGen(
            ANDROZOO_ROOT / "data",
            ANDROZOO_ROOT / "trainLabels.csv",
            min_size,
            max_size,
            tr_vl_ts,
            _split,
        )
        datasets.append(d)

    assert all(d.tr_idx == datasets[0].tr_idx for d in datasets), "Dumb piece of shit."
    return lambda: chain(*[d() for d in datasets])


def raw_byte_file_to_str(p: Path) -> str:
    """Converts a raw byte file to a string of utf-8 characters."""
    return "".join([BYTE_TO_UTF8[b] for b in p.read_bytes()])


def microsoft_byte_file_to_str(p: Path) -> str:
    """Converts the microsoft hex dump file to a string of utf-8 characters."""
    ignore = {"??", "NaN", "nan"}
    df = pd.read_csv(p, header=None, sep=" ", dtype=str, index_col=0)
    try:
        data_ = [int(str(s), 16) for _, row in df.iterrows() for s in row if str(s) not in ignore]
    except TypeError as e:
        print(df)
        raise e
    return "".join([BYTE_TO_UTF8[d] for d in data_])


def info(dataset: Dataset) -> list[dict[str, float]]:
    """Analysis."""
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
    """Analysis."""
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


def prep_microsoft():
    for f in tqdm(MICROSOFT_ROOT / "train"):
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


def _process_androzoo_file(f: Path) -> None:
    if f.suffix == ".txt":
        return
    s = raw_byte_file_to_str(f)
    with open(f.with_suffix(".txt"), "w") as fp:
        fp.write(s)
    f.unlink()


def prep_androzoo() -> None:

    selected = ANDROZOO_ROOT / "selected.json"
    if selected.exists():
        with open(selected, "r") as fp:
            d = list(json.load(fp).items())
        files, labels = list(zip(*d))
        pd.DataFrame({"Id": files, "Class": labels}).to_csv(ANDROZOO_ROOT / "trainLabels.csv")
        selected.unlink()

    files = list((ANDROZOO_ROOT / "data").iterdir())
    with mp.Pool(16) as pool:
        pool.map(_process_androzoo_file, files)


if __name__ == "__main__":
    prep_androzoo()
