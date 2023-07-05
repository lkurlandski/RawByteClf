"""
"""

from collections import namedtuple
from itertools import chain, islice
from pathlib import Path
from pprint import pprint
import subprocess
import sys
from typing import Literal

from datasets import Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm


MICROSOFT_ROOT = Path("/home/lk3591/Documents/code/RawByteClf/data")


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
        n: int = None,
        min_size: int = 0,
        max_size: int = sys.maxsize,
        split: Literal["train", "test"] = "train",
        separator: str = "",
    ) -> None:
        self.min_size = min_size
        self.max_size = max_size
        self.separator = separator
        files = (p for p in (MICROSOFT_ROOT / split).iterdir() if p.suffix == ".txt")
        self.files = sorted(list(files))[0:n]
        self.keys = pd.read_csv("data/trainLabels.csv", index_col=0).to_dict()["Class"]
        self.iteration = 0

    def __call__(self):
        return iter(self)

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.files)

    def __next__(self):
        if self.iteration >= len(self):
            raise StopIteration

        f = self.files[self.iteration]
        with open(f, encoding="utf-8") as handle:
            s = handle.read()
        if self.separator != "":
            s = s.replace("", self.separator)[len(self.separator) : -1 * len(self.separator)]

        l = self.keys.get(f.stem, None)
        self.iteration += 1

        if self.min_size <= len(s) <= self.max_size:
            return {"text": s, "label": l, "file": f.as_posix()}
        return next(self)


class DatasetGen:
    def __init__(
        self,
        n_sorel: int = 1,
        n_windows: int = 1,
        min_size: int = 0,
        max_size: int = sys.maxsize,
    ) -> None:
        self.min_size = min_size
        self.max_size = max_size

        p_sorel = Path("/home/lk3591/Documents/datasets/Sorel/processed/")
        f_sorel = self.get_files(p_sorel)
        if len(f_sorel) > n_sorel:
            f_sorel = f_sorel[:n_sorel]
        l_sorel = [1] * len(f_sorel)

        p_windows = Path("/home/lk3591/Documents/datasets/Windows/processed/")
        f_windows = self.get_files(p_windows)
        if len(f_windows) > n_windows:
            f_windows = f_windows[:n_windows]
        l_windows = [0] * len(f_windows)

        self.files = f_sorel + f_windows
        self.labels = l_sorel + l_windows
        idx = np.flip(np.argsort([f.stat().st_size for f in self.files], kind="stable"))
        self.files = [self.files[i] for i in idx]
        self.labels = [self.labels[i] for i in idx]
        self.iteration = 0

    def get_files(self, p: Path):
        files = chain((p / "train").iterdir(), (p / "test").iterdir())
        files = [f for f in files if self.min_size <= f.stat().st_size <= self.max_size]
        return sorted(files)

    def __call__(self):
        return iter(self)

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.files)

    def __next__(self):
        if self.iteration >= len(self):
            raise StopIteration
        f = self.files[self.iteration]
        l = self.labels[self.iteration]
        self.iteration += 1
        return {"text": byte_string(f), "label": l, "file": f.as_posix()}


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


def convert_microsoft_to_utf_bytes(split: Literal["train", "test"]):
    files = set(p for p in (MICROSOFT_ROOT / split).iterdir())
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
    ...
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
