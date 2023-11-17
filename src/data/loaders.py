"""
High-level loading API for the datasets.
"""

from collections import Counter
import os
from pathlib import Path
from pprint import pprint
import sys
from typing import Optional

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# pylint: disable=wrong-import-position

from datasets import concatenate_datasets, Dataset, DatasetDict, Features, Value

from src.cfg import INPUT_PATH, OUTPUT_PATH


def apply_categorical_encoding(
    dataset: Dataset, min_freq: Optional[int] = None, top_k: Optional[int] = None
) -> Dataset:
    raise NotImplementedError()
    assert bool(min_freq) != bool(top_k), "Only one selection method can be used."
    ITER_SIZE = 1024
    PRE = "is_"

    labels = Counter()
    for d in dataset.iter(ITER_SIZE):
        labels.update(d["labels"])

    categories = {
        f"{PRE}{l}": Value("bool")
        for l, n in labels.most_common(top_k)
        if (min_freq is None or n >= min_freq)
    }

    features = Features(
        {
            "name": Value("string"),
            "bytes": Value("binary"),
            "size": Value("int64"),
            "length": Value("int64"),
        }
        | categories
    )

    dataset.add_column()

    for d in dataset.iter(ITER_SIZE):
        for c in categories:
            d[c] = c[len(PRE) :] in d["labels"]


def tr_vl_ts_split(dataset: Dataset, vl_size: float, ts_size: float) -> DatasetDict:
    dataset = dataset.train_test_split(test_size=ts_size)
    dataset["ts"] = dataset.pop("test")
    d = dataset.pop("train").train_test_split(test_size=vl_size / (1 - ts_size))
    dataset["tr"] = d.pop("train")
    dataset["vl"] = d.pop("test")
    return dataset


def get_sorel_dataset(subset: Optional[int] = None) -> DatasetDict:
    dataset = concatenate_datasets([Dataset.load_from_disk(p) for p in INPUT_PATH.glob("sorel_*")])
    if subset:
        dataset = dataset.select(range(subset))
    dataset = tr_vl_ts_split(dataset, vl_size=0.1, ts_size=0.1)
    return dataset


def get_bodmas_dataset(subset: Optional[int] = None) -> DatasetDict:
    dataset = Dataset.load_from_disk(INPUT_PATH / "bodmas_pe").class_encode_column("labels")
    if subset:
        dataset = dataset.select(range(subset))
    dataset = tr_vl_ts_split(dataset, vl_size=0.1, ts_size=0.1)
    return dataset
