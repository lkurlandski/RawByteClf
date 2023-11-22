"""
High-level loading API for the datasets.
"""

from collections import Counter
from functools import partial
import os
from pathlib import Path
from pprint import pprint
import random
import sys
from typing import Optional, Protocol

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# pylint: disable=wrong-import-position

from datasets import concatenate_datasets, ClassLabel, Dataset, DatasetDict, Features, Value
import numpy as np

from src.cfg import INPUT_PATH, OUTPUT_PATH
from src.data.cfg import BODMAS_LABELS_FILE, BODMAS_DIST_FILE
from src.data.utils import print_dataset


ITER_SIZE = 1024


class GetDataset(Protocol):
    def __call__(self, *args, **kwds) -> tuple[DatasetDict, Counter]:
        ...


def tr_vl_ts_split_with_guarentees(
    dataset: Dataset,
    vl_size: float,
    ts_size: float,
    samples_per_class: int = 1,
) -> DatasetDict:
    """
    Guarentees that at least `samples_per_class` samples from each class are present in each split.
    """
    y = np.array(dataset["labels"])
    values, counts = np.unique(y, return_counts=True)
    if any(counts < (samples_per_class * 3)):
        raise ValueError("Not enough samples per class.")

    tr_dist, tr_idx = {v: 0 for v in values}, []
    vl_dist, vl_idx = {v: 0 for v in values}, []
    ts_dist, ts_idx = {v: 0 for v in values}, []
    for i, l in enumerate(y):
        if ts_dist[l] < samples_per_class:
            ts_dist[l] += 1
            ts_idx.append(i)
        elif vl_dist[l] < samples_per_class:
            vl_dist[l] += 1
            vl_idx.append(i)
        elif tr_dist[l] < samples_per_class:
            tr_dist[l] += 1
            tr_idx.append(i)
        else:
            r = random.uniform(0, 1)
            if 0 <= r < ts_size:
                ts_dist[l] += 1
                ts_idx.append(i)
            elif ts_size <= r < ts_size + vl_size:
                vl_dist[l] += 1
                vl_idx.append(i)
            else:
                tr_dist[l] += 1
                tr_idx.append(i)

    d = DatasetDict()
    d["tr"] = dataset.select(tr_idx)
    d["vl"] = dataset.select(vl_idx)
    d["ts"] = dataset.select(ts_idx)
    return d


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


def get_bodmas_dataset(
    subset: Optional[int] = None,
    min_freq: Optional[int] = None,
    top_k: Optional[int] = None,
) -> DatasetDict:
    """Expect additional computation if min_freq or top_k is not None."""

    samples_per_class = 1
    ts_size = 0.1
    vl_size = 0.1
    min_freq = samples_per_class * 3 if min_freq is None else min_freq

    def map_id2label_fn(examples: dict[str, list]) -> dict[str, list]:
        examples["labels"] = [id2label[int(i)] for i in examples["labels"]]
        return examples


    dataset = Dataset.load_from_disk(INPUT_PATH / "bodmas_pe")
    # dataset.cleanup_cache_files()  # TODO: uncomment after testing complete.

    dist: Counter[int, int] = Counter(dataset["labels"])
    id2label: dict[int, str] = {i: n for i, n in enumerate(dataset.info.features["labels"].names)}
    label2id: dict[str, int] = {n: int(i) for i, n in id2label.items()}
    unknowns = label2id["unknown"], label2id["Unknown"]
    keep = [l for l, n in dist.most_common(top_k) if (n >= min_freq and l not in unknowns)]

    dataset = dataset.filter(lambda exs: [e in keep for e in exs["labels"]], batched=True)
    dataset = dataset.cast_column("labels", Value("string"))
    dataset = dataset.map(map_id2label_fn, batched=True)

    dist = Counter(dataset["labels"])

    dataset = dataset.class_encode_column("labels")
    dataset = dataset.select(range(subset)) if subset else dataset
    dataset = tr_vl_ts_split_with_guarentees(dataset, vl_size, ts_size, samples_per_class)
    return dataset, dist
