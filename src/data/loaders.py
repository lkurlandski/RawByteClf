"""
High-level loading API for the datasets.
"""

from collections import Counter
import json
import os
from pathlib import Path
from pprint import pprint
import sys
from typing import Optional, Protocol

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# pylint: disable=wrong-import-position

from datasets import concatenate_datasets, ClassLabel, Dataset, DatasetDict, Features, Value

from src.cfg import INPUT_PATH, OUTPUT_PATH
from src.data.cfg import BODMAS_LABELS_FILE, BODMAS_DIST_FILE


ITER_SIZE = 1024


class GetDataset(Protocol):
    def __call__(self, *args, **kwds) -> tuple[DatasetDict, Counter]:
        ...


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

    def filter_fn(examples: list) -> list[bool]:
        return [e in keep for e in examples["labels"]]

    def map_fn(examples: dict[str, list]) -> dict[str, list]:
        examples["labels"] = [id2label[i] for i in examples["labels"]]
        return examples

    dataset = Dataset.load_from_disk(INPUT_PATH / "bodmas_pe")
    # dataset.cleanup_cache_files()  # TODO: remove this after testing...
    with open(BODMAS_DIST_FILE, "r") as fp:
        d: dict = json.load(fp)
        d.pop("benign")
        dist = Counter(d)

    if min_freq or top_k:
        id2label = {i: n for i, n in enumerate(dataset.info.features["labels"].names)}
        id2label.update({str(i): n for i, n in id2label.items()})

        dist = Counter((d["labels"] for d in dataset.select_columns(["labels"])))
        keep = [l for l, n in dist.most_common(top_k) if (min_freq is None or n >= min_freq)]
        keep = set(keep) if len(keep) > 50 else keep

        # It is critical to cast the column to string before applying map function.
        dataset = dataset.filter(filter_fn, batched=True)
        dataset = dataset.cast_column("labels", Value("string"))
        dataset = dataset.map(map_fn, batched=True)

        dist = Counter()
        for d in dataset.select_columns("labels").iter(ITER_SIZE):
            dist.update(d["labels"])
        dataset = dataset.class_encode_column("labels")

    if subset:
        dataset = dataset.select(range(subset))
    dataset = tr_vl_ts_split(dataset, vl_size=0.1, ts_size=0.1)
    return dataset, dist
