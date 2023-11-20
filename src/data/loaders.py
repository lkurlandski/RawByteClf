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

from datasets import concatenate_datasets, ClassLabel, Dataset, DatasetDict, Features, Value

from src.cfg import INPUT_PATH, OUTPUT_PATH
from src.data.cfg import BODMAS_LABELS_FILE


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

    dataset = Dataset.load_from_disk(INPUT_PATH / "bodmas_pe")
    if min_freq or top_k:
        id2label = {i: n for i, n in enumerate(dataset.info.features["labels"].names)}

        dist = Counter((d["labels"] for d in dataset.select_columns(["labels"])))
        keep = [l for l, n in dist.most_common(top_k) if (min_freq is None or n >= min_freq)]
        keep = set(keep) if len(keep) > 50 else keep

        dataset = dataset.filter(lambda exs: [e in keep for e in exs["labels"]], batched=True)
        dataset = dataset.map(
            lambda exs: exs.update({"labels": [id2label[i] for i in exs["labels"]]}),
            batched=True,
        )
        dataset = dataset.cast_column("labels", Value("string"))
        dataset = dataset.class_encode_column("labels")

    if subset:
        dataset = dataset.select(range(subset))
    dataset = tr_vl_ts_split(dataset, vl_size=0.1, ts_size=0.1)
    return dataset
