"""
High-level loading API for the datasets.
"""

from collections import Counter
import os
import random
import sys
import time
from typing import Optional, Protocol

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# pylint: disable=wrong-import-position

from datasets import (
    concatenate_datasets,
    Dataset,
    DatasetDict,
    IterableDataset,
    Value,
)
import numpy as np

from src.data.utils import balance_imbalanced_dataset
from src.cfg import INPUT_PATH


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

    vl_size = vl_size / len(dataset) if isinstance(vl_size, int) else vl_size
    ts_size = ts_size / len(dataset) if isinstance(ts_size, int) else ts_size

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


def tr_vl_ts_split(dataset: Dataset, vl_size: float | int, ts_size: float | int) -> DatasetDict:
    vl_size = vl_size / len(dataset) if isinstance(vl_size, int) else vl_size
    ts_size = ts_size / len(dataset) if isinstance(ts_size, int) else ts_size
    assert vl_size < 1.0 and ts_size < 1.0, f"{vl_size=} and {ts_size=} must be less than 1.0."

    dataset = dataset.train_test_split(test_size=ts_size)
    dataset["ts"] = dataset.pop("test")
    d = dataset.pop("train").train_test_split(test_size=vl_size / (1 - ts_size))
    dataset["tr"] = d.pop("train")
    dataset["vl"] = d.pop("test")
    return dataset


def get_sorel_dataset(
    subset: Optional[int] = None, vl_size: int | float = None, ts_size: int | float = None
) -> DatasetDict:
    files = [INPUT_PATH / f"sorel_pe_{i}" for i in range(0, 32)]
    if vl_size is None:
        vl_size = 10000 if subset is None else 0.1
    if ts_size is None:
        ts_size = 10000 if subset is None else 0.1

    print(f"Loading SOREL ({subset=} {vl_size=} {ts_size=})...", flush=True)
    dataset = Dataset.load_from_disk(files.pop(0))
    while (subset is None or len(dataset) < subset) and files:
        try:
            dataset = concatenate_datasets([dataset, Dataset.load_from_disk(files.pop(0))])
        except FileNotFoundError as err:
            print(err)
        except IndexError:
            break

    if subset:
        dataset = dataset.select(range(subset))
    dataset = tr_vl_ts_split(dataset, vl_size=vl_size, ts_size=ts_size)
    return dataset


def get_bodmas_dataset(
    subset: Optional[int] = None,
    min_freq: Optional[int] = None,
    top_k: Optional[int] = None,
    ts_size: int | float = 0.1,
    vl_size: int | float = 0.1,
) -> tuple[Dataset, Counter]:
    """Expect additional computation if min_freq or top_k is not None."""

    samples_per_class = 1
    min_freq = samples_per_class * 3 if min_freq is None else min_freq

    print(f"Loading BODMAS ({subset=} {vl_size=} {ts_size=} {min_freq=} {top_k=})...", flush=True)

    def map_id2label_fn(examples: dict[str, list]) -> dict[str, list]:
        examples["labels"] = [id2label[int(i)] for i in examples["labels"]]
        return examples

    dataset = Dataset.load_from_disk(INPUT_PATH / "bodmas_pe")
    # dataset.cleanup_cache_files()  # TODO: uncomment after testing complete.

    # Select the samples that meet the top_k and/or min_freq requirements
    dist: Counter[int, int] = Counter(dataset["labels"])
    id2label: dict[int, str] = {i: n for i, n in enumerate(dataset.info.features["labels"].names)}
    label2id: dict[str, int] = {n: int(i) for i, n in id2label.items()}
    unknowns = label2id["unknown"], label2id["Unknown"]
    keep = [l for l, n in dist.most_common(top_k) if (n >= min_freq and l not in unknowns)]
    dataset = dataset.filter(lambda exs: [e in keep for e in exs["labels"]], batched=True)

    # Convert the int labels to str, then replace them with their class name
    dataset = dataset.cast_column("labels", Value("string"))
    dataset = dataset.map(map_id2label_fn, batched=True)
    dist = Counter(dataset["labels"])

    dataset = dataset.class_encode_column("labels")
    dataset = dataset.select(range(subset)) if subset else dataset
    dataset = tr_vl_ts_split_with_guarentees(dataset, vl_size, ts_size, samples_per_class)
    return dataset, dist


def test_balance_imbalanced_dataset() -> None:
    def test_iteration_time(dataset: Dataset | IterableDataset, b: int = 1, n: int = 16) -> int:
        s = time.time()
        for i, d in enumerate(dataset.iter(b)):
            if i >= n:
                break
        return time.time() - s

    B = 1
    N = 1
    S = 1

    dataset, dist = get_bodmas_dataset()
    dataset: Dataset = dataset["tr"]
    dataset = dataset.to_iterable_dataset(S)

    print(f"{dist=}\n{'-' * 88}")
    print(f"{dataset=}")
    if hasattr(dataset, "__len__"):
        print(f"{len(dataset)=}")
    print(f"{test_iteration_time(dataset, B, N)=}")

    dataset, dist = balance_imbalanced_dataset(dataset, dist, 0.5, check=False)
    print(f"{dist=}\n{'-' * 88}")
    print(f"{dataset=}")
    if hasattr(dataset, "__len__"):
        print(f"{len(dataset)=}")
    dataset.flatten_indices()
    print(f"{test_iteration_time(dataset, B, N)=}")


if __name__ == "__main__":
    ...
