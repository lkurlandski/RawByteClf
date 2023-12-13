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
from tqdm import tqdm

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


def get_sorel_dataset(subset: Optional[int] = None, vl_size: int | float = None, ts_size: int | float = None) -> DatasetDict:
    files = [INPUT_PATH / f"sorel_pe_{i}" for i in range(0, 32)]
    if vl_size is None:
        vl_size = 10000 if subset is None else 0.1
    if ts_size is None:
        ts_size = 10000 if subset is None else 0.1

    print(f"Loading SOREL ({subset=} {vl_size=} {ts_size=})...", flush=True)
    dataset = Dataset.load_from_disk(files.pop(0))
    while ((subset is None or len(dataset) < subset) and files):
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
) -> DatasetDict:
    """Expect additional computation if min_freq or top_k is not None."""

    samples_per_class = 1
    ts_size = 0.1
    vl_size = 0.1
    min_freq = samples_per_class * 3 if min_freq is None else min_freq

    print(f"Loading BODMAS ({subset=} {vl_size=} {ts_size=} {min_freq=} {top_k=})...", flush=True)

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


def test():
    from src.learn.train import DEPTH
    from src.learn.utils import get_tokenizer_object, get_fast_tokenizer, tokenize_fn, preprocess_a

    max_length = 4096
    task = "mlm"
    tokenizer = get_fast_tokenizer(get_tokenizer_object(), max_length=max_length)
    dataset = get_sorel_dataset(subset=1024)["tr"].to_iterable_dataset()
    print(dataset.column_names)
    print_dataset(dataset, 16)

    dataset = dataset.map(
        partial(
            preprocess_a,
            max_length=max_length * DEPTH if task in ("mlm", "clm") else max_length,
        ),
        batched=True,
    )
    print(dataset.column_names)
    print_dataset(dataset, 16)
    # Converts the "text" column into a "input_ids" column.
    # Additional rows are added for language modeling.
    remove_columns = ["name", "bytes", "labels", "size", "length", "text"]
    dataset = dataset.map(
        partial(
            tokenize_fn,  # The partial function here is picky (`tokenizer` must be arg not kwd).
            tokenizer,
            truncation=True,
            max_length=max_length,
            return_overflowing_tokens=task in ("mlm", "clm"),
        ),
        batched=True,
        batch_size=16,
        remove_columns=remove_columns,
    )
    print(dataset.column_names)
    # print(len(dataset))
    for i, d in enumerate(dataset):
        print(len(d["input_ids"]))
        if i == 16:
            break

    sys.exit(0)


if __name__ == "__main__":
    d = get_sorel_dataset()
    print(d)
    sys.exit(0)
