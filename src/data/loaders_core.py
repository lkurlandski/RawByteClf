"""
Core file operations to construct labeled datasets for classification tasks.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import chain
import math
import os
from pathlib import Path
from pprint import pprint, pformat
import random
import sys
import time
from typing import Literal, Optional
import warnings

# pylint: disable=wrong-import-position
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# pylint: enable=wrong-import-position

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm

from src.utils import get_max_keys_from_dict, flatten, get_unique_files
from src.data.cfg import (
    SOREL_PATH,
    BODMAS_PATH,
    BODMAS_LABELS_FILE,
    DATASET_TO_FILES,
    SOREL_META_CSV,
    ELF_CLASSIFICATION_DATASETS,
    PACKING_ROOTS,
    SOREL_CLARAVY_CACHE,
    SOREL_AVCLASS_CACHE,
    SOREL_AVCLASS_FAMILY_CACHE,
)
from src.data.detect_packing_sorel import PackingMap, universal_packing_map
from src.data.label_datasets import (
    get_label_mapping_virus_total_reports_sorel,
    get_label_mapping_virus_total_reports_elf,
    ThreatLabelExtractor,
    ThreatLabelRefiner,
)
from src.data.labeling import FilterArgs, Labeler, Label


MIN_SAMPLES_PER_CLASS_PER_SPLIT = 1

################################################################################
# Utilities
################################################################################


SplitNames = Literal["tr", "vl", "ts"]
FilesAndLabels = tuple[list[os.PathLike], Optional[Sequence[int]]]


@dataclass
class Materials:
    files: dict[SplitNames, list[os.PathLike]]
    labels: Optional[dict[SplitNames, Sequence[int | Sequence[int]]]] = None
    id2label: dict[int, str] = None
    label2id: dict[str, int] = None
    dist: Counter[str, int] = None

    @property
    def problem_type(self) -> Optional[Literal["single_label_classification", "multi_label_classification"]]:
        if self.labels is None:
            return None
        if isinstance(self.labels["tr"][0], (Sequence, np.ndarray)):
            return "multi_label_classification"
        if isinstance(self.labels["tr"], (Sequence, np.ndarray)):
            return "single_label_classification"
        raise RuntimeError(f"Invalid labels: {type(self.labels['tr'])=} {type(self.labels['tr'][0])=}")

    @property
    def num_classes(self) -> Optional[int]:
        if self.id2label is None:
            return None
        return len(self.dist)

    @property
    def dist_tr(self) -> Counter:
        if self.labels is None:
            return None
        return self.get_split_dist("tr")

    @property
    def dist_vl(self) -> Counter:
        if self.labels is None:
            return None
        return self.get_split_dist("vl")

    @property
    def dist_ts(self) -> Counter:
        if self.labels is None:
            return None
        return self.get_split_dist("ts")

    def imbalance(self, split: Optional[SplitNames] = None) -> Optional[float]:
        if self.problem_type is None:
            return None
        if split is None:
            return self.dist.most_common(1)[0][1] / self.dist.most_common()[-1][1]
        if self.problem_type == "single_label_classification":
            c = Counter(self.labels[split])
            return c.most_common(1)[0][1] / c.most_common()[-1][1]
        if self.problem_type == "multi_label_classification":
            c = Counter(chain.from_iterable(self.labels[split]))
            return c.most_common(1)[0][1] / c.most_common()[-1][1]
        raise RuntimeError(f"Invalid problem type: {self.problem_type=}")

    def get_split_dist(self, split: SplitNames) -> Counter:
        if self.labels is None:
            return None
        if self.problem_type == "single_label_classification":
            return Counter([self.id2label[int(i)] for i in self.labels[split]])
        if self.problem_type == "multi_label_classification":
            return Counter([self.id2label[int(i)] for i in chain.from_iterable(self.labels[split])])
        raise RuntimeError(f"Invalid problem type: {self.problem_type=}")

    def __repr__(self):
        return (
            f"len(tr)={len(self.files['tr'])}\n"
            f"len(vl)={len(self.files['vl'])}\n"
            f"len(ts)={len(self.files['ts'])}\n"
            f"num_classes={len(self.id2label) if self.id2label is not None else None}\n"
            f"dist={pformat(self.dist) if self.dist is not None else None}"
        )


def get_tr_vl_ts_files_and_labels(
    files: list[os.PathLike],
    labels: np.ndarray | list[int | Sequence[int]],
    idx: Optional[dict[SplitNames, Sequence[int]]] = None,
    tr_idx: Optional[Sequence[int]] = None,
    vl_idx: Optional[Sequence[int]] = None,
    ts_idx: Optional[Sequence[int]] = None,
) -> tuple[dict[SplitNames, list[os.PathLike]], dict[SplitNames, np.ndarray]]:

    if idx is not None:
        if tr_idx is not None or vl_idx is not None or ts_idx is not None:
            raise ValueError("Cannot specify both `idx` and `tr_idx`, `vl_idx`, `ts_idx`.")
        tr_idx, vl_idx, ts_idx = idx["tr"], idx["vl"], idx["ts"]

    files = {
        "tr": [files[i] for i in tr_idx],
        "vl": [files[i] for i in vl_idx],
        "ts": [files[i] for i in ts_idx],
    }

    if not isinstance(labels[0], Sequence):
        labels = np.array(labels) if isinstance(labels, list) else labels
        labels = {
            "tr": labels[tr_idx],
            "vl": labels[vl_idx],
            "ts": labels[ts_idx],
        }
    else:
        labels = {
            "tr": [labels[i] for i in tr_idx],
            "vl": [labels[i] for i in vl_idx],
            "ts": [labels[i] for i in ts_idx],
        }

    return files, labels


def select_k_for_each_class(labels: list[int | str], k: int) -> list[int]:
    unique = set(labels)
    count = {s : 0 for s in unique}
    idx = []
    for i, l in enumerate(labels):
        if count[l] < k:
            count[l] += 1
            idx.append(i)
    return idx


def select_k_for_each_class_multilabel(
    labels: list[list[int | str]],
    k: int,
    max_iter: int = 10,
    strict: bool = False,
) -> list[int]:
    unique = set(chain.from_iterable(labels))
    count = {s : 0 for s in unique}
    idx = set()

    # Repeat for a maximum of `max_iter` iterations before greedily adding samples.
    pbar = tqdm(range(max_iter), total=max_iter, desc="Attempting to find a precise solution...", leave=False)
    for j in pbar:
        for i, label in tqdm(enumerate(labels), total=len(labels), leave=False):
            # If the count for all labels in the label is less than `k`, add the sample.
            if all(count[l] < k for l in label):
                for l in label:
                    count[l] += 1
                idx.add(i)
            elif any(count[l] < k for l in label):
                # If less than j labels have reached their quota, add the sample.
                if sum(1 for l in label if count[l] >= k) < j:
                    for l in label:
                        count[l] += 1
                    idx.add(i)

        done = sum(1 for l in unique if count[l] >= k)
        pbar.set_description(f"Found {done} / {len(unique)}")
        if done == len(unique):
            break

    if any(count[l] < k for l in unique):
         for i, label in tqdm(enumerate(labels), total=len(labels), leave=False):
            if i in idx:
                continue

            if any(count[l] < k for l in label):
                for l in label:
                    count[l] += 1
                idx.add(i)

            if all(count[l] >= k for l in unique):
                break

    if any(count[l] != k for l in unique):
        max_ = Counter(count).most_common(1)[0]
        min_ = Counter(count).most_common(None)[-1]
        message = f"Could not perfectly return {k} for each class. Min: ({min_}), Max: ({max_})"
        warnings.warn(message)
        if strict:
            raise RuntimeError(message)

    return list(idx)


################################################################################
# Splitting Datasets
################################################################################


def compute_integer_sizes(
    total: int,
    tr_size: int | float,
    vl_size: int | float,
    ts_size: int | float,
) -> tuple[int, int, int]:
    """
    Given float or integer sizes for a train/test/validation split, returns the integer sizes of
    each split.
    """
    types = [type(x) for x in (tr_size, vl_size, ts_size)]
    if len(set(types)) != 1:
        raise TypeError("The semantics of using both float and int is not well defined.")

    if types[0] == int:
        return tr_size, vl_size, ts_size

    tr_size = math.floor(total * tr_size)
    vl_size = math.floor(total * vl_size)
    ts_size = math.floor(total * ts_size)

    # There are likely some leftovers because of floor function.
    # Place them in the test/validation set.
    if ts_size > 0:
        ts_size += (total - tr_size - vl_size - ts_size)
    if vl_size > 0:
        vl_size += (total - tr_size - vl_size - ts_size)

    if tr_size + vl_size + ts_size != total:
        raise ValueError(f"Sum of splits should equal {total=}: {tr_size=} {vl_size=} {ts_size=}")

    return tr_size, vl_size, ts_size


def compute_float_sizes(
    total: int,
    tr_size: int | float,
    vl_size: int | float,
    ts_size: int | float,
) -> tuple[int, int, int]:
    """
    Given float or integer sizes for a train/test/validation split, returns the float proportions
    of each split.
    """
    types = [type(x) for x in (tr_size, vl_size, ts_size)]
    if len(set(types)) != 1:
        raise TypeError("The semantics of using both float and int is not well defined.")

    if types[0] == float:
        return tr_size, vl_size, ts_size

    return tr_size / total, vl_size / total, ts_size / total


def tr_vl_ts_split_idx(
    total: int,
    tr_size: int | float,
    vl_size: int | float,
    ts_size: int | float,
) -> dict[SplitNames, np.ndarray]:
    """
    Returns train/validation/test indices for a collection with `total` elements.
    """
    types = [type(x) for x in (tr_size, vl_size, ts_size)]
    if len(set(types)) != 1:
        raise TypeError("The semantics of using both float and int is not well defined.")

    tr_size, vl_size, ts_size = compute_integer_sizes(total, tr_size, vl_size, ts_size)
    collection = np.array(list(range(total)))

    if ts_size > 0:
        tr_vl, ts = train_test_split(collection, test_size=ts_size, train_size=tr_size + vl_size)
    else:
        tr_vl = collection
        ts = []

    if vl_size > 0:
        tr, vl = train_test_split(tr_vl, test_size=vl_size, train_size=tr_size)
    else:
        tr = tr_vl
        vl = []

    return {"tr": tr[0:tr_size], "vl": vl, "ts": ts}


def tr_vl_ts_split(
    collection: Sequence,
    tr_size: int | float,
    vl_size: int | float,
    ts_size: int | float,
) -> dict[SplitNames, Sequence]:
    """
    Returns train/validation/test sets for a collection.
    """

    idx = tr_vl_ts_split_idx(len(collection), tr_size, vl_size, ts_size)
    try:  # Try to slice with a numpy array
        return {s: collection[indices] for s, indices in idx.items()}
    except TypeError:  # If that fails, use list comprehension
        return {s: [collection[i] for i in indices] for s, indices in idx.items()}


def tr_vl_ts_split_idx_guarentee(
    labels: Sequence[int],
    tr_size: int | float,
    vl_size: int | float,
    ts_size: int | float,
    samples_per_class: int = 1,
) -> dict[SplitNames, np.ndarray]:
    """
    Returns train/validation/test indices for a collection, with guarentees that at least
    `samples_per_class` samples from each class are present in each split.
    """

    num_splits = 2 if vl_size == 0 or ts_size == 0 else 3
    values, counts = np.unique(labels, return_counts=True)

    # Verify there are enough samples per class to guarentee each split has representative samples.
    if any(counts < (samples_per_class * num_splits)):
        raise ValueError(
            f"Not enough samples per class to create {num_splits} splits."
            f"dist={pformat(Counter(labels))}"
        )

    # Get both integer and floating point representations of the size of the splits.
    if any(isinstance(x, float) for x in (tr_size, vl_size, ts_size)):
        tr_size, vl_size, ts_size = compute_integer_sizes(len(labels), tr_size, vl_size, ts_size)
    total = tr_size + vl_size + ts_size
    tr_prop, vl_prop, ts_prop = compute_float_sizes(total, tr_size, vl_size, ts_size)

    # Verify that the splits are large enough to contain `samples_per_class` samples for each class.
    for split, size in zip(["tr", "vl", "ts"], [tr_size, vl_size, ts_size]):
        if 0 < size < len(values) * samples_per_class:
            raise ValueError(
                f"The {split=} set with {size=} is too small to contain {samples_per_class} samples"
                f" per class for {len(values)} classes."
            )

    # For each element in the collection, assign it to a particular split if that split has not
    # reached its quota for that class.
    tr_dist, tr_idx = {v: 0 for v in values}, []
    vl_dist, vl_idx = {v: 0 for v in values}, []
    ts_dist, ts_idx = {v: 0 for v in values}, []
    for i, l in enumerate(labels):
        if tr_size > 0 and tr_dist[l] < samples_per_class:
            tr_dist[l] += 1
            tr_idx.append(i)
        elif vl_size > 0 and vl_dist[l] < samples_per_class:
            vl_dist[l] += 1
            vl_idx.append(i)
        elif ts_size > 0 and ts_dist[l] < samples_per_class:
            ts_dist[l] += 1
            ts_idx.append(i)

    # Add the remaining samples to the splits if they have not already been added.
    added = set(tr_idx) | set(vl_idx) | set(ts_idx)
    for i, l in enumerate(labels):
        if i in added:
            continue

        r = random.uniform(0, sum((tr_prop, vl_prop, ts_prop)))
        if (0 <= r < ts_prop) and (len(ts_idx) < ts_size):
            ts_dist[l] += 1
            ts_idx.append(i)
        elif (ts_prop <= r < vl_prop + ts_prop) and (len(vl_idx) < vl_size):
            vl_dist[l] += 1
            vl_idx.append(i)
        elif (ts_prop + vl_prop <= r < tr_prop + vl_prop + ts_prop) and (len(tr_idx) < tr_size):
            tr_dist[l] += 1
            tr_idx.append(i)

    assert set.intersection(set(tr_idx), set(vl_idx), set(ts_idx)) == set(), "Indices are not mutually exclusive."

    tr_idx = np.array(tr_idx, dtype=np.int32)
    vl_idx = np.array(vl_idx, dtype=np.int32)
    ts_idx = np.array(ts_idx, dtype=np.int32)
    return {"tr": tr_idx, "vl": vl_idx, "ts": ts_idx}


def tr_vl_ts_split_idx_multilabel_guarentee(
    labels: Sequence[Sequence[int]],
    tr_size: int | float,
    vl_size: int | float,
    ts_size: int | float,
    samples_per_class: int = 1,
) -> dict[SplitNames, np.ndarray]:
    """
    Returns train/validation/test indices for a collection, with guarentees that at least
    `samples_per_class` samples from each class are present in each split.
    """

    num_splits = 2 if vl_size == 0 or ts_size == 0 else 3
    values, counts = np.unique(list(flatten(labels)), return_counts=True)

    # Verify there are enough samples per class to guarentee each split has representative samples.
    if any(counts < (samples_per_class * num_splits)):
        raise ValueError(
            f"Not enough samples per class to create {num_splits} splits."
            f"dist={pformat(Counter(labels))}"
        )

    # Get both integer and floating point representations of the size of the splits.
    if any(isinstance(x, float) for x in (tr_size, vl_size, ts_size)):
        tr_size, vl_size, ts_size = compute_integer_sizes(len(labels), tr_size, vl_size, ts_size)
    total = tr_size + vl_size + ts_size
    tr_prop, vl_prop, ts_prop = compute_float_sizes(total, tr_size, vl_size, ts_size)

    # Verify that the splits are large enough to contain `samples_per_class` samples for each class.
    for split, size in zip(["tr", "vl", "ts"], [tr_size, vl_size, ts_size]):
        if 0 < size < len(values) * samples_per_class:
            raise ValueError(
                f"The {split=} set with {size=} is too small to contain {samples_per_class} samples"
                f" per class for {len(values)} classes."
            )

    # For each element in the collection, assign it to a particular split if that split has not
    # reached its quota for that class.
    tr_dist, tr_idx = {v: 0 for v in values}, []
    vl_dist, vl_idx = {v: 0 for v in values}, []
    ts_dist, ts_idx = {v: 0 for v in values}, []
    for i, ls in enumerate(labels):
        if tr_size > 0 and any(tr_dist[l] < samples_per_class for l in ls):
            for l in ls:
                tr_dist[l] += 1
            tr_idx.append(i)
        elif vl_size > 0 and any(vl_dist[l] < samples_per_class for l in ls):
            for l in ls:
                vl_dist[l] += 1
            vl_idx.append(i)
        elif ts_size > 0 and any(ts_dist[l] < samples_per_class for l in ls):
            for l in ls:
                ts_dist[l] += 1
            ts_idx.append(i)

    if tr_size > 0 and any(tr_dist[l] < samples_per_class for l in values):
        raise ValueError(f"Failed to create an equitable split.\n{tr_dist=}\n{vl_dist=}\n{ts_dist=}")
    if vl_size > 0 and any(vl_dist[l] < samples_per_class for l in values):
        raise ValueError(f"Failed to create an equitable split.\n{tr_dist=}\n{vl_dist=}\n{ts_dist=}")
    if ts_size > 0 and any(ts_dist[l] < samples_per_class for l in values):
        raise ValueError(f"Failed to create an equitable split.\n{tr_dist=}\n{vl_dist=}\n{ts_dist=}")

    # Add the remaining samples to the splits if they have not already been added.
    added = set(tr_idx) | set(vl_idx) | set(ts_idx)
    for i, ls in enumerate(labels):
        if i in added:
            continue

        r = random.uniform(0, sum((tr_prop, vl_prop, ts_prop)))
        if (0 <= r < ts_prop) and (len(ts_idx) < ts_size):
            for l in ls:
                ts_dist[l] += 1
            ts_idx.append(i)
        elif (ts_prop <= r < vl_prop + ts_prop) and (len(vl_idx) < vl_size):
            for l in ls:
                vl_dist[l] += 1
            vl_idx.append(i)
        elif (ts_prop + vl_prop <= r < tr_prop + vl_prop + ts_prop) and (len(tr_idx) < tr_size):
            for l in ls:
                tr_dist[l] += 1
            tr_idx.append(i)

    assert set.intersection(set(tr_idx), set(vl_idx), set(ts_idx)) == set(), "Indices are not mutually exclusive."

    tr_idx = np.array(tr_idx, dtype=np.int32)
    vl_idx = np.array(vl_idx, dtype=np.int32)
    ts_idx = np.array(ts_idx, dtype=np.int32)
    return {"tr": tr_idx, "vl": vl_idx, "ts": ts_idx}


################################################################################
# Filtering Datasets
################################################################################


def filter_file_label_map(
    files_and_labels: dict[os.PathLike, str],
    top_k: Optional[int] = None,
    min_freq: int = 1,
    max_imbalance_ratio: Optional[int] = None,
    min_size: int = 0,
    max_size: int = sys.maxsize,
    must_exist: bool = True,
) -> dict[os.PathLike, str]:
    """
    Remove samples from the file label map whose labels are not in the top_k most frequent labels
    or whose frequency is less than min_freq. The default values do not filter at all.
    """
    # Remove files that do not exist on the current system.
    if must_exist:
        files_and_labels = {
            f: l for f, l in files_and_labels.items() if os.path.exists(f)
        }

    # Remove files that are too small or too large, if they exist.
    if min_size > 0 or max_size < sys.maxsize:
        files_and_labels = {
            f: l for f, l in files_and_labels.items() if
            (not os.path.exists(f) or (min_size <= os.path.getsize(f) <= max_size))
        }

    # Remove files not in the top_k most prolific classes or files without min_freq examples.
    dist = Counter(files_and_labels.values())
    keep = [l for l, n in dist.most_common(top_k) if n >= min_freq]
    files_and_labels = {f: l for f, l in files_and_labels.items() if l in keep}

    # Remove some files from prolific classes to prevent the ratio exceeding max_imbalance_ratio.
    if max_imbalance_ratio is not None:
        dist = Counter(files_and_labels.values())
        min_n = dist.most_common(None)[-1][1]
        remove = []
        for f, l in files_and_labels.items():
            if dist[l] / min_n > max_imbalance_ratio:
                remove.append(f)
            dist[l] -= 1
        remove = set(remove)
        files_and_labels = {f: l for f, l in files_and_labels.items() if f not in remove}

    return files_and_labels


def filter_file_label_map_multilabel(
    files_and_labels: dict[os.PathLike, tuple[str]],
    top_k: Optional[int] = None,
    min_freq: int = 1,
    min_size: int = 0,
    max_size: int = sys.maxsize,
    must_exist: bool = True,
) -> dict[os.PathLike, str]:
    """
    Remove samples from the file label map whose labels are not in the top_k most frequent labels
    or whose frequency is less than min_freq. The default values do not filter at all.
    """
    if must_exist:  # Remove files that do not exist on the current system.
        files_and_labels = {
            f: l for f, l in files_and_labels.items() if os.path.exists(f)
        }
    if min_size > 0 or max_size < sys.maxsize:  # Remove files that are too small or too large, if they exist.
        files_and_labels = {
            f: l for f, l in files_and_labels.items() if
            (not os.path.exists(f) or (min_size <= os.path.getsize(f) <= max_size))
        }

    dist: Counter[str, int] = Counter(chain.from_iterable(files_and_labels.values()))
    keep: list[str] = [l for l, n in dist.most_common(top_k) if n >= min_freq]
    files_and_labels: dict[Path, str] = {
        file: tuple(l for l in labels if l in keep) for file, labels in files_and_labels.items()
    }
    files_and_labels: dict[Path, str] = {
        file: labels for file, labels in files_and_labels.items() if labels
    }
    return files_and_labels


def filter_packed_files(
    files: list[str],
    packing_protocol: Literal["yes", "no", "any", "unk"],
    root: Optional[str | Path | list[str | Path]] = None,
) -> list[str]:

    if packing_protocol == "any":
        return files


    def path_to_key(f: Path) -> str:
        return f.stem

    def str_to_key(f: str) -> str:
        return os.path.splitext(os.path.basename(f))[0]


    if isinstance(files[0], str):
        fn = str_to_key
    elif isinstance(files[0], Path):
        fn = path_to_key
    else:
        raise TypeError(f"Invalid type: {type(files[0])}")


    num_workers = min(len(os.sched_getaffinity(0)), 20) if os.environ.get("DEBUG") != "1" else 8
    print(f"Getting packing map ({root=})")
    t_0 = time.time()
    is_packed = universal_packing_map(root, lazy=False, chunked=True, num_workers=num_workers)
    print(f"Got packing map ({len(is_packed)=}) in {round(time.time() - t_0, 2)} seconds.")


    n_negative = sum(1 for f in files if is_packed.get(fn(f)) is False)
    n_positive = sum(1 for f in files if is_packed.get(fn(f)) is True)
    n_unknown = sum(1 for f in files if is_packed.get(fn(f)) is None)
    print(f"Packing Distribution for {len(files)=}:")
    print(f"\t{n_negative=}")
    print(f"\t{n_positive=}")
    print(f"\t{n_unknown=}")


    if packing_protocol == "no":
        return [f for f in files if is_packed.get(fn(f)) is False]
    if packing_protocol == "yes":
        return [f for f in files if is_packed.get(fn(f)) is True]
    if packing_protocol == "unk":
        return [f for f in files if is_packed.get(fn(f)) is None]

    raise ValueError(f"Invalid: {packing_protocol=}")


################################################################################
# Load File-label Maps
################################################################################


def get_bodmas_file_label_map() -> dict[os.PathLike, str]:
    """
    Get the files and labels associated with the BODMAS corpus.
    """

    files = list(sorted(DATASET_TO_FILES["binaries"]["bodmas_pe"]()))
    labels = pd.read_csv(BODMAS_LABELS_FILE).set_index("sha")["family"].to_dict()
    files_and_labels = {
        f.as_posix() : labels[f.stem] for f in files
        if labels[f.stem] not in (np.NaN, "unknown", "Unknown")
    }
    return files_and_labels


def _get_sorel_file_label_map() -> dict[os.PathLike, Label]:

    filter_args = FilterArgs(
        class_=(None, 2),
        file=(1, 2),
        fam=(1, 2),
        beh=(None, 2),
        unk=(None, 2),
        pack=(None, 1),
        vuln=(None, 1),
    )
    labeler = Labeler(
        SOREL_CLARAVY_CACHE,
        SOREL_AVCLASS_CACHE,
        SOREL_AVCLASS_FAMILY_CACHE,
        filter_args
    )()
    files_and_labels = {
        str(f): labeler.data.get(f.stem) for f in DATASET_TO_FILES["binaries"]["sorel_pe"]()
    }
    files_and_labels = {
        f: l for f, l in files_and_labels.items() if l is not None and l.is_labeled
    }
    return files_and_labels


def get_sorel_file_label_map(name: str) -> dict[os.PathLike, tuple[str]]:

    files_and_labels = _get_sorel_file_label_map()
    files_and_labels = {f: getattr(l, name) for f, l in files_and_labels.items()}
    files_and_labels = {f: l for f, l in files_and_labels.items() if l is not None}
    return files_and_labels


def _get_elf_file_label_map() -> dict[os.PathLike, Label]:

    raise NotImplementedError()
    # pylint: disable=unreachable
    filter_args = FilterArgs(
        class_=(None, 2),
        file=(1, 2),
        fam=(1, 2),
        beh=(None, 2),
        unk=(None, 2),
        pack=(None, 1),
        vuln=(None, 1),
    )
    labeler = Labeler(
        ELF_CLARAVY_CACHE,  # pylint: disable=undefined-variable
        ELF_AVCLASS_CACHE,  # pylint: disable=undefined-variable
        ELF_AVCLASS_FAMILY_CACHE,  # pylint: disable=undefined-variable
        filter_args
    )()
    files = chain.from_iterable(
        (DATASET_TO_FILES["binaries"][d]() for d in ELF_CLASSIFICATION_DATASETS)
    )
    files_and_labels = {
        str(f): labeler.data.get(f.stem) for f in files
    }
    files_and_labels = {
        f: l for f, l in files_and_labels.items() if l is not None and l.is_labeled
    }
    return files_and_labels


def get_elf_file_label_map(name: str) -> dict[os.PathLike, tuple[str]]:

    raise NotImplementedError()
    # pylint: disable=unreachable
    files_and_labels = _get_elf_file_label_map()
    files_and_labels = {f: getattr(l, name) for f, l in files_and_labels.items()}
    files_and_labels = {f: l for f, l in files_and_labels.items() if l is not None}
    return files_and_labels


################################################################################
# Load Materials Core
################################################################################


def _get_materials_pretrain(
    files: list[os.PathLike],
    tr_size: int | float,
    vl_size: int | float,
    ts_size: int | float,
    packing_protocol: Literal["yes", "no", "any", "unk"] = "any",
    packing_root: Optional[Path | list[Path]] = None,
    remove: tuple[str] = tuple(),
) -> Materials:
    files = sorted(map(lambda p: p.as_posix(), DATASET_TO_FILES["binaries"]["sorel_pe"]()))
    files = filter_packed_files(files, packing_protocol, root=packing_root)
    remove = set(remove)
    files = [f for f in files if f not in remove and os.path.basename(f).split(".")[0] not in remove]

    if tr_size == -1 or (isinstance(tr_size, int) and tr_size >= (len(files) - vl_size - ts_size)):
        tr_size = len(files) - vl_size - ts_size

    tr_vl_ts_files = tr_vl_ts_split(files, tr_size, vl_size, ts_size)
    return Materials(files=tr_vl_ts_files)


def _get_materials_clf(
    files_and_labels: dict[str, str],
    tr_size: int | float,
    vl_size: int | float,
    ts_size: int | float,
    top_k: Optional[int] = None,
    min_freq: Optional[int] = None,
    max_imbalance_ratio: Optional[int] = None,
    min_size: int = 0,
    packing_protocol: Literal["yes", "no", "any", "unk"] = "any",
    packing_root: Optional[Path | list[Path]] = None,
    must_exist: bool = True,
) -> Materials:

    if min_freq is None:
        if vl_size == 0 or ts_size == 0:
            min_freq = MIN_SAMPLES_PER_CLASS_PER_SPLIT * 2
        else:
            min_freq = MIN_SAMPLES_PER_CLASS_PER_SPLIT * 3

    files_to_keep = filter_packed_files(list(files_and_labels.keys()), packing_protocol, root=packing_root)
    files_and_labels = {f: files_and_labels[f] for f in files_to_keep}

    # Filter out the files that are not in the top_k most frequent labels
    files_and_labels = filter_file_label_map(
        files_and_labels,
        top_k=top_k,
        min_freq=min_freq,
        max_imbalance_ratio=max_imbalance_ratio,
        min_size=min_size,
        must_exist=must_exist,
    )

    # Final collection of data items
    dist: Counter[str, int] = Counter(files_and_labels.values())
    label2id = {l: i for i, l in enumerate(dist.keys())}
    id2label = {i: l for l, i in label2id.items()}

    files = list(files_and_labels.keys())
    labels = np.array([label2id[files_and_labels[f]] for f in files])

    files, labels = shuffle(files, labels)

    idx = tr_vl_ts_split_idx_guarentee(
        labels, tr_size, vl_size, ts_size, MIN_SAMPLES_PER_CLASS_PER_SPLIT
    )

    files, labels = get_tr_vl_ts_files_and_labels(files, labels, idx)
    return Materials(files, labels, id2label, label2id, dist)


def _get_materials_clf_few_shot_learning(
    files_and_labels: dict[str, str],
    tr_samples_per_class: int,
    vl_min_samples_per_class: int = 1,
    vl_max_samples_per_class: int = 10,
    top_k: Optional[int] = None,
    min_size: int = 0,
    packing_protocol: Literal["yes", "no", "any", "unk"] = "any",
    packing_root: Optional[Path | list[Path]] = None,
    must_exist: bool = True,
    **kwds,
) -> Materials:
    if invalid := set(kwds) - {"min_freq", "max_imbalance_ratio"}:
        raise TypeError(f"Function got some unexpected keyword argument(s): {invalid}")

    # First, remove the files that we do not want to use.
    files_to_keep = filter_packed_files(list(files_and_labels.keys()), packing_protocol, root=packing_root)
    files_and_labels = {f: files_and_labels[f] for f in files_to_keep}
    files_and_labels = filter_file_label_map(
        files_and_labels,
        top_k=top_k,
        min_freq=tr_samples_per_class + vl_min_samples_per_class,
        min_size=min_size,
        must_exist=must_exist,
    )

    # Get a distribution of the labels and their mappings to/from label IDs.
    dist: Counter[str, int] = Counter(files_and_labels.values())
    label2id: dict[str, int] = {l: i for i, l in enumerate(dist.keys())}
    id2label: dict[int, str] = {i: l for l, i in label2id.items()}

    # Get all of the files and their labels, and shuffle them.
    files = list(files_and_labels.keys())
    labels = np.array([label2id[files_and_labels[f]] for f in files])
    files, labels = shuffle(files, labels)

    # Get indices for the train, validation, and test (empty) sets.
    tr_idx = select_k_for_each_class(labels, k=tr_samples_per_class)
    vl_idx_and_label = [(i, labels[i]) for i in range(len(files)) if i not in tr_idx]
    vl_counts = Counter()
    vl_idx = []
    for i, l in vl_idx_and_label:
        if vl_counts[l] < vl_max_samples_per_class:
            vl_idx.append(i)
            vl_counts[l] += 1
        if all(vl_counts[l] >= vl_min_samples_per_class for l in dist):
            break
    ts_idx = []

    # Update the distribution to reflect the samples not included.
    files, labels = get_tr_vl_ts_files_and_labels(files, labels, None, tr_idx, vl_idx, ts_idx)
    dist = Counter(id2label[i] for i in chain.from_iterable(labels.values()))
    return Materials(files, labels, id2label, label2id, dist)


def _get_materials_clf_multilabel(
    files_and_labels: dict[str, str],
    tr_size: int | float,
    vl_size: int | float,
    ts_size: int | float,
    top_k: Optional[int] = None,
    min_freq: Optional[int] = None,
    max_imbalance_ratio: Optional[int] = None,
    min_size: int = 0,
    packing_protocol: Literal["yes", "no", "any", "unk"] = "any",
    packing_root: Optional[Path | list[Path]] = None,
    must_exist: bool = True,
) -> Materials:

    if min_freq is None:
        if vl_size == 0 or ts_size == 0:
            min_freq = MIN_SAMPLES_PER_CLASS_PER_SPLIT * 2
        else:
            min_freq = MIN_SAMPLES_PER_CLASS_PER_SPLIT * 3

    files_to_keep = filter_packed_files(list(files_and_labels.keys()), packing_protocol, root=packing_root)
    files_and_labels = {f: files_and_labels[f] for f in files_to_keep}

    # Filter out the files that are not in the top_k most frequent labels
    files_and_labels = filter_file_label_map_multilabel(
        files_and_labels,
        top_k=top_k,
        min_freq=min_freq,
        min_size=min_size,
        must_exist=must_exist,
    )

    # Final collection of data items
    dist: Counter[str, int] = Counter(chain.from_iterable(files_and_labels.values()))
    label2id = {l: i for i, l in enumerate(dist.keys())}
    id2label = {i: l for l, i in label2id.items()}

    files = list(files_and_labels.keys())
    labels = [tuple(label2id[l] for l in files_and_labels[f]) for f in files]

    files, labels = shuffle(files, labels)
    idx = tr_vl_ts_split_idx_multilabel_guarentee(
        labels, tr_size, vl_size, ts_size, MIN_SAMPLES_PER_CLASS_PER_SPLIT
    )
    materials = Materials(
        {s: [files[i] for i in idx[s]] for s in idx},
        {s: [labels[i] for i in idx[s]] for s in idx},
        id2label,
        label2id,
        dist,
    )
    if max_imbalance_ratio is None:
        return materials

    print("Filtering out some samples to prevent imbalance...")
    new_materials = {"files": {}, "labels": {}}
    for split, size in [("tr", tr_size), ("vl", vl_size), ("ts", ts_size)]:
        if size == 0:
            new_materials["files"][split] = []
            new_materials["labels"][split] = []
            continue

        labels: list[tuple[int]] = materials.labels[split]
        dist = materials.get_split_dist(split)
        min_n = min(dist.values())
        remove = []
        remove_counter = Counter()
        for i, label in enumerate(labels):
            if all(dist[id2label[l]] / min_n > max_imbalance_ratio for l in label):
                remove.append(i)
                for l in label:
                    l = id2label[l]
                    remove_counter.update([l])
                    dist[l] -= 1

        remove = set(remove)
        new_materials["files"][split] = [f for i, f in enumerate(materials.files[split]) if i not in remove]
        new_materials["labels"][split] = [l for i, l in enumerate(materials.labels[split]) if i not in remove]
        print(f"Removed {len(remove)} samples from {len(remove_counter)} classes for {split=}.")

    dist = Counter()
    for labels in new_materials["labels"].values():
        dist.update(map(lambda i: id2label[i], chain.from_iterable(labels)))
    return Materials(new_materials["files"], new_materials["labels"], id2label, label2id, dist)


def _get_materials_clf_multilabel_few_shot_learning(
    files_and_labels: dict[str, str],
    tr_samples_per_class: int,
    vl_min_samples_per_class: int = 1,
    top_k: Optional[int] = None,
    min_size: int = 0,
    packing_protocol: Literal["yes", "no", "any", "unk"] = "any",
    packing_root: Optional[Path | list[Path]] = None,
    must_exist: bool = True,
    **kwds,
) -> Materials:
    if invalid := set(kwds) - {"min_freq", "max_imbalance_ratio", "vl_max_samples_per_class"}:
        raise TypeError(f"Function got some unexpected keyword argument(s): {invalid}")

    raise NotImplementedError("This needs a bit more work. The unit tests are not passing.")  # pylint: disable=unreachable

    # First, remove the files that we do not want to use.
    files_to_keep = filter_packed_files(list(files_and_labels.keys()), packing_protocol, root=packing_root)
    files_and_labels = {f: files_and_labels[f] for f in files_to_keep}
    files_and_labels = filter_file_label_map_multilabel(
        files_and_labels,
        top_k=top_k,
        min_freq=tr_samples_per_class + vl_min_samples_per_class,
        min_size=min_size,
        must_exist=must_exist,
    )

    # Get a distribution of the labels and their mappings to/from label IDs.
    dist: Counter[str, int] = Counter(chain.from_iterable(files_and_labels.values()))
    label2id: dict[str, int] = {l: i for i, l in enumerate(dist.keys())}
    id2label: dict[int, str] = {i: l for l, i in label2id.items()}

    # Get all of the files and their labels, and shuffle them.
    files = list(files_and_labels.keys())
    labels = [tuple(label2id[l] for l in files_and_labels[file]) for file in files]
    files, labels = shuffle(files, labels)

    # Get indices for the train, validation, and test (empty) sets.
    tr_idx = select_k_for_each_class_multilabel(labels, k=tr_samples_per_class)
    tr_idx = set(tr_idx)

    vl_idx_and_label = [(i, labels[i]) for i in range(len(files)) if i not in tr_idx]
    _idx = select_k_for_each_class_multilabel([l for _, l in vl_idx_and_label], k=vl_min_samples_per_class)
    vl_idx = [vl_idx_and_label[i][0] for i in _idx]

    ts_idx = []

    assert set.intersection(set(tr_idx), set(vl_idx), set(ts_idx)) == set(), "Indices are not mutually exclusive."

    # Update the distribution to reflect the samples not included.
    files, labels = get_tr_vl_ts_files_and_labels(files, labels, None, tr_idx, vl_idx, ts_idx)
    dist = Counter()
    for l in labels.values():
        dist.update(id2label[i] for i in chain.from_iterable(l))
    return Materials(files, labels, id2label, label2id, dist)


################################################################################
# Load Materials Endpoints
################################################################################


def get_materials_pretrain_sorel(
    tr_size: int | float,
    vl_size: int | float,
    ts_size: int | float,
    remove_clf_files: bool = True,
    **kwds,
) -> Materials:
    files = sorted(map(lambda p: p.as_posix(), DATASET_TO_FILES["binaries"]["sorel_pe"]()))
    remove = []
    if remove_clf_files:
        files_and_labels = _get_sorel_file_label_map() | get_bodmas_file_label_map()
        remove = [os.path.basename(f).split(".")[0] for f in files_and_labels.keys()]
    return _get_materials_pretrain(
        files, tr_size, vl_size, ts_size, packing_root=PACKING_ROOTS["sorel_pe"], remove=remove, **kwds
    )


def get_materials_clf_bodmas(
    tr_size: Optional[int | float],
    vl_size: Optional[int | float],
    ts_size: Optional[int | float],
    tr_samples_per_class: Optional[int],
    **kwds,
) -> Materials:
    files_and_labels = get_bodmas_file_label_map()
    kwds["packing_root"] = PACKING_ROOTS["bodmas_pe"]
    if tr_samples_per_class is not None:
        return _get_materials_clf_few_shot_learning(files_and_labels, tr_samples_per_class, **kwds)
    return _get_materials_clf(files_and_labels, tr_size, vl_size, ts_size, **kwds)


def get_materials_clf_sorel(
    tr_size: Optional[int | float],
    vl_size: Optional[int | float],
    ts_size: Optional[int | float],
    tr_samples_per_class: Optional[int],
    name: str,
    **kwds,
) -> Materials:
    files_and_labels = get_sorel_file_label_map(name)
    kwds["packing_root"] = PACKING_ROOTS["sorel_pe"]

    if name in ("fam", "file"):  # We consider these single-label classification tasks.
        files_and_labels = {f: l[0] for f, l in files_and_labels.items()}
        if tr_samples_per_class is not None:
            return _get_materials_clf_few_shot_learning(files_and_labels, tr_samples_per_class, **kwds)
        return _get_materials_clf(files_and_labels, tr_size, vl_size, ts_size, **kwds)

    if tr_samples_per_class is not None:
        return _get_materials_clf_multilabel_few_shot_learning(files_and_labels, tr_samples_per_class, **kwds)
    return _get_materials_clf_multilabel(files_and_labels, tr_size, vl_size, ts_size, **kwds)


def get_materials_pretrain_elf(
    tr_size: int | float,
    vl_size: int | float,
    ts_size: int | float,
    remove_clf_files: bool = True,
    **kwds,
) -> Materials:
    raise NotImplementedError()
    # pylint: disable=unreachable
    files = chain.from_iterable((DATASET_TO_FILES["binaries"][d]() for d in ELF_CLASSIFICATION_DATASETS))
    files = sorted(get_unique_files(list(map(str, files))))
    packing_root = [PACKING_ROOTS[d] for d in ELF_CLASSIFICATION_DATASETS]
    remove = []
    if remove_clf_files:
        files_and_labels = _get_elf_file_label_map()
        remove = [os.path.basename(f).split(".")[0] for f in files_and_labels.keys()]
    return _get_materials_pretrain(
        files, tr_size, vl_size, ts_size, packing_root=packing_root, remove=remove, **kwds
    )


def get_materials_clf_elf(
    tr_size: Optional[int | float],
    vl_size: Optional[int | float],
    ts_size: Optional[int | float],
    tr_samples_per_class: Optional[int],
    name: str,
    **kwds,
) -> Materials:
    raise NotImplementedError()
    # pylint: disable=unreachable
    files_and_labels = get_elf_file_label_map(name)
    kwds["packing_root"] = [PACKING_ROOTS[d] for d in ELF_CLASSIFICATION_DATASETS]

    if name in ("fam", "file"):  # We consider these single-label classification tasks.
        files_and_labels = {f: l[0] for f, l in files_and_labels.items()}
        if tr_samples_per_class is not None:
            return _get_materials_clf_few_shot_learning(files_and_labels, tr_samples_per_class, **kwds)
        return _get_materials_clf(files_and_labels, tr_size, vl_size, ts_size, **kwds)

    if tr_samples_per_class is not None:
        return _get_materials_clf_multilabel_few_shot_learning(files_and_labels, tr_samples_per_class, **kwds)
    return _get_materials_clf_multilabel(files_and_labels, tr_size, vl_size, ts_size, **kwds)


# def get_materials_clf_bodmas_balanced_slice(
#     tr_size: int,
#     vl_size: int,
#     min_freq: Optional[int] = None,
#     top_k: Optional[int] = None,
#     balance_tr_set: bool = True,
# ) -> Materials:
#     """Returns small slices for the BODMAS training dataset. The validation set is consistent
#     accross all slices.
#     """

#     num_splits = 2
#     min_freq = MIN_SAMPLES_PER_CLASS_PER_SPLIT * num_splits if min_freq is None else min_freq

#     files_and_labels = get_bodmas_file_label_map()
#     files_and_labels = filter_file_label_map(files_and_labels, top_k=top_k, min_freq=min_freq)

#     files = files_and_labels.keys()
#     labels = files_and_labels.values()

#     dist: Counter[str, int] = Counter(labels)
#     label2id: dict[str, int] = {l: i for i, l in enumerate(dist.keys())}
#     id2label: dict[int, str] = {i: l for l, i in label2id.items()}

#     # Forces the validation set to be consistent across slices of various sizes.
#     idx = tr_vl_ts_split_idx(len(files_and_labels), len(files_and_labels) - vl_size, vl_size, 0)

#     # Balance the training set so that each class has the same number of samples.
#     if balance_tr_set:
#         samples_per_cls = tr_size / len(dist)
#         if not samples_per_cls.is_integer():
#             raise ValueError("Cannot balance tr_set because train size is not divisible by number of classes")
#         tr_sub_idx = select_k_for_each_class([labels[i] for i in idx["tr"]], k=samples_per_cls)
#     # The tr_set itself is already random, so we can just take the first ones.
#     else:
#         tr_sub_idx = list(range(tr_size))

#     assert len(tr_sub_idx) == tr_size
#     idx["tr"] = idx["tr"][tr_sub_idx]

#     files, labels = get_tr_vl_ts_files_and_labels(files, labels, None, idx["tr"], idx["vl"], [])
#     return Materials(files, labels, id2label, label2id, dist)


if __name__ == "__main__":
    # materials = _get_materials_clf_multilabel_few_shot_learning(
    #     get_sorel_file_label_map("beh"),
    #     1,
    #     vl_min_samples_per_class=1,
    #     vl_max_samples_per_class=10,
    #     top_k=None,
    # )

    materials = _get_materials_clf_multilabel(
        get_sorel_file_label_map("beh"),
        tr_size=0.8,
        vl_size=0.1,
        ts_size=0.1,
        must_exist=False,
        max_imbalance_ratio=100,
    )
