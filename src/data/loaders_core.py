"""
Core file operations to construct labeled datasets for classification tasks.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import chain, cycle, islice
import json
import math
import os
from pathlib import Path
from pprint import pprint, pformat
import random
import sys
from statistics import mean, median
from typing import Callable, Literal, Optional

# pylint: disable=wrong-import-position
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# pylint: enable=wrong-import-position

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm

from src.utils import get_max_keys_from_dict
from src.data.cfg import SOREL_PATH, BODMAS_LABELS_FILE, DATASET_TO_FILES, SOREL_META_CSV
from src.data.detect_packing_sorel import PackingMap
from src.data.label_datasets import (
    get_label_mapping_virus_total_reports_sorel,
    ThreatLabelExtractor,
    ThreatLabelRefiner,
)


MIN_SAMPLES_PER_CLASS_PER_SPLIT = 1


SplitNames = Literal["tr", "vl", "ts"]
FilesAndLabels = tuple[list[os.PathLike], Optional[Sequence[int]]]


@dataclass
class Materials:
    files: dict[SplitNames, list[os.PathLike]]
    labels: Optional[dict[SplitNames, Sequence[int]]] = None
    id2label: dict[int, str] = None
    label2id: dict[str, int] = None
    dist: Counter[str, int] = None

    def __repr__(self):
        return (
            f"len(tr)={len(self.files['tr'])}\n"
            f"len(vl)={len(self.files['vl'])}\n"
            f"len(ts)={len(self.files['ts'])}\n"
            f"num_classes={len(self.id2label) if self.id2label is not None else None}\n"
            f"dist={pformat(self.dist) if self.dist is not None else None}"
        )


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


def select_k_for_each_class(labels: list[int | str], k: int) -> list[int]:
    unique = set(labels)
    count = {s : 0 for s in unique}
    idx = []
    for i, l in enumerate(labels):
        if count[l] < k:
            count[l] += 1
            idx.append(i)
    return idx


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

    num_splits = 2 if vl_size == 0 or tr_size == 0 else 3
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
        elif ts_size > 0 and tr_dist[l] < samples_per_class:
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


def tr_vl_ts_split_guarentee(
    collection: Sequence,
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

    collection, labels = shuffle(collection, labels)
    idx = tr_vl_ts_split_idx_guarentee(labels, tr_size, vl_size, ts_size, samples_per_class)
    try:  # Try to slice with a numpy array
        return {s: collection[indices] for s, indices in idx.items()}
    except TypeError:  # If that fails, use list comprehension
        return {s: [collection[i] for i in indices] for s, indices in idx.items()}


def get_tr_vl_ts_files_and_labels(
    files: list[os.PathLike],
    labels: np.ndarray,
    idx: Optional[dict[SplitNames, Sequence[int]]] = None,
    tr_idx: Optional[Sequence[int]] = None,
    vl_idx: Optional[Sequence[int]] = None,
    ts_idx: Optional[Sequence[int]] = None,
) -> tuple[dict[SplitNames, list[os.PathLike]], dict[SplitNames, np.ndarray]]:

    if idx is not None:
        if tr_idx is not None or vl_idx is not None or ts_idx is not None:
            raise ValueError("Cannot specify both `idx` and `tr_idx`, `vl_idx`, `ts_idx`.")
        tr_idx, vl_idx, ts_idx = idx["tr"], idx["vl"], idx["ts"]

    labels = np.array(labels) if isinstance(labels, list) else labels
    files = {
        "tr": [files[i] for i in tr_idx],
        "vl": [files[i] for i in vl_idx],
        "ts": [files[i] for i in ts_idx],
    }
    labels = {
        "tr": labels[tr_idx],
        "vl": labels[vl_idx],
        "ts": labels[ts_idx],
    }
    return files, labels


def filter_file_label_map(
    files_and_labels: dict[os.PathLike, str],
    top_k: Optional[int] = None,
    min_freq: int = 1,
    min_size: int = 0,
) -> dict[os.PathLike, str]:
    """
    Remove samples from the file label map whose labels are not in the top_k most frequent labels
    or whose frequency is less than min_freq. The default values do not filter at all.
    """

    files_and_labels = {
        f: l for f, l in files_and_labels.items()
        if os.path.exists(f) and os.path.getsize(f) >= min_size
    }
    dist: Counter[str, int] = Counter(files_and_labels.values())
    keep: list[str] = [l for l, n in dist.most_common(top_k) if n >= min_freq]
    files_and_labels: dict[Path, str] = {
        f: l for f, l in files_and_labels.items()
        if l in keep
    }
    return files_and_labels


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


def get_sorel_virus_total_file_label_map() -> dict[os.PathLike, str]:
    """
    Get the files and labels from VirusTotal reports for the SOREL dataset.
    """

    files = sorted(list(map(str, DATASET_TO_FILES["reports"]["sorel_pe"]())))
    extractor = ThreatLabelExtractor.build("category")
    refiner = ThreatLabelRefiner.build("top", k=1)
    files_and_labels = get_label_mapping_virus_total_reports_sorel(files, extractor, refiner)
    files_and_labels = {
        f: l[0] for f, l in files_and_labels.items()
        if isinstance(l, (list, tuple))
    }
    sorel_path = os.path.join(SOREL_PATH.as_posix(), "binaries")
    files_and_labels = {os.path.join(sorel_path, sha): l for sha, l in files_and_labels.items()}
    return files_and_labels


def get_sorel_original_labels_file_label_map(**kwds) -> dict[os.PathLike, str]:
    df = pd.read_csv(SOREL_META_CSV, **kwds)
    df = df[df["is_malware"] == 1]
    df = df.drop(columns=["is_malware", "rl_fs_t", "rl_ls_const_positives"])
    df = df.set_index("sha256")
    d = df.to_dict(orient="index")
    d = {sha: get_max_keys_from_dict(labels) for sha, labels in d.items()}
    d = {sha: labels[0] for sha, labels in d.items() if len(labels) == 1}
    sorel_path = os.path.join(SOREL_PATH.as_posix(), "binaries")
    files_and_labels = {os.path.join(sorel_path, sha): l for sha, l in d.items()}
    return files_and_labels


def get_materials_pretrain_sorel(
    tr_size: int | float,
    vl_size: int | float,
    ts_size: int | float,
    remove_packed: bool = False,
) -> Materials:
    files = sorted(map(lambda p: p.as_posix(), DATASET_TO_FILES["binaries"]["sorel_pe"]()))
    if remove_packed:
        is_packed = PackingMap(
            lazy=False, chunked=True, num_workers=len(os.sched_getaffinity(0)),
        )
        print(f"Packing Negative, Positive, and Unknown: {len(files)=}")
        files = [f for f in files if is_packed.get(os.path.splitext(os.path.basename(f))[0]) is not None]
        print(f"Packing Negative and Positive: {len(files)=}")
        files = [f for f in files if is_packed[os.path.splitext(os.path.basename(f))[0]] is False]
        print(f"Packing Negative: {len(files)=}")

    tr_vl_ts_files = tr_vl_ts_split(files, tr_size, vl_size, ts_size)
    return Materials(files=tr_vl_ts_files)


def get_materials_clf(
    files_and_labels: dict[str, str],
    tr_size: int | float,
    vl_size: int | float,
    ts_size: int | float,
    top_k: Optional[int] = None,
    min_freq: Optional[int] = None,
    min_size: int = 0,
) -> Materials:

    num_splits = 2 if vl_size == 0 or tr_size == 0 else 3
    min_freq = MIN_SAMPLES_PER_CLASS_PER_SPLIT * num_splits if min_freq is None else min_freq

    # Filter out the files that are not in the top_k most frequent labels
    files_and_labels = filter_file_label_map(files_and_labels, top_k, min_freq, min_size)

    # Final collection of data items
    dist: Counter[str, int] = Counter(files_and_labels.values())
    label2id = {l: i for i, l in enumerate(dist.keys())}
    id2label = {i: l for l, i in label2id.items()}

    files = list(files_and_labels.keys())
    labels = np.array([label2id[files_and_labels[f]] for f in files])

    idx = tr_vl_ts_split_idx_guarentee(
        labels, tr_size, vl_size, ts_size, MIN_SAMPLES_PER_CLASS_PER_SPLIT
    )

    files, labels = get_tr_vl_ts_files_and_labels(files, labels, idx)
    return Materials(files, labels, id2label, label2id, dist)


def get_materials_clf_bodmas(
    tr_size: int | float,
    vl_size: int | float,
    ts_size: int | float,
    **kwds,
) -> Materials:

    files_and_labels = get_bodmas_file_label_map()
    return get_materials_clf(files_and_labels, tr_size, vl_size, ts_size, **kwds)


def get_materials_clf_bodmas_with_k_samples_per_class_in_train_set(
    tr_samples_per_class: int,
    vl_samples_per_class: Optional[int] = None,
    top_k: Optional[int] = None,
) -> Materials:
    """
    Returns a balanced BODMAS dataset with the same number of samples for each class in the
    train set. The remainder of the samples are allocated to the validation set.
    """
    _vl_samples_per_class = 1 if vl_samples_per_class is None else vl_samples_per_class
    min_freq = tr_samples_per_class + _vl_samples_per_class

    files_and_labels = get_bodmas_file_label_map()
    files_and_labels = filter_file_label_map(files_and_labels, top_k=top_k, min_freq=min_freq)

    dist: Counter[str, int] = Counter(files_and_labels.values())
    label2id: dict[str, int] = {l: i for i, l in enumerate(dist.keys())}
    id2label: dict[int, str] = {i: l for l, i in label2id.items()}

    files = list(files_and_labels.keys())
    labels = list(files_and_labels.values())
    labels = np.array([label2id[l] for l in labels], dtype=np.int32)
    files, labels = shuffle(files, labels)

    tr_idx = select_k_for_each_class(labels, k=tr_samples_per_class)
    if vl_samples_per_class is None:
        vl_idx = [i for i in range(len(files_and_labels)) if i not in tr_idx]
    else:
        # _labels: [(IDX of original file/labels data structure, label)]
        _labels = [(i, l) for i, l in enumerate(labels) if i not in tr_idx]
        _vl_idx = select_k_for_each_class([l for _, l in _labels], k=vl_samples_per_class)
        vl_idx = [i for j, (i, l) in enumerate(_labels) if j in _vl_idx]
    ts_idx = []

    files, labels = get_tr_vl_ts_files_and_labels(files, labels, None, tr_idx, vl_idx, ts_idx)

    tr_dist = Counter(labels["tr"])
    vl_dist = Counter(labels["vl"])
    assert len(tr_dist) == len(vl_dist), f"{len(tr_dist)=} != {len(vl_dist)=}"
    assert all(tr_dist[l] == tr_samples_per_class for l in tr_dist), f"tr_dist={pformat(tr_dist)}"
    assert vl_samples_per_class is None or all(vl_dist[l] == vl_samples_per_class for l in vl_dist), f"vl_dist={pformat(vl_dist)}"

    dist = Counter(id2label[i] for i in chain.from_iterable(labels.values()))
    return Materials(files, labels, id2label, label2id, dist)


def get_materials_clf_bodmas_balanced_slice(
    tr_size: int,
    vl_size: int,
    min_freq: Optional[int] = None,
    top_k: Optional[int] = None,
    balance_tr_set: bool = True,
) -> Materials:
    """Returns small slices for the BODMAS training dataset. The validation set is consistent
    accross all slices.
    """

    num_splits = 2
    min_freq = MIN_SAMPLES_PER_CLASS_PER_SPLIT * num_splits if min_freq is None else min_freq

    files_and_labels = get_bodmas_file_label_map()
    files_and_labels = filter_file_label_map(files_and_labels, top_k=top_k, min_freq=min_freq)

    files = files_and_labels.keys()
    labels = files_and_labels.values()

    dist: Counter[str, int] = Counter(labels)
    label2id: dict[str, int] = {l: i for i, l in enumerate(dist.keys())}
    id2label: dict[int, str] = {i: l for l, i in label2id.items()}

    # Forces the validation set to be consistent across slices of various sizes.
    idx = tr_vl_ts_split_idx(len(files_and_labels), len(files_and_labels) - vl_size, vl_size, 0)

    # Balance the training set so that each class has the same number of samples.
    if balance_tr_set:
        samples_per_cls = tr_size / len(dist)
        if not samples_per_cls.is_integer():
            raise ValueError("Cannot balance tr_set because train size is not divisible by number of classes")
        tr_sub_idx = select_k_for_each_class([labels[i] for i in idx["tr"]], k=samples_per_cls)
    # The tr_set itself is already random, so we can just take the first ones.
    else:
        tr_sub_idx = list(range(tr_size))

    assert len(tr_sub_idx) == tr_size
    idx["tr"] = idx["tr"][tr_sub_idx]

    files, labels = get_tr_vl_ts_files_and_labels(files, labels, None, idx["tr"], idx["vl"], [])
    return Materials(files, labels, id2label, label2id, dist)


def get_materials_clf_sorel(
    tr_size: int | float,
    vl_size: int | float,
    ts_size: int | float,
    **kwds,
) -> Materials:

    files_and_labels = get_sorel_original_labels_file_label_map()
    return get_materials_clf(files_and_labels, tr_size, vl_size, ts_size, **kwds)


class GetMaterialsClfLengthExtrapolation:

    def __init__(
        self,
        classes: list[str],
        tr_size: int,
        vl_size: int,
        tr_length_cutoffs: list[int],
        cache_root: Path = None,
    ) -> None:
        self.classes = classes
        self.tr_size = tr_size
        self.vl_size = vl_size
        self.tr_length_cutoffs = tr_length_cutoffs
        self.cache_root = cache_root

        self.tr_samples_per_class = int(tr_size // len(classes))
        self.vl_samples_per_class = int(vl_size // len(classes))

    def cache_path(self, tr_length_cutoff: int) -> Path:
        return (
            self.cache_root /
            f"tr_length_cutoffs--{'_'.join(map(str, sorted(self.tr_length_cutoffs)))}" /
            f"tr_size--{self.tr_size}" /
            f"vl_size--{self.vl_size}" /
            f"tr_length_cutoff--{tr_length_cutoff}"
        )

    def tr_cache_path(self, tr_length_cutoff: int) -> Path:
        return self.cache_path(tr_length_cutoff) / "tr.csv"

    def vl_cache_path(self, tr_length_cutoff: int) -> Path:
        return self.cache_path(tr_length_cutoff) / "vl.csv"

    def id2label_cache_path(self, tr_length_cutoff: int) -> Path:
        return self.cache_path(tr_length_cutoff) / "id2label.json"

    def label2id_cache_path(self, tr_length_cutoff: int) -> Path:
        return self.cache_path(tr_length_cutoff) / "label2id.json"

    def dist_cache_path(self, tr_length_cutoff: int) -> Path:
        return self.cache_path(tr_length_cutoff) / "dist.json"

    def load_from_cache(self, tr_length_cutoff: int):
        if not self.cache_path(tr_length_cutoff).exists():
            return None

        print("Getting length extrapolcation dataset from cache file.")
        df = pd.read_csv(self.tr_cache_path(tr_length_cutoff), index_col=None)
        tr_files = df["files"].tolist()
        tr_labels = df["labels"].tolist()
        df = pd.read_csv(self.vl_cache_path(tr_length_cutoff), index_col=None)
        vl_files = df["files"].tolist()
        vl_labels = df["labels"].tolist()
        with open(self.id2label_cache_path(tr_length_cutoff), "r") as fp:
            id2label = json.load(fp)
            id2label = {int(i) : l for i, l in id2label.items()}
        with open(self.label2id_cache_path(tr_length_cutoff), "r") as fp:
            label2id = json.load(fp)
        with open(self.dist_cache_path(tr_length_cutoff), "r") as fp:
            dist = Counter(json.load(fp))

        files = {
            "tr": tr_files,
            "vl": vl_files,
            "ts": [],
        }
        labels = {
            "tr": np.array(tr_labels, dtype=np.int32),
            "vl": np.array(vl_labels, dtype=np.int32),
            "ts": np.array([], dtype=np.int32),
        }
        return Materials(files, labels, id2label, label2id, dist)

    def get_tr_and_ts_idx(
        self,
        tr_cutoff: int,
        files: list[os.PathLike],
        labels: np.ndarray,
        label2id: dict[str, int],
    ) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
        tr_idx = {}
        vl_idx = {}
        for c in tqdm(self.classes, desc="Getting tr and vl idx for each class"):
            c_encoded = label2id[c]
            idx = np.where(labels == c_encoded)[0].tolist()
            tr_idx[c], vl_idx[c] = train_test_split(
                idx, test_size=self.vl_samples_per_class, random_state=0
            )
            tr_idx_within_cuttoff = [i for i in tr_idx[c] if os.path.getsize(files[i]) <= tr_cutoff]
            tr_idx[c] = tr_idx_within_cuttoff

        tr_samples_per_class = min(min(len(v) for v in tr_idx.values()), self.tr_samples_per_class)
        tr_idx = {c: v[0:tr_samples_per_class] for c, v in tr_idx.items()}
        return tr_idx, vl_idx

    def __call__(self, tr_length_cutoff: int) -> Materials:
        if (materials := self.load_from_cache(tr_length_cutoff)) is not None:
            return materials

        print("Building length extrapolcation dataset and saving to cache file.")
        files_and_labels = get_sorel_original_labels_file_label_map(nrows=None)
        files_and_labels = {
            f: l for f, l in files_and_labels.items()
            if l in self.classes and os.path.exists(f)
        }

        dist: Counter[str, int] = Counter(files_and_labels.values())
        label2id: dict[str, int] = {l: i for i, l in enumerate(dist.keys())}
        id2label: dict[int, str] = {i: l for l, i in label2id.items()}

        files = list(files_and_labels.keys())
        labels = list(files_and_labels.values())
        labels = np.array([label2id[l] for l in labels], dtype=np.int32)

        # Calculate the split for the smallest length cutoff, then use this number of samples
        # for the larger splits to ensure the training size is consistent across all splits.
        tr_idx, _ = self.get_tr_and_ts_idx(min(self.tr_length_cutoffs), files, labels, label2id)
        min_training_sizes_per_class_across_cutoffs = [len(x) for x in tr_idx.values()]
        assert len(set(min_training_sizes_per_class_across_cutoffs)) == 1
        min_training_size_per_class_across_cutoffs = min_training_sizes_per_class_across_cutoffs[0]

        tr_idx, vl_idx = self.get_tr_and_ts_idx(tr_length_cutoff, files, labels, label2id)
        tr_idx = {
            c: x[0:min(min_training_size_per_class_across_cutoffs, self.tr_samples_per_class)]
            for c, x in tr_idx.items()
        }

        # Finally, get the files and labels for the train and validation sets.
        tr_idx = sum(tr_idx.values(), [])
        tr_files = [files[i] for i in tr_idx]
        tr_labels = labels[tr_idx]
        vl_idx = sum(vl_idx.values(), [])
        vl_files = [files[i] for i in vl_idx]
        vl_labels = labels[vl_idx]

        # Save to the cache file.
        self.cache_path(tr_length_cutoff).mkdir(exist_ok=True, parents=True)
        pd.DataFrame(
            {"files": tr_files, "labels": tr_labels.tolist()}
        ).to_csv(self.tr_cache_path(tr_length_cutoff), index=None)
        pd.DataFrame(
            {"files": vl_files, "labels": vl_labels.tolist()}
        ).to_csv(self.vl_cache_path(tr_length_cutoff), index=None)
        with open(self.id2label_cache_path(tr_length_cutoff), "w") as fp:
            json.dump(id2label, fp)
        with open(self.label2id_cache_path(tr_length_cutoff), "w") as fp:
            json.dump(label2id, fp)

        files, labels = get_tr_vl_ts_files_and_labels(files, labels, None, tr_idx, vl_idx, [])
        dist = Counter({c: len(tr_files) / len(self.classes) for c in self.classes})
        return Materials(files, labels, id2label, label2id, dist)


def get_materials_clf_sorel_length_extrapolation(
    tr_length_cutoff: int,
    tr_size: int = 120000,
    vl_size: int = 12000,
    tr_length_cutoffs: list[int] = tuple(list(range(2**17, 2**20 + 1, 2**17))),
) -> Materials:

    classes = ("spyware", "worm", "dropper", "file_infector", "downloader", "adware")
    cache_root = Path("./cache/length_extrapolation_dataset_sorel_original_labels/")
    getter = GetMaterialsClfLengthExtrapolation(classes, tr_size, vl_size, tr_length_cutoffs, cache_root)
    return getter(tr_length_cutoff)


def get_materials_clf_goodware_vs_malware(
    tr_size: int | float,
    vl_size: int | float,
    ts_size: int | float,
    ratio: Optional[float] = None,
    oversample: bool = False,
    undersample: bool = False,
    min_size: int = 0,
) -> Materials:

    def filter_fn(f: Path) -> bool:
        return f.stat().st_size >= min_size and f.suffix == ".exe"

    id2label = {0: "benign", 1: "malware"}
    label2id = {"benign": 0, "malware": 1}

    mal_files = list(filter(filter_fn, DATASET_TO_FILES["binaries"]["bodmas_pe"]()))
    ben_files = list(filter(filter_fn, DATASET_TO_FILES["binaries"]["local_pe"]()))
    cur_ratio = len(ben_files) / (len(ben_files) + len(mal_files))

    if ratio is None or cur_ratio == ratio:
        files = mal_files + ben_files
        labels = np.array([1] * len(mal_files) + [0] * len(ben_files))
        files, labels = shuffle(files, labels)
        idx = tr_vl_ts_split_idx(len(files), tr_size, vl_size, ts_size)
        files, labels = get_tr_vl_ts_files_and_labels(files, labels, idx)
        dist = Counter({"benign": len(ben_files), "malware": len(mal_files)})
        return Materials(files, labels, id2label, label2id, dist)

    n_ben = ratio * len(mal_files) / (1 - ratio)
    assert n_ben.is_integer(), n_ben
    n_ben = int(n_ben)

    n_mal = len(ben_files) * (1 - ratio) / ratio
    assert n_mal.is_integer(), n_mal
    n_mal = int(n_mal)

    if oversample and undersample:
        n_ben = int((n_ben + n_mal) / 2)
        n_mal = int((n_ben + n_mal) / 2)

    if oversample:
        ben_files = list(islice(cycle(ben_files), n_ben))
    if undersample:
        mal_files = mal_files[0:int(n_mal)]

    files = mal_files + ben_files
    labels = np.array([1] * len(mal_files) + [0] * len(ben_files))
    files, labels = shuffle(files, labels)
    if all(isinstance(x, int) for x in (tr_size, vl_size, ts_size)):
        s = tr_size + vl_size + ts_size
        files = files[0:s]
        labels = labels[0:s]

    idx = tr_vl_ts_split_idx(len(files), tr_size, vl_size, ts_size)
    files, labels = get_tr_vl_ts_files_and_labels(files, labels, idx)
    dist = Counter({"benign": n_ben, "malware": n_mal})
    return Materials(files, labels, id2label, label2id, dist)


def test_get_materials_clf_sorel_length_extrapolation():

    print("-" * 100)
    print("STARTING TESTS")
    print("-" * 100)

    TR_LENGTH_CUTOFFS = tuple(list(range(2 ** 17, 2 ** 20 + 1, 2 ** 17)))
    TR_SIZE = 1200
    TS_SIZE = 120

    for cutoff in TR_LENGTH_CUTOFFS:
        print("-" * 100)
        print(f"{cutoff=}")

        materials = get_materials_clf_sorel_length_extrapolation(
            tr_length_cutoff=cutoff,
            tr_size=TR_SIZE,
            vl_size=TS_SIZE,
            tr_length_cutoffs=TR_LENGTH_CUTOFFS,
        )

        lengths = [
            os.path.getsize(f) for f in chain(
                materials.files["tr"],
                materials.files["vl"],
                materials.files["ts"],
            )
        ]
        print(f"{len(lengths)=}")
        print(f"{min(lengths)=}")
        print(f"{max(lengths)=}")
        print(f"{mean(lengths)=}")
        print(f"{median(lengths)=}")
        if max(lengths) > cutoff:
            print("We have a problem, Watson.")


def build_length_extrapolation_cache_files(tr_length_cutoff: int):

    print(tr_length_cutoff)

    tr_length_cutoffs = [
        (2 ** 17),
        (2 ** 18),
        (2 ** 18) + (2 ** 17),
        (2 ** 19),
        (2 ** 19) + (2 ** 17),
        (2 ** 19) + (2 ** 18),
        (2 ** 19) + (2 ** 18) + (2 ** 17),
        (2 ** 20),
    ]

    get_materials_clf_sorel_length_extrapolation(
        tr_length_cutoff=tr_length_cutoff,
        tr_size=120000,
        vl_size=12000,
        tr_length_cutoffs=tr_length_cutoffs,
    )


def test():

    for f in [get_bodmas_file_label_map, get_sorel_virus_total_file_label_map, get_sorel_original_labels_file_label_map]:
        print(f.__name__)
        files_and_labels = f()
        print(f"{len(files_and_labels)=} {next(iter(files_and_labels.items()))=}")

    materials = get_materials_clf_bodmas(0.8, 0.2, 0.0, top_k=10)
    print(materials)

    materials = get_materials_clf_bodmas(25000, 5000, 5000)
    print(materials)

    materials = get_materials_clf_bodmas(25000, 5000, 0, min_freq=20)
    print(materials)

    # materials = get_materials_clf_bodmas_balanced_slice()
    # materials = get_materials_clf_bodmas_with_k_samples_per_class_in_train_set()
    # materials = get_length_extrapolation_dataset_sorel_original_labels(2**17, tr_size=100, ts_size=10)
    # materials = get_length_extrapolation_dataset_sorel_original_labels(2**18, tr_size=100, ts_size=10)


def test_get_materials_clf_bodmas_with_k_samples_per_class_in_train_set():

    # materials = get_materials_clf_bodmas_with_k_samples_per_class_in_train_set(1, None)
    # print(materials)
    # print("-" * 88)

    # materials = get_materials_clf_bodmas_with_k_samples_per_class_in_train_set(1, 1)
    # print(materials)
    # print("-" * 88)

    # materials = get_materials_clf_bodmas_with_k_samples_per_class_in_train_set(1, 2)
    # print(materials)
    # print("-" * 88)

    # materials = get_materials_clf_bodmas_with_k_samples_per_class_in_train_set(3, 2)
    # print(materials)
    # print("-" * 88)

    files = {"tr": [], "vl": []}
    for seed in [0, 1, 2, 3, 42]:
        random.seed(seed)
        np.random.seed(seed)
        materials = get_materials_clf_bodmas_with_k_samples_per_class_in_train_set(3, 2)
        files["tr"].append(materials.files["tr"])
        files["vl"].append(materials.files["vl"])

    print()


if __name__ == "__main__":
    test_get_materials_clf_bodmas_with_k_samples_per_class_in_train_set()
