"""
Core file operations to construct labeled datasets for classification tasks.
"""

from __future__ import annotations
from collections import defaultdict, Counter
from collections.abc import Sequence, Iterable
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from itertools import chain
import math
import os
from pathlib import Path
import pickle
from pprint import pprint, pformat
import random
import sys
import time
from typing import Callable, Literal, Optional
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

from src.enums import LiftLevel, SplitMode, DatasetName
from src.utils import flatten, get_unique_files, rglob, unique_value
from src.data.cfg import (
    BODMAS_LABELS_FILE,
    DATASET_TO_FILES,
    ELF_CLASSIFICATION_DATASETS,
    PACKING_ROOTS,
    SOREL_CLARAVY_CACHE,
    SOREL_AVCLASS_CACHE,
    SOREL_AVCLASS_FAMILY_CACHE,
    TIMESTAMPS_FILES,
    VALID_TIMESTAMP_RANGES,
    DIGESTS_FILES,
    LIFT_LEVEL_EXTENSIONS,
)
from src.data.detect_packing_sorel import universal_packing_map
from src.data.labeling import FilterArgs, Labeler, Label
from src.data.utils import get_sha_timestamp_map, get_sha_digest_map, get_data_from_archives


MIN_SAMPLES_PER_CLASS_PER_SPLIT = 1

################################################################################
# Utilities
################################################################################


SplitNames = Literal["tr", "vl", "ts"]
FilesAndLabels = tuple[list[os.PathLike], Optional[Sequence[int]]]


class ArchivedFile:

    def __init__(self, archive: str | Path, name: str) -> None:
        self.archive = archive
        self.name = name

    def __repr__(self) -> str:
        return f"ArchiveFile({self.archive}, {self.name})"

    def __str__(self) -> str:
        return f"{str(self.archive)} --> {self.name}"

    @staticmethod
    def list_from_archives(archives: list[str | Path]) -> list[ArchivedFile]:
        archived_files = []
        for archive in archives:
            for name, _ in get_data_from_archives(archives=[archive], names=True, contents=False):
                archived_files.append(ArchivedFile(archive, name))
        return archived_files

    @staticmethod
    def is_archive_list_contiguous(archive_list: list[ArchivedFile | tuple]) -> bool:
        encountered = set()
        prev_a = None
        for i in archive_list:
            if isinstance(i, ArchivedFile):
                a = i.archive
            else:
                a = i[0]
            if prev_a is None:
                prev_a = a
                encountered.add(a)
                continue
            if a != prev_a and a in encountered:
                return False
            prev_a = a
            encountered.add(a)
        return True

    @staticmethod
    def make_archive_list_contiguous(archive_list: list[ArchivedFile | tuple], sort: bool = True) -> list[ArchivedFile]:
        # NOTE: Added sorting of the archives to improve reproducibility, however,
        # some experiments were already run under the more random conditions.
        archive_files_map = defaultdict(list)
        for af in archive_list:
            archive_files_map[af.archive].append(af)

        archives = list(archive_files_map.keys())
        random.shuffle(archives)
        archives = sorted(archives) if sort else archives

        new = []
        for archive in archives:
            arr = archive_files_map[archive]
            arr = sorted(arr, key=lambda af: af.name) if sort else arr
            new.extend(arr)

        assert len(new) == len(archive_list)
        return new


@dataclass
class Materials:
    files: dict[SplitNames, list[str | ArchivedFile]]
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

    @property
    def shas_tr(self) -> Counter:
        return self.get_split_shas("tr")

    @property
    def shas_vl(self) -> Counter:
        return self.get_split_shas("vl")

    @property
    def shas_ts(self) -> Counter:
        return self.get_split_shas("ts")

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

    def get_split_shas(self, split: SplitNames) -> list[str]:
        files = self.files[split]
        if len(files) == 0:
            return []

        shas = [""] * len(files)

        def archived_file_to_sha(af: ArchivedFile) -> str:
            return os.path.basename(af.name).split(".")[0]

        def str_file_to_sha(f: str) -> str:
            return os.path.basename(f).split(".")[0]

        to_sha = archived_file_to_sha if isinstance(files[0], ArchivedFile) else str_file_to_sha

        for i, f in enumerate(files):
            shas[i] = to_sha(f)

        return shas

    def __repr__(self):
        if self.dist is not None:
            dists = {s: self.get_split_dist(s) for s in ["tr", "vl", "ts"] if len(self.labels[s]) > 0}
            dist = defaultdict(dict)
            for s, d in dists.items():  # s: split, d: class distribution
                for k, v in d.items():  # k: class label, v: class count
                    dist[k][s] = v
            dist = dict(dist)
        else:
            dist = None
        return (
            f"len(tr)={len(self.files['tr'])}\n"
            f"len(vl)={len(self.files['vl'])}\n"
            f"len(ts)={len(self.files['ts'])}\n"
            f"problem_type={self.problem_type}\n"
            f"num_classes={len(self.id2label) if self.id2label is not None else None}\n"
            f"dist={pformat(dist) if dist is not None else None}"
        )

    def convert_files_to_archived_file(self, file_to_archive_map: dict[str, str]) -> Materials:
        for split in self.files:
            length = len(self.files[split])
            for i in range(length):
                f = self.files[split][i]
                a = file_to_archive_map[f]
                af = ArchivedFile(a, f)
                self.files[split][i] = af
        return self

    def convert_files_suffix(self, suffix: str) -> Materials:
        def convert_archived_file_suffix(af: ArchivedFile) -> ArchivedFile:
            if af.name.endswith(suffix):
                return af
            return ArchivedFile(af.archive, af.name + suffix)

        def convert_string_file_suffix(f: str) -> str:
            if f.endswith(suffix):
                return f
            return f + suffix

        def convert_pathlib_file_suffix(f: Path) -> Path:
            if f.suffix == suffix:
                return f
            return f.with_suffix(suffix)

        if isinstance(self.files["tr"][0], ArchivedFile):
            convert = convert_archived_file_suffix
        elif isinstance(self.files["tr"][0], Path):
            convert = convert_pathlib_file_suffix
        else:
            convert = convert_string_file_suffix

        for split in self.files:
            self.files[split] = [convert(f) for f in self.files[split]]

        return self

    def spacially_bias(self, ratio: float, minority_class: str = "mal", splits: tuple[str] = ("tr", "vl", "ts")) -> Materials:

        for split in splits:
            if len(self.files[split]) == 0:
                continue

            d_i = {str(file): int(label) for file, label in zip(self.files[split], self.labels[split])}
            d_f = spacially_bias(d_i, ratio, self.label2id[minority_class])
            self.files[split]  = list(d_f.keys())
            self.labels[split] = np.array(list(d_f.values()))

        self.dist = self.dist_tr | self.dist_vl | self.dist_ts

        return self


def spacially_bias(
    files_and_labels: dict[str, str | int],
    ratio: float,
    minority_class: str = "mal",
    check: bool = True,
) -> dict[str, str]:
    """
    Manipulate the distribution adding/removing samples.

    Args:
        files_and_labels (dict): a dictionary of samples and their corresponding labels.
        ratio (float): the desired ratio of the minority class relative to the number of samples in the split.
        minority_class (str): the name of the class for which there should be less of.
        check (bool): whether to check the resulting distribution for correctness.

    Returns:
        dict: the new distribution with some samples removed.
    """

    files       = np.array(list(files_and_labels.keys()))
    labels      = np.array(list(files_and_labels.values()))
    dist        = Counter(labels)
    num_samples = len(files_and_labels)

    if num_samples <= 2:
        raise ValueError(f"{num_samples=}")
    if ratio <= 0.0 or ratio >= 1.0:
        raise ValueError(f"{ratio=}")
    if minority_class not in dist:
        raise ValueError(f"{minority_class=}")
    if not isinstance(labels[0], (int, str, np.integer)) or len(dist) != 2:
        raise ValueError(f"Only binary classification permitted. {type(labels[0])=} {dist=}")

    # The class labels for the classes which should respectively contain more and less samples.
    majority_class = unique_value([l for l in dist if l != minority_class])

    # The class that we should actually be removing samples from to reach the target ratio.
    # If we're already at the target ratio, do nothing and just return the current samples.
    if dist[minority_class] / num_samples > ratio:
        removal_class = minority_class
        tgt_count     = int((ratio * dist[majority_class]) / (1 - ratio))
    elif dist[minority_class] / num_samples < ratio:
        removal_class = majority_class
        tgt_count     = int(dist[minority_class] * (1 - ratio) / ratio)
    else:
        return files_and_labels

    # Determine the number of samples to remove.
    num_remove = dist[removal_class] - tgt_count
    if num_remove == 0:
        return files_and_labels
    if num_remove < 0:
        raise RuntimeError(f"{num_remove=}")

    # Determine which samples to remove.
    candidates = np.where(labels == removal_class)[0]
    remove     = np.random.choice(candidates, size=num_remove, replace=False)
    if len(np.unique(remove)) != num_remove:
        raise RuntimeError(f"{len(np.unique(remove))=} != {num_remove}")

    # Remove the samples and return the dataset.
    files  = np.delete(files,  remove).tolist()
    labels = np.delete(labels, remove).tolist()

    if check:
        c = Counter(labels)
        r = round(c[minority_class] / len(labels), 4)
        if not math.isclose(r, ratio, abs_tol=0.025):  # 2.5% tolerance
            raise RuntimeError(f"The resulting ratio, {r} after biasing is not close to the target ratio {ratio}.")
        if not (c[minority_class] == dist[minority_class] or c[majority_class] == dist[majority_class]):
            raise RuntimeError(f"Biasing lost samples from both classes: initial={dict(dist)} final={dict(c)}.")

    return {file: label for file, label in zip(files, labels)}


def is_temporal_classwise(materials: Materials, sha_timestamp_map: dict[str, int], raise_if_not: bool = True) -> bool:
    """Checks whether the materials are split temporally within each class.
    """
    timestamps = {}
    for split, files in materials.files.items():
        t = [sha_timestamp_map[Path(f).stem] for f in files]
        timestamps[split] = np.sort(np.array(t, np.int64))
    for i in materials.id2label:
        idx = {split: np.where(labels == i)[0] for split, labels in materials.labels.items()}
        cuttoffs = [
            timestamps["tr"][idx["tr"]].max(),
            timestamps["vl"][idx["vl"]].min(),
            timestamps["vl"][idx["vl"]].max(),
            timestamps["ts"][idx["ts"]].min(initial=np.iinfo(np.int64).max),
        ]
        result = bool(np.all(np.diff(cuttoffs) > 0))
        if not result:
            if raise_if_not:
                raise RuntimeError(f"Materials are not temporal (class {materials.id2label[i]}): {cuttoffs=}")
            return False

    return True


def is_temporal_absolute(materials: Materials, sha_timestamp_map: dict[str, int], raise_if_not: bool = True) -> bool:
    """Checks whether the materials are split temporally over the entire datasets.
    """
    timestamps = {}
    for split, files in materials.files.items():
        t = [sha_timestamp_map[Path(f).stem] for f in files]
        timestamps[split] = np.sort(np.array(t, dtype=np.int64))
    cuttoffs = [
        timestamps["tr"].max(),
        timestamps["vl"].min(),
        timestamps["vl"].max(),
        timestamps["ts"].min(initial=np.iinfo(np.int64).max),
    ]
    result = bool(np.all(np.diff(cuttoffs) > 0))
    if not result and raise_if_not:
        raise RuntimeError(f"Materials are not absolutely temporal: {cuttoffs=}")
    return result


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


def distribute_elements_to_meet_proportions(
    A: Iterable | int,
    B: Iterable | int,
    C: Iterable | int,
    c: int,
    p1: float,
    p2: float,
    p3: float,
) -> tuple[int, int, int]:

    def get_counts():
        count_A = len(A) if isinstance(A, Iterable) else A
        count_B = len(B) if isinstance(B, Iterable) else B
        count_C = len(C) if isinstance(C, Iterable) else C
        return count_A, count_B, count_C

    count_A, count_B, count_C = get_counts()

    total_elements = count_A + count_B + count_C + c
    total_ratio = p1 + p2 + p3

    # Calculate the target number of elements in each set
    target_A = (p1 / total_ratio) * total_elements
    target_B = (p2 / total_ratio) * total_elements
    target_C = (p3 / total_ratio) * total_elements

    # Current counts of elements in each set
    count_A, count_B, count_C = get_counts()

    for _ in range(c):
        # Calculate current ratios
        current_total = count_A + count_B + count_C
        current_ratio_A = count_A / current_total if current_total != 0 else 0
        current_ratio_B = count_B / current_total if current_total != 0 else 0
        current_ratio_C = count_C / current_total if current_total != 0 else 0

        # Calculate the difference from the target ratios
        diff_A = target_A - (current_ratio_A * total_elements)
        diff_B = target_B - (current_ratio_B * total_elements)
        diff_C = target_C - (current_ratio_C * total_elements)

        # Add the element to the set which is furthest from its target ratio
        if diff_A >= diff_B and diff_A >= diff_C:
            count_A += 1
        elif diff_B >= diff_A and diff_B >= diff_C:
            count_B += 1
        else:
            count_C += 1

    return count_A, count_B, count_C


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
    split_mode: SplitMode = SplitMode.RANDOM,
    timestamps: Optional[Sequence[int]] = None,
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
    assert total == len(labels), f"{total=} should equal {len(labels)=}"

    # Verify that the splits are large enough to contain `samples_per_class` samples for each class.
    for split, size in zip(["tr", "vl", "ts"], [tr_size, vl_size, ts_size]):
        if 0 < size < len(values) * samples_per_class:
            raise ValueError(
                f"The {split=} set with {size=} is too small to contain {samples_per_class} samples"
                f" per class for {len(values)} classes."
            )


    # These stuctures contain the class distribution and indices for each split.
    tr_dist, tr_idx = {v: 0 for v in values}, []
    vl_dist, vl_idx = {v: 0 for v in values}, []
    ts_dist, ts_idx = {v: 0 for v in values}, []


    if split_mode == SplitMode.RANDOM:
        # If not using a temporal split, add a certain number of samples to every split,
        # then add the remaining samples randomly.
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
            else:
                warnings.warn(f"Trouble adding sample to a split {i=} {l=}.")
                if ts_size > 0:
                    ts_dist[l] += 1
                    ts_idx.append(i)
                elif vl_size > 0:
                    vl_dist[l] += 1
                    vl_idx.append(i)
                elif tr_size > 0:
                    tr_dist[l] += 1
                    tr_idx.append(i)


    elif split_mode == SplitMode.TEMPORAL_CLASSWISE:
        # If using the temporal split, add the samples based on their timestamp.
        # The number of samples in the tr, vl, and ts sets should approximately match the
        # tr_prop, vl_prop, and ts_prop values (although in reality this will likely be innaccurate).

        timestamps = np.array(timestamps)
        if not np.all(np.diff(timestamps) >= 0):
            raise ValueError("Timestamps are not sorted. Files, labels and timestamps need to be placed in temporal order.")

        classes_and_counts_org = {cls: cnt for cls, cnt in zip(values, counts)}
        tr_dist_fin, vl_dist_fin, ts_dist_fin = {}, {}, {}
        for cls in values:
            tr_cnt, vl_cnt, ts_cnt = distribute_elements_to_meet_proportions(
                tr_dist[cls],
                vl_dist[cls],
                tr_dist[cls],
                classes_and_counts_org[cls],
                tr_prop,
                vl_prop,
                ts_prop,
            )
            tr_dist_fin[cls] = tr_cnt
            vl_dist_fin[cls] = vl_cnt
            ts_dist_fin[cls] = ts_cnt

            assert tr_dist_fin[cls] >= tr_dist[cls], f"{tr_dist_fin[cls]=} should be greater or equal to {tr_dist[cls]} for {cls=}."
            assert vl_dist_fin[cls] >= vl_dist[cls], f"{vl_dist_fin[cls]=} should be greater or equal to {vl_dist[cls]} for {cls=}."
            assert ts_dist_fin[cls] >= ts_dist[cls], f"{ts_dist_fin[cls]=} should be greater or equal to {ts_dist[cls]} for {cls=}."

        # Since the labels are sorted according to timestamp, we simply add them to the appropriate split.
        for i, l in enumerate(labels):
            if tr_dist[l] < tr_dist_fin[l]:
                tr_dist[l] += 1
                tr_idx.append(i)
            elif vl_dist[l] < vl_dist_fin[l]:
                vl_dist[l] += 1
                vl_idx.append(i)
            elif ts_dist[l] < ts_dist_fin[l]:
                ts_dist[l] += 1
                ts_idx.append(i)
            else:
                warnings.warn(f"Trouble adding sample to a split {i=} {l=}.")
                if ts_size > 0:
                    ts_dist[l] += 1
                    ts_idx.append(i)
                elif vl_size > 0:
                    vl_dist[l] += 1
                    vl_idx.append(i)
                elif tr_size > 0:
                    tr_dist[l] += 1
                    tr_idx.append(i)

        assert len(tr_dist) == len(tr_dist_fin), f"{len(tr_dist)=} should equal {len(tr_dist_fin)=}"
        assert len(vl_dist) == len(vl_dist_fin), f"{len(vl_dist)=} should equal {len(vl_dist_fin)=}"
        assert len(ts_dist) == len(ts_dist_fin), f"{len(ts_dist)=} should equal {len(ts_dist_fin)=}"


    elif split_mode == SplitMode.TEMPORAL_ABSOLUTE:
        timestamps = np.array(timestamps)

        tr_idx = list(range(0, tr_size))
        vl_idx = list(range(tr_size, tr_size + vl_size))
        ts_idx = list(range(tr_size + vl_size, tr_size + vl_size + ts_size))

        # Ensure that the split is strict by grouping elements identical timestamps into the same set.
        if ts_size != 0 or vl_size == 0:
            raise NotImplementedError(f"Not implemented for {tr_size=} {vl_size=} {ts_size=}")

        if vl_size > 0:
            k = vl_idx[0]                     # first element idx in vl set
            t = timestamps[k]                 # earliest timestamp in vl set
            if timestamps[tr_idx[-1]] == t:   # the last element in tr set has the same timestamp
                i = np.where(timestamps == t)[0]  # indicies with the same timestamp
                n_lss = len(np.where(i < k)[0])   # number of indices to the left
                n_grt = len(np.where(i > k)[0])   # number of indices to the right
                if n_lss < n_grt:  # more samples with this timestamp in the vl set
                    warnings.warn(f"Moving {len(i)} samples to the vl set because of overlapping timestamps.")
                    tr_idx = tr_idx[:tr_idx.index(i[0])]
                    vl_idx = [j for j in i if j < vl_idx[0]] + vl_idx
                else:              # more samples with this timestamp in the tr set
                    warnings.warn(f"Moving {len(i)} samples to the tr set because of overlapping timestamps.")
                    tr_idx = tr_idx + [j for j in i.tolist() if j > tr_idx[-1]]
                    vl_idx = vl_idx[vl_idx.index(i[-1]) + 1:]

        tr_dist = Counter(labels[tr_idx])
        vl_dist = Counter(labels[vl_idx])
        ts_dist = Counter(labels[ts_idx])
        if (vl_size > 0 and set(tr_dist) != set(vl_dist)) or (ts_size > 0 and set(tr_dist) != set(ts_dist)):
            raise RuntimeError(
                f"Creating a temporal split with {tr_size=} {vl_size=} {ts_size=} "
                "resulted in one of the splits losing all representatives of one or more classes "
                f"(tr_dist={pformat(tr_dist)} vl_dist={pformat(vl_dist)} ts_dist={pformat(ts_dist)})."
            )

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
    split_mode: SplitMode = SplitMode.RANDOM,
    files_and_timestamps: Optional[dict[os.PathLike, int]] = None,
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
    if must_exist and (min_size > 0 or max_size < sys.maxsize):
        files_and_labels = {
            f: l for f, l in files_and_labels.items() if
            (not os.path.exists(f) or (min_size <= os.path.getsize(f) <= max_size))
        }

    if split_mode != SplitMode.RANDOM:
        files_and_labels = {
            f: l for f, l in files_and_labels.items() if files_and_timestamps.get(f) is not None
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


def get_bodmas_sha_label_map() -> dict[str, str]:
    return pd.read_csv(BODMAS_LABELS_FILE).set_index("sha")["family"].to_dict()


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


def _get_sorel_sha_label_map() -> dict[str, Label]:

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
    return {sha : label for sha, label in labeler.data.items() if label is not None and label.is_labeled}


def get_sorel_sha_label_map(name: str) -> dict[str, tuple[str]]:

    shas_and_labels = _get_sorel_sha_label_map()
    shas_and_labels = {f: getattr(l, name) for f, l in shas_and_labels.items()}
    shas_and_labels = {f: l for f, l in shas_and_labels.items() if l is not None}
    return shas_and_labels


def get_sorel_file_label_map(name: str) -> dict[os.PathLike, Label]:

    shas_and_labels = get_sorel_sha_label_map(name)
    files = DATASET_TO_FILES["binaries"]["sorel_pe"]()
    files_and_labels = {str(f): shas_and_labels.get(f.stem) for f in files}
    return {f: l for f, l in files_and_labels.items() if l is not None}


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
    files = filter_packed_files(files, packing_protocol, root=packing_root)
    remove = set(remove)
    files = [f for f in files if f not in remove and os.path.basename(f).split(".")[0] not in remove]

    if tr_size == -1 or (isinstance(tr_size, int) and tr_size >= (len(files) - vl_size - ts_size)):
        tr_size = len(files) - vl_size - ts_size

    files.sort()  # Sort after filtering, as the list will be smaller
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
    split_mode: SplitMode = SplitMode.RANDOM,
    timestamps_file: Optional[Path] = None,
) -> Materials:

    if min_freq is None:
        if vl_size == 0 or ts_size == 0:
            min_freq = MIN_SAMPLES_PER_CLASS_PER_SPLIT * 2
        else:
            min_freq = MIN_SAMPLES_PER_CLASS_PER_SPLIT * 3

    files_to_keep = filter_packed_files(list(files_and_labels.keys()), packing_protocol, root=packing_root)
    files_and_labels = {f: files_and_labels[f] for f in files_to_keep}

    if split_mode != SplitMode.RANDOM:
        shas_and_timestamps = get_sha_timestamp_map(timestamps_file)
        files_and_timestamps = {f: shas_and_timestamps.get(os.path.basename(f).split(".")[0]) for f in files_and_labels}
    else:
        files_and_timestamps = None

    # Filter out the files that are not in the top_k most frequent labels
    files_and_labels = filter_file_label_map(
        files_and_labels,
        top_k=top_k,
        min_freq=min_freq,
        max_imbalance_ratio=max_imbalance_ratio,
        min_size=min_size,
        must_exist=must_exist,
        split_mode=split_mode,
        files_and_timestamps=files_and_timestamps,
    )
    # Remove the timestamps for files that have been filtered out.
    if split_mode != SplitMode.RANDOM:
        files_and_timestamps = {
            f: files_and_timestamps[f] for f in files_and_labels if f in files_and_timestamps
        }

    # Final collection of data items
    dist: Counter[str, int] = Counter(files_and_labels.values())
    label2id = {l: i for i, l in enumerate(dist.keys())}
    id2label = {i: l for l, i in label2id.items()}

    files = list(files_and_labels.keys())
    labels = np.array([label2id[files_and_labels[f]] for f in files])

    if split_mode != SplitMode.RANDOM:
        # For the temporal split, we reorder the collections in a temporal fashion,
        # ie, the files and labels are ordered according to their timestamps.
        timestamps = np.array([files_and_timestamps[f] for f in files])
        sort_idx = np.argsort(timestamps)
        files = [files[i] for i in sort_idx]
        labels = labels[sort_idx]
        timestamps = timestamps[sort_idx]
    else:
        timestamps = None
        files, labels = shuffle(files, labels)

    idx = tr_vl_ts_split_idx_guarentee(
        labels, tr_size, vl_size, ts_size,
        MIN_SAMPLES_PER_CLASS_PER_SPLIT, split_mode, timestamps,
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
    split_mode: SplitMode = SplitMode.RANDOM,
    timestamps_file: Optional[Path] = None,
    **kwds,
) -> Materials:
    if invalid := set(kwds) - {"min_freq", "max_imbalance_ratio"}:
        raise TypeError(f"Function got some unexpected keyword argument(s): {invalid}")

    # First, remove the files that we do not want to use.
    files_to_keep = filter_packed_files(list(files_and_labels.keys()), packing_protocol, root=packing_root)
    files_and_labels = {f: files_and_labels[f] for f in files_to_keep}

    if split_mode != SplitMode.RANDOM:
        shas_and_timestamps = get_sha_timestamp_map(timestamps_file)
        files_and_timestamps = {f: shas_and_timestamps.get(os.path.basename(f).split(".")[0]) for f in files_and_labels}
    else:
        files_and_timestamps = None

    files_and_labels = filter_file_label_map(
        files_and_labels,
        top_k=top_k,
        min_freq=tr_samples_per_class + vl_min_samples_per_class,
        min_size=min_size,
        must_exist=must_exist,
        split_mode=split_mode,
        files_and_timestamps=files_and_timestamps,
    )
    # Remove the timestamps for files that have been filtered out.
    if split_mode != SplitMode.RANDOM:
        files_and_timestamps = {
            f: files_and_timestamps[f] for f in files_and_labels if f in files_and_timestamps
        }

    # Get a distribution of the labels and their mappings to/from label IDs.
    dist: Counter[str, int] = Counter(files_and_labels.values())
    label2id: dict[str, int] = {l: i for i, l in enumerate(dist.keys())}
    id2label: dict[int, str] = {i: l for l, i in label2id.items()}

    # Get all of the files and their labels, and shuffle them.
    files = list(files_and_labels.keys())
    labels = np.array([label2id[files_and_labels[f]] for f in files])

    # Order the files, labels, and timestamps according to the timestamps.
    if split_mode != SplitMode.RANDOM:
        timestamps = np.array([files_and_timestamps[f] for f in files])
        sort_idx = np.argsort(timestamps)
        files = [files[i] for i in sort_idx]
        labels = labels[sort_idx]
        timestamps = timestamps[sort_idx]
    else:
        timestamps = None
        files, labels = shuffle(files, labels)

    # Get indices for the train, validation, and test (empty) sets.
    tr_idx = select_k_for_each_class(labels, k=tr_samples_per_class)
    vl_idx_and_label = [(i, labels[i]) for i in range(len(files)) if i not in tr_idx]
    # Reverse the iteration so that the last samples are used for the validation set,
    # which is only meaningful if the collections were sorted for temporal consistency.
    vl_counts = Counter()
    vl_idx = []
    for i, l in reversed(vl_idx_and_label):
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

    if max_imbalance_ratio is not None:
        warnings.warn(f"Using an inaccurate heuristic algorithm to set a maximum imbalance ratio of {max_imbalance_ratio}.")

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
    # pylint: disable=unreachable

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
    files = list(map(lambda p: p.as_posix(), DATASET_TO_FILES["binaries"]["sorel_pe"]()))
    remove = []
    if remove_clf_files:
        files_and_labels = _get_sorel_sha_label_map() | get_bodmas_file_label_map()
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

    kwds["packing_root"] = PACKING_ROOTS["bodmas_pe"]
    kwds["timestamps_file"] = TIMESTAMPS_FILES[DatasetName.BODMAS]

    files_and_labels = get_bodmas_file_label_map()

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
    if kwds.get("temporal", False) and name not in ("fam", "file"):
        raise NotImplementedError()

    kwds["packing_root"] = PACKING_ROOTS["sorel_pe"]
    kwds["timestamps_file"] = TIMESTAMPS_FILES[DatasetName.SOREL]

    files_and_labels = get_sorel_file_label_map(name)

    if name in ("fam", "file"):
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
    files = get_unique_files(list(map(str, files)))
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


################################################################################
# Load Materials Endpoints For ESP
################################################################################


# FIXME: added a flag `purge_empty_samples` which should be removed at some point.
# Instead, the empty samples should be purged from the data caches directly.
# This flag has not been added to the caches either, so this will cause confusion.

DISABLE_ESP_CACHE = os.environ.get("DISABLE_ESP_CACHE") == "1"


def esp_cache_file(get_materials: Callable, lift_level: LiftLevel, **kwds) -> Path:
    identifier = "--".join([f"{k}={kwds[k]}" for k in sorted(kwds)])
    b = identifier.encode("utf-8")
    h = hashlib.sha256(b).hexdigest()
    parent = Path("./cache") / "materials"
    parent.mkdir(parents=True, exist_ok=True)
    return parent / f"{get_materials.__name__}--{lift_level.value}--{h}.pkl"


def _get_materials_esp_lm(
    lift_level: LiftLevel,
    tr_size: Optional[float | int] = None,
    vl_size: float | int = 4096,
    ts_size: float | int = 0,
    lift_level_ddp: LiftLevel = LiftLevel.DEC,
    purge_empty_samples: bool = False,
    verbose: bool = True,
) -> Materials:
    # pylint: disable=multiple-statements
    # vl_size = int(os.environ.get("LMLM_GET_MATERIALS_ESP_LM_VL_SIZE", vl_size))  # FIXME: remove

    lift_level     = LiftLevel(lift_level)
    lift_level_ddp = LiftLevel(lift_level_ddp)

    if lift_level == LiftLevel.ALL or lift_level_ddp == LiftLevel.ALL:
        raise NotImplementedError(f"Cannot use LiftLevel.ALL directly.")

    cache_file = esp_cache_file(
        _get_materials_esp_lm,
        lift_level,
        tr_size=tr_size,
        vl_size=vl_size,
        ts_size=ts_size,
        lift_level_ddp=lift_level_ddp.value,
    )

    if not DISABLE_ESP_CACHE and cache_file.exists():
        print(f"Loading materials from cache: {cache_file}.")
        with open(cache_file, "rb") as fp:
            return pickle.load(fp)

    print("\tGetting SHAs that could be used for finetuning.")
    timestamps_files = (TIMESTAMPS_FILES[DatasetName.ASSEMBLAGE], TIMESTAMPS_FILES[DatasetName.WINDOWS])
    shas_for_finetuning = set(chain(
        _get_sorel_sha_label_map().keys(),
        get_sha_timestamp_map(timestamps_files),
    ))

    # Empty set if we aren't doing the skip empty samples.
    shas_that_are_empty = set()
    if purge_empty_samples:
        shas_that_are_empty = set(Path("./cache/empty_shas.txt").read_text().split("\n"))

    shas_that_are_valid: Optional[set[str]] = None
    if lift_level in (LiftLevel.ALL, LiftLevel.NOP):
        shas_that_are_valid = set(Path("./cache/processedShas.txt").read_text().split("\n"))

    sha_digest_map = {}
    for dnm in DatasetName:
        _sha_digest_map = get_sha_digest_map(DIGESTS_FILES[dnm][lift_level_ddp])
        sha_digest_map.update(_sha_digest_map)

    archives = map(Path, rglob("./data", "*.zip", True))
    archives = sorted(a for a in archives if a.parent.name == lift_level.value)

    skipped_cause_empty                 = 0
    skipped_cause_invalid               = 0
    skipped_cause_finetuning            = 0
    skipped_cause_duplicates            = 0
    skipped_cause_duplicates_finetuning = 0
    archived_files: list[ArchivedFile] = []
    finetuning_digests: set[str] = set(sha_digest_map[s] for s in shas_for_finetuning if s in sha_digest_map)
    present_digests: set[str] = set()
    for archive in tqdm(archives, leave=False, desc="Filtering..."):
        for name, _ in get_data_from_archives(archives=[archive], names=True, contents=False):
            s = name.split(".")[0]
            if s in shas_that_are_empty:
                skipped_cause_empty += 1
                continue
            if shas_that_are_valid is not None and s not in shas_that_are_valid:
                skipped_cause_invalid += 1
                continue
            if s in shas_for_finetuning:
                skipped_cause_finetuning += 1
                continue
            d = sha_digest_map[s]
            if d in finetuning_digests:
                skipped_cause_duplicates_finetuning += 1
                continue
            if d in present_digests:
                skipped_cause_duplicates += 1
                continue
            archived_files.append(ArchivedFile(archive, name))
    print(f"\tAcquired {len(archived_files)} for pretraining.")
    if verbose: print(f"\tSkipped {skipped_cause_empty=} due to empty file.")
    if verbose: print(f"\tSkipped {skipped_cause_invalid=} due to invalid file (not present in RAW, DIS, and DEC collections).")
    if verbose: print(f"\tSkipped {skipped_cause_finetuning=} due to finetuning.")
    if verbose: print(f"\tSkipped {skipped_cause_duplicates_finetuning=} due to duplicates with finetuning.")
    if verbose: print(f"\tSkipped {skipped_cause_duplicates=} due to duplicates.")

    archived_files.sort(key=lambda af: af.name)
    tr_size = len(archived_files) - vl_size if tr_size is None else tr_size
    tr_vl_ts_files = tr_vl_ts_split(archived_files, tr_size, vl_size, ts_size)
    materials = Materials(files=tr_vl_ts_files)
    materials = materials.convert_files_suffix(LIFT_LEVEL_EXTENSIONS[lift_level])

    if not DISABLE_ESP_CACHE:
        print(f"Saving materials to cache: {cache_file}.")
        with open(cache_file, "wb") as fp:
            pickle.dump(materials, fp)

    return materials


def get_materials_esp_clm(
    lift_level: LiftLevel,
    tr_size: Optional[float | int] = None,
    vl_size: float | int = 4096,
    ts_size: float | int = 0,
    lift_level_ddp: LiftLevel = LiftLevel.DEC,
    purge_empty_samples: bool = False,
    verbose: bool = True,
) -> Materials:
    # vl_size = int(os.environ.get("LMLM_GET_MATERIALS_ESP_CLM_VL_SIZE", vl_size))  # FIXME: remove
    return _get_materials_esp_lm(
        lift_level,
        tr_size,
        vl_size,
        ts_size,
        lift_level_ddp,
        purge_empty_samples,
        verbose,
    )


def get_materials_esp_mlm(
    lift_level: LiftLevel,
    tr_size: Optional[float | int] = None,
    vl_size: float | int = 4096,
    ts_size: float | int = 0,
    lift_level_ddp: LiftLevel = LiftLevel.DEC,
    purge_empty_samples: bool = False,
    verbose: bool = True,
) -> Materials:
    # vl_size = int(os.environ.get("LMLM_GET_MATERIALS_ESP_MLM_VL_SIZE", vl_size))  # FIXME: remove
    return _get_materials_esp_lm(
        lift_level,
        tr_size,
        vl_size,
        ts_size,
        lift_level_ddp,
        purge_empty_samples,
        verbose,
    )


def get_materials_esp_det(
    lift_level: LiftLevel,
    tr_size: float = 0.80,
    vl_size: float = 0.20,
    ts_size: float = 0.00,
    ratio_pre_split: Optional[float] = None,
    ratio_pos_split: Optional[float] = 0.50,
    lift_level_ddp: Optional[LiftLevel] = LiftLevel.DEC,
    purge_empty_samples: bool = True,
    timestamp_early: int = int(datetime(2010, 1, 1, 0, 0, 0, 0, timezone.utc).timestamp()),
    timestamp_late: int = int(datetime(2020, 1, 1, 0, 0, 0, 0, timezone.utc).timestamp()),
    verbose: bool = True,
) -> Materials:
    """

    Args:
     timestamp_early: int
      Beginning of time - int(datetime(1970,  1, 1, 0, 0, 0, 0, timezone.utc).timestamp())
      BODMAS end of collection - int(datetime(2020, 10, 1, 0, 0, 0, 0, timezone.utc).timestamp())
    """
    # pylint: disable=multiple-statements

    lift_level     = LiftLevel(lift_level)
    lift_level_ddp = LiftLevel(lift_level_ddp) if lift_level_ddp is not None else None

    if lift_level == LiftLevel.ALL or lift_level_ddp == LiftLevel.ALL:
        raise NotImplementedError(f"Cannot use LiftLevel.ALL directly.")

    cache_file = esp_cache_file(
        get_materials_esp_det,
        lift_level,
        tr_size=tr_size,
        vl_size=vl_size,
        ts_size=ts_size,
        ratio_pre_split=ratio_pre_split,
        ratio_pos_split=ratio_pos_split,
        lift_level_ddp=lift_level_ddp.value if lift_level_ddp is not None else None,
        timestamp_early=timestamp_early,
        timestamp_late=timestamp_late,
    )

    if not DISABLE_ESP_CACHE and cache_file.exists():
        print(f"Loading materials from cache: {cache_file}.")
        with open(cache_file, "rb") as fp:
            return pickle.load(fp)

    # Due to the way the data is distributed temporally, spacially biasing the data
    # before the train test split is formed can result in individual splits with
    # different ratios of malware vs goodware. Spacially biasing the splits after
    # splitting results in the number of samples in each split being difficult to
    # set, but this is probably better than having different ratios. The tr/vl/ts
    # sizes need to be tuned to see whats its going to look like after the fact.

    # Get data for each dataset.
    timestamps_files   = {dnm: TIMESTAMPS_FILES[dnm] for dnm in DatasetName}
    sha_timestamp_maps = {dnm: get_sha_timestamp_map(timestamps_files[dnm]) for dnm in DatasetName}
    directories        = {dnm: Path(f"./data/{dnm.value}/{lift_level.value}") for dnm in DatasetName}
    archives           = {dnm: sorted(map(Path, rglob(directories[dnm], "*.zip", True))) for dnm in DatasetName}

    # Remove files with no timestamp, an invalid timestamp, or a timestamp outside the range.
    print("\tRemoving files that do not meet timestamp requirements.")
    files = {}
    for dnm in DatasetName:
        lower = max(VALID_TIMESTAMP_RANGES[dnm][0], timestamp_early)
        upper = min(VALID_TIMESTAMP_RANGES[dnm][1], timestamp_late)
        fs = [f for f, _ in tqdm(get_data_from_archives(archives[dnm], names=True, contents=False), leave=False)]
        if verbose: print(f"\t\t{dnm.value}: {len(fs)=} -->", end=" ")
        fs = [f for f in fs if sha_timestamp_maps[dnm].get(Path(f).stem) is not None]
        if verbose: print(f"{len(fs)=} -->", end=" ")
        fs = [f for f in fs if lower <= sha_timestamp_maps[dnm].get(Path(f).stem) <= upper]
        if verbose: print(f"{len(fs)=}")
        files[dnm] = fs

    # Remove files that turned out to be empty.
    if purge_empty_samples:
        print("\tRemoving files that are empty.")
        shas_that_are_empty = set(Path("./cache/empty_shas.txt").read_text().split("\n"))
        for dnm in DatasetName:
            fs = files[dnm]
            if verbose: print(f"\t\t{len(fs)=} -->", end=" ")
            fs = [f for f in fs if f.split(".")[0] not in shas_that_are_empty]
            if verbose: print(f"{len(fs)=}")
            files[dnm] = fs

    if lift_level == LiftLevel.NOP:
        print("Removing files that aren't present in the RAW, DIS, and DEC collections.")
        shas_that_are_valid = set(Path("./cache/processedShas.txt").read_text().split("\n"))
        for dnm in DatasetName:
            fs = files[dnm]
            if verbose: print(f"\t\t{len(fs)=} -->", end=" ")
            fs = [f for f in fs if f.split(".")[0] in shas_that_are_valid]
            if verbose: print(f"{len(fs)=}")
            files[dnm] = fs

    # Convert everything to an array.
    for dnm in DatasetName:
        files[dnm] = np.array(files[dnm])

    # Remove files that are identical.
    if lift_level_ddp is not None:
        digests_files   = {dnm: DIGESTS_FILES[dnm][lift_level_ddp] for dnm in DatasetName}
        sha_digest_maps = {dnm: get_sha_digest_map(digests_files[dnm]) for dnm in DatasetName}

        print("\tRemoving files that are duplicates.")
        present: dict[str, set[str]] = {"ben": set(), "mal": set()}
        noisy: set[str] = set()
        for dnm in DatasetName:
            if dnm in (DatasetName.ASSEMBLAGE, DatasetName.WINDOWS):
                k = "ben"
                j = "mal"
            elif dnm in (DatasetName.BODMAS, DatasetName.SOREL):
                k = "mal"
                j = "ben"
            else:
                raise ValueError(f"Invalid dataset name: {dnm=}")

            # Sort the files by timestamp, therefore, when we remove duplicates, we keep the earliest.
            fs = files[dnm]
            ts = np.array([sha_timestamp_maps[dnm][Path(f).stem] for f in fs], dtype=np.int64)
            idx = np.argsort(ts)
            fs = fs[idx]
            rm = set()
            if verbose: print(f"\t\t{dnm.value}: {len(fs)=} -->", end=" ")

            for i, f in enumerate(fs):
                s = Path(f).stem
                d = sha_digest_maps[dnm][s]

                if d in present[j]:
                    noisy.add(d)

                if d in present[k]:
                    rm.add(i)
                present[k].add(d)

            fs = np.delete(fs, list(rm))
            files[dnm] = fs
            if verbose: print(f"{len(fs)=}")

        if noisy:
            if verbose: print(f"\tDetected {len(noisy)} non-unique samples leaking across different classes.")
            for dnm in DatasetName:
                fs = files[dnm]
                rm = set()
                for i, f in enumerate(fs):
                    s = Path(f).stem
                    d = sha_digest_maps[dnm][s]
                    if d in noisy:
                        rm.add(i)
                fs = np.delete(fs, list(rm))
                files[dnm] = fs
                print(f"\t\t{dnm.value}: {len(rm)=}.")

    # Create file-label map for malware detection. Remove spacial bias.
    print("\tGetting file label map.")
    files_and_labels = {}
    for dnm in DatasetName:
        if dnm in (DatasetName.ASSEMBLAGE, DatasetName.WINDOWS):
            k = "ben"
        elif dnm in (DatasetName.BODMAS, DatasetName.SOREL):
            k = "mal"
        else:
            raise ValueError(f"Invalid dataset name: {dnm=}")
        for f in files[dnm]:
            files_and_labels[f] = k

    if ratio_pre_split is not None and ratio_pre_split > 0:
        print(f"\tSpacially biasing the entire corpus to {ratio_pre_split}")
        if verbose: print(f"\t\tdist = {dict(Counter(files_and_labels.values()))} --> ", end=" ")
        files_and_labels = spacially_bias(files_and_labels, ratio_pre_split, minority_class="mal")
        if verbose: print(f"{dict(Counter(files_and_labels.values()))}")

    # Get the dataset materials.
    print("\tAcquiring raw materials.")
    materials = _get_materials_clf(
        files_and_labels=files_and_labels,
        tr_size=tr_size,
        vl_size=vl_size,
        ts_size=ts_size,
        must_exist=False,
        split_mode=SplitMode.TEMPORAL_ABSOLUTE,
        timestamps_file=list(timestamps_files.values()),
    )

    if ratio_pos_split is not None and ratio_pos_split > 0:
        print(f"\tSpacially biasing each split to {ratio_pos_split}")
        _value_1 = f"{materials.dist_tr=}"
        _value_2 = f"{materials.dist_vl=}"
        _value_3 = f"{materials.dist_ts=}"
        materials = materials.spacially_bias(ratio_pos_split, minority_class="mal")
        if verbose: print(f"\t\t{_value_1} --> {materials.dist_tr=}")
        if verbose: print(f"\t\t{_value_2} --> {materials.dist_vl=}")
        if verbose: print(f"\t\t{_value_3} --> {materials.dist_ts=}")

    # Verify that the temporal split was formed correctly.
    print("\tVerifying temporal bias.")
    sha_timestamp_map = {k: v for d in sha_timestamp_maps.values() for k, v in d.items()}
    is_temporal_absolute(materials, sha_timestamp_map, raise_if_not=True)

    # Replace the logical files with real ArchivedFile objects that can be accessed.
    if verbose: print("\tConverting to ArchivedFile.")
    file_to_archive_map = {}
    for dnm in DatasetName:
        prefix_archive_map = {a.stem: a for a in archives[dnm]}
        name_length = len(next(iter(prefix_archive_map.keys())))
        for f in tqdm(files[dnm], leave=False):
            if (prefix := f.split(".")[0][0:name_length]) in prefix_archive_map:
                file_to_archive_map[f] = prefix_archive_map[prefix]
            else:
                raise RuntimeError(f"Could not find the archive containing {f=} where {archives[dnm]=}")
    materials = materials.convert_files_to_archived_file(file_to_archive_map)
    materials = materials.convert_files_suffix(LIFT_LEVEL_EXTENSIONS[lift_level])

    print("\tComputing final ratios.")
    def fmt(t: float, b: float) -> str:
        if b == 0:
            return "nan"
        return f"{round(100 * t / b)}%"

    n_tr = len(materials.files["tr"])
    n_vl = len(materials.files["vl"])
    n_ts = len(materials.files["ts"])
    n_to = n_tr + n_vl + n_ts
    n_tr_mal = np.sum(materials.labels["tr"] == materials.label2id["mal"])
    n_tr_ben = np.sum(materials.labels["tr"] == materials.label2id["ben"])
    n_vl_mal = np.sum(materials.labels["vl"] == materials.label2id["mal"])
    n_vl_ben = np.sum(materials.labels["vl"] == materials.label2id["ben"])
    n_ts_mal = np.sum(materials.labels["ts"] == materials.label2id["mal"])
    n_ts_ben = np.sum(materials.labels["ts"] == materials.label2id["ben"])

    if verbose: print(f"\t\tto: tr-vl-ts = {fmt(n_tr, n_to)} {fmt(n_vl, n_to)} {fmt(n_ts, n_to)}")
    if verbose: print(f"\t\ttr: mal-ben  = {fmt(n_tr_mal, n_tr)} {fmt(n_tr_ben, n_tr)}")
    if verbose: print(f"\t\tvl: mal-ben  = {fmt(n_vl_mal, n_vl)} {fmt(n_vl_ben, n_vl)}")
    if verbose: print(f"\t\tts: mal-ben  = {fmt(n_ts_mal, n_ts)} {fmt(n_ts_ben, n_ts)}")

    if not DISABLE_ESP_CACHE:
        print(f"Saving materials to cache: {cache_file}.")
        with open(cache_file, "wb") as fp:
            pickle.dump(materials, fp)

    return materials


def _get_materials_esp_clf(
    sha_label_map: dict[str, str],
    problem_type: Literal["single_label_classification", "multi_label_classification"],
    lift_level: LiftLevel,
    tr_size: float = 0.80,
    vl_size: float = 0.20,
    ts_size: float = 0.00,
    lift_level_ddp: Optional[LiftLevel] = LiftLevel.DEC,
    purge_empty_samples: bool = True,
    verbose: bool = True,
    **kwds,
) -> Materials:
    # pylint: disable=multiple-statements

    lift_level     = LiftLevel(lift_level)
    lift_level_ddp = LiftLevel(lift_level_ddp) if lift_level_ddp is not None else None

    if lift_level == LiftLevel.ALL or lift_level_ddp == LiftLevel.ALL:
        raise NotImplementedError(f"Cannot use LiftLevel.ALL directly.")

    cache_file = esp_cache_file(
        _get_materials_esp_clf,
        lift_level,
        problem_type=problem_type,
        tr_size=tr_size,
        vl_size=vl_size,
        ts_size=ts_size,
        lift_level_ddp=lift_level_ddp.value if lift_level_ddp is not None else None,
        **kwds,
    )

    if not DISABLE_ESP_CACHE and cache_file.exists():
        print(f"Loading materials from cache: {cache_file}.")
        with open(cache_file, "rb") as fp:
            return pickle.load(fp)

    if purge_empty_samples:
        print(f"\tRemoving files that are empty ({len(sha_label_map)} --> ", end="")
        shas_that_are_empty = set(Path("./cache/empty_shas.txt").read_text().split("\n"))
        sha_label_map = {s: l for s, l in sha_label_map.items() if s.split(".")[0] not in shas_that_are_empty}
        print(f"{len(sha_label_map)})")

    if lift_level == LiftLevel.NOP:
        print(f"\tRemoving files that aren't present in the RAW, DIS, and DEC collections ({len(sha_label_map)} --> ", end="")
        shas_that_are_valid = set(Path("./cache/processedShas.txt").read_text().split("\n"))
        sha_label_map = {s: l for s, l in sha_label_map.items() if s.split(".")[0] in shas_that_are_valid}
        print(f"{len(sha_label_map)})")

    dnm = DatasetName.SOREL
    directory = Path(f"./data/{dnm.value}/{lift_level.value}")
    archives  = sorted(map(Path, rglob(directory, "*.zip", True)))

    print("\tReading file names.")
    files = (f for f, _ in get_data_from_archives(archives, names=True, contents=False))
    files = (f for f in files if sha_label_map.get(Path(f).stem) is not None)
    files = np.array(list(tqdm(files, leave=False)))
    sha_label_map = {Path(f).stem: sha_label_map[Path(f).stem] for f in files}
    print(f"{len(files)=}")
    print(f"{len(sha_label_map)=}")

    if lift_level_ddp is not None:
        # Generate a map containing the most popular label for each digest.
        # When selecting one sample for each digest, we'll ensure it has this label.
        print("\tComputing the most popular label for each digest.")
        digests_file     = DIGESTS_FILES[dnm][lift_level_ddp]
        sha_digest_map   = get_sha_digest_map(digests_file)
        digest_label_map = defaultdict(Counter)
        for s, l in sha_label_map.items():
            d = sha_digest_map[s]
            digest_label_map[d].update([l])
        digest_label_map = {d: c.most_common(1)[0][0] for d, c in digest_label_map.items()}

        # Remove samples that hash to the same digest.
        # Keep a random sample with the most popular label.
        print("\tRemoving files that are duplicates.")
        if verbose: print(f"\t\t{len(files)=} -->", end=" ")
        rm = set()
        digests_added = set()
        for i, f in enumerate(files):
            s = Path(f).stem
            d = sha_digest_map[s]
            l = sha_label_map[s]
            if d in digests_added:
                rm.add(i)
                continue
            if l != digest_label_map[d]:
                rm.add(i)
                continue
            digests_added.add(d)
        files = np.delete(files, list(rm))
        if verbose: print(f"{len(files)=}")

        # Re-create the sha_label_map
        print("\tRe-creating the sha_label_map.")
        sha_label_map = {}
        for f in files:
            s = Path(f).stem
            d = sha_digest_map[s]
            l = digest_label_map[d]
            sha_label_map[s] = l

    # Get the dataset materials.
    print("\tAcquiring raw materials.")
    if problem_type == "single_label_classification":
        materials = _get_materials_clf(
            sha_label_map,
            tr_size,
            vl_size,
            ts_size,
            must_exist=False,
            **kwds,
        )
    elif problem_type == "multi_label_classification":
        materials = _get_materials_clf_multilabel(
            sha_label_map,
            tr_size,
            vl_size,
            ts_size,
            must_exist=False,
            **kwds,
        )

    # Replace the logical files with real ArchivedFile objects that can be accessed.
    print("\tConverting to ArchivedFile.")
    file_to_archive_map = {}
    if all(len(a.stem) == 3 for a in archives):
        # Because every sample here is from SOREL and is from an archive whose name
        # consists of three hex characters, we can do this without a double for-loop.
        prefix_archive_map = {a.stem: a for a in archives}
        for f in tqdm(files):
            if (prefix := f.split(".")[0][0:3]) in prefix_archive_map:
                file_to_archive_map[f.split(".")[0]] = prefix_archive_map[prefix]
            else:
                raise RuntimeError(f"Could not find the archive containing {f=} where {archives=}")
    else:
        # This is the slow version with double for loop.
        # We shouldn't really every hit this branch of code, but I'm remiss to delete it, so we'll raise.
        raise RuntimeError("None but devils play past here.")
        file_to_archive_map = {}
        for f in tqdm(files):
            for a in archives:
                if f.startswith(a.stem):
                    file_to_archive_map[f.split(".")[0]] = a
                    break
            else:
                raise RuntimeError(f"Could not find the archive containing {f=} where {archives=}")
    materials = materials.convert_files_to_archived_file(file_to_archive_map)
    materials = materials.convert_files_suffix(LIFT_LEVEL_EXTENSIONS[lift_level])

    if not DISABLE_ESP_CACHE:
        print(f"Saving materials to cache: {cache_file}.")
        with open(cache_file, "wb") as fp:
            pickle.dump(materials, fp)

    return materials


def get_materials_esp_fam(
    lift_level: LiftLevel,
    tr_size: float = 0.80,
    vl_size: float = 0.20,
    ts_size: float = 0.00,
    lift_level_ddp: LiftLevel = LiftLevel.DEC,
    min_freq: Optional[int] = 10,
    max_imbalance_ratio: Optional[int] = 100,
    top_k: Optional[int] = None,
    purge_empty_samples: bool = True,
    verbose: bool = True,
) -> Materials:
    sha_label_map = {s: l[0] for s, l in get_sorel_sha_label_map("fam").items()}

    # This ensures that using other de-duplication lift levels will use the same classes
    if lift_level_ddp != LiftLevel.DEC:
        file = Path("./cache/usenix_fam_classes.txt")
        include = set(file.read_text().split())
        sha_label_map = {s: l for s, l in sha_label_map.items() if l in include}

    # min_freq = int(os.environ.get("LMLM_GET_MATERIALS_ESP_FAM_MIN_FREQ", min_freq))  # FIXME: remove
    # max_imbalance_ratio = int(os.environ.get("LMLM_GET_MATERIALS_ESP_FAM_MAX_IMBALANCE_RATIO", max_imbalance_ratio))  # FIXME: remove
    # top_k = int(os.environ.get("LMLM_GET_MATERIALS_ESP_FAM_TOP_K", top_k))  # FIXME: remove

    return _get_materials_esp_clf(
        sha_label_map,
        "single_label_classification",
        lift_level,
        tr_size,
        vl_size,
        ts_size,
        lift_level_ddp,
        purge_empty_samples,
        verbose,
        min_freq=min_freq,
        max_imbalance_ratio=max_imbalance_ratio,
        top_k=top_k,
    )


def get_materials_esp_beh(
    lift_level: LiftLevel,
    tr_size: float = 0.80,
    vl_size: float = 0.20,
    ts_size: float = 0.00,
    lift_level_ddp: LiftLevel = LiftLevel.DEC,
    min_freq: Optional[int] = 100,
    max_imbalance_ratio: Optional[int] = 10,
    top_k: Optional[int] = None,
    purge_empty_samples: bool = True,
    verbose: bool = True,
) -> Materials:
    sha_label_map = get_sorel_sha_label_map("beh")

    # This ensures that using other de-duplication lift levels will use the same classes
    if lift_level_ddp != LiftLevel.DEC:
        file = Path("./cache/usenix_beh_classes.txt")
        include = set(file.read_text().split())
        sha_label_map = {s: l for s, l in sha_label_map.items() if all(i in include for i in l)}

    # min_freq = int(os.environ.get("LMLM_GET_MATERIALS_ESP_BEH_MIN_FREQ", min_freq))  # FIXME: remove
    # max_imbalance_ratio = int(os.environ.get("LMLM_GET_MATERIALS_ESP_BEH_MAX_IMBALANCE_RATIO", max_imbalance_ratio))  # FIXME: remove
    # top_k = int(os.environ.get("LMLM_GET_MATERIALS_ESP_BEH_TOP_K", top_k))  # FIXME: remove

    return _get_materials_esp_clf(
        sha_label_map,
        "multi_label_classification",
        lift_level,
        tr_size,
        vl_size,
        ts_size,
        lift_level_ddp,
        purge_empty_samples,
        verbose,
        min_freq=min_freq,
        max_imbalance_ratio=max_imbalance_ratio,
        top_k=top_k,
    )
