"""
High-level loading API for pytorch datasets.

Loading the files asynchronously is signficantly faster, but seems to blow up.

asyncronous_loading:
    Asynchronous loading can be orders of magnitude faster, but has a quirky memory leak
    (technically the memory leak could also exist in the synchronous version, but it seems
    most liklely to be a result of my implementation of asynchronous loading). The memory
    leak seems to be caused when reading a large number of files, e.g., it occurs when reading
    600000 files at sequence length 512, but not for only 500000 files. The leak does not seem
    connected to the number of bytes being read, e.g., it does not occur when reading 262144
    bytes from 100000 files.
dtype:
    - "bytes" stores data as bytes
    - "np" stores data as a numpy.ndarray
    - "pt" stores data as a ByteTensor
    For short sequences, "bytes" requires marginally less memory than "np" and "pt", e.g.,
    around 6% less for sequences of length 512. For long sequences, the overhead is negligible.

"""

from argparse import ArgumentParser
from abc import ABC
import asyncio
from collections import Counter, UserDict
from collections.abc import Iterable, Sequence
from datetime import datetime
from functools import partial
import gc
from itertools import cycle, islice
import json
import math
import os
from pathlib import Path
from pprint import pprint, pformat
import random
import sys
from statistics import mean, median
import time
from typing import Callable, Literal, Optional
import warnings

if __name__ == "__main__":
    print(f"STARTING @{datetime.now()}\n{'-' * 88}", flush=True)
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# pylint: enable=wrong-import-position

import numpy as np
import pandas as pd
import psutil
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from torch import ByteTensor, LongTensor, Tensor
from torch.utils.data import (
    ConcatDataset,
    Dataset,
    IterableDataset,
    Subset,
    get_worker_info,
    random_split,
)
from tqdm import tqdm

from src.utils import get_max_keys_from_dict, to_long_tensor
from src.data.cfg import SOREL_PATH, BODMAS_LABELS_FILE, DATASET_TO_FILES, SOREL_META_CSV
from src.data.label_datasets import (
    get_label_mapping_virus_total_reports_sorel,
    ThreatLabelExtractor,
    ThreatLabelRefiner,
)
from src.data.loaders_core import Materials, SplitNames
from src.data.utils import read_binary_files, read_binary_files_asynch


class BinaryDataset(ABC):

    def __init__(
        self,
        files: Sequence[os.PathLike],
        labels: Optional[Sequence[int] | np.ndarray | Tensor] = None,
        max_length: Optional[int] = None,
        preprocess_fn: Callable[[LongTensor], LongTensor] = to_long_tensor,
        in_memory_dtype: Literal["bytes", "np", "pt"] = "bytes",
        asynch: bool = True,
        asynch_chunk_size: int = 500000,
        disable_tqdm: bool = True,
        id2label: Optional[dict[int, str]] = None,
        label2id: Optional[dict[str, int]] = None,
        dist: Optional[Counter[str, int]] = None,
    ) -> None:

        self.files = list(map(str, files))
        self.labels = torch.tensor(labels, dtype=torch.long) if isinstance(labels, (Sequence, np.ndarray)) else None
        self.max_length = max_length
        self.preprocess_fn = preprocess_fn
        self.in_memory_dtype = in_memory_dtype
        self.asynch = asynch
        self.asynch_chunk_size = asynch_chunk_size
        self.disable_tqdm = disable_tqdm
        self._id2label = id2label
        self._label2id = label2id
        self._dist = dist if dist is not None else self.get_dist()

    # TODO: if the IterableBinaryDataset has __len__, could this interfere with
    # how third-party code treats the dataset? We want them to treat it as a
    # subclass of IterableDataset, but third party code might only check for
    # structural subtypes.
    def __len__(self) -> int:
        return len(self.files)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"\t{len(self)=}\n"
            f"\tlen(self.labels)={len(self.labels) if self.labels is not None else None}\n"
            f"\t{self.max_length=}\n"
            f"\t{self.preprocess_fn=}\n"
            f"\t{self.asynch=}\n"
            f"\t{self.asynch_chunk_size=}\n"
            f"\t{self.in_memory_dtype=}\n"
            ")"
        )

    def get_dist(self):
        if self.labels is None:
            return None
        if isinstance(self.labels, (Tensor, np.ndarray)):
            labels = self.labels.tolist()
        return Counter([self.id2label[i] for i in labels])

    @property
    def dist(self) -> Counter[str, int]:
        return self._dist

    @property
    def id2label(self) -> dict[int, str]:
        if self._id2label is not None:
            return self._id2label
        raise NotImplementedError()

    @property
    def label2id(self) -> dict[str, int]:
        if self._label2id is not None:
            return self._label2id
        raise NotImplementedError()

    @property
    def num_classes(self) -> int:
        return len(self.dist)


class MapBinaryDataset(Dataset, BinaryDataset):
    """
    Standard dataset. Loads binaries into memory when initialized.
    """

    def __init__(
        self,
        files: Sequence[os.PathLike],
        labels: Optional[Sequence[int] | np.ndarray | Tensor] = None,
        max_length: Optional[int] = None,
        preprocess_fn: Callable[[LongTensor], LongTensor] = to_long_tensor,
        in_memory_dtype: Literal["bytes", "np", "pt"] = "bytes",
        asynch: bool = True,
        asynch_chunk_size: int = 500000,
        disable_tqdm: bool = True,
        id2label: Optional[dict[int, str]] = None,
        label2id: Optional[dict[str, int]] = None,
        dist: Optional[Counter[str, int]] = None,
    ) -> None:
        super().__init__(
            files,
            labels,
            max_length,
            preprocess_fn,
            in_memory_dtype,
            asynch,
            asynch_chunk_size,
            disable_tqdm,
            id2label,
            label2id,
            dist,
        )

        if self.asynch:
            loop = asyncio.get_event_loop()
            future = read_binary_files_asynch(
                self.files,
                self.max_length,
                self.in_memory_dtype,
                self.disable_tqdm,
                self.asynch_chunk_size,
            )
            self.x: list[bytes | np.ndarray | ByteTensor] = loop.run_until_complete(future)
        else:
            self.x = read_binary_files(self.files, self.max_length, self.in_memory_dtype)

    def __getitem__(self, i: int) -> dict[Literal["name", "labels", "input_ids"], str | LongTensor]:
        r = {
            "name": str(self.files[i]).split("/")[-1],
            "input_ids": self.preprocess_fn(self.x[i]),
        }

        if self.labels is not None:
            r["labels"] = self.labels[i]

        return r


class LazyMapBinaryDataset(Dataset, BinaryDataset):
    """
    Lazy loads large chunks of binaries once in-memory cache of data is drained.
    Needs to paired with a sampler that will select indices in contiguous chunks.
    Otherwise, this is simply a much less efficient version of the MapBinaryDataset.
    """

    def __init__(
        self,
        files: Sequence[os.PathLike],
        labels: Optional[Sequence[int] | np.ndarray | Tensor] = None,
        max_length: Optional[int] = None,
        preprocess_fn: Callable[[LongTensor], LongTensor] = to_long_tensor,
        in_memory_dtype: Literal["bytes", "np", "pt"] = "bytes",
        asynch: bool = True,
        asynch_chunk_size: int = 500000,
        disable_tqdm: bool = True,
        id2label: Optional[dict[int, str]] = None,
        label2id: Optional[dict[str, int]] = None,
        dist: Optional[Counter[str, int]] = None,
        chunk_size: Optional[int] = None
    ) -> None:
        super().__init__(
            files,
            labels,
            max_length,
            preprocess_fn,
            in_memory_dtype,
            asynch,
            asynch_chunk_size,
            disable_tqdm,
            id2label,
            label2id,
            dist,
        )
        self.chunk_size = self.asynch_chunk_size if chunk_size is None else chunk_size
        self.x: list[Optional[bytes | np.ndarray | ByteTensor]] = [None for _ in range(len(self))]

    def __getitem__(self, i: int) -> dict[Literal["name", "labels", "input_ids"], str | LongTensor]:

        r = {"name": str(self.files[i]).split("/")[-1]}
        if self.labels is not None:
            r["labels"] = self.labels[i]

        if self.x[i] is None:
            # Clean up the data from the last chunk
            self.x = [None for _ in range(len(self))]
            gc.collect()
            # Fetch the data for this chunk
            files = self.files[i : i + self.chunk_size]
            if self.asynch:
                loop = asyncio.get_event_loop()
                future = read_binary_files_asynch(
                    files,
                    self.max_length,
                    self.in_memory_dtype,
                    self.disable_tqdm,
                    self.asynch_chunk_size,
                )
                x = loop.run_until_complete(future)
            else:
                x = read_binary_files(files, self.max_length, self.in_memory_dtype)
            self.x[i : i + self.chunk_size] = x

        r["input_ids"] = self.preprocess_fn(self.x[i])

        return r

    def __len__(self) -> int:
        return len(self.files)


class IterableBinaryDataset(IterableDataset, BinaryDataset):
    """
    Lazy loads large chunks of binaries once in-memory cache of data is drained.
    Implemented in a pseudo-iterable fashion forcing accesses to data to be
    sequential, rather than random.
    Additional care taken to ensure that multiple workers in the dataloader do
    not interfere with each other.
    Unfortunately, using num_workers > 0 and asynch together causes the asynchronous
    file reading to hang, so this utility requires num_workers == 0.
    """

    def __init__(
        self,
        files: Sequence[os.PathLike],
        labels: Optional[Sequence[int] | np.ndarray | Tensor] = None,
        max_length: Optional[int] = None,
        preprocess_fn: Callable[[LongTensor], LongTensor] = to_long_tensor,
        in_memory_dtype: Literal["bytes", "np", "pt"] = "bytes",
        asynch: bool = True,
        asynch_chunk_size: int = 500000,
        disable_tqdm: bool = True,
        id2label: Optional[dict[int, str]] = None,
        label2id: Optional[dict[str, int]] = None,
        dist: Optional[Counter[str, int]] = None,
        chunk_size: Optional[int] = None
    ) -> None:
        super().__init__(
            files,
            labels,
            max_length,
            preprocess_fn,
            in_memory_dtype,
            asynch,
            asynch_chunk_size,
            disable_tqdm,
            id2label,
            label2id,
            dist,
        )
        self.chunk_size = self.asynch_chunk_size if chunk_size is None else chunk_size

        # Initialized after call to __iter__. These are unique to each process when num_workers > 1.
        # Each sequences will have the same length. Their meaning is self-evident.
        # idx is used by each process to index specific value within the sequence.
        self.my_length: int = None
        self.my_files: list[str] = None
        self.my_labels: Optional[LongTensor] = None
        self.my_x: list[Optional[bytes | np.ndarray | ByteTensor]] = None
        self.my_idx: int = None

    def __iter__(self):
        self.set_my_local_attributes()
        return self

    def set_my_local_attributes(self) -> None:
        worker_info = get_worker_info()

        if worker_info is None:
            start = 0
            end = len(self)
        else:
            per_worker = int(math.ceil((len(self) - 0) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start = 0 + worker_id * per_worker
            end = min(start + per_worker, len(self))

        self.my_length = end - start
        self.my_files = self.files[start:end]
        self.my_labels = self.labels[start:end] if self.labels is not None else None
        self.my_x = [None for _ in range(self.my_length)]
        self.my_idx = 0

    def __next__(self):
        if self.my_idx >= len(self.my_files):
            raise StopIteration()

        r = {"name": str(self.my_files[self.my_idx]).split("/")[-1]}
        if self.my_labels is not None:
            r["labels"] = self.my_labels[self.my_idx]

        if self.my_x[self.my_idx] is None:
            # Clean up the data from the last chunk
            self.my_x = [None for _ in range(self.my_length)]
            gc.collect()
            # Fetch the data for this chunk
            files = self.my_files[self.my_idx : self.my_idx + self.chunk_size]
            if self.asynch:
                loop = asyncio.get_event_loop()
                future = read_binary_files_asynch(
                    files,
                    self.max_length,
                    self.in_memory_dtype,
                    self.disable_tqdm,
                    self.asynch_chunk_size,
                )
                x = loop.run_until_complete(future)
            else:
                x = read_binary_files(files, self.max_length, self.in_memory_dtype)
            self.my_x[self.my_idx : self.my_idx + self.chunk_size] = x

        r["input_ids"] = self.preprocess_fn(self.my_x[self.my_idx])

        self.my_idx += 1
        return r


MapBinaryDatasetDict = dict[SplitNames, MapBinaryDataset]
IterableBinaryDatasetDict = dict[SplitNames, IterableBinaryDataset]


def print_dataset_pt(dataset: MapBinaryDatasetDict | IterableBinaryDatasetDict) -> None:
    for split, d in dataset.items():
        print(f"{split} -- {d}")


def get_dataset_pt(
    materials: Materials, streaming: bool = False, **kwds,
) -> MapBinaryDatasetDict | IterableBinaryDatasetDict:
    BinaryDatasetClass = IterableBinaryDataset if streaming else MapBinaryDataset
    dataset = {
        split: BinaryDatasetClass(
            materials.files[split],
            materials.labels[split],
            id2label=materials.id2label,
            label2id=materials.label2id,
            dist=materials.dist,
            **kwds,
        )
        for split in ["tr", "vl", "ts"]
    }
    return dataset


def test_pytorch_style_datasets():

    random.seed(0)
    np.random.seed(0)
    torch.random.manual_seed(0)

    DISABLE_TQDM = False
    MAX_LENGTH = 4096
    IN_MEMORY_DTYPE = "pt"
    ASYNCH = True
    N_SAMPLES = 1000
    ASYNCH_CHUNK_SIZE = 50

    files = list(sorted(DATASET_TO_FILES["binaries"]["bodmas_pe"]()))
    labels = pd.read_csv(BODMAS_LABELS_FILE).set_index("sha")["family"].to_dict()
    files_and_labels = {
        f : labels[f.stem] for f in files
        if labels[f.stem] not in (np.NaN, "unknown", "Unknown")
    }
    DIST: Counter[str, int] = Counter(files_and_labels.values())
    LABEL2ID = {l: i for i, l in enumerate(DIST.keys())}
    ID2LABEL = {i: l for l, i in LABEL2ID.items()}

    FILES = list(files_and_labels.keys())[:N_SAMPLES]
    LABELS = list(LABEL2ID[l] for l in files_and_labels.values())[:N_SAMPLES]

    def PREPROCESS_FN(x: LongTensor) -> LongTensor:
        return x + 5

    map_dataset = MapBinaryDataset(
        FILES,
        LABELS,
        MAX_LENGTH,
        PREPROCESS_FN,
        IN_MEMORY_DTYPE,
        ASYNCH,
        ASYNCH_CHUNK_SIZE,
        DISABLE_TQDM,
        ID2LABEL,
        LABEL2ID,
        DIST,
    )
    iterable_dataset = IterableBinaryDataset(
        FILES,
        LABELS,
        MAX_LENGTH,
        PREPROCESS_FN,
        IN_MEMORY_DTYPE,
        ASYNCH,
        ASYNCH_CHUNK_SIZE,
        DISABLE_TQDM,
        ID2LABEL,
        LABEL2ID,
        DIST,
    )

    BATCH_SIZE = 4
    SHUFFLE = False
    NUM_WORKERS = 4

    print("Testing the Map-style Dataset")
    map_out = {}
    dataloader = DataLoader(map_dataset, BATCH_SIZE, SHUFFLE, num_workers=NUM_WORKERS)
    for i, inputs in tqdm(enumerate(dataloader), total=N_SAMPLES // BATCH_SIZE):
        for l, n in zip(inputs["labels"], inputs["name"]):
            assert n not in map_out, f"{i=} {n=}"
            map_out[n] = l.item()

    print("Testing the Iterable-style Dataset")
    dataloader = DataLoader(iterable_dataset, BATCH_SIZE, num_workers=NUM_WORKERS)
    iterable_out = {}
    for i, inputs in tqdm(enumerate(dataloader), total=N_SAMPLES // BATCH_SIZE):
        for l, n in zip(inputs["labels"], inputs["name"]):
            assert n not in iterable_out, f"{i=} {n=}"
            iterable_out[n] = l.item()

    assert map_out == iterable_out, "Outputs differ."


if __name__ == "__main__":
    ...
