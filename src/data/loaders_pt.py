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

# pylint: disable=wrong-import-position
print(f"Entered {__file__=}")

from abc import ABC
import asyncio
from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import datetime
import gc
from itertools import cycle, islice
import math
import os
from pathlib import Path
from pprint import pprint
import sys
import time
from typing import Callable, Literal, Optional

if __name__ == "__main__":
    print(f"STARTING @{datetime.now()}\n{'-' * 88}", flush=True)
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# pylint: enable=wrong-import-position

import numpy as np
import pandas as pd
import psutil
import torch
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
import warnings

from src.utils import batched
from src.data.cfg import BODMAS_LABELS_FILE, DATASET_TO_FILES
from src.data.label_datasets import (
    get_label_mapping_virus_total_reports,
    ThreatLabelExtractor,
    ThreatLabelRefiner,
)
from src.data.utils import _tr_vl_ts_split, _tr_vl_ts_split_with_guarentees


# TODO: move these functions somewhere else...

def preprocess_fn_add_cls_token(x: LongTensor, cls_token_id: int) -> LongTensor:
    return torch.cat([torch.tensor([cls_token_id], dtype=torch.long), x])


def preprocess_fn_shift_token_idx(x: LongTensor, shift: int) -> LongTensor:
    return x + shift


DEFAULT_ASYNCH_CHUNK_SIZE = 500000
DEFAULT_IN_MEMORY_DTYPE = "pt"
DEFAULT_DISABLE_TQDM = True
DEFAULT_MIN_SAMPLES_PER_CLASS = 1

def read_binary_file(
    f: Path,
    max_length: Optional[int] = None,
    in_memory_dtype: Literal["bytes", "np", "pt"] = DEFAULT_IN_MEMORY_DTYPE,
) -> bytes | np.ndarray | ByteTensor:
    """
    Will warn with "UserWarning: The given buffer is not writable...", which can
    safely be ignored because we don't care about modifying the bytes object.
    """
    with open(f, "rb") as fp:
        b = fp.read(max_length)

    if in_memory_dtype == "bytes":
        return b
    elif in_memory_dtype == "np":
        return np.frombuffer(b, dtype=np.uint8)
    elif in_memory_dtype == "pt":
        return torch.frombuffer(b, dtype=torch.uint8)

    raise ValueError(f"Unknown {in_memory_dtype=}")


async def read_binary_file_asynch(
    f: Path,
    max_length: Optional[int] = None,
    in_memory_dtype: Literal["bytes", "np", "pt"] = DEFAULT_IN_MEMORY_DTYPE,
) -> bytes | np.ndarray | ByteTensor:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, read_binary_file, f, max_length, in_memory_dtype)


async def read_binary_files_asynch(
    files: list[str],
    max_length: Optional[int] = None,
    in_memory_dtype: Literal["bytes", "np", "pt"] = DEFAULT_IN_MEMORY_DTYPE,
    disable_tqdm: bool = DEFAULT_DISABLE_TQDM,
    asynch_chunk_size: int = DEFAULT_ASYNCH_CHUNK_SIZE,
) -> None:
    file_chunks = batched(files, asynch_chunk_size)

    iterable = file_chunks
    if not disable_tqdm:
        n_chunks = math.ceil(len(files) / asynch_chunk_size)
        iterable = tqdm(
            file_chunks,
            desc=f"Asynchronously loading {len(files)} files in {n_chunks} chunks...",
            total=n_chunks,
        )

    x = []
    for files in iterable:
        tasks = [read_binary_file_asynch(f, max_length, in_memory_dtype) for f in files]
        x_i = await asyncio.gather(*tasks)
        x.extend(x_i)
    return x


def read_binary_files(
    files: list[str],
    max_length: Optional[int] = None,
    in_memory_dtype: Literal["bytes", "np", "pt"] = "bytes",
    disable_tqdm: bool = DEFAULT_DISABLE_TQDM,
) -> list[bytes | np.ndarray | ByteTensor]:

    iterable = files
    if not disable_tqdm:
        iterable = tqdm(
            files,
            desc=f"Synchronously loading {len(files)} files...",
        )

    return [read_binary_file(f, max_length, in_memory_dtype) for f in iterable]


def to_long_tensor(x: bytes | np.ndarray | ByteTensor) -> LongTensor:
    if isinstance(x, bytes):
        return torch.frombuffer(x, dtype=torch.uint8).to(torch.long)
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(torch.long)
    if isinstance(x, Tensor):
        return x.to(torch.long)
    raise TypeError(f"Unexpected type: {type(x)=}")


class BinaryDataset(ABC):

    def __init__(
        self,
        files: Sequence[os.PathLike],
        labels: Optional[Sequence[int]] = None,
        max_length: Optional[int] = None,
        preprocess_fn: Callable[[LongTensor], LongTensor] = lambda x: x,
        in_memory_dtype: Literal["bytes", "np", "pt"] = "pt",
        asynch: bool = True,
        asynch_chunk_size: int = 500000,
        id2label: Optional[dict[int, str]] = None,
        label2id: Optional[dict[str, int]] = None,
    ) -> None:

        self.files = list(map(str, files))
        self.labels = torch.tensor(labels, dtype=torch.long) if isinstance(labels, Sequence) else None
        self.max_length = max_length
        self.preprocess_fn = preprocess_fn
        self.asynch = asynch
        self.asynch_chunk_size = asynch_chunk_size
        self.in_memory_dtype = in_memory_dtype
        self._id2label = id2label
        self._label2id = label2id
        self._dist = self.get_dist()

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
        labels: Optional[Sequence[int]] = None,
        max_length: Optional[int] = None,
        preprocess_fn: Callable[[LongTensor], LongTensor] = lambda x: x,
        in_memory_dtype: Literal["bytes", "np", "pt"] = "pt",
        asynch: bool = True,
        asynch_chunk_size: int = 500000,
        id2label: Optional[dict[int, str]] = None,
        label2id: Optional[dict[str, int]] = None,
    ) -> None:
        super().__init__(
            files,
            labels,
            max_length,
            preprocess_fn,
            in_memory_dtype,
            asynch,
            asynch_chunk_size,
            id2label,
            label2id,
        )

        if self.asynch:
            loop = asyncio.get_event_loop()
            future = read_binary_files_asynch(
                self.files,
                self.max_length,
                self.in_memory_dtype,
                DEFAULT_DISABLE_TQDM,
                self.asynch_chunk_size,
            )
            self.x: list[bytes | np.ndarray | ByteTensor] = loop.run_until_complete(future)
        else:
            self.x = read_binary_files(self.files, self.max_length, self.in_memory_dtype)

    def __getitem__(self, i: int) -> dict[Literal["name", "labels", "input_ids"], str | LongTensor]:
        r = {"name": str(self.files[i]).split("/")[-1]}

        x_i = to_long_tensor(self.x[i])
        x_i = self.preprocess_fn(x_i)[0:self.max_length]
        r["input_ids"] = x_i

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
        labels: Optional[Sequence[int]] = None,
        max_length: Optional[int] = None,
        preprocess_fn: Callable[[LongTensor], LongTensor] = lambda x: x,
        in_memory_dtype: Literal["bytes", "np", "pt"] = "pt",
        asynch: bool = True,
        asynch_chunk_size: int = 500000,
        id2label: Optional[dict[int, str]] = None,
        label2id: Optional[dict[str, int]] = None,
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
            id2label,
            label2id,
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
                    DEFAULT_DISABLE_TQDM,
                    self.asynch_chunk_size,
                )
                x = loop.run_until_complete(future)
            else:
                x = read_binary_files(files, self.max_length, self.in_memory_dtype)
            self.x[i : i + self.chunk_size] = x

        x_i = to_long_tensor(self.x[i])
        x_i = self.preprocess_fn(x_i)[0:self.max_length]
        r["input_ids"] = x_i

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
        labels: Optional[Sequence[int]] = None,
        max_length: Optional[int] = None,
        preprocess_fn: Callable[[LongTensor], LongTensor] = lambda x: x,
        in_memory_dtype: Literal["bytes", "np", "pt"] = "pt",
        asynch: bool = True,
        asynch_chunk_size: int = 500000,
        id2label: Optional[dict[int, str]] = None,
        label2id: Optional[dict[str, int]] = None,
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
            id2label,
            label2id,
        )
        self.chunk_size = self.asynch_chunk_size if chunk_size is None else chunk_size

        # Initialized after call to __iter__. These are unique to each process when num_workers > 1.
        # Each sequences will have the same length. Their meaning is self-evident.
        # idx is used by each process to index specific value within the sequence.
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
                    DEFAULT_DISABLE_TQDM,
                    self.asynch_chunk_size,
                )
                x = loop.run_until_complete(future)
            else:
                x = read_binary_files(files, self.max_length, self.in_memory_dtype)
            self.my_x[self.my_idx : self.my_idx + self.chunk_size] = x

        x_i = to_long_tensor(self.my_x[self.my_idx])
        x_i = self.preprocess_fn(x_i)[0:self.max_length]
        r["input_ids"] = x_i

        self.my_idx += 1
        return r


def tr_vl_ts_split_with_guarentees(
    dataset: BinaryDataset,
    vl_size: float,
    ts_size: float,
    samples_per_class: int = 1,
) -> dict[Literal["tr", "vl", "ts"], BinaryDataset]:
    """
    Guarentees that at least `samples_per_class` samples from each class are present in each split.
    """
    y = np.array(dataset.labels)
    idx = _tr_vl_ts_split_with_guarentees(y, len(dataset), vl_size, ts_size, samples_per_class)
    return {split: Subset(dataset, idx[split]) for split in ["tr", "vl", "ts"]}


def tr_vl_ts_split(
    dataset: Dataset,
    vl_size: float | int,
    ts_size: float | int,
) -> dict[Literal["tr", "vl", "ts"], BinaryDataset]:

    if isinstance(vl_size, float) and isinstance(ts_size, float):
        l = 1.0
    elif isinstance(vl_size, int) and isinstance(ts_size, int):
        l = len(dataset)
    else:
        raise TypeError(f"{vl_size=} and {ts_size=} must be both int or both float.")

    splits = random_split(dataset, [l - vl_size - ts_size, vl_size, ts_size])
    return {
        "tr": splits[0],
        "vl": splits[1],
        "ts": splits[2],
    }


def get_sorel_dataset(
    subset: Optional[int] = None,
    vl_size: int | float = None,
    ts_size: int | float = None,
    max_length: Optional[int] = None,
    streaming: bool = False,
    **kwds,
) -> dict[Literal["tr", "vl", "ts"], IterableBinaryDataset | Subset[BinaryDataset]]:
    if vl_size is None:
        vl_size = 10000 if subset is None else 0.1
    if ts_size is None:
        ts_size = 10000 if subset is None else 0.1

    mem = psutil.virtual_memory()
    print(f"MEM: {round(mem.used / (1024 ** 2), 2)} / {round(mem.total / (1024 ** 2), 2)} MB")
    print(f"Loading SOREL ({subset=} {vl_size=} {ts_size=})...", flush=True)

    # Paths are usually ~100 characters long, so as a str take ~269 bytes of memory.
    # For 10M files, this equates to ~2.69GB of memory for the paths alone, which is
    # why we convert them to str instead of using pathlib.Path objects.
    files = sorted(map(lambda p: p.as_posix(), DATASET_TO_FILES["binaries"]["sorel_pe"]()))[:subset]

    if streaming:  # Iterable datasets do not support splits, so we create separate ones.
        files = _tr_vl_ts_split(files, vl_size, ts_size)
        return {
            "tr": IterableBinaryDataset(files["tr"], max_length=max_length, **kwds),
            "vl": IterableBinaryDataset(files["vl"], max_length=max_length, **kwds),
            "ts": IterableBinaryDataset(files["ts"], max_length=max_length, **kwds),
        }

    dataset = MapBinaryDataset(files, max_length=max_length, **kwds)
    dataset = tr_vl_ts_split(dataset, vl_size, ts_size)
    return dataset


def get_classification_dataset(
    files_and_labels: dict[str, str],
    subset: Optional[int] = None,
    min_freq: Optional[int] = None,
    top_k: Optional[int] = None,
    ts_size: int | float = 0.1,
    vl_size: int | float = 0.1,
    **kwds,
) -> tuple[dict[Literal["tr", "vl", "ts"], BinaryDataset], Counter[str, int]]:
    print(
        f"Loading classification dataset ({subset=} {vl_size=} {ts_size=} {min_freq=} {top_k=})...",
        flush=True,
    )

    min_freq = DEFAULT_MIN_SAMPLES_PER_CLASS * 3 if min_freq is None else min_freq

    # TODO: implement streaming by applying to split to the files themselves rather than the dataset.
    if kwds.pop("streaming", False):
        warnings.warn("Streaming not supported")

    # Select a subset
    files_and_labels = {
        f: l for i, (f, l) in enumerate(files_and_labels.items())
        if (not isinstance(subset, int) or i < subset)
    }

    # Filter out the files that are not in the top_k most frequent labels
    dist: Counter[str, int] = Counter(files_and_labels.values())
    keep = [l for l, n in dist.most_common(top_k) if (n >= min_freq)]
    files_and_labels = {f: l for f, l in files_and_labels.items() if l in keep}

    # Final collection of data items
    dist: Counter[str, int] = Counter(files_and_labels.values())
    label2id = {l: i for i, l in enumerate(dist.keys())}
    id2label = {i: l for l, i in label2id.items()}

    files = list(files_and_labels.keys())
    labels = [label2id[files_and_labels[f]] for f in files]

    dataset = MapBinaryDataset(
        files,
        labels,
        id2label=id2label,
        label2id=label2id,
        **kwds,
    )
    dataset = tr_vl_ts_split_with_guarentees(dataset, vl_size, ts_size, DEFAULT_MIN_SAMPLES_PER_CLASS)
    dist = Counter(files_and_labels.values())

    return dataset, dist


def get_bodmas_dataset(
    subset: Optional[int] = None,
    min_freq: Optional[int] = None,
    top_k: Optional[int] = None,
    ts_size: int | float = 0.1,
    vl_size: int | float = 0.1,
    **kwds,
) -> tuple[dict[Literal["tr", "vl", "ts"], BinaryDataset], Counter[str, int]]:

    # Get the files and labels, then create a mapping for each file to its label
    files = list(sorted(DATASET_TO_FILES["binaries"]["bodmas_pe"]()))
    labels = pd.read_csv(BODMAS_LABELS_FILE).set_index("sha")["family"].to_dict()
    files_and_labels = {
        f : labels[f.stem] for f in files
        if labels[f.stem] not in (np.NaN, "unknown", "Unknown")
    }
    return get_classification_dataset(
        files_and_labels, subset, min_freq, top_k, ts_size, vl_size, **kwds
    )


def get_sorel_dataset_clf(
    subset: Optional[int] = None,
    min_freq: Optional[int] = None,
    top_k: Optional[int] = None,
    ts_size: int | float = 0.1,
    vl_size: int | float = 0.1,
    **kwds,
) -> tuple[dict[Literal["tr", "vl", "ts"], BinaryDataset], Counter[str, int]]:

    # pseudo subset when getting the labels, as many of them will be None
    files = sorted(list(map(str, DATASET_TO_FILES["reports"]["sorel_pe"]())))

    t = time.time()
    print("Getting labels from Virus Total reports...")
    label_map = get_label_mapping_virus_total_reports(
        files,
        ThreatLabelExtractor.build("category"),
        ThreatLabelRefiner.build("top", k=1),
        True,
    )
    print(f"Acquired labels in {time.time() - t}")


    def filter_fn(f: os.PathLike) -> bool:
        f = Path(f)
        return f.stat().st_size >= 2 ** 14 and label_map.get(f.stem, None) is not None


    files = list(filter(filter_fn, files))[0:subset]
    files_and_labels = {f: label_map[Path(f).stem][0] for f in files}
    return get_classification_dataset(
        files_and_labels, subset, min_freq, top_k, ts_size, vl_size, **kwds
    )


def get_goodware_vs_malware_dataset(
    n_ben: Optional[int] = None,
    n_mal: Optional[int] = None,
    mal_to_ben_ratio: Optional[float] = None,
    ts_size: int | float = 0.1,
    vl_size: int | float = 0.1,
    **kwds,
):
    """
    n_mal: 9030
    """

    def filter_fn(f: Path) -> bool:
        return f.stat().st_size >= 2 ** 14 and f.suffix == ".exe"

    mal_files = list(filter(filter_fn, DATASET_TO_FILES["binaries"]["bodmas_pe"]()))
    ben_files = list(filter(filter_fn, DATASET_TO_FILES["binaries"]["local_pe"]()))

    if mal_to_ben_ratio is not None:
        n_mal = len(mal_files)
        n_ben = n_mal * mal_to_ben_ratio
        ben_files = ben_files[0:n_ben]
    else:
        mal_files = mal_files[0:n_mal]
        ben_files = ben_files[0:n_ben]

    dataset_mal = BinaryDataset(mal_files, [1] * len(mal_files), **kwds)
    dataset_ben = BinaryDataset(ben_files, [0] * len(ben_files), **kwds)

    dataset = ConcatDataset([dataset_mal, dataset_ben])
    return tr_vl_ts_split(dataset, vl_size, ts_size)


def test_timing():
    """
    Memory analysis:
        mprof run python {SCRIPT.py}
        mprof plot --output={PLOT.png}
    """

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--n_files", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--asynch", action="store_true")
    parser.add_argument("--dtype", default="bytes")
    args = parser.parse_args()

    print(f"{args.n_files=}")
    print(f"{args.max_length=}")
    print(f"{args.asynch=}")
    print(f"{args.dtype=}")

    m_i = psutil.virtual_memory()
    print(f"{m_i.total=}")
    print(f"{m_i.used=}")

    files = sorted(list(DATASET_TO_FILES["binaries"]["bodmas_pe"]()))
    files = islice(cycle(files), args.n_files if isinstance(args.n_files, int) else len(files))
    dataset = BinaryDataset(
        files,
        labels=None,
        max_length=args.max_length,
        keep_in_memory=True,
        asyncronous_loading=args.asynch,
        in_memory_dtype=args.dtype,
        length=args.n_files,
    )

    m_f = psutil.virtual_memory()
    print(f"{m_f.total=}")
    print(f"{m_f.used=}")

    true_mem_delta = m_f.used - m_i.used
    expt_mem_delta = sum(
        min(
            f.stat().st_size,
            args.max_length if isinstance(args.max_length, int) else sys.maxsize)
        for f in files
    )
    mem_discrepancy = true_mem_delta - expt_mem_delta

    print(f"{true_mem_delta=}")
    print(f"{expt_mem_delta=}")
    print(f"{mem_discrepancy=}")

    print(f"{dataset=}")


def test():

    from functools import partial
    import random

    from torch.utils.data import DataLoader


    random.seed(0)
    np.random.seed(0)
    torch.random.manual_seed(0)


    global DEFAULT_DISABLE_TQDM
    DEFAULT_DISABLE_TQDM = False

    MAX_LENGTH = 4096
    IN_MEMORY_DTYPE = "pt"
    ASYNCH = True
    N_SAMPLES = 1000
    ASYNCH_CHUNK_SIZE = 50
    CHUNK_SIZE = 100

    files = list(sorted(DATASET_TO_FILES["binaries"]["bodmas_pe"]()))
    labels = pd.read_csv(BODMAS_LABELS_FILE).set_index("sha")["family"].to_dict()
    files_and_labels = {
        f : labels[f.stem] for f in files
        if labels[f.stem] not in (np.NaN, "unknown", "Unknown")
    }
    dist: Counter[str, int] = Counter(files_and_labels.values())
    LABEL2ID = {l: i for i, l in enumerate(dist.keys())}
    ID2LABEL = {i: l for l, i in LABEL2ID.items()}

    FILES = list(files_and_labels.keys())[:N_SAMPLES]
    LABELS = list(LABEL2ID[l] for l in files_and_labels.values())[:N_SAMPLES]

    fn_1 = partial(preprocess_fn_shift_token_idx, shift=5)
    fn_2 = partial(preprocess_fn_add_cls_token, cls_token_id=3)
    PREPROCESS_FN = lambda x: fn_2(fn_1(x))

    map_dataset = MapBinaryDataset(
        FILES,
        LABELS,
        MAX_LENGTH,
        PREPROCESS_FN,
        IN_MEMORY_DTYPE,
        ASYNCH,
        ASYNCH_CHUNK_SIZE,
        ID2LABEL,
        LABEL2ID,
    )
    iterable_dataset = IterableBinaryDataset(
        FILES,
        LABELS,
        MAX_LENGTH,
        PREPROCESS_FN,
        IN_MEMORY_DTYPE,
        ASYNCH,
        ASYNCH_CHUNK_SIZE,
        ID2LABEL,
        LABEL2ID,
        CHUNK_SIZE,
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

    sys.exit(0)

    map_dataset_tr, map_dataset_ts = random_split(map_dataset, lengths=[0.8, 0.2])
    iterable_dataset_tr, iterable_dataset_ts = random_split(iterable_dataset, lengths=[0.8, 0.2])

    print(f"{map_dataset_tr=}")
    print(f"{map_dataset_ts=}")
    print(f"{iterable_dataset_tr=}")
    print(f"{iterable_dataset_ts=}")

    print("Testing the Map-style Dataset")
    map_out = {}
    dataloader = DataLoader(map_dataset_tr, BATCH_SIZE, SHUFFLE, num_workers=NUM_WORKERS)
    for i, inputs in tqdm(enumerate(dataloader), total=N_SAMPLES // BATCH_SIZE):
        for l, n in zip(inputs["labels"], inputs["name"]):
            assert n not in map_out, f"{i=} {n=}"
            map_out[n] = l.item()

    print("Testing the Iterable-style Dataset")
    dataloader = DataLoader(iterable_dataset_tr, BATCH_SIZE, num_workers=NUM_WORKERS)
    iterable_out = {}
    for i, inputs in tqdm(enumerate(dataloader), total=N_SAMPLES // BATCH_SIZE):
        for l, n in zip(inputs["labels"], inputs["name"]):
            assert n not in iterable_out, f"{i=} {n=}"
            iterable_out[n] = l.item()

    assert map_out == iterable_out, "Outputs differ."


if __name__ == "__main__":
    dataset, dist = get_sorel_dataset_clf(subset=10000, max_length=512, top_k=10)
    # dataset, dist = get_bodmas_dataset(top_k=10, max_length=512)

    for s in dataset:
        print(s)
        print(f"{dataset[s].dataset}")
        print(f"{dataset[s].dataset.labels}")
        print(f'{dataset[s].dataset[0]["name"]}')
        print(f'{dataset[s][0]["name"]}')

    print(dataset["tr"].dataset.labels.tolist())

    # test()
