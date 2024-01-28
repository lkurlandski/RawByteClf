"""
High-level loading API for pytorch datasets.
"""

# pylint: disable=wrong-import-position
print(f"Entered {__file__=}")

import asyncio
from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import datetime
import gc
from itertools import cycle, islice
import math
import os
from pathlib import Path
import random
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
from torch.utils.data import ConcatDataset, Dataset, Subset, random_split
from tqdm import tqdm

from src.data.cfg import BODMAS_LABELS_FILE, DATASET_TO_FILES
from src.data.utils import _tr_vl_ts_split_with_guarentees, stream_sorel_meta
from src.data.loaders_hf import get_sorel_dataset as get_sorel_dataset_hf
from src.data.prepare_datasets import s3_dataset_generator


def batched(iterable: Iterable, n: int):
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def read_binary_file(
    f: Path,
    max_length: Optional[int] = None,
    dtype: Literal["bytes", "np", "pt"] = "bytes",
) -> bytes | np.ndarray | torch.ByteTensor:
    """
    Args:
        dtype: "UserWarning: The given buffer is not writable..."
    """
    with open(f, "rb") as fp:
        b = fp.read(max_length)

    if dtype == "bytes":
        return b
    elif dtype == "np":
        return np.frombuffer(b, dtype=np.uint8)
    elif dtype == "pt":
        return torch.frombuffer(b, dtype=torch.uint8)

    raise ValueError(f"Unknown {dtype=}")


async def read_binary_file_async(
    f: Path,
    max_length: Optional[int] = None,
    dtype: Literal["bytes", "np", "pt"] = "bytes",
) -> bytes | np.ndarray | torch.ByteTensor:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, read_binary_file, f, max_length, dtype)


# FIXME: add len(SPECIALS) !!
def preprocess_fn_add_cls_token(x: torch.LongTensor, cls_token_id: int) -> torch.LongTensor:
    return torch.cat([torch.tensor([cls_token_id], dtype=torch.long), x])


class BinaryDataset(Dataset):
    """
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
        "bytes" stores data as bytes
        "np" stores data as a numpy.ndarray
        "pt" stores data as a torch.ByteTensor
        For short sequences, "bytes" requires marginally less memory than "np" and "pt", e.g.,
        around 6% less for sequences of length 512. For long sequences, the overhead is negligible.
    """

    def __init__(
        self,
        files: Iterable[os.PathLike] | Sequence[os.PathLike],
        labels: Optional[Iterable[int] | Sequence[int]] = None,
        max_length: Optional[int] = None,
        keep_in_memory: bool = True,
        preprocess_fn: Callable[[torch.LongTensor], torch.LongTensor] = lambda x: x,
        id2label: Optional[dict[int, str]] = None,
        label2id: Optional[dict[str, int]] = None,
        asyncronous_loading: bool = True,
        asynch_chunk_size: int = 100000,
        in_memory_dtype: Literal["bytes", "np", "pt"] = "bytes",
        length: Optional[int] = None,
    ) -> None:

        if hasattr(files, "__len__"):
            self._length = len(files)
        elif hasattr(labels, "__len__"):
            self._length = len(labels)
        elif length is not None:
            self._length = length
        else:
            raise RuntimeError("Could not determine length of dataset.")

        if not keep_in_memory:
            if not isinstance(files, Sequence):
                raise ValueError("Files must be a sequence if keep_in_memory=False.")
            if self.labels is not None and not isinstance(labels, Sequence):
                raise ValueError("Labels must be a sequence if keep_in_memory=False.")

        self.files = files
        self.labels = torch.tensor(labels, dtype=torch.long) if isinstance(labels, Sequence) else None
        self.max_length = max_length
        self.keep_in_memory = keep_in_memory
        self.preprocess_fn = preprocess_fn
        self.in_memory_dtype = in_memory_dtype
        self.asynch_chunk_size = asynch_chunk_size

        self._id2label = id2label
        self._label2id = label2id
        self._dist = None

        if self.keep_in_memory:
            if asyncronous_loading:
                loop = asyncio.get_event_loop()
                loop.run_until_complete(self.load_dataset_into_memory_async())
            else:
                x = []
                for f in tqdm(files):
                    x.append(read_binary_file(f, max_length, in_memory_dtype))
                self.x = x

    def __getitem__(self, i: int) -> tuple[torch.LongTensor, torch.LongTensor]:
        r = {"name": str(self.files[i]).split("/")[-1]}

        if self.labels is not None:
            r["labels"] = torch.tensor(self.labels[i], dtype=torch.long)

        if self.keep_in_memory:
            x_i = self.x[i]
            if isinstance(x_i, bytes):
                x_i: torch.ByteTensor = torch.frombuffer(x_i, dtype=torch.uint8)
            elif isinstance(x_i, np.ndarray):
                x_i: torch.ByteTensor = torch.from_numpy(x_i)
        else:
            x_i: torch.ByteTensor = read_binary_file(self.files[i], self.max_length, dtype="pt")

        x_i: torch.LongTensor = x_i.to(torch.long)
        x_i = self.preprocess_fn(x_i)[0:self.max_length]
        r["input_ids"] = x_i

        return r

    def __len__(self) -> int:
        return self._length

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return (
            "BinaryDataset(\n"
            f"\t{len(self.files)=}\n"
            f"\t{len(self.labels) if self.labels else None=}\n"
            f"\t{self.max_length=}\n"
            f"\t{self.keep_in_memory=}\n"
            f"\t{self.preprocess_fn=}\n"
            ")"
        )

    async def load_dataset_into_memory_async(self) -> list[np.ndarray]:
        file_chunks = batched(self.files, self.asynch_chunk_size)
        # file_chunks = list(file_chunks)  # FIXME
        n_chunks = math.ceil(len(self) / self.asynch_chunk_size)
        desc = f"Asynchronously loading {len(self)} files in {n_chunks} chunks..."
        x = []
        for files in tqdm(file_chunks, desc=desc, total=n_chunks):
            tasks = [read_binary_file_async(f, self.max_length, self.in_memory_dtype) for f in files]
            x_i = await asyncio.gather(*tasks)
            x.extend(x_i)
            # gc.collect()
        assert len(x) == len(self)
        self.x = x

    @property
    def dist(self) -> Counter[str, int]:
        if self._dist is not None:
            return self._dist
        self._dist = Counter([self.id2label[i] for i in self.labels])
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


class StreamingBinaryDataset(BinaryDataset):

    def __init__(
        self,
        hashes: list[str],
        labels: Optional[list[int] | int] = None,
        max_length: Optional[int] = None,
        keep_in_memory: bool = True,  # pylint: disable=unused-argument
        preprocess_fn: Callable[[torch.LongTensor], torch.LongTensor] = lambda x: x,
        id2label: Optional[dict[int, str]] = None,
        label2id: Optional[dict[str, int]] = None,
        asyncronous_loading: bool = True,  # pylint: disable=unused-argument
    ) -> None:

        if isinstance(labels, list):
            if not isinstance(labels[0], int):
                raise TypeError(f"labels must be a list of ints, not {type(labels[0])=}")
            self.labels = labels
        elif isinstance(labels, int):
            self.labels = [labels] * len(hashes)
        else:
            self.labels = None

        iterable = s3_dataset_generator(hashes, max_length=max_length, errors=2)
        samples = list(tqdm(iterable, desc="Streaming dataset...", total=len(hashes)))
        kept = set(s["name"] for s in samples)

        self.x = [np.frombuffer(s["bytes"], dtype=np.uint8) for s in samples]
        self.files = [f for f in hashes if f in kept]
        self.max_length = max_length
        self.preprocess_fn = preprocess_fn
        self.keep_in_memory = True

        self._id2label = id2label
        self._label2id = label2id
        self._dist = None

    def __len__(self) -> int:
        return len(self.x)



class StaticBinaryDataset(BinaryDataset):

    def __init__(
        self,
        files: list[str],
        x: list[np.ndarray],
        labels: Optional[list[int] | int] = None,
        max_length: Optional[int] = None,
        keep_in_memory: bool = True,  # pylint: disable=unused-argument
        preprocess_fn: Callable[[torch.LongTensor], torch.LongTensor] = lambda x: x,
        id2label: Optional[dict[int, str]] = None,
        label2id: Optional[dict[str, int]] = None,
        asyncronous_loading: bool = True,  # pylint: disable=unused-argument
    ) -> None:

        if isinstance(labels, list):
            if not isinstance(labels[0], int):
                raise TypeError(f"labels must be a list of ints, not {type(labels[0])=}")
            self.labels = labels
        elif isinstance(labels, int):
            self.labels = [labels] * len(x)
        else:
            self.labels = None

        self.x = x
        self.files = files
        self.max_length = max_length
        self.preprocess_fn = preprocess_fn
        self.keep_in_memory = True

        self._id2label = id2label
        self._label2id = label2id
        self._dist = None

    def __len__(self) -> int:
        return len(self.x)


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
    vl_size = vl_size / len(dataset) if isinstance(vl_size, int) else vl_size
    ts_size = ts_size / len(dataset) if isinstance(ts_size, int) else ts_size
    assert vl_size < 1.0 and ts_size < 1.0, f"{vl_size=} and {ts_size=} must be less than 1.0."

    splits = random_split(dataset, [1 - vl_size - ts_size, vl_size, ts_size])
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
    **kwds,
) -> dict[Literal["tr", "vl", "ts"], BinaryDataset]:
    if vl_size is None:
        vl_size = 10000 if subset is None else 0.1
    if ts_size is None:
        ts_size = 10000 if subset is None else 0.1

    print(f"Loading SOREL ({subset=} {vl_size=} {ts_size=})...", flush=True)

    files = list(sorted(DATASET_TO_FILES["binaries"]["sorel_pe"]()))
    if files:
        print(f"Found SOREL binaries on disk. Loading from binaries...", flush=True)
        dataset = BinaryDataset(files, **kwds)
        dataset = tr_vl_ts_split(dataset, vl_size, ts_size)
        return dataset

    BATCH_SIZE = 1024
    print(
        f"Could not locate SOREL binaries on disk. Loading from HF dataset with {BATCH_SIZE=}. "
        f"Note that this can will take a long time...",
        flush=True,
    )
    dataset = get_sorel_dataset_hf(subset, 1, 1)
    files = []
    x = []
    for s in ["tr", "vl", "ts"]:
        for d in tqdm(dataset[s].iter(batch_size=BATCH_SIZE), total=len(dataset[s]) / BATCH_SIZE):
            files.extend(d["name"])
            x.extend([np.frombuffer(b[:max_length], dtype=np.uint8) for b in d["bytes"]])

    dataset = StaticBinaryDataset(files, x=x, max_length=max_length, **kwds)

    return tr_vl_ts_split(dataset, vl_size=vl_size, ts_size=ts_size)


def get_bodmas_dataset(
    subset: Optional[int] = None,
    min_freq: Optional[int] = None,
    top_k: Optional[int] = None,
    ts_size: int | float = 0.1,
    vl_size: int | float = 0.1,
    **kwds,
) -> tuple[dict[Literal["tr", "vl", "ts"], BinaryDataset], Counter[str, int]]:

    samples_per_class = 1
    min_freq = samples_per_class * 3 if min_freq is None else min_freq

    print(f"Loading BODMAS ({subset=} {vl_size=} {ts_size=} {min_freq=} {top_k=})...", flush=True)

    # Get the files and labels, then create a mapping for each file to its label
    files = list(sorted(DATASET_TO_FILES["binaries"]["bodmas_pe"]()))
    labels = pd.read_csv(BODMAS_LABELS_FILE).set_index("sha")["family"].to_dict()
    files_and_labels = {
        f : labels[f.stem] for f in files
        if labels[f.stem] not in (np.NaN, "unknown", "Unknown")
    }
    del files, labels

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

    dataset = BinaryDataset(
        files,
        labels,
        id2label=id2label,
        label2id=label2id,
        **kwds,
    )
    dataset = tr_vl_ts_split_with_guarentees(dataset, vl_size, ts_size, samples_per_class)
    dist = Counter(files_and_labels.values())

    return dataset, dist


def get_goodware_vs_malware_dataset():
    raise NotImplementedError()

    dataset_mal = BinaryDataset(
        list(filter(filter_fn, DATASET_TO_FILES["binaries"]["bodmas_pe"]()))[0:9030],
        1,
        args.max_length,
        False,
        partial(preprocess_fn_add_cls_token, cls_token_id=tokenizer.cls_token_id),
    )
    print(f"{dataset_mal=}")
    dataset_ben = BinaryDataset(
        list(filter(filter_fn, DATASET_TO_FILES["binaries"]["local_pe"]())),
        0,
        args.max_length,
        False,
        partial(preprocess_fn_add_cls_token, cls_token_id=tokenizer.cls_token_id),
    )
    print(f"{dataset_ben=}")
    dataset = ConcatDataset([dataset_mal, dataset_ben])
    dataset = random_split(dataset, [0.8, 0.1, 0.1])
    dataset = {"tr": dataset[0], "vl": dataset[1], "ts": dataset[2]}


def test():
    # def filter_fn(f: Path):
    #     return f.stat().st_size > 0 and f.suffix == ".exe"

    # from src.data.cfg import DATASET_TO_FILES
    # dataset_mal = BinaryDataset(list(filter(filter_fn, DATASET_TO_FILES["binaries"]["bodmas_pe"]()))[0:256], 1, 1024, True)
    # dataset_ben = BinaryDataset(list(filter(filter_fn, DATASET_TO_FILES["binaries"]["local_pe"]()))[0:256], 0, 1024, True)

    # print(dataset_mal[0])
    # print(dataset_ben[0])

    # dataset = get_bodmas_dataset(max_length=1024)
    # for k, v in dataset.items():
    #     print(k, len(v))

    # import time

    # start = time.time()

    # files = list(DATASET_TO_FILES["binaries"]["bodmas_pe"]())
    # labels = list(range(len(files)))
    # max_length = 65536

    # dataset = BinaryDataset(files, labels, max_length, True, asyncronous_loading=True)

    # for i in range(10):
    #     print(dataset[i]["input_ids"].tolist()[0:16])
    # print()

    # print(f"Time taken: {time.time() - start:.2f}s")
    from argparse import ArgumentParser
    import time
    import math

    parser = ArgumentParser()
    parser.add_argument("--n_files", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--asynch", action="store_true")
    parser.add_argument("--dtype", default="bytes")
    args = parser.parse_args()

    BYTES_OBJ_OVERHEAD = 33

    print(f"{args.n_files=}")
    print(f"{args.max_length=}")
    print(f"{args.asynch=}")
    print(f"{args.dtype=}")

    m_i = psutil.virtual_memory()
    print(f"{m_i.total=}")
    print(f"{m_i.used=}")

    files = sorted(list(DATASET_TO_FILES["binaries"]["bodmas_pe"]()))
    files = islice(cycle(files), args.n_files if isinstance(args.n_files, int) else len(files))

    # files = list(files)  # FIXME
    # print(f"{len(files)=}")

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

    isinstance(dataset, BinaryDataset)

    true_mem_delta = m_f.used - m_i.used
    expt_mem_delta = sum(
        min(f.stat().st_size, args.max_length if isinstance(args.max_length, int) else sys.maxsize) + BYTES_OBJ_OVERHEAD
        for f in files
    )
    mem_discrepancy = true_mem_delta - expt_mem_delta

    print(f"{true_mem_delta=}")
    print(f"{expt_mem_delta=}")
    print(f"{mem_discrepancy=}")

    # p = Path("./mem.log")
    # if not p.exists():
    #     p.write_text("subset,max_length,true_mem_delta,expt_mem_delta,mem_discrepancy\n")
    # with open(p, "a") as fp:
    #     fp.write(f"{SUBSET},{MAX_LENGTH},{true_mem_delta},{expt_mem_delta},{mem_discrepancy}\n") 

    # print(dataset)
    # for d in dataset["tr"]:
    #     print(d)
    #     break
    
    time.sleep(2)


if __name__ == "__main__":
    test()
