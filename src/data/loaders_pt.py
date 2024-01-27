"""
High-level loading API for pytorch datasets.

# TODO:
    - create a BinaryDatasetDict class that has common properties across different splits.
"""

# pylint: disable=wrong-import-position
print(f"Entered {__file__=}")

import asyncio
from collections import Counter
from datetime import datetime
from itertools import islice
import os
from pathlib import Path
import random
import sys
from typing import Callable, Literal, Optional

if __name__ == "__main__":
    print(f"STARTING @{datetime.now()}\n{'-' * 88}", flush=True)
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# pylint: enable=wrong-import-position

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, Dataset, Subset, random_split
from tqdm import tqdm

from src.data.cfg import BODMAS_LABELS_FILE, DATASET_TO_FILES
from src.data.utils import _tr_vl_ts_split_with_guarentees, stream_sorel_meta
from src.data.loaders_hf import get_sorel_dataset as get_sorel_dataset_hf
from src.data.prepare_datasets import s3_dataset_generator


def read_binary_file(f: Path, max_length: Optional[int] = None, dtype: np.dtype = np.uint8) -> np.ndarray:
    with open(f, "rb") as fp:
        b = fp.read(max_length)
    x = np.frombuffer(b, dtype=dtype)
    return x


async def read_binary_file_async(f: Path, max_length: Optional[int] = None, dtype: np.dtype = np.uint8) -> np.ndarray:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, read_binary_file, f, max_length, dtype)


def preprocess_fn_add_cls_token(x: torch.LongTensor, cls_token_id: int) -> torch.LongTensor:
    return torch.cat([torch.tensor([cls_token_id], dtype=torch.long), x])


class BinaryDataset(Dataset):
    """
    files: list[Path]
    labels: list[int]
    max_length: int
    keep_in_memory: bool
    preprocess_fn: Callable[[torch.LongTensor], torch.LongTensor]
    x: list[np.ndarray]

    asynchronous_loading is about 16x faster than single-threaded loading.
    """

    def __init__(
        self,
        files: list[Path],
        labels: Optional[list[int] | int] = None,
        max_length: Optional[int] = None,
        keep_in_memory: bool = True,
        preprocess_fn: Callable[[torch.LongTensor], torch.LongTensor] = lambda x: x,
        id2label: Optional[dict[int, str]] = None,
        label2id: Optional[dict[str, int]] = None,
        asyncronous_loading: bool = True,
    ) -> None:
        self.files = files
        if isinstance(labels, list):
            if not isinstance(labels[0], int):
                raise TypeError(f"labels must be a list of ints, not {type(labels[0])=}")
            self.labels = labels
        elif isinstance(labels, int):
            self.labels = [labels] * len(files)
        else:
            self.labels = None
        self.max_length = max_length
        self.keep_in_memory = keep_in_memory
        self.preprocess_fn = preprocess_fn

        if self.keep_in_memory:
            if asyncronous_loading:
                loop = asyncio.get_event_loop()
                loop.run_until_complete(self.load_dataset_into_memory_async())
            else:
                self.x = []
                for f in tqdm(self.files, desc="Loading dataset into memory..."):
                    self.x.append(read_binary_file(f, max_length))

        self._id2label = id2label
        self._label2id = label2id
        self._dist = None

    def __getitem__(self, i: int) -> tuple[torch.LongTensor, torch.LongTensor]:
        r = {"name": self.files[i].name if isinstance(self.files[i], Path) else self.files[i].split("/")[-1]}

        if self.labels is not None:
            y_i = self.labels[i]
            y_i = torch.tensor(y_i, dtype=torch.long)
            r["labels"] = y_i

        if self.keep_in_memory:
            x_i = self.x[i]
        else:
            x_i = read_binary_file(self.files[i], self.max_length)
        x_i = torch.tensor(x_i, dtype=torch.long)
        x_i = self.preprocess_fn(x_i)[0:self.max_length]
        r["input_ids"] = x_i

        return r

    def __len__(self) -> int:
        return len(self.files)

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
        tasks = [read_binary_file_async(f, self.max_length) for f in self.files]
        self.x = await asyncio.gather(*tasks)

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
    dataset = get_sorel_dataset_hf(subset, 1, 1)

    files = []
    x = []
    for s in ["tr", "vl", "ts"]:
        for d in dataset[s].iter(batch_size=1024):
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


    dataset = get_sorel_dataset(10000, 0.1, 0.1)
    print(dataset)
    for d in dataset["tr"]:
        print(d)
        break


if __name__ == "__main__":
    test()
