"""
High-level loading API for huggingface datasets.
"""

import asyncio
from collections.abc import Generator
from functools import partial
import os
from pprint import pformat
import sys
from typing import Optional

# pylint: disable=wrong-import-position
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# pylint: enable=wrong-import-position

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    Value,
    Features,
)
import numpy as np

from src.data.loaders_core import (
    ClfMaterials,
    SplitNames,
)
from src.data.loaders_pt import read_binary_files_asynch, read_binary_files


FEATURES = Features({"name": Value("string"), "bytes": Value("binary"), "labels": Value("int32")})


def classification_generator(
    files: list[os.PathLike],
    labels: Optional[np.ndarray] = None,
    max_length: Optional[int] = None,
    asynch: bool = True,
    asynch_chunk_size: int = 500000,
) -> Generator:
    kwds = {"max_length": max_length, "in_memory_dtype": "bytes", "disable_tqdm": True}

    data: Optional[bytes] = [None for _ in range(len(files))]
    for i in range(len(files)):
        if data[i] is None:
            data = [None for _ in range(len(files))]
            idx = list(range(i, min(i + asynch_chunk_size, len(files))))
            files_chunk = [files[j] for j in idx]
            if asynch:
                loop = asyncio.get_event_loop()
                future = read_binary_files_asynch(files_chunk, asynch_chunk_size=asynch_chunk_size, **kwds)
                data_chunk = loop.run_until_complete(future)
            else:
                data_chunk = read_binary_files(files_chunk,  **kwds)

            for j, d in zip(idx, data_chunk):
                data[j] = d

        r = {"bytes": data[i], "name": str(files[i]).split("/")[-1].split(".")[0]}
        if labels is not None and labels[i] is not None:
            r["labels"] = labels[i]

        assert all(x is not None for x in r.values())

        yield r


def print_dataset_hf(dataset: DatasetDict | IterableDatasetDict):
    cache_files = []
    for split, d in dataset.items():
        d: Dataset | IterableDataset
        print(f"{split} -- {d.info}")
        cache_files.extend([list(f.values())[0] for f in d.cache_files])
    print("Cache Files:")
    for f in cache_files:
        print(f, "\\")


def get_dataset_hf(
    materials: ClfMaterials, streaming: bool = False, num_shards: Optional[int] = None, **kwds,
) -> DatasetDict | IterableDataset:
    datasets: dict[SplitNames, Dataset] = {}
    for split in ["tr", "vl", "ts"]:
        generator = partial(
            classification_generator,
            materials.tr_vl_ts_files_and_labels[split][0],
            materials.tr_vl_ts_files_and_labels[split][1],
            **kwds,
        )
        datasets[split] = Dataset.from_generator(generator, features=FEATURES)

    if streaming:
        num_shards = 1 if num_shards is None else num_shards
        dataset = IterableDatasetDict()
        for split in datasets:
            dataset[split] = datasets[split].to_iterable_dataset(num_shards)
    else:
        dataset = DatasetDict(datasets)

    return dataset


def test():
    ...
    # materials = get_materials_clf_bodmas(0.8, 0.1, 0.1, top_k=10)
    # dataset, id2label, label2id, dist = get_dataset_hf(materials, max_length=65536)


if __name__ == "__main__":
    test()
