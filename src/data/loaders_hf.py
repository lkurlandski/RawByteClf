"""
High-level loading API for huggingface datasets.
"""

import asyncio
from collections.abc import Generator
from functools import partial
import os
from pathlib import Path
from pprint import pformat
import sys
from typing import Optional
import zipfile

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
    Sequence,
)
import numpy as np
import pandas as pd

from src.data.loaders_core import (
    Materials,
    ArchivedFile,
    SplitNames,
)
from src.data.utils import read_binary_files_asynch, read_binary_files


FEATURES_CLM = Features({"name": Value("string"), "bytes": Value("binary")})
FEATURES_CLF = Features({"name": Value("string"), "bytes": Value("binary"), "labels": Value("int32")})
FEATURES_CLF_MULTILABEL = Features({"name": Value("string"), "bytes": Value("binary"), "labels": Sequence(Value("int32"))})
DF_CLM = pd.DataFrame({"name": [""], "bytes": [b""]}).drop(index=0)
DF_CLF = pd.DataFrame({"name": [""], "bytes": [b""], "labels": [0]}).drop(index=0)
DF_CLF_MULTILABEL = pd.DataFrame({"name": [""], "bytes": [b""], "labels": [[0]]}).drop(index=0)


def generator_from_zipfiles(
    files: list[ArchivedFile],
    labels: Optional[np.ndarray] = None,
    max_length: Optional[int] = None,
) -> Generator[dict[str, str | bytes | int], None, None]:

    zp = None

    try:

        archive: str = ""
        for i, archived_file in enumerate(files):

            if archived_file.archive != archive:
                archive = archived_file.archive
                zp = zipfile.ZipFile(archive, "r")  # pylint: disable=consider-using-with

            b = zp.read(archived_file.name)[0:max_length]
            n = archived_file.name.split("/")[-1].split(".")[0]

            r = {"bytes": b, "name": n}
            if labels is not None and labels[i] is not None:
                r["labels"] = labels[i]

            assert all(x is not None for x in r.values())

            yield r

    finally:
        if isinstance(zp, zipfile.ZipFile):
            zp.close()


def generator(
    files: list[os.PathLike],
    labels: Optional[np.ndarray] = None,
    max_length: Optional[int] = None,
    asynch: bool = True,
    asynch_chunk_size: int = 500000,
) -> Generator[dict[str, str | bytes | int], None, None]:
    kwds = {"max_length": max_length, "in_memory_dtype": "bytes", "disable_tqdm": True}

    data: Optional[bytes] = [None for _ in range(len(files))]
    for i in range(len(files)):  # pylint: disable=consider-using-enumerate
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


def print_dataset_hf(dataset: DatasetDict | IterableDatasetDict, n_samples: int = 0):

    print(f"Dataset: {dataset.__class__.__name__}")

    for split, d in dataset.items():
        d: Dataset | IterableDataset
        print(f" {split=}")
        print(f"  splits={d.info.splits}")
        print(f"  features={d.info.features}")

        if n_samples > 0:
            for i, items in enumerate(d):
                fields = list(items.keys())
                types = [type(v) for v in items.values()]
                shapes = []
                for v in items.values():
                    if hasattr(v, "shape"):
                        shape = v.shape
                    elif isinstance(v, list):
                        shape = len(v)
                    else:
                        shape = None
                    shapes.append(shape)
                print(f"  row={i}  {fields=}  {types=}  {shapes=}")

                if i == n_samples - 1:
                    break

    if isinstance(dataset, DatasetDict):
        files = [list(f.values())[0] for f in d.cache_files for d in dataset.values()]
        print(" Cache Files: [")
        for f in files:
            print(f"  {str(f)},")
        print(" ]")


# TODO: treat the train and test sets differently because we don't really need an iterable test set.
def get_dataset_hf(
    materials: Materials,
    streaming: bool = False,
    num_shards: Optional[int] = None,
    **kwds,
) -> DatasetDict | IterableDataset:
    """
    Should the streaming version first generate the Dataset from the raw files then call
    to_iterable_dataset? Or should it just generate the dataset from the raw files? The former
    will get the asycnhronous file-reading out of the way before the training loop and before the
    DataLoader multiprocessing phase. The latter will have the asynchrounous file-reading happen
    during the training loop such that multiple processes are using different asyncio event loops.

    This would look something like the below:
    """
    if materials.problem_type is None:
        features = FEATURES_CLM
        df = DF_CLM
    elif materials.problem_type == "single_label_classification":
        features = FEATURES_CLF
        df = DF_CLF
    elif materials.problem_type == "multi_label_classification":
        features = FEATURES_CLF_MULTILABEL
        df = DF_CLF_MULTILABEL
    else:
        raise RuntimeError(f"Unknown problem type: {materials.problem_type}")

    datasets: dict[SplitNames, Dataset] = {}
    for split in ["tr", "vl", "ts"]:
        if not materials.files[split]:  # Empty split for datasets > 2.14
            datasets[split] = Dataset.from_pandas(df.copy(), features=features)
            continue

        kwds["files"] = materials.files[split]
        if materials.labels is not None:
            kwds["labels"] = materials.labels[split]

        if isinstance(materials.files[split][0], ArchivedFile):
            gen = generator_from_zipfiles
        elif isinstance(materials.files[split][0], (str, Path)):
            gen = generator

        datasets[split] = Dataset.from_generator(gen, features=features, gen_kwargs=kwds)

    if streaming:
        if num_shards is None or num_shards == 0:
            num_shards = 1
        dataset = IterableDatasetDict()
        for split in ["tr", "vl", "ts"]:
            if len(datasets[split]) == 0:
                continue
            dataset[split] = datasets[split].to_iterable_dataset(num_shards)
    else:
        dataset = DatasetDict(datasets)

    return dataset

    # num_shards = 1 if num_shards is None else num_shards
    # datasets: dict[SplitNames, IterableDataset] = {}
    # for split in ["tr", "vl", "ts"]:
    #     # if not materials.files[split]:  # Empty split for datasets > 2.14
    #     #     datasets[split] = Dataset.from_pandas(df.copy(), features=features)
    #     #     continue
    #     generator = partial(
    #         classification_generator,
    #         materials.files[split],
    #         materials.labels[split] if materials.labels is not None else None,
    #         **kwds,
    #     )
    #     gen_kwargs={"shards": shards}
    #     datasets[split] = IterableDataset.from_generator(generator, features=features)
    # return DatasetDict(datasets)


def test():
    ...
    # materials = get_materials_clf_bodmas(0.8, 0.1, 0.1, top_k=10)
    # dataset, id2label, label2id, dist = get_dataset_hf(materials, max_length=65536)


if __name__ == "__main__":
    test()
