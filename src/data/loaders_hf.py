"""
High-level loading API for huggingface datasets.
"""

import asyncio
from collections.abc import Generator
from functools import partial
from hashlib import md5
import multiprocessing as mp
import os
from pathlib import Path
from pprint import pformat
import shutil
import sys
from tempfile import NamedTemporaryFile
from typing import Optional
import warnings
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
from tqdm import tqdm

from src.cfg import SYSTEM
from src.enums import System
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


USE_FAST_STORAGE = True


def generator_from_zipfiles(
    files: list[ArchivedFile],
    labels: Optional[np.ndarray] = None,
    max_length: Optional[int] = None,
    preserve_order: bool = False,
) -> Generator[dict[str, str | bytes | int], None, None]:
    # If we're on SPORC, we can copy archives to /tmp before reading for decreased IO latency.
    # If all the archives fit comfortably in /tmp, we can just copy them all at once.
    # This is currently set to 1/2 TB, meaning that two of these processes can work independently
    # without wreacking havoc on that particular node, but its not guarenteed to work.
    # On SPORC, reading from /shared results in a processing speed of about ~10 samples/s.
    # Reading from /tmp using the iterative approach gives ~?? samples/s.
    # Pre-copying the archives leads to a read speed of ~200 samples/s.
    # The above stats only seem to pertain to the data used with the lift_level stuff...
    # when working with full-length binaries, the copy approach feels much slower.

    # Half a terabyte, or 500G.
    HTB = 0.50 * 2 ** 40

    # Sorted list of unique archives and their asoociated sizes.
    archives = sorted(set(af.archive for af in files))
    sizes    = [os.stat(af.archive).st_size for af in files]

    # Map from the original archive file in the input list to its real location.
    # If we do any copying, the real location will be different from the original.
    archive_path_map = {a: a for a in archives}

    # Figure out what we're going to do and set things up.
    shdcopy = SYSTEM == System.SPORC and all(s <= HTB for s in sizes) and USE_FAST_STORAGE
    precopy = None
    tmppath = None
    if shdcopy:
        precopy = sum(sizes) <= HTB
        tmppath = Path(f"/tmp/lk3591/{os.urandom(16).hex()}")
        tmppath.mkdir(parents=True)
        archive_path_map = {a: tmppath / md5(str(a).encode()).hexdigest() for a in archives}

    # Double check the integrity of the map.
    if len(set(archive_path_map.values())) != len(archive_path_map):
        raise RuntimeError("A hash collision has occurred.")

    print(f"\ngenerator_from_zipfiles: {shdcopy=} {precopy=} {len(files)=} {len(archives)=} {tmppath=}")

    # Perform the pre-copying.
    if shdcopy and precopy:
        iterable = [(src, dst) for src, dst in archives.items() if not dst.exists()]
        with mp.Pool(min(16, len(os.sched_getaffinity(0)))) as pool:
            pool.starmap(shutil.copy2, iterable)

    # Sort the archives so we can read them in a contiguous fashion.
    if not (contiguous := ArchivedFile.is_archive_list_contiguous(files)):
        warnings.warn(f"Non-contiguous files will result in slow reading. Sorting: {not preserve_order}.")
        if not preserve_order:
            files = ArchivedFile.make_archive_list_contiguous(files)
            if not ArchivedFile.is_archive_list_contiguous(files):
                raise RuntimeError("Detected non-contiguous files.")
            contiguous = True

    zp = None

    try:

        archive: str = ""
        for i, af in enumerate(files):

            # When the currently opened ZipFile does not contain the next data blob, we open up a new one.
            # If we're using the copy-system, we get the next archive location from the archive_path_map.
            # If we've got contiguous files, we can delete the temporary archive because we won't need it.
            if archive_path_map[af.archive] != archive:
                if shdcopy:
                    if os.path.isfile(archive) and os.path.exists(archive) and contiguous:
                        if os.path.dirname(archive) != str(tmppath):
                            raise RuntimeError(f"Attempting to remove perminent data: {archive=}")
                        os.unlink(archive)
                    archive = archive_path_map[af.archive]
                    if not precopy:
                        shutil.copy2(af.archive, archive)
                else:
                    archive = af.archive
                zp = zipfile.ZipFile(archive, "r")  # pylint: disable=consider-using-with

            b = zp.read(af.name)[0:max_length]
            n = af.name.split("/")[-1].split(".")[0]

            r = {"bytes": b, "name": n}
            if labels is not None and labels[i] is not None:
                r["labels"] = labels[i]

            assert all(x is not None for x in r.values())

            yield r

    finally:
        # Close the zipfile and clean up temporary directories.
        if isinstance(zp, zipfile.ZipFile):
            zp.close()
        if tmppath is not None:
            shutil.rmtree(tmppath)


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
