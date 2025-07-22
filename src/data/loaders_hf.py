"""
High-level loading API for huggingface datasets.
"""

import asyncio
from collections.abc import Generator
from copy import deepcopy
from functools import partial
from hashlib import md5
import multiprocessing as mp
import os
from pathlib import Path
from pprint import pformat
import shutil
import sys
from tempfile import NamedTemporaryFile
import time
from typing import Any, Optional
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
    concatenate_datasets,
)
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.cfg import TMP_DIR
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
    use_fast_storage: bool = USE_FAST_STORAGE,
) -> Generator[dict[str, str | bytes | int], None, None]:
    print()

    print(f"generator_from_zipfiles: {max_length=} {preserve_order=} {use_fast_storage=}.")

    # If True, then we are going to copy the archives from one location to another.
    precopy = os.environ.get("LMLM_CAN_PRECOPY_ZIPFILES", "0") == "1" and use_fast_storage
    tmppath = TMP_DIR / os.urandom(16).hex()
    print(f"generator_from_zipfiles: {precopy=} {tmppath=}")

    # Sorted list of unique archives.
    archives = sorted(set(af.archive for af in files))

    # Map from the original archive file in the input list to its real location.
    archive_path_map = {a: a for a in archives}

    # Set up the temporary directory and the archive map to it.
    if precopy:
        tmppath.mkdir(parents=True)
        archive_path_map = {a: tmppath / md5(str(a).encode()).hexdigest() for a in archives}

    # Double check the integrity of the map.
    if len(set(archive_path_map.values())) != len(archive_path_map):
        raise RuntimeError("A hash collision has occurred.")

    # Since we may be sorting the files, we cannot rely on the ordered nature of the arrays.
    name_label_map = None
    if labels is not None:
        name_label_map: dict[str, int] = {}
        for a, l in zip(files, labels, strict=True):
            name_label_map[a.name.split(".")[0]] = l
        assert len(name_label_map) == len(files) == len(labels), f"{len(name_label_map)=} {len(files)=} {len(labels)=}"
        del labels

    # Sort the archives so we can read them in a contiguous fashion.
    contiguous = ArchivedFile.is_archive_list_contiguous(files)
    sort_when_making_archives_contiguous = None
    if not contiguous and not preserve_order:
        sort_when_making_archives_contiguous = os.environ.get("SORT_WHEN_MAKING_ARCHIVES_CONTIGUOUS", "1") == "1"
        files = ArchivedFile.make_archive_list_contiguous(files, sort=sort_when_making_archives_contiguous)
        if not ArchivedFile.is_archive_list_contiguous(files):
            raise RuntimeError("Detected non-contiguous files.")
        contiguous = True
    print(f"generator_from_zipfiles: {contiguous=} {sort_when_making_archives_contiguous=}")

    # We keep track of the order we'll encounter the archives as well as the
    # index of the archive we're currently processing.
    archives_in_order = list({af.archive: None for af in files}.keys())
    archive_idx = 0

    # The current archive and the opened ZipFile object for reading from it.
    # When the currently opened ZipFile does not contain the next data blob, we open up a new one.
    zp: Optional[zipfile.ZipFile] = None
    archive: Optional[str] = None

    t_fin: float = None

    try:

        for i, af in enumerate(files):  # pylint: disable=unused-variable

            # If the archive associated with this sample is not the one we have open, then we need to open it.
            if archive_path_map[af.archive] != archive:

                # Close the open archive before reading the next one.
                if isinstance(zp, zipfile.ZipFile):
                    zp.close()

                # If copying and contiguous, we'll never encounter this archive again, so remove it.
                # NOTE: it would probably be faster if we deleted in batches too.
                if precopy and contiguous and isinstance(archive, (str, Path)):
                    if os.path.dirname(archive) != str(tmppath):
                        raise RuntimeError(f"Attempting to remove perminent data: {archive=}")
                    os.unlink(archive)

                # If copying, copy a batch of archives to faster storage location every 256 archives.
                if precopy and archive_idx % 256 == 0:
                    archives_to_copy = archives_in_order[archive_idx: archive_idx + 256]
                    iterable = [(src, archive_path_map[src]) for src in archives_to_copy]
                    t_ini = time.time()
                    with mp.Pool(min(16, len(os.sched_getaffinity(0)))) as pool:
                        pool.starmap(shutil.copy2, iterable)
                    samples_per_second = (i / (t_ini - t_fin)) if t_fin is not None else None
                    samples_per_second = round(samples_per_second) if samples_per_second is not None else None
                    t_fin = time.time()
                    time_for_batch_copy = round(t_fin - t_ini)
                    print(f"generator_from_zipfiles: {time_for_batch_copy=}s {samples_per_second=}")

                # We set the archive variable and open it for reading.
                archive_idx += 1
                archive = archive_path_map[af.archive]
                zp = zipfile.ZipFile(archive, "r")  # pylint: disable=consider-using-with

            b = zp.read(af.name)[0:max_length]
            n = af.name.split("/")[-1].split(".")[0]

            r = {"bytes": b, "name": n}
            if name_label_map is not None and name_label_map[n] is not None:
                r["labels"] = name_label_map[n]

            assert all(x is not None for x in r.values())

            yield r

    finally:
        # Close the zipfile and clean up temporary directories.
        if isinstance(zp, zipfile.ZipFile):
            zp.close()
        if tmppath.exists():
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

        r = {"bytes": data[i], "name": str(files[i]).split("/")[-1].split(".")[0]}  # pylint: disable=use-maxsplit-arg
        if labels is not None and labels[i] is not None:
            r["labels"] = labels[i]

        assert all(x is not None for x in r.values())

        yield r


def is_dataset_empty(dataset: Dataset | IterableDataset) -> bool:
    if isinstance(dataset, Dataset):
        return dataset.num_rows == 0
    if isinstance(dataset, IterableDataset):
        try:
            next(iter(dataset))
            return False
        except StopIteration:
            return True
    if dataset is None:
        return True

    raise TypeError(type(dataset))


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


def merge_generator(
    raw: Dataset | IterableDataset,
    dis: Dataset | IterableDataset,
    dec: Dataset | IterableDataset,
    check: bool = True,
    disable_tqdm: bool = False,
) -> Generator[dict[str, str | bytes | int], None, None]:
    """
    Merges three datasets with different representations of the same binary file into a single dataset.

    Args:
        raw: The raw bytes of the binary file.
        dis: The disassembled representation of the binary file.
        dec: The decompiled representation of the binary file.
        check: Whether to check that the three datasets correspond to the same binary file.

    Notes:
        The three datasets must have the same number of samples and the same order of samples.
            If check is False and the above condition is not met, then the dataset output
            will be completely incorrect. If check is True and the above condition is not met,
            then the function will (hopefully) raise an exception.
        The output dataset will have a single feature for the `labels` key,
            no feature for the `name` key, and three features for each other key, which will be
            preceeded by either "raw_", "dis_", or "dec_" to indicate the source of the data.
            Note that the merging of the lables from each dataset will result in completely
            incorrect data if the labels are not the same for each dataset, i.e., the labels
            must be for a classification task, not a languague modeling one.
    """

    iterable = zip(raw, dis, dec)
    iterable = iterable if disable_tqdm else tqdm(iterable, desc=f"Merging Datasets ({check=})")
    for d_raw, d_dis, d_dec in iterable:

        keys: set[str]

        if check:
            name   = None
            labels = None
            keys   = None
            for d in [d_raw, d_dis, d_dec]:
                name = d["name"] if name is None else name
                if d["name"] != name:
                    raise RuntimeError(f"Mismatched names: {name} != {d['name']}")
                labels = d["labels"] if labels is None else labels
                if d["labels"] != labels:
                    raise RuntimeError(f"Mismatched labels: {labels} != {d['labels']}")
                keys = set(d.keys()) if keys is None else keys
                if set(d.keys()) != keys:
                    raise RuntimeError(f"Mismatched keys: {keys} != {set(d.keys())}")
                if "input_ids" not in keys:
                    raise RuntimeError("Missing input_ids")
        else:
            keys = set(d_raw.keys())

        keys = tuple(keys)
        new = {}
        for k in keys:
            if k == "name":
                continue
            if k == "labels":
                new[k] = d_raw[k]
                continue
            new[f"raw_{k}"] = d_raw[k]
            new[f"dis_{k}"] = d_dis[k]
            new[f"dec_{k}"] = d_dec[k]

        yield new


def merge_generator_fast(
    raw: Dataset | IterableDataset,
    dis: Dataset | IterableDataset,
    dec: Dataset | IterableDataset,
    check: bool = True,
    batch_size: int = 1024,
    disable_tqdm: bool = False,
) -> Generator[dict[str, str | bytes | int], None, None]:
    """
    Batched (fast) version of the merge_generator function.
    """

    iterable = zip(raw.iter(batch_size), dis.iter(batch_size), dec.iter(batch_size))
    iterable = iterable if disable_tqdm else tqdm(iterable, desc=f"Merging Datasets ({check=})")
    for d_raw, d_dis, d_dec in iterable:

        d_raw: dict[str, list[Any]]
        d_dis: dict[str, list[Any]]
        d_dec: dict[str, list[Any]]

        keys = None
        for d in (d_raw, d_dis, d_dec):
            if keys is None:
                keys = set(d.keys())
            else:
                if not set(d.keys()) == keys:
                    raise RuntimeError("Mismatched keys.")
        keys = tuple(keys)

        length = None
        for d in (d_raw, d_dis, d_dec):
            for k in keys:
                if length is None:
                    length = len(d[k])
                else:
                    if len(d[k]) != length:
                        raise RuntimeError("Mismatched lengths.")

        if check:
            for i in range(length):
                name = None
                labels = None
                for d in (d_raw, d_dis, d_dec):
                    if name is None:
                        name = d["name"][i]
                    if d["name"][i] != name:
                        raise RuntimeError(f"Mismatched names: {name} != {d['name'][i]}")
                    if labels is None:
                        labels = d["labels"][i]
                    if d["labels"][i] != labels:
                        raise RuntimeError(f"Mismatched labels: {labels} != {d['labels'][i]}")

        for i in range(length):
            new = {}

            for k in keys:
                if k == "name":
                    continue
                if k == "labels":
                    new[k] = d_raw[k][i]
                new[f"raw_{k}"] = d_raw[k][i]
                new[f"dis_{k}"] = d_dis[k][i]
                new[f"dec_{k}"] = d_dec[k][i]

            yield new


def merge_raw_dis_dec_datasets_slow(
    raw: DatasetDict | IterableDatasetDict,
    dis: DatasetDict | IterableDatasetDict,
    dec: DatasetDict | IterableDatasetDict,
) -> DatasetDict | IterableDataset:

    # Validate the input to make sure the datasets are of the same type.
    cl = None
    for d in [raw, dis, dec]:
        if isinstance(d, DatasetDict):
            t = Dataset
            disable_tqdm = False
        elif isinstance(d, IterableDatasetDict):
            t = IterableDataset
            disable_tqdm = True
        else:
            raise TypeError(f"Expected DatasetDict or IterableDatasetDict. Received {type(d)}")
        cl = t if cl is None else cl
        if cl != t:
            raise TypeError("Expected only DatasetDict or IterableDatasetDict. Receieved both.")

    features = None
    df = pd.DataFrame()
    if raw["tr"].features is not None:
        # End features of the output dataset.
        # No idea why I am doing this, it doesn't seem to work with the IterableDatasets.
        features = {}
        for k, v in dict(raw["tr"].features).items():
            if k == "name":
                continue
            if k == "labels":
                features[k] = v
                continue
            features[f"raw_{k}"] = v
            features[f"dis_{k}"] = v
            features[f"dec_{k}"] = v
        features = Features(features)

        # Empty dataframe for the case when one of the splits is empty.
        # No idea why I need to cast these to object types, but its the only thing that works.
        df = {}
        types = {}
        for k, v in features.items():
            if "labels" in k:
                if isinstance(v, Value):
                    df[k] = [0]
                    types[k] = object
                elif isinstance(v, Sequence):
                    df[k] = [[0]]
                    types[k] = object
                else:
                    raise TypeError(f"Expected Value or Sequence. Received {type(v)}")
            if "input_ids" in k:
                df[k] = [0]
                types[k] = object
            if "attention_mask" in k:
                df[k] = [0]
                types[k] = object
            if "token_type_ids" in k:
                df[k] = [0]
                types[k] = object
        df = pd.DataFrame(df).astype(types).drop(index=0)

    dataset = {}
    for s in raw.keys():
        if is_dataset_empty(raw[s]) or is_dataset_empty(dis[s]) or is_dataset_empty(dec[s]):
            dataset[s] = cl.from_pandas(df.copy(), features=features)
            continue
        # gen = partial(merge_generator, raw[s], dis[s], dec[s], True, disable_tqdm)
        gen = partial(merge_generator_fast, raw[s], dis[s], dec[s], True, 64, disable_tqdm)
        dataset[s] = cl.from_generator(gen, features)

    return dataset


def merge_raw_dis_dec_datasets(
    raw: DatasetDict | IterableDatasetDict,
    dis: DatasetDict | IterableDatasetDict,
    dec: DatasetDict | IterableDatasetDict,
) -> DatasetDict | IterableDataset:

    assert set(raw.keys()) == set(dis.keys()) == set(dec.keys())
    splits = tuple(raw.keys())

    dataset = {}
    for s in splits:
        assert raw[s].column_names == dis[s].column_names == dec[s].column_names, "mismatched keys"
        assert list(raw[s]["name"]) == list(dis[s]["name"]) == list(dec[s]["name"]), "mismatched names"

        # names = raw[s]["name"]
        raw[s] = raw[s].remove_columns("name")
        dis[s] = dis[s].remove_columns("name")
        dec[s] = dec[s].remove_columns("name")

        labels = raw[s]["labels"]
        raw[s] = raw[s].remove_columns("labels")
        dis[s] = dis[s].remove_columns("labels")
        dec[s] = dec[s].remove_columns("labels")

        keep_cols = [k for k in raw[s].column_names if k not in {"name"}]
        def _prefix(ds, pref):
            return ds.rename_columns({k: f"{pref}_{k}" for k in keep_cols})

        merged = [_prefix(raw[s], "raw"), _prefix(dis[s], "dis"), _prefix(dec[s], "dec")]
        merged = concatenate_datasets(merged, axis=1,)
        merged = merged.add_column("labels", labels)
        # merged = merged.add_column("name", names)

        dataset[s] = merged

    return DatasetDict(dataset)


def test():
    ...


if __name__ == "__main__":
    test()
