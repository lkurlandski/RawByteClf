"""
Utility functions.
"""

from argparse import ArgumentParser, Namespace
from collections import Counter
import csv
import bz2
import gzip
from io import BufferedReader
import logging
import lzma
import math
from pathlib import Path
from typing import ClassVar, Generator, NamedTuple, Optional
import warnings
import zlib

from datasets import (
    interleave_datasets,
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
)
from datasets.utils.logging import set_verbosity, disable_progress_bar, enable_progress_bar
from tqdm import tqdm
import py7zr

from src.data.cfg import SOREL_META_CSV, DATASET_NAMES


def print_dataset(
    dataset: Dataset | DatasetDict | IterableDataset | IterableDatasetDict,
    n: Optional[int] = None,
) -> None:
    def f(ds):
        print(f"{ds=}")
        print(f"{ds.info=}")
        if not n or n <= 0:
            return
        for i, d in enumerate(ds):
            if i >= n:
                break
            print(f"{d['name']=}|{d['size']=}|{d['length']}|{d['labels']=}")

    if isinstance(dataset, DatasetDict) or isinstance(dataset, IterableDatasetDict):
        for ds in dataset:
            f(dataset[ds])
    else:
        f(dataset)


class PerDatasetArgumentParser(ArgumentParser):
    shortcuts: ClassVar[dict[str, list[str]]] = {
        "bodmas": ["bodmas_pe"],
        "local": ["local_pe", "local_elf", "local_macho"],
        "malware_bazaar": [
            "malware_bazaar_dll",
            "malware_bazaar_elf",
            "malware_bazaar_exe",
            "malware_bazaar_macho",
        ],
        "sorel": ["sorel_pe"],
        "virus_share": [
            "virus_share_dll",
            "virus_share_elf",
            "virus_share_exe",
            "virus_share_macho",
        ],
        "virus_total": [
            "virus_total_dll",
            "virus_total_elf",
            "virus_total_exe",
            "virus_total_macho",
        ],
    }

    def __init__(self) -> None:
        super().__init__()
        self.add_argument(
            "--datasets",
            nargs="*",
            default=["all"],
            choices=["all"] + list(self.shortcuts.keys()) + DATASET_NAMES,
        )

    def parse_args(self, args=None, namespace=None) -> Namespace:
        args = super().parse_args()

        datasets = []
        if "all" in args.datasets:
            datasets = DATASET_NAMES

        for k, v in self.shortcuts.items():
            if k in args.datasets:
                datasets.extend(v)

        ignore = ["all"] + list(self.shortcuts.keys())
        datasets.extend([d for d in args.datasets if d not in ignore])
        datasets = list(sorted(set(datasets)))
        args.datasets = datasets
        return args


class SorelSample(NamedTuple):
    sha256: str
    is_malware: bool
    rl_fs_t: float
    rl_ls_const_positives: int
    adware: bool
    flooder: bool
    ransomware: bool
    dropper: bool
    spyware: bool
    packed: bool
    crypto_miner: bool
    file_infector: bool
    installer: bool
    worm: bool
    downloader: bool


def stream_sorel_meta(meta: Path = SOREL_META_CSV) -> Generator[SorelSample, None, None]:
    with open(meta, "r", encoding="utf-8") as file:
        csv_reader = csv.reader(file)
        _ = next(csv_reader, None)
        for row in csv_reader:
            sample = SorelSample(
                sha256=row[0],
                is_malware=bool(int(row[1])),
                rl_fs_t=float(row[2]),
                rl_ls_const_positives=int(row[3]),
                adware=bool(int(row[4])),
                flooder=bool(int(row[5])),
                ransomware=bool(int(row[6])),
                dropper=bool(int(row[7])),
                spyware=bool(int(row[8])),
                packed=bool(int(row[9])),
                crypto_miner=bool(int(row[10])),
                file_infector=bool(int(row[11])),
                installer=bool(int(row[12])),
                worm=bool(int(row[13])),
                downloader=bool(int(row[14])),
            )

            yield sample


def decompress_fp(fp: BufferedReader) -> bytes:
    SIG_GZIP = b"\x1f\x8b\x08"
    SIG_BZIP2 = b"\x42\x5a\x68"
    SIG_LZMA = b"\xfd7zXZ\x00"
    SIG_ZLIB = b"\x78\x01"
    SIG_7Z = b"7z"

    fp.seek(0)
    signature = fp.read(10)
    fp.seek(0)

    if signature.startswith(SIG_GZIP):
        with gzip.open(fp, "rb") as compressed_file:
            return compressed_file.read()

    if signature.startswith(SIG_BZIP2):
        with bz2.BZ2File(fp, "rb") as compressed_file:
            return compressed_file.read()

    if signature.startswith(SIG_LZMA):
        with lzma.open(fp, "rb") as compressed_file:
            return compressed_file.read()

    if signature.startswith(SIG_ZLIB):
        return zlib.decompress(fp.read())

    if signature.startswith(SIG_7Z):
        with py7zr.SevenZipFile(fp, mode="r") as archive:
            file_list = archive.getnames()
            if len(file_list) != 1:
                raise ValueError("The 7zip archive does not contain a single file.")
            return archive.read(file_list[0])

    fp.seek(0)
    return fp.read()


def decompress(
    file_or_file_pointer: str | Path | bytes | BufferedReader, outfile: Optional[Path] = None
) -> bytes:
    if isinstance(file_or_file_pointer, (str, Path, bytes)):
        with open(file_or_file_pointer, "rb") as fp:
            b = decompress_fp(fp)
    else:
        b = decompress_fp(file_or_file_pointer)

    if outfile:
        with open(outfile, "rb") as fp:
            fp.write(b)

    return b


def balance_imbalanced_dataset(
    dataset: Dataset | IterableDataset,
    dist: Counter,
    smoothing_factor: float = 1,
    check: bool = True,
) -> tuple[IterableDataset | Dataset, Counter]:
    """Oversample a dataset to balance the classes.

    Args:
        dataset: The dataset to balance.
        dist: The class distribution of the dataset.
        smoothing_factor: The amount of smoothing to apply when oversampling.
            1 means the classes will be fully balanced.

    Returns:
        A tuple of the balanced dataset and its class distribution.
    """

    warnings.warn(
        "This function adds and indices mapping to the dataset, making iterating over it slow."
        "The dataset will have to be reindexed using flatten_indices() to be useful."
    )

    assert 0 < smoothing_factor <= 1, f"{smoothing_factor=} must be between 0 and 1."
    if not isinstance(dataset, IterableDataset):
        warnings.warn("The dataset is not an IterableDataset, so this might take a long time...")

    id2label = {i: l for i, l in enumerate(dataset.info.features["labels"].names)}
    label2id = {l: i for i, l in enumerate(id2label.values())}

    # Extract one dataset for each class
    # It is critical for the datasets to have non null features, otherwise the
    # interleave_datasets function will take hours to complete, ie, .cast()
    set_verbosity(logging.CRITICAL)
    disable_progress_bar()
    datasets = []
    for l in tqdm(list(dist.keys()), desc="Filtering..."):
        label_or_id = (l, label2id.get(l))
        d = dataset.filter(
            lambda exs: [e in label_or_id for e in exs["labels"]], batched=True
        ).cast(dataset.features)
        datasets.append(d)
    set_verbosity(logging.INFO)
    enable_progress_bar()

    if check:
        for d in tqdm(datasets, desc="Checking..."):
            if isinstance(dataset, Dataset):
                assert len(d) > 0, "Some classes have no samples."
            elif isinstance(dataset, IterableDataset):
                assert bool(next(iter(d), False)), "Some classes have no samples."

    probabilities = class_probabilities_with_smoothing(dist, smoothing_factor)

    f = max(dist.values()) / max(probabilities)
    new_dist = Counter({l: int(v * p * f) for (l, v), p in zip(dist.items(), probabilities)})

    print(f"Interleaving...")
    dataset = interleave_datasets(datasets, probabilities, stopping_strategy="all_exhausted")

    return dataset, new_dist


def class_probabilities_with_smoothing(
    dist: Counter, smoothing_factor: float = 1
) -> dict[str, float]:
    ratio = {k: (1 / math.pow(l, smoothing_factor)) for k, l in dist.items()}
    s = sum(ratio.values())
    return {k: v / s for k, v in ratio.items()}
