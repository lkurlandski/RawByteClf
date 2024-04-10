"""
Utility functions.
"""

from argparse import ArgumentParser, Namespace
import asyncio
from copy import deepcopy
import csv
import bz2
import gzip
from io import BufferedReader
from itertools import islice
import lzma
import math
import os
from pathlib import Path
import sys
import time
from typing import ClassVar, Generator, NamedTuple, Literal, Optional
import warnings
import zlib

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
)
import numpy as np
import torch
from torch import ByteTensor
from tqdm import tqdm
import py7zr

from src.utils import batched
from src.data.cfg import SOREL_META_CSV, DATASET_NAMES, DATASET_TO_FILES


DEFAULT_ASYNCH_CHUNK_SIZE = 500000
DEFAULT_DISABLE_TQDM = True


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

    if isinstance(dataset, (DatasetDict, IterableDatasetDict)):
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


class Decompressor:
    # Signatures, for specific compression methods.
    SIG_GZIP = b"\x1f\x8b\x08"
    SIG_BZIP2 = b"\x42\x5a\x68"
    SIG_LZMA = b"\xfd7zXZ\x00"
    SIG_ZLIB = b"x\x9c"
    SIG_7Z = b"7z"

    # Return codes for specific compression methods.
    NONE = 0
    GZIP = 1
    BZIP2 = 2
    LZMA = 3
    ZLIB = 4
    S7Z = 5

    def __init__(self, alg: Optional[int] = None, must_decompress: bool = False) -> None:
        """A universal decompression utility.

        Args:
            alg (Optional[int], optional): An optional integer indicating the expected compression
                algorithm of input data. Defaults to None.
            must_decompress (bool, optional): If True, raises a RuntimeError if the data cannot be
                decompressed using a known algorithm. Defaults to False.
        """
        if alg == Decompressor.S7Z:
            raise NotImplementedError("7z decompression is not supported yet.")
        self.alg = alg
        self.must_decompress = must_decompress

    def __call__(self, data: str | Path | BufferedReader | bytes, outfile: Optional[Path] = None) -> tuple[int, bytes]:
        if isinstance(data, (str, Path)):
            with open(data, "rb") as fp:
                alg, b = self.decompress(fp)
        elif isinstance(data, bytes):
            fp = BufferedReader(data)
            alg, b = self.decompress(fp)
        else:
            fp = data
            alg, b = self.decompress(fp)

        if outfile:
            with open(outfile, "rb") as fp:
                fp.write(b)

        return alg, b

    def decompress(self, fp: BufferedReader) -> tuple[int, bytes]:
        fp.seek(0)
        signature = fp.read(10)
        fp.seek(0)

        # Try to detect specific signatures and decompress using a targeting method.
        if (self.alg is None or self.alg == Decompressor.GZIP) and signature.startswith(Decompressor.SIG_GZIP):
            return Decompressor._gzip(fp)
        if (self.alg is None or self.alg == Decompressor.BZIP2) and signature.startswith(Decompressor.SIG_BZIP2):
            return Decompressor._bzip2(fp)
        if (self.alg is None or self.alg == Decompressor.LZMA) and signature.startswith(Decompressor.SIG_LZMA):
            return Decompressor._lzma(fp)
        if (self.alg is None or self.alg == Decompressor.ZLIB) and signature.startswith(Decompressor.SIG_ZLIB):
            return Decompressor._zlib(fp)
        if (self.alg is None or self.alg == Decompressor.S7Z) and signature.startswith(Decompressor.SIG_7Z):
            return Decompressor._py7zr(fp)

        # Raise an error if a specific algorithm is requested but the data does not match the signature.
        if self.alg is not None:
            raise RuntimeError(f"Could not decompress the data using {self.alg=}")

        # Brute-force decompression methods if all else fails.
        fns = [
            Decompressor._bzip2,
            Decompressor._gzip,
            Decompressor._lzma,
            Decompressor._zlib,
            # Decompressor._py7zr,  # "7z decompression is not supported yet.")
        ]
        for fn in fns:
            try:
                return fn(fp)
            except Exception as err:
                print(err)
                fp.seek(0)

        if self.must_decompress:
            raise RuntimeError("Could not decompress the data using any method.")

        # Return the original file if no decompression method works.
        fp.seek(0)
        return Decompressor.NONE, fp.read()

    @staticmethod
    def _gzip(fp: BufferedReader) -> tuple[int, bytes]:
        with gzip.open(fp, "rb") as compressed_file:
            return Decompressor.GZIP, compressed_file.read()

    @staticmethod
    def _bzip2(fp: BufferedReader) -> tuple[int, bytes]:
        with bz2.BZ2File(fp, "rb") as compressed_file:
            return Decompressor.BZIP2, compressed_file.read()

    @staticmethod
    def _lzma(fp: BufferedReader) -> tuple[int, bytes]:
        with lzma.open(fp, "rb") as compressed_file:
            return Decompressor.LZMA, compressed_file.read()

    @staticmethod
    def _zlib(fp: BufferedReader) -> tuple[int, bytes]:
        return Decompressor.ZLIB, zlib.decompress(fp.read())

    @staticmethod
    def _py7zr(fp: BufferedReader) -> tuple[int, bytes]:
        raise NotImplementedError("7z decompression is not supported yet.")
        with py7zr.SevenZipFile(fp, mode="r") as archive:
            file_list = archive.getnames()
            if len(file_list) != 1:
                raise ValueError("The 7zip archive does not contain a single file.")
        return Decompressor.S7Z, archive.read(file_list[0])


def read_binary_file(
    f: Path,
    max_length: Optional[int] = None,
    in_memory_dtype: Literal["bytes", "np", "pt"] = "bytes",
) -> bytes | np.ndarray | ByteTensor:
    """
    Will warn with "UserWarning: The given buffer is not writable...", which can
    safely be ignored because we don't care about modifying the bytes object.
    """
    with open(f, "rb") as fp:
        b = fp.read(max_length)

    if in_memory_dtype == "bytes":
        return b
    if in_memory_dtype == "np":
        return np.frombuffer(b, dtype=np.uint8)
    if in_memory_dtype == "pt":
        return torch.frombuffer(b, dtype=torch.uint8)

    raise ValueError(f"Unknown {in_memory_dtype=}")


async def read_binary_file_asynch(
    f: Path,
    max_length: Optional[int] = None,
    in_memory_dtype: Literal["bytes", "np", "pt"] = "bytes",
) -> bytes | np.ndarray | ByteTensor:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, read_binary_file, f, max_length, in_memory_dtype)


async def read_binary_files_asynch(
    files: list[str],
    max_length: Optional[int] = None,
    in_memory_dtype: Literal["bytes", "np", "pt"] = "bytes",
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
    for batch_files in iterable:
        tasks = [read_binary_file_asynch(f, max_length, in_memory_dtype) for f in batch_files]
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


def time_decompressor(n: int = 10000):
    files = islice(DATASET_TO_FILES["binaries"]["sorel_pe"](), n)
    decompress = Decompressor(Decompressor.ZLIB)

    sizes = []
    times = []
    for f in tqdm(files, total=n):
        t_i = time.time()
        try:
            _, b = decompress(f)
        except Exception as err:
            continue
        t_f = time.time()
        times.append(t_f - t_i)
        sizes.append((f.stat().st_size, len(b)))

    print(f"Num files: {len(sizes)=}")
    print(f"Average decompression time: {np.mean(times)}")
    print(f"Average compressed size: {np.mean(s[0] for s in sizes)}")
    print(f"Average uncompressed size: {np.mean(s[1] for s in sizes)}")


if __name__ == "__main__":
    time_decompressor(100000)
