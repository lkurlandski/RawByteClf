"""
Tools for boundary identification of functions.

The functions for producing the map are definetly not optimized.
"""

from __future__ import annotations
from array import array
from argparse import ArgumentParser
from collections import UserDict
from functools import partial
from itertools import islice
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
from pprint import pprint, pformat
import re
import sys
import time
from typing import Optional, Iterable, Literal
import zipfile

import numpy as np
from tqdm import tqdm

# pylint: disable=wrong-import-position
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# pylint: enable=wrong-import-position

from src.utils import rglob
from src.data.cfg import FUNCTION_BOUNDARIES_FILES, DatasetName
from src.data.utils import get_data_from_archives


def dis_file_to_exe_func_bounds(content: Path | str | bytes, errors: Literal["raise", "pass"] = "raise") -> np.ndarray:
    """
    Take a disassembly file and return the function boundaries with respect to the original executable file.

    If no functions were found, the saved array is empty, so we reshape to have two columns.
    """
    if errors not in {"raise", "pass"}:
        raise ValueError(f"Invalid value for errors: {errors}")

    try:

        if isinstance(content, Path):
            text = content.read_text()
        elif isinstance(content, bytes):
            text = content.decode("utf-8", errors="ignore")
        elif isinstance(content, str):
            text = content
        else:
            raise TypeError(f"Invalid type for content: {type(content)}")

        bounds = []
        for func in text.split("\n\n"):
            func = func.strip()
            lines = func.split("\n")

            signature = lines[0]  # pylint: disable=unused-variable
            body = lines[1:]
            if not body:
                continue

            first = body[0].split("\t")
            lower = int(first[1].strip(), 16)

            final = body[-1].split("\t")
            addr  = int(final[1].strip(), 16)
            leng  = len(final[3].strip().split(" "))
            upper = addr + leng

            bounds.append([lower, upper])

        bounds = np.array(bounds, dtype=np.uint32)
        bounds = np.reshape(bounds, (len(bounds), 2))
        return bounds

    except Exception:  # pylint: disable=broad-except
        if errors == "raise":
            raise
        return None


def dis_files_archive_to_exe_func_bounds_map(file: Path) -> dict[str, np.ndarray]:
    data = {}
    for name, content in get_data_from_archives([file], names=True, contents=True):
        bounds = dis_file_to_exe_func_bounds(content, errors="pass")
        sha = name.split(".")[0]
        data[sha] = bounds
    return data


def dis_files_archives_to_exe_func_bounds_map(files: Iterable[Path], num_workers: Optional[int] = 1, disable_tqdm: bool = False) -> dict[str, np.ndarray]:
    files = files if disable_tqdm else tqdm(files)

    if num_workers > 1:
        with mp.Pool(num_workers) as pool:
            data_per_archive = list(pool.imap(dis_files_archive_to_exe_func_bounds_map, files))
    else:
        data_per_archive = map(dis_files_archive_to_exe_func_bounds_map, files)

    data = {}
    for d in data_per_archive:
        data.update(d)
    return data


class EXEFuncBoundsMap(UserDict):
    """
    This is slow as fuck for a large number of samples. Using memory mapping does not seem to help.
    """

    def __init__(self, data: dict[str, np.ndarray] = None, shas: Optional[list[str]] = None, allow_missing_shas: bool = False) -> None:

        if shas is not None:
            shas = set(shas)
            data = {k: v for k, v in data.items() if k in shas}
            if not allow_missing_shas and len(data) != len(shas):
                raise ValueError(f"Missing {len(shas) - len(data)} shas!")

        for k, v in data.items():
            if not isinstance(k, str):
                raise TypeError(f"Invalid type for key: {type(k)}")
            if not isinstance(v, np.ndarray):
                raise TypeError(f"Invalid type for value: {type(v)}")
            if v.ndim != 2 or v.shape[1] != 2:
                raise ValueError(f"Invalid shape for value: {v.shape}")
            if v.dtype != np.uint32:
                raise ValueError(f"Invalid dtype for value: {v.dtype}")

        super().__init__(data)

    @classmethod
    def from_files(cls, files: list[Path], shas: Optional[list[str]] = None, allow_missing_shas: bool = False) -> EXEFuncBoundsMap:
        """
        Load the function boundaries from a list of files.

        If no functions were found, the saved array might be empty, so we reshape to have two columns.
        """

        data = {}
        for f in files:
            print(f"Loading {f} ... ", end="")
            t_initital = time.time()

            # Load the data from the file. If shas is provided, only load the data for those shas.
            # Since np.load returns a lazy map, this greatly reduces the latency and memory usage.
            # The __init__ method is responsible for ensuring that all shas are present.
            d: dict[str, np.ndarray] = np.load(f, mmap_mode=None)
            if shas is not None:
                d = {s: d[s] for s in shas if s in d}
            else:
                d = dict(d)

            d = {k: np.reshape(v, (len(v), 2)) if v.shape == (0,) else v for k, v in d.items()}
            data.update(d)

            t_final = time.time()
            print(f"done. Elapsed time: {round(t_final - t_initital)}s")
        return cls(data, shas, allow_missing_shas)

    @classmethod
    def from_dataset_name(cls, dnms: Optional[list[DatasetName]] = None, shas: Optional[list[str]] = None, allow_missing_shas: bool = False) -> EXEFuncBoundsMap:
        dnms = dnms if dnms is not None else list(DatasetName)
        files = [FUNCTION_BOUNDARIES_FILES[dnm] for dnm in dnms]
        return cls.from_files(files, shas, allow_missing_shas)

    def get_stats(self, r: Optional[int] = None) -> dict[str, float]:
        """
        Get statistics about the function boundaries.

        On samples from the ass, bod, win, and a subset of samples from the sor corpus totalling to 150K:
          fun-num-med: 39
          fun-num-men: 220
          fun-num-std: 653274
          fun-len-med: 74
          fun-len-men: 864
          fun-len-std: 3111
        """

        d = {}

        num_funs = np.array([len(b) for b in self.values()])
        d["fun-num-med"] = np.median(num_funs)
        d["fun-num-men"] = np.mean(num_funs)
        d["fun-num-std"] = np.std(num_funs)

        all_bounds = np.concatenate(list(self.values()), axis=0)
        lengths = all_bounds[:, 1] - all_bounds[:, 0]
        d["fun-len-med"] = np.median(lengths)
        d["fun-len-men"] = np.mean(lengths)
        d["fun-len-std"] = np.std(lengths)

        if r is not None:
            d = {k: round(v, r) for k, v in d.items()}

        return d


def main():
    parser = ArgumentParser()
    parser.add_argument("--outfile", type=Path, required=True)
    parser.add_argument("--inarchives", type=Path, required=True)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--subset", type=int, default=None)
    args = parser.parse_args()

    print(f"args={pformat(args.__dict__)}")

    if args.outfile.suffix != ".npz":
        raise ValueError(f"Invalid extension for outfile: {args.outfile.suffix}. Expected: `.npz`.")

    archives = sorted(rglob(args.inarchives, "*.zip"))
    bounds = dis_files_archives_to_exe_func_bounds_map(archives, args.num_workers)

    failed = set()
    i = None
    for i, (s, b) in enumerate(bounds.items()):
        if b is None:
            failed.add(s)

    print(f"Finished: {i + 1 - len(failed)} / {i + 1}")

    np.savez_compressed(args.outfile, **bounds)


def test():
    ...


if __name__ == "__main__":
    main()
