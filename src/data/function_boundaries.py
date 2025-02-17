"""
Tools for boundary identification of functions.

The functions for producing the map are definetly not optimized.
"""

from array import array
from argparse import ArgumentParser
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

        return np.array(bounds, dtype=np.uint32)

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


def get_exe_func_bounds_map(
    dnm: Optional[str | DatasetName] = None,
    shas: Optional[list[str]] = None,
    allow_missing_shas: bool = False,
) -> dict[str, np.ndarray]:
    dnms = [DatasetName(dnm)] if dnm is not None else list(DatasetName)

    data = {}
    for dnm in dnms:  # pylint: disable=redefined-argument-from-local
        d = np.load(FUNCTION_BOUNDARIES_FILES[dnm])
        data.update(dict(d))

    if shas is not None:
        shas = set(shas)
        data = {k: v for k, v in data.items() if k in shas}
        if not allow_missing_shas and len(data) != len(shas):
            raise ValueError(f"Missing {len(shas) - len(data)} shas!")

    return data


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
