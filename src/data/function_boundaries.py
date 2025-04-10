"""
Tools for boundary identification of functions.

The functions for producing the map are definetly not optimized.

NOTE: The physical addresses in the disassembly files cannot always be trusted.
For example, 0007ad60610f055472513ded6bbc47130f77804dee7046a18d479409f3e2bbad, contains
sections named "UPX", which contains references to functions that are out of bounds.
"""

from __future__ import annotations
import asyncio
from array import array
from argparse import ArgumentParser
from collections import UserDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
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

    func      = None
    signature = None
    first     = None
    lower     = None
    final     = None
    upper     = None

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
            leng  = len(final[3].strip().split(" ")) if len(final) == 4 else 0
            upper = addr + leng

            bounds.append([lower, upper])

        bounds = np.array(bounds, dtype=np.uint32)
        bounds = np.reshape(bounds, (len(bounds), 2))
        return bounds

    except Exception:  # pylint: disable=broad-except
        if errors == "raise":
            try:
                print("func=\n********\n", func, "\n********\n", sep="", end="")
                print("signature=\n********\n", signature, "\n********\n", sep="", end="")
                print("first=\n********\n", first, "\n********\n", sep="", end="")
                print("lower=\n********\n", lower, "\n********\n", sep="", end="")
                print("final=\n********\n", final, "\n********\n", sep="", end="")
                print("upper=\n********\n", upper, "\n********\n", sep="", end="")
            except Exception:
                pass
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
                for s in shas:
                    if s not in data:
                        print(s)
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

    @staticmethod
    def _load_npz(file_path: Path, shas: Optional[list[str]], auto: bool) -> dict[str, np.ndarray]:
        """
        This executes in a separate *process*, so keep it self‑contained.
        """
        if auto and shas is not None:
            if file_path.name == "function_boundaries.npz":
                shas_subset = shas
            elif file_path.parent.name == "function_boundaries":
                h = file_path.stem
                shas_subset = [s for s in shas if s.startswith(h)]
            else:
                raise RuntimeError(f"Unexpected file: {file_path}")
        else:
            shas_subset = None

        raw = np.load(file_path, mmap_mode=None)

        if shas_subset is not None:
            raw = {s: raw[s] for s in shas_subset if s in raw}
        else:
            raw = dict(raw)

        return {k: np.reshape(v, (len(v), 2)) if v.shape == (0,) else v for k, v in raw.items()}

    @classmethod
    def from_files(cls, files: list[Path], shas: Optional[list[str]] = None, allow_missing_shas: bool = False, auto: bool = False) -> EXEFuncBoundsMap:
        """
        Load the function boundaries from a list of files.

        If no functions were found, the saved array might be empty, so we reshape to have two columns.
        """

        if shas is not None:
            shas = sorted(deepcopy(shas))

        data = {}
        for f in files:
            print(f"Loading {f} ... ", end="")
            t_initital = time.time()

            # If auto is True, we assume two types of files: "function_boundaries.npz" and "function_boundaries/hh.npz"
            # where hh are hex values indicating which files are contained in the archive. This basically let's us
            # slightly optimize the search within the npz file by only checking a subset of the shas.
            if auto and shas is not None:
                if f.name == "function_boundaries.npz":
                    shas_to_look_for = shas
                elif f.parent.name == "function_boundaries":
                    h = f.stem
                    shas_to_look_for = [s for s in shas if s.startswith(h)]
                else:
                    raise RuntimeError(f"Unexpected file: {f}")

            # Load the data from the file. If shas is provided, only load the data for those shas.
            # Since np.load returns a lazy map, this greatly reduces the latency and memory usage.
            # The __init__ method is responsible for ensuring that all shas are present.
            d: dict[str, np.ndarray] = np.load(f, mmap_mode=None)
            if shas is not None:
                d = {s: d[s] for s in shas_to_look_for if s in d}
            else:
                d = dict(d)

            d = {k: np.reshape(v, (len(v), 2)) if v.shape == (0,) else v for k, v in d.items()}
            data.update(d)

            t_final = time.time()
            print(f"done. Elapsed time: {round(t_final - t_initital)}s")
        return cls(data, shas, allow_missing_shas)

    @classmethod
    async def from_files_async(cls, files: list[Path], shas: Optional[list[str]] = None, allow_missing_shas: bool = False, auto: bool = False) -> EXEFuncBoundsMap:
        """
        Same as from_files, but using async IO to load the files in parallel.
        """
        # Freeze and sort the SHA list once – shared by all workers
        if shas is not None:
            shas = sorted(deepcopy(shas))

        async def _load_one(f: Path) -> dict[str, np.ndarray]:
            """Blocking file read wrapped in a thread via asyncio.to_thread."""
            t0 = time.time()

            # Decide which SHAs this file could contain
            if auto and shas is not None:
                if f.name == "function_boundaries.npz":
                    shas_subset = shas
                elif f.parent.name == "function_boundaries":
                    h = f.stem
                    shas_subset = [s for s in shas if s.startswith(h)]
                else:
                    raise RuntimeError(f"Unexpected file: {f}")
            else:
                shas_subset = None  # means “take everything”

            # ── Actual blocking work in a background thread ────────────────
            def _blocking_read() -> dict[str, np.ndarray]:
                raw = np.load(f, mmap_mode=None)

                if shas_subset is not None:
                    raw = {s: raw[s] for s in shas_subset if s in raw}
                else:
                    raw = dict(raw)  # materialise the lazy map

                # Ensure empty arrays come back with shape (0, 2)
                return {
                    k: np.reshape(v, (len(v), 2)) if v.shape == (0,) else v
                    for k, v in raw.items()
                }

            data_chunk = await asyncio.to_thread(_blocking_read)
            print(f"Loading {f} … ", end="", flush=True)
            print(f"done. Elapsed: {round(time.time() - t0)} s")
            return data_chunk

        # Kick off all reads at once
        chunks = await asyncio.gather(*(_load_one(f) for f in files))

        # Flatten the list of dicts into a single dict
        merged: dict[str, np.ndarray] = {}
        for c in chunks:
            merged.update(c)

        return cls(merged, shas, allow_missing_shas)

    @classmethod
    def from_files_multiprocessing(cls, files: list[Path], shas: Optional[list[str]] = None, allow_missing_shas: bool = False, auto: bool = False, num_workers: int = 4) -> EXEFuncBoundsMap:
        """
        Same as from_files, but using async IO to load the files in parallel.
        """
        if shas is not None:
            shas = sorted(deepcopy(shas))

        num_workers = min(num_workers, os.cpu_count())
        num_workers = max(num_workers, 1)
        print(f"Using {num_workers} worker process{'es' if num_workers > 1 else ''}.")

        t0_global = time.time()
        merged: dict[str, np.ndarray] = {}

        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = {
                pool.submit(EXEFuncBoundsMap._load_npz, f, shas, auto): f for f in files
            }

            for fut in as_completed(futures):
                f = futures[fut]
                try:
                    t0 = time.time()
                    chunk = fut.result()  # raises if the worker blew up
                    merged.update(chunk)
                    print(
                        f"{f} → {len(chunk):>5} keys "
                        f"(elapsed {round(time.time() - t0)} s)"
                    )
                except Exception as e:
                    # Surface the exact file that failed
                    raise RuntimeError(f"Worker failed on {f}: {e}") from e

        print(f"TOTAL elapsed: {round(time.time() - t0_global)} s")

        # Construct your return type (unchanged from original signature)
        return cls(merged, shas, allow_missing_shas)

    @classmethod
    def from_dataset_name(cls, dnms: Optional[list[DatasetName]] = None, shas: Optional[list[str]] = None, allow_missing_shas: bool = False, impl: Literal["naive", "asyncio", "multiprocessing"] = "multiprocessing") -> EXEFuncBoundsMap:
        dnms = dnms if dnms is not None else list(DatasetName)
        files = []
        for dnm in dnms:
            f = FUNCTION_BOUNDARIES_FILES[dnm]
            if f.is_file():
                files.append(f)
            else:
                files.extend(sorted(f.rglob("*.npz")))

        if impl == "naive":
            return cls.from_files(files, shas, allow_missing_shas, auto=True)
        if impl == "asyncio":
            return asyncio.run(cls.from_files_async(files, shas, allow_missing_shas, auto=True))
        if impl == "multiprocessing":
            return cls.from_files_multiprocessing(files, shas, allow_missing_shas, auto=True, num_workers=os.cpu_count())

        raise ValueError(f"Invalid impl: {impl}")

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


def bounds_total_slack_space(bounds: np.ndarray) -> int:
    """
    Compute the total number of bytes outside of any function boundaries (assuming
    addresses range from 0 up to the maximum 'end' value found in the bounds array).
    """
    if len(bounds) == 0:
        return 0

    sorted_bounds = bounds[np.argsort(bounds[:, 0])]

    # We'll merge overlapping ranges to find the total coverage.
    merged = []
    current_start, current_end = sorted_bounds[0]

    for i in range(1, len(sorted_bounds)):
        nxt_start, nxt_end = sorted_bounds[i]
        if nxt_start <= current_end:
            # Overlaps or touches, extend the current_end if needed
            current_end = max(current_end, nxt_end)
        else:
            # No overlap, push the previous interval and reset
            merged.append((current_start, current_end))
            current_start, current_end = nxt_start, nxt_end
    # Add the last interval
    merged.append((current_start, current_end))

    # Calculate total covered size
    covered = sum((end - start) for start, end in merged)

    # The highest address
    max_end = merged[-1][1]

    # Slack = all addresses up to max_end minus the covered addresses
    slack = max_end - covered
    return slack


def bounds_contain_totally_overlapping_functions(bounds: np.ndarray) -> bool:
    """
    Check if any function lies completely within another function.
    NOTE: this algorithm is O(N^2) and can be slow for large arrays. Use with care.
    """
    # Sort by start (and possibly by end) to simplify checks
    sorted_bounds = bounds[np.lexsort((bounds[:, 1], bounds[:, 0]))]

    # Naive O(N^2) approach: check each pair
    for i in range(len(sorted_bounds)):
        start_i, end_i = sorted_bounds[i]
        for j in range(len(sorted_bounds)):
            if i == j:
                continue
            start_j, end_j = sorted_bounds[j]
            # Check if i is contained in j (i.e. start_j <= start_i and end_i <= end_j)
            if start_j <= start_i and end_i <= end_j:
                return True
    return False


def main():
    parser = ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--inarchives", type=Path, required=True)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--subset", type=int, default=None)
    args = parser.parse_args()

    print(f"args={pformat(args.__dict__)}")

    args.outdir.mkdir(exist_ok=True)
    archives = sorted(rglob(args.inarchives, "*.zip"))
    bounds = dis_files_archives_to_exe_func_bounds_map(archives, args.num_workers)

    failed = set()
    i = None
    for i, (s, b) in enumerate(bounds.items()):
        if b is None:
            failed.add(s)
            print(f"Failed: {s}")
    bounds = {k: v for k, v in bounds.items() if k not in failed}

    print(f"Finished: {i + 1 - len(failed)} / {i + 1}")

    for i in range(256):
        h = format(i, "02x")
        f = args.outdir / f"{h}.npz"
        d = {k: v for k, v in bounds.items() if k.startswith(h)}
        np.savez_compressed(f, **d)


def test():
    ...


if __name__ == "__main__":
    main()
