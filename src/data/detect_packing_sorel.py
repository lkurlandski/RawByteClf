"""
Detect whether executable samples are packed or not using Detect-It-Easy (DiE).

Usage
-----
1. Prepare the script for embarassingly parallel execution by creating a list of shas
for each worker to process. Use either the filter_mode or num_shards method of splitting
the SOREL collection across workers (--filter_mode=4 is recommended for load balancing).

    python detect_packing_sorel.py --dataset=DATASET --prepare --filter_mode=FILTER_MODE

or

    python detect_packing_sorel.py --dataset=DATASET --prepare --num_shards=NUM_SHARDS

2. Run the script to download, analyze, and save the results for each sample.

The SOREL binaries will be temporarily saved in the P_DOWNLOAD directory, but will be
deleted after processing each file. Datasets like BODMAS will not be downloaded and/or
deleted because they are assumed to already exist on disk.

    | -- P_ROOT
        | -- P_DOWNLOAD
            | -- sha.exe
            ...

You can run the program for every file at once in a single process (no parallelism)

    python detect_packing_sorel.py --dataset=DATASET --run

or split the load across the individual workers using the sharded approach you prepared for
in step 1 (recommended).

    python detect_packing_sorel.py --dataset=DATASET --run --filter_idx=FILTER_IDX

or

    python detect_packing_sorel.py --dataset=DATASET --run --shard_idx=SHARD_IDX

Its recommended to use a tool like GNU-parallel to run all shards at once, e.g.,

    parallel --bar -j 48 'python src/data/detect_packing_sorel.py --dataset=DATASET --run --filter_idx={1} --filter_mode=4 > ./logs/packing_4_{1}.log 2>&1' ::: $(printf "%04x\n" {0..65535})

or

    parallel --bar -j 48 'python src/data/detect_packing_sorel.py --dataset=DATASET --run --filter_idx={1} --filter_mode=2 > ./logs/packing_2_{1}.log 2>&1' ::: $(printf "%02x\n" {0..255})

or

    parallel --bar -j 16 'python src/data/detect_packing_sorel.py --dataset=DATASET --run --shard_idx={1} --num_shards=16 > ./logs/packing_16_{1}.log 2>&1' ::: {0..15}

Either way, this will produce a set of JSON-ish files for each sample in the dataset

    | -- P_ROOT
        | -- P_RAW
            | -- recursive
                    | -- 0
                        | -- sha.txt
                        ...
                    ...
                    | -- f
                        | -- sha.txt
                        ...
                ...
            | -- deep
                ...
            | -- heuristic
                ...

We use subdirectories 0, 1, 2, ..., e, f to prevent Errno 28, which can occur when a directory
contains so many files that the OS runs in hash collision issues. Notably, both diec's stdout
and stderr are piped to the same output file. If an error occurs, the file is not valid JSON,
hence our decision to use the .txt extension instead of .json.

If you want DO NOT want to ignore files that have already been processed (process them again),
then use the --dont_ignore_complete flag.

    python detect_packing_sorel.py --dataset=DATASET --run --dont_ignore_complete

3. After all the samples have been processed, merge the results from each mode of DIEC
into a single JSON file for each sample.

    python detect_packing_sorel.py --dataset=DATASET --merge

This will produce one JSON files for each sample in the SOREL collection

    | -- P_ROOT
        | -- P_MERGED
            | -- 0
                | -- sha.json
                ...
            ...
            | -- f
                | -- sha.json
                ...

4. Finally, consolidate the results into a single file for the entire collection
for usage in downstream ML tasks.

    python detect_packing_sorel.py --dataset=DATASET --consolidate_partials
    python detect_packing_sorel.py --dataset=DATASET --consolidate_final


This will produce a single JSON file for the entire SOREL collection (and some temporary files).

    | -- P_ROOT
        | -- P_CONSOLIDATED
            | -- output.json
            | -- tmp_0.json
            ...
"""

from argparse import ArgumentParser
from collections.abc import Iterable
from collections import UserDict
from functools import partial
import gc
import hashlib
from itertools import chain, islice
from io import BytesIO
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
from pprint import pformat, pprint
import subprocess
import sys
import tempfile
import time
from typing import Generator, Literal, Optional, Protocol

# pylint: disable=wrong-import-position
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# pylint: enable=wrong-import-position

import psutil
from tqdm import tqdm

from src.data.cfg import (
    SOREL_PATH,
    DATASET_TO_FILES,
    SOREL_BUCKET,
    SOREL_PREFIX,
    PACKING_ROOTS,
)
from src.data.prepare_datasets import s3_dataset_generator
from src.data.utils import stream_sorel_meta, Decompressor


HEX = tuple(hex(i)[2:] for i in range(16))
DIEC_MODES = ("recursive", "deep", "heuristic")
DiecMode = Literal["recursive", "deep", "heuristic"]


ALL_TYPES = [
    "Linker",
    "Compiler",
    "Tool",
    "Format",
    "Packer",
    "Sign",
    "Certificate",
    "Protection",
    "Library",
    "Data",
    "Installer",
    "Protector",
    "Cryptor",
    "Virus",
    "sfx",
    "source",
    "Archive",
    "Image",
    "patcher",
    "GameEngine",
    "Player",
    "Crypter",
    "Joiner",
    "Converter",
    "audio",
    "scrambler",
    "emulator",
    "script",
    "camera",
    "other",
    "debug",
    "extender",
    "keygen",
]
OBFUSCATION_TYPES = [
    "Packer",
    "Protector",
    "Protection",
    "Crypter",
    "Cryptor",
    "patcher",
    "scrambler",
    "sfx",
    "Archive",
    "Joiner",
]
assert all(t in ALL_TYPES for t in OBFUSCATION_TYPES), "Stupid."
NAME_TO_GENERIC = (
    "Cryptor detected",
    "Packer detected",
)


def find_files_with_null(directory: Path | str) -> list[os.PathLike]:
    command = ['grep', '-l', '-r', 'null', str(directory)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as err:
        print(f"{err.stdout=}\n{err.stderr=}")
        raise err
    file_paths = result.stdout.splitlines()
    return file_paths


class ShasStreamer(Protocol):

    def __call__(self) -> Generator[str, None, None]:
        ...


def sorel_shas() -> Generator[str, None, None]:
    for s in stream_sorel_meta():
        yield s.sha256


def basal_shas(name: str) -> Generator[str, None, None]:
    for f in DATASET_TO_FILES["binaries"][name]():
        yield f.stem


def assemblage_shas() -> Generator[str, None, None]:
    return basal_shas("assemblage_pe")


def windows_shas() -> Generator[str, None, None]:
    return basal_shas("windows_pe")


def bodmas_shas() -> Generator[str, None, None]:
    return basal_shas("bodmas_pe")


def virus_share_elf_shas() -> Generator[str, None, None]:
    return basal_shas("virus_share_elf")


def malware_bazaar_elf_shas() -> Generator[str, None, None]:
    return basal_shas("malware_bazaar_elf")


def virus_total_elf_shas() -> Generator[str, None, None]:
    return basal_shas("virus_total_elf")


class SampleStreamer(Protocol):

    def __call__(self, shas: list[str]) -> Generator[tuple[bytes | Path, str], None, None]:
        ...


def sorel_streamer(shas: list[str]) -> Generator[tuple[bytes, str], None, None]:
    generator = s3_dataset_generator(
        files=shas,
        num_bytes=sys.maxsize,
        max_length=sys.maxsize,
        bucket=SOREL_BUCKET,
        prefix=SOREL_PREFIX,
        errors=2,
        decompress=Decompressor(Decompressor.ZLIB, must_decompress=True),
    )
    for sample in generator:
        yield sample["bytes"], sample["name"]


def basal_file_streamer(shas: list[str], name: str) -> Generator[tuple[Path, str], None, None]:
    sha_map = {f.stem : f for f in DATASET_TO_FILES["binaries"][name]() if f.stem in shas}
    for s in shas:
        yield sha_map[s], s


def assemblage_streamer(shas: list[str]) -> Generator[tuple[Path, str], None, None]:
    return basal_file_streamer(shas, "assemblage_pe")


def windows_streamer(shas: list[str]) -> Generator[tuple[Path, str], None, None]:
    return basal_file_streamer(shas, "windows_pe")


def bodmas_streamer(shas: list[str]) -> Generator[tuple[Path, str], None, None]:
    return basal_file_streamer(shas, "bodmas_pe")


def virus_share_elf_streamer(shas: list[str]) -> Generator[tuple[Path, str], None, None]:
    return basal_file_streamer(shas, "virus_share_elf")


def malware_bazaar_elf_streamer(shas: list[str]) -> Generator[tuple[Path, str], None, None]:
    return basal_file_streamer(shas, "malware_bazaar_elf")


def virus_total_elf_streamer(shas: list[str]) -> Generator[tuple[Path, str], None, None]:
    return basal_file_streamer(shas, "virus_total_elf")


class PackingAnalyzerDirectory:

    def __init__(self, p_root: Path) -> None:
        self.p_root = Path(p_root)
        self.p_prep = self.p_root / "prep"
        self.p_download = self.p_root / "download"
        self.p_raw = self.p_root / "raw"
        self.p_modes = {m: self.p_raw / m for m in DIEC_MODES}
        self.p_merged = self.p_root / "merged"
        self.p_consolidated = self.p_root / "consolidated"

    def mkdir(self) -> None:
        self.p_root.mkdir(exist_ok=True)
        self.p_prep.mkdir(exist_ok=True)
        self.p_raw.mkdir(exist_ok=True)
        for p in self.p_modes.values():
            p.mkdir(exist_ok=True)
            for h in HEX:
                (p / h).mkdir(exist_ok=True)
        self.p_merged.mkdir(exist_ok=True)
        for h in HEX:
            (self.p_merged / h).mkdir(exist_ok=True)
        self.p_consolidated.mkdir(exist_ok=True)


class PackingAnalyzer:

    """
    Note: This class was intended to be used to filter out binaries that are obfuscated in ways
    that make certain things difficult to do. Although the name of the class suggests that it
    only looks for 'packers', it also looks for other forms of obfuscation...
    """

    def __init__(
        self,
        p_root: Path,
        all_shas: Optional[ShasStreamer] = None,
        streamer: Optional[SampleStreamer] = None,
        filter_mode: Optional[int] = None,
        filter_idx: Optional[int] = None,
        num_shards: Optional[int] = None,
        shard_idx: Optional[int] = None,
        diec_timeout: int = 10,
        merge_chunk_size: int = 100000,
        consolidate_chunk_size: int = 100000,
    ) -> None:
        self.paths = PackingAnalyzerDirectory(p_root)
        self.all_shas = all_shas
        self.streamer = streamer
        self.filter_mode = filter_mode
        self.filter_idx = filter_idx
        self.num_shards = num_shards
        self.shard_idx = shard_idx
        self.diec_timeout = diec_timeout
        self.merge_chunk_size = merge_chunk_size
        self.consolidate_chunk_size = consolidate_chunk_size

        if (self.filter_mode is not None) and (self.num_shards is not None):
            raise ValueError("Must use filter or shard API, not both.")

    def __call__(self, ignore_complete: bool = False) -> None:
        self.paths.mkdir()
        self.prepare(ignore_complete)
        self.run(ignore_complete)
        self.merge(ignore_complete)
        self.consolidate_partials()
        self.consolidate_final()

    def mkdir(self) -> None:
        self.paths.mkdir()

    def prepare(self, ignore_complete: bool = False) -> None:

        for f in self.paths.p_prep.iterdir():
            f.unlink()

        shas = sorted(islice(self.all_shas(), None))
        print(f"{len(shas)=}")

        if ignore_complete:
            completed = self.infer_completed_samples_run()
            print(f"Ignoring {len(completed)} completed files")
        else:
            completed = set()

        if isinstance(self.num_shards, int):
            shard_size = (len(shas) // self.num_shards) + 1
            print(f"{shard_size=}")
            for shard_idx in range(self.num_shards):
                idx_start = shard_idx * shard_size
                idx_end = (shard_idx + 1) * shard_size
                shard_file = self.paths.p_prep / f"packingPrep_{shard_idx}.txt"
                print(str(shard_file))
                with open(shard_file, "w") as fp:
                    for i in range(idx_start, min(idx_end, len(shas) - 1)):
                        fp.write(f"{shas[i]}\n")
            return

        if isinstance(self.filter_mode, int):
            num_filters = 16 ** self.filter_mode
            print(f"{num_filters=}")
            start = 0
            finish = len(shas)
            for filter_idx in range(16 ** self.filter_mode):
                filter_ = hex(filter_idx)[2:]
                filter_ = ("0" * (self.filter_mode - len(filter_))) + filter_
                filter_file = self.paths.p_prep / f"packingPrep_{filter_}.txt"
                print(str(filter_file))
                for i in range(start, len(shas)):
                    if shas[i][0:self.filter_mode] != filter_:
                        if i == start:
                            raise RuntimeError("This should never happen. Try reducing the filter_mode value.")
                        finish = i
                        break

                with open(filter_file, "w") as fp:
                    for f in shas[start:finish]:
                        if f not in completed:
                            fp.write(f"{f}\n")

                start = finish
                finish = len(shas)
            return

        raise RuntimeError()

    def run(self, ignore_complete: bool = False) -> None:

        if isinstance(self.shard_idx, int):
            idx = self.shard_idx
        elif isinstance(self.filter_idx, str):
            idx = self.filter_idx
        else:
            idx = None

        # Get the files for this shard (or all files)
        if idx is not None:
            file = self.paths.p_prep / f"packingPrep_{idx}.txt"
            with open(file, "r") as fp:
                shas = [l.strip() for l in fp.readlines()]
            if ignore_complete:
                pass  # Already handled in the preparation :)
        else:
            shas = sorted(islice(self.all_shas(), None))
            if ignore_complete:
                completed = self.infer_completed_samples_run()
                shas = [s for s in shas if s not in completed]
        print(f"{len(shas)=}")

        for data, sha in tqdm(self.streamer(shas), total=len(shas)):
            self.analyze_sample(data, sha)

    def merge(self, ignore_complete: bool = False) -> None:

        iterables = []
        for d in self.paths.p_modes.values():
            for h in HEX:
                p = d / h
                iterables.append(p.iterdir())
        files = list(f for f in tqdm(chain.from_iterable(iterables), desc="Initial Scan..."))
        shas = set(file.stem for file in files)
        print(f"{len(files)} reports from {len(shas)} unique files.")

        if ignore_complete:
            print("Locating merged files...")
            complete = self.infer_completed_samples_merge()
            print(f"Found {len(complete)=}")
            files = [f for f in files if f.stem not in complete]
            shas = set(file.stem for file in files)
            print(f"{len(files)} reports from {len(shas)} unique files.")

        del iterables, files

        errors: tuple[list[str], list[str], list[str]] = ([], [], [])
        pbar = tqdm(shas)
        for sha in pbar:
            pbar.set_description(f"Processing: {sha}")
            files = {alg: (path / sha[0] / sha).with_suffix(".txt") for alg, path in self.paths.p_modes.items()}
            data = {}
            for alg, file in files.items():
                s = str(Path(file.parent.parent.name) / file.name[0] / file.name)
                if not file.exists():
                    print(f"File not found: {s}")
                    d = None
                    errors[0].append(s)
                elif file.stat().st_size == 0:
                    print(f"File is empty: {s}")
                    d = None
                    errors[1].append(s)
                else:

                    with open(file, "r") as fp:
                        raw = fp.read()
                    content = raw[raw.find("{"):raw.rfind("}") + 1].strip()

                    try:
                        d = json.loads(content)
                    except json.JSONDecodeError as err:
                        print(f"JSONDecodeError: {s}")
                        print(err)
                        print(f"*****{content}*****")
                        errors[2].append(s)
                        d = None

                data[alg] = d

            outfile = (self.paths.p_merged / sha[0] / sha).with_suffix(".json")
            with open(outfile, "w") as fp:
                json.dump(data, fp, indent=4)

        print("ERRORS\n------")
        print(f"\tFile not found: {len(errors[0])}")
        print(f"\tFile is empty: {len(errors[1])}")
        print(f"\tJSONDecodeError: {len(errors[2])}")

        print("Logging to logs/merge_errors.log")
        with open("logs/merge_errors.log", "w") as fp:
            fp.write("File not found\n")
            for s in errors[0]:
                fp.write(f"{s}\n")
            fp.write("\nFile is empty\n")
            for s in errors[1]:
                fp.write(f"{s}\n")
            fp.write("\nJSONDecodeError\n")
            for s in errors[2]:
                fp.write(f"{s}\n")

    def consolidate_partials(self) -> None:

        def packeds_decision(packeds: list[bool]) -> bool:
            return any(packeds)

        def packers_decision(packers: list[str]) -> list[str]:
            packers = [p if p not in NAME_TO_GENERIC else "Generic" for p in packers]
            return list(set(packers) - {""})

        def parse_values_blob(values: list[dict], obfuscation: str) -> tuple[bool, list[str]]:
            packeds: list[bool] = []
            packers: list[str] = []
            for value in values:
                if "values" in value:
                    packed, packer = parse_values_blob(value.get("values"), obfuscation)
                elif value.get("type") == obfuscation:
                    packed = True
                    packer = [value.get("name", "")]
                else:
                    packed = False
                    packer = [""]

                packeds.append(packed)
                packers.extend(packer)

            return packeds_decision(packeds), packers_decision(packers)

        def parse_detects_blob(detects: list[dict], obfuscation: str) -> tuple[bool, list[str]]:
            packeds: list[bool] = []
            packers: list[str] = []

            for detect in detects:
                packed, packer = parse_values_blob(detect.get("values", []), obfuscation)
                packeds.append(packed)
                packers.extend(packer)

            return packeds_decision(packeds), packers_decision(packers)

        def save_partial(output: dict, i: int):
            mem_0 = psutil.virtual_memory().used
            with open(self.paths.p_consolidated / f"tmp_{i}.json", "w") as fp:
                json.dump(output, fp, indent=4)
            output.clear()
            gc.collect()
            mem_1 = psutil.virtual_memory().used
            print(f"Partial file: {i}. Freed: {round((mem_1 - mem_0) / 1e6)} MB. Used: {round(mem_1 / 1e6)} MB")


        # Remove old temporary files.
        for f in self.paths.p_consolidated.glob("tmp_*.json"):
            f.unlink()

        # Iterate over the merged JSON files.
        output = {}
        total = sum(1 for _ in tqdm(self.paths.p_merged.rglob("*.json"), desc="Initial Scan..."))
        pbar = tqdm(self.paths.p_merged.rglob("*.json"), total=total)
        for i, file in enumerate(pbar):
            sha = file.stem
            pbar.set_description(f"Processing: {sha}")

            # Process the data from the merged JSON file into a concise summary and store it in memory.
            with open(file, "r") as fp:
                data = json.load(fp)
            output[sha] = {}
            for mode, d in data.items():
                if d is None:
                    output[sha][mode] = None
                    continue
                output[sha][mode] = {"is_obfuscated": False}
                for obfuscation in OBFUSCATION_TYPES:
                    obfuscated, obfuscator = parse_detects_blob(d.get("detects", []), obfuscation)
                    output[sha][mode][obfuscation] = {
                        "obfuscated": obfuscated,
                        "obfuscator": obfuscator,
                    }
                    output[sha][mode]["is_obfuscated"] = output[sha][mode]["is_obfuscated"] or obfuscated

            # Write the output to a temporary file and clear up in-memory data structures.
            if (i + 1) % self.consolidate_chunk_size == 0:
                save_partial(output, i)

        if output:
            save_partial(output, i + len(output) if total > self.consolidate_chunk_size else i)  # pylint: disable=undefined-loop-variable

    def consolidate_final(self) -> None:

        # Parse the temporary files and consolidate them into a single JSON file.
        files = sorted(self.paths.p_consolidated.glob("tmp_*.json"), key=lambda f: int(f.stem.split("_")[1]))
        with open(self.paths.p_consolidated / "output.json", "w") as fp_w:
            fp_w.write("{")

            i = None
            for i, f in enumerate(tqdm(files, desc="Merging temporary consolidation files.")):
                with open(f, "r") as fp_r:
                    lines = fp_r.readlines()

                for j, line in enumerate(lines):
                    if j == 0:  # Skip the initial bracket
                        fp_w.write("\n")
                        continue
                    if j == len(lines) - 1:  # Skip the final bracket
                        break
                    if j == len(lines) - 2:  # Add a comma to the end of the line
                        if i == len(files) - 1:  # Skip if this is the final file
                            pass
                        else:
                            line = line.rstrip("\n") + ","
                    fp_w.write(line)

            if i is None:
                raise RuntimeError()

            if i != (len(files) - 1):
                fp_w.write("\n")
            fp_w.write("}")

    def analyze_sample(self, data: bytes | Path, sha: str) -> None:

        def args(mode: str, file: Path | str) -> list[str]:
            return ["diec", f"--{mode}scan", "--json", str(file)]


        if isinstance(data, bytes):
            file = (self.paths.p_download / sha).with_suffix(".exe")
            with open(file, "wb") as fp:
                fp.write(data)
            unlink = True
        elif isinstance(data, (Path, str)):
            file = Path(data)
            unlink = False
        else:
            raise TypeError(f"Unacceptable type: {type(data)}")


        for mode in DIEC_MODES:
            outfile = self.paths.p_modes[mode] / sha[0] / f"{sha}.txt"
            try:
                subprocess.run(
                    args(mode, file),
                    stdout=open(outfile, "w"),
                    timeout=self.diec_timeout,
                    check=True,
                    capture_output=False,
                )
            except subprocess.TimeoutExpired:
                print(f"TimeoutExpired: {mode} {sha}")
            except subprocess.CalledProcessError as err:
                if "SIGSEGV" in str(err):
                    print(f"SIGSEGV 11: {mode} {sha}")
                else:
                    raise err
            except OSError as err:
                if "Errno 28" in str(err):
                    print(f"Errno 28: {mode} {sha}")
                else:
                    raise err

        if unlink:
            file.unlink()

    def infer_completed_samples_run(self, all_modes: bool = True) -> set[str]:
        completed = set()

        for d in tqdm(self.paths.p_modes.values(), total=3, desc="Scanning for completed..."):
            c = set()
            for h in tqdm(HEX, leave=False):
                c.update(f.stem for f in (d / h).iterdir() if f.stat().st_size > 0)

            if not completed:
                completed = c
            else:
                if all_modes:
                    completed = completed.intersection(c)
                else:
                    completed = completed.union(c)

        return completed

    def infer_completed_samples_merge(self, all_modes: bool = True) -> set:

        iterable = [(self.paths.p_merged / h, all_modes) for h in HEX]
        with mp.Pool(len(HEX)) as pool:
            results = list(tqdm(
                pool.starmap(_infer_completed_samples_merge, iterable),
                total=len(iterable),
                desc="Scanning for completed merged...",
            ))
        completed = set()
        for result in results:
            completed.update(result)
        return completed


def _infer_completed_samples_merge(p: Path, all_modes: bool = True) -> set[str]:
    completed = set(f.stem for f in p.iterdir() if f.stat().st_size > 0)
    if all_modes:
        empty = next(p.iterdir(), None) is None
        if not empty:
            null_files = set(Path(f).stem for f in find_files_with_null(str(p)))
            completed.difference_update(null_files)
    return completed


class PackingMap(UserDict):
    """
    Map SHA256 to whether the corresponding sample is packed or not.

    Usage
    -----

    To minimize memory usage, booleanize every individual report on the fly:
    >>> packing_map = PackingMap(lazy=True, chunked=False, num_workers=None)

    To speed up the process a little bit, perform this in parallel:
    >>> packing_map = PackingMap(lazy=True, chunked=True, num_workers=16)
    # Elapsed time: 24.29 seconds

    If you have a lot of extra memory to spare, you can load the entire JSON then booleanize:
    >>> packing_map = PackingMap(lazy=False, chunked=False, num_workers=None)
    # Elapsed time: 88.93 seconds

    To reduce those memory requirements, you can perform JSON loading in chunks:
    >>> packing_map = PackingMap(lazy=False, chunked=True, num_workers=None)
    # Elapsed time: 102.94 seconds

    To speed up the chunked processing, you can parallelize it at the expense of extra memory:
    >>> packing_map = PackingMap(lazy=False, chunked=True, num_workers=16)
    # Elapsed time: 16.63 seconds

    Note that `PackingMap(lazy=True, chunked=True, num_workers=None)` is functionally
    identical to the `chunked=False` version.
    """

    def __init__(
        self,
        root: Path | str = str(SOREL_PATH / "diec"),
        include: tuple[DiecMode] = tuple(DIEC_MODES),
        obfuscations: tuple[str] = tuple(OBFUSCATION_TYPES),
        lazy: bool = False,
        chunked: bool = False,
        num_workers: Optional[int] = None,
    ) -> None:
        if isinstance(num_workers, int) and chunked is False:
            print("`chunked` is False, but multiple workers were requested. Setting `chunked` to True.")
            chunked = True

        self._cache_file = None

        self.p_consolidated = PackingAnalyzerDirectory(root).p_consolidated
        self.include = tuple(include)
        self.obfuscations = tuple(obfuscations)
        self.lazy = lazy
        self.chunked = chunked
        self.num_workers = num_workers

        for f in self.partial_files + [self.complete_file]:
            with open(f, "rb") as fp:
                fp.seek(-1, 2)
                if not fp.read(1) == "}".encode():
                    raise ValueError(f"File is invalid JSON: {f}")

        if self.num_workers is not None:
            self.num_workers = min(self.num_workers, len(self.partial_files))

        if self.cache_file.exists() and self.cache_file.stat().st_size > 0:
            print(f"Getting the packing map from {self.cache_file=}")
            with open(self.cache_file, "rb") as fp:
                packing_map = pickle.load(fp)
        else:
            print(f"Building packing map and saving to {self.cache_file=}")
            self.cache_file.parent.mkdir(exist_ok=True, parents=True)
            packing_map = self.get_packing_map()
            with open(self.cache_file, "wb") as fp:
                pickle.dump(packing_map, fp)

        super().__init__(packing_map)

    @property
    def cache_file(self) -> Path:
        if self._cache_file is not None:
            return self._cache_file

        # This is idiotic but I don't care
        b_1 = str(self.include).encode()
        b_2 = str(self.obfuscations).encode()
        with open(self.partial_files[0], "rb") as fp:
            fp.seek(64)
            b_3 = fp.read(64)
        with open(self.partial_files[-1], "rb") as fp:
            fp.seek(64)
            b_4 = fp.read(64)
        b = b_1 + b_2 + b_3 + b_4
        h = hashlib.sha256(b).hexdigest()
        self._cache_file = Path("./cache") / "packing_map" / f"{h}.pkl"
        return self._cache_file

    @property
    def partial_files(self) -> list[os.PathLike]:
        files = list(self.p_consolidated.glob("tmp_*.json"))
        files.sort(key=lambda f: int(f.stem.split("_")[1]))
        files = [str(f) for f in files]
        return files

    @property
    def complete_file(self) -> os.PathLike:
        return str(self.p_consolidated / "output.json")

    def get_packing_map(self) -> dict[str, bool]:
        if self.chunked:
            files = self.partial_files
        else:
            files = [self.complete_file]

        if self.lazy:
            if self.chunked:
                fn = self.get_packing_map_lazy
            else:
                fn = partial(self.get_packing_map_lazy, disable_tqdm=False)
        else:
            fn = self.get_packing_map_fast

        if isinstance(self.num_workers, int):
            with mp.Pool(self.num_workers) as pool:
                results: list[dict] = list(pool.map(fn, files))
        else:
            results = [fn(f) for f in tqdm(files)]

        packing_map = {}
        for r in results:
            packing_map.update(r)

        return packing_map

    def get_packing_map_fast(self, file: str):
        with open(file, "r") as fp:
            d = json.load(fp)
        return {sha: self.get_packing_report(v) for sha, v in d.items()}

    def get_packing_map_lazy(self, file: str, disable_tqdm: bool = True):

        args = ["wc", "-l", file]
        result = subprocess.run(args, check=True, capture_output=True)
        total = int(result.stdout.split()[0])

        packing_map = {}
        blob = []
        brace_op = 0
        brace_cl = 0

        with open(file, "r") as fp:

            if not disable_tqdm:
                pbar = tqdm(enumerate(fp), total=total)
                iterable = pbar
            else:
                pbar = None
                iterable = enumerate(fp)

            for i, line in iterable:
                if i == 0:  # Skip the initial bracket
                    continue
                if i == total:  # Skip the final bracket
                    break

                line = line.strip()

                if brace_op == 0:  # Identify and strip the SHA
                    sha = line.split(":")[0].replace('"', "")
                    line = line.split(":")[1].strip()
                    if pbar is not None:
                        pbar.set_description(f"Processing: {sha}")

                brace_op += line.count("{")
                brace_cl += line.count("}")
                blob.append(line)

                if line.rstrip(",") == "}" and brace_op == brace_cl:  # End of the blob
                    blob = "".join(blob).rstrip(",")
                    d = json.loads(blob)
                    p = self.get_packing_report(d)
                    packing_map[sha] = p

                    blob = []
                    brace_op = 0
                    brace_cl = 0

        return packing_map

    def get_packing_report(self, report: dict[DiecMode, Optional[dict]]) -> bool:
        for mode in self.include:
            if report[mode] is None:
                continue
            if self.obfuscations == OBFUSCATION_TYPES:
                if report[mode]["is_obfuscated"]:
                    return True
                continue
            for obfuscation in self.obfuscations:
                if report[mode][obfuscation] is None:
                    continue
                if report[mode][obfuscation]["obfuscated"]:
                    return True
        return False


def universal_packing_map(roots: Optional[Path | list[Path]] = None, **kwds) -> dict[str, bool]:
    if roots is None:
        roots = list(PACKING_ROOTS.values()) if roots is None else roots
    elif isinstance(roots, (Path, str)):
        roots = [roots]
    elif isinstance(roots, Iterable):
        roots = list(roots)
        if not isinstance(roots[0], (Path, str)):
            raise TypeError(f"Unacceptable type: {type(roots[0])}")

    all_maps = {}
    for root in roots:
        m = PackingMap(root, **kwds)
        all_maps.update(dict(m))
    return all_maps


def not_packed_list(root: str, outfile: Path) -> None:
    m = PackingMap(root)
    notpacked = sorted([k for k, v in m.items() if not v])
    with open(outfile, "w") as fp:
        for s in notpacked:
            fp.write(f"{s}\n")


def unpack(
    data: str | Path | bytes,
    outfile: Optional[str | Path] = None,
    return_file: bool = True,
    return_bytes: bool = False,
    errors: int = 0,
) -> tuple[Optional[Path], Optional[bytes]]:

    infile = None
    tmpfile = None
    inbytes = None
    outbytes = None

    if isinstance(data, (str, Path)):
        infile = data
    elif isinstance(data, BytesIO):
        data.seek(0)
        inbytes = data.read()
    elif isinstance(data, bytes):
        inbytes = data
    else:
        raise TypeError(f"Unacceptable type: {type(data)}")

    if infile is None:
        tmpfile = Path(tempfile.mkstemp())
        with open(tmpfile, "wb") as fp:
            fp.write(inbytes)

    if outfile is None:
        if return_file:
            outfile = Path(tempfile.mkstemp())
    else:
        outfile = Path(outfile)

    args = ["upx", "-d"]
    if outfile is not None:
        args.extend(["-o", str(outfile)])
    args.append(str(infile))

    try:
        subprocess.run(args, check=True, capture_output=True)
    except subprocess.CalledProcessError as err:
        if errors == 0:
            return None, None
        raise err

    if return_bytes and outfile is not None:
        with open(outfile, "rb") as fp:
            outbytes = fp.read()

    if not return_file:
        outfile.unlink()
        outfile = None

    if tmpfile is not None:
        tmpfile.unlink()

    return outfile, outbytes


def unpack_samples(
    samples: list[Path | str],
    outdir: Optional[Path] = None,
    outfiles: Optional[list[Path]] = None,
    overwrite: bool = False,
    return_files: bool = False,
    return_bytes: bool = True,
    errors: int = 0,
) -> Generator[tuple[Optional[Path], Optional[bytes]], None, None]:
    if outdir is None and outfiles is None:
        outdir = Path(tempfile.mkdtemp())
        outfiles = (outdir / f.stem for f in samples)
        print(f"Saving files to: {outdir=}")
    elif overwrite:
        print("Overwriting files...")
        outfiles = (f for f in samples)
    elif outfiles is not None:
        pass
    outfiles: list[Path | str]

    pbar = tqdm(zip(samples, outfiles), total=len(samples))
    for f_in, f_out in pbar:
        pbar.set_description(f_in.stem)
        yield unpack(f_in, f_out, return_files, return_bytes, errors)
        if not return_files:
            Path(f_out).unlink()

    if not return_files and outdir is not None:
        outdir.rmdir()


def main():

    parser = ArgumentParser()
    parser.add_argument("--dataset", choices=["sorel_pe", "bodmas_pe", "virus_share_elf", "malware_bazaar_elf", "virus_total_elf", "assemblage_pe", "windows_pe"], required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--consolidate_partials", action="store_true")
    parser.add_argument("--consolidate_final", action="store_true")
    parser.add_argument("--dont_ignore_complete", action="store_true")
    parser.add_argument("--diec_timeout", type=int, default=10)
    parser.add_argument("--filter_mode", type=int, default=None,
        help="Parallel with 16 ** `filter_mode` processes. Required for --prepare and --run.")
    parser.add_argument("--filter_idx", type=str, default=None,
        help="Required for --run.")
    parser.add_argument("--num_shards", type=int, default=None,
        help="Parallel with `num_shards` processes. Required for --prepare and --run.")
    parser.add_argument("--shard_idx", type=int, default=None,
        help="Required for --run.")
    args = parser.parse_args()

    if args.dataset == "sorel_pe":
        p_root = PACKING_ROOTS["sorel_pe"]
        all_shas = sorel_shas
        streamer = sorel_streamer
    elif args.dataset == "windows_pe":
        p_root = PACKING_ROOTS["windows_pe"]
        all_shas = windows_shas
        streamer = windows_streamer
    elif args.dataset == "assemblage_pe":
        p_root = PACKING_ROOTS["assemblage_pe"]
        all_shas = assemblage_shas
        streamer = assemblage_streamer
    elif args.dataset == "bodmas_pe":
        p_root = PACKING_ROOTS["bodmas_pe"]
        all_shas = bodmas_shas
        streamer = bodmas_streamer
    elif args.dataset == "virus_share_elf":
        p_root = PACKING_ROOTS["virus_share_elf"]
        all_shas = virus_share_elf_shas
        streamer = virus_share_elf_streamer
    elif args.dataset == "malware_bazaar_elf":
        p_root = PACKING_ROOTS["malware_bazaar_elf"]
        all_shas = malware_bazaar_elf_shas
        streamer = malware_bazaar_elf_streamer
    elif args.dataset == "virus_total_elf":
        p_root = PACKING_ROOTS["virus_total_elf"]
        all_shas = virus_total_elf_shas
        streamer = virus_total_elf_streamer

    analyzer = PackingAnalyzer(
        p_root,
        all_shas,
        streamer,
        args.filter_mode,
        args.filter_idx,
        args.num_shards,
        args.shard_idx,
        args.diec_timeout,
    )

    analyzer.mkdir()

    if args.prepare:
        t = time.time()
        analyzer.prepare(not args.dont_ignore_complete)
        print(f"Elapsed time: {time.time() - t:.2f} seconds")

    if args.run:
        t = time.time()
        analyzer.run(not args.dont_ignore_complete)
        print(f"Elapsed time: {time.time() - t:.2f} seconds")

    if args.merge:
        t = time.time()
        analyzer.merge(not args.dont_ignore_complete)
        print(f"Elapsed time: {time.time() - t:.2f} seconds")

    if args.consolidate_partials:
        t = time.time()
        analyzer.consolidate_partials()
        print(f"Elapsed time: {time.time() - t:.2f} seconds")

    if args.consolidate_final:
        t = time.time()
        analyzer.consolidate_final()
        print(f"Elapsed time: {time.time() - t:.2f} seconds")


if __name__ == "__main__":
    main()
