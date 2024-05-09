"""
Detect whether samples of SOREL collection are packed or not using Detect-It-Easy (DiE).

Usage
-----
1. Prepare the script for embarassingly parallel execution by creating a list of shas
for each worker to process. Use either the filter_mode or num_shards method of splitting
the SOREL collection across workers (filter_mode=4 is recommended for better load balancing).

    python detect_packing_sorel.py --prepare --filter_mode=FILTER_MODE

or

    python detect_packing_sorel.py --prepare --num_shards=NUM_SHARDS

2. Run the script to download, analyze, and save the results for each sample.

The SOREL binaries will be temporarily saved in the P_DOWNLOAD directory, but will be
deleted after processing each file.

    | -- P_ROOT
        | -- P_DOWNLOAD
            | -- sha.exe
            ...

You can run the program for every file at once in a single process (no parallelism)

    python detect_packing_sorel.py --run

or split the load across the individual workers using the sharded approach you prepared for
in step 1 (recommended).

    python detect_packing_sorel.py --run --filter_idx=FILTER_IDX

or

    python detect_packing_sorel.py --run --shard_idx=SHARD_IDX

Its recommended to use a tool like GNU-parallel to run all shards at once, e.g.,

    parallel --bar -j 48 'python src/data/detect_packing_sorel.py --run --filter_idx={1} --filter_mode=4 > ./logs/packing_python_{1}.log 2>&1' ::: $(printf "%04x\n" {0..65535})

Either way, this will produce a set of JSON-ish files for each sample in the SOREL collection

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

    python detect_packing_sorel.py --run --dont_ignore_complete

3. After all the samples have been processed, merge the results from each mode of DIEC
into a single JSON file for each sample.

    python detect_packing_sorel.py --merge

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

    python detect_packing_sorel.py --consolidate

This will produce a single JSON file for the entire SOREL collection (and some temporary files).

    | -- P_ROOT
        | -- P_CONSOLIDATED
            | -- output.json
            | -- tmp_0.json
            ...
"""

from argparse import ArgumentParser
import asyncio
from collections.abc import Iterable
from collections import UserDict
import gc
from itertools import chain, islice
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Literal, Optional

# pylint: disable=wrong-import-position
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# pylint: enable=wrong-import-position

import psutil
from tqdm import tqdm

from src.utils import batched
from src.data.cfg import SOREL_BUCKET, SOREL_PREFIX
from src.data.prepare_datasets import s3_dataset_generator
from src.data.utils import stream_sorel_meta, Decompressor, read_binary_files_asynch, write_binary_files_asynch


DIEC_MODES = ("recursive", "deep", "heuristic")
DiecMode = Literal["recursive", "deep", "heuristic"]
DIEC_TIMEOUT = 10
MERGE_CHUNK_SIZE = 100000
CONSOLIDATE_CHUNK_SIZE = 100000
DECOMPRESSOR = Decompressor(Decompressor.ZLIB, must_decompress=True)
HEX = tuple(hex(i)[2:] for i in range(16))

P_ROOT = Path("/home/lk3591/Documents/datasets/Sorel/diec")
P_PREP = P_ROOT / "prep"
P_DOWNLOAD = P_ROOT / "download"
P_RAW = P_ROOT / "raw"
P_MODES = {m: P_RAW / m for m in DIEC_MODES}
P_MERGED = P_ROOT / "merged"
P_CONSOLIDATED = P_ROOT / "consolidated"


def analyze_sample(b: bytes, sha: str) -> None:

    def args(mode: str) -> list[str]:
        return ["diec", f"--{mode}scan", "--json", str(file)]

    file = (P_DOWNLOAD / sha).with_suffix(".exe")
    with open(file, "wb") as fp:
        fp.write(b)

    for mode in DIEC_MODES:
        outfile = P_MODES[mode] / sha[0] / f"{sha}.txt"
        try:
            subprocess.run(
                args(mode),
                stdout=open(outfile, "w"),
                timeout=DIEC_TIMEOUT,
                check=True,
                capture_output=False,
            )
        except subprocess.TimeoutExpired:
            print(f"TimeoutExpired: {mode} {sha}")
        except OSError as err:
            if "Errno 28" in str(err):
                print(f"Errno 28: {mode} {sha}")
            else:
                raise err

    file.unlink()


def analyze_samples(files: Iterable[str]) -> None:

    generator = s3_dataset_generator(
        files=files,
        num_bytes=sys.maxsize,
        max_length=sys.maxsize,
        bucket=SOREL_BUCKET,
        prefix=SOREL_PREFIX,
        errors=2,
        decompress=DECOMPRESSOR,
    )

    pbar = tqdm(generator, total=len(files))
    for sample in pbar:
        pbar.set_description(f"Processing: {sample['name']}")
        analyze_sample(sample["bytes"], sample["name"])


def infer_completed_samples(all_modes: bool = True) -> set[str]:
    completed = set()

    for d in P_MODES.values():
        c = set()
        for h in HEX:
            c.update(f.stem for f in (d / h).iterdir() if f.stat().st_size > 0)

        if not completed:
            completed = c
        else:
            if all_modes:
                completed = completed.intersection(c)
            else:
                completed = completed.union(c)

    return completed


def find_files_with_null(directory: Path | str) -> list[os.PathLike]:
    command = ['grep', '-l', '-r', 'null', str(directory)]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    file_paths = result.stdout.splitlines()
    return file_paths


def process_hex(h: str, all_modes: bool = True) -> set[str]:
    p = P_MERGED / h
    completed = set(f.stem for f in p.iterdir() if f.stat().st_size > 0)
    if all_modes:
        null_files = set(Path(f).stem for f in find_files_with_null(str(p)))
        completed.difference_update(null_files)
    return completed


def infer_completed_samples_merge(all_modes: bool = True) -> set:
    """
    Takes about 5 minutes when all_modes=True.
    """
    iterable = [(h, all_modes) for h in HEX]
    with mp.Pool(len(HEX)) as pool:
        results = list(tqdm(
            pool.starmap(process_hex, iterable),
            total=len(iterable),
            desc="Scanning for completed merged...",
        ))
    completed = set()
    for result in results:
        completed.update(result)
    return completed


def consolidate():


    def packeds_decision(packeds: list[bool]) -> bool:
        return any(packeds)


    def packers_decision(packers: list[str]) -> list[str]:
        packers = [p if p != "Packer detected" else "Heursitic" for p in packers]
        return list(set(packers) - {""})


    def parse_values_blob(values: list[dict]) -> tuple[bool, list[str]]:
        packeds: list[bool] = []
        packers: list[str] = []
        for value in values:
            if "values" in value:
                packed, packer = parse_values_blob(value.get("values"))
            elif value.get("type") == "Packer":
                packed = True
                packer = [value.get("name", "")]
            else:
                packed = False
                packer = [""]

            packeds.append(packed)
            packers.extend(packer)

        return packeds_decision(packeds), packers_decision(packers)


    def parse_detects_blob(detects: list[dict]) -> tuple[bool, list[str]]:
        packeds: list[bool] = []
        packers: list[str] = []

        for detect in detects:
            packed, packer = parse_values_blob(detect.get("values", []))
            packeds.append(packed)
            packers.extend(packer)

        return packeds_decision(packeds), packers_decision(packers)


    # Remove old temporary files.
    for f in P_CONSOLIDATED.glob("tmp_*.json"):
        f.unlink()

    # Iterate over the merged JSON files.
    output = {}
    total = sum(1 for _ in tqdm(P_MERGED.rglob("*.json"), desc="Initial Scan..."))
    pbar = tqdm(P_MERGED.rglob("*.json"), total=total)
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
            packed, packer = parse_detects_blob(d.get("detects", []))
            output[sha][mode] = {"packed": packed, "packer": packer}

        # Write the output to a temporary file and clear up in-memory data structures.
        if (i + 1) % CONSOLIDATE_CHUNK_SIZE == 0:
            mem_0 = psutil.virtual_memory().used
            with open(P_CONSOLIDATED / f"tmp_{i}.json", "w") as fp:
                json.dump(output, fp, indent=4)
            output = {}
            gc.collect()
            mem_1 = psutil.virtual_memory().used
            print(f"Partial file: {i}. Freed: {round((mem_1 - mem_0) / 1e6)} MB. Used: {round(mem_1 / 1e6)} MB")


    # Parse the temporary files and consolidate them into a single JSON file.
    files = sorted(P_CONSOLIDATED.glob("tmp_*.json"), key=lambda f: int(f.stem.split("_")[1]))
    with open(P_CONSOLIDATED / "output.json", "w") as fp_w:
        fp_w.write("{")

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

        fp_w.write("\n}")


def merge(ignore_complete: bool):

    iterables = []
    for d in P_MODES.values():
        for h in HEX:
            p = d / h
            iterables.append(p.iterdir())
    files = list(f for f in tqdm(chain.from_iterable(iterables), desc="Initial Scan..."))
    shas = set(file.stem for file in files)
    print(f"{len(files)} reports from {len(shas)} unique files.")

    if ignore_complete:
        print("Locating merged files...")
        complete = infer_completed_samples_merge()
        print(f"Found {len(complete)=}")
        files = [f for f in files if f.stem not in complete]
        shas = set(file.stem for file in files)
        print(f"{len(files)} reports from {len(shas)} unique files.")

    del iterables, files

    errors: tuple[list[str], list[str], list[str]] = ([], [], [])
    pbar = tqdm(shas)
    for sha in pbar:
        pbar.set_description(f"Processing: {sha}")
        files = {alg: (path / sha[0] / sha).with_suffix(".txt") for alg, path in P_MODES.items()}
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

        outfile = (P_MERGED / sha[0] / sha).with_suffix(".json")
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


def run(filter_idx: Optional[int], shard_idx: Optional[int], ignore_complete: bool) -> None:
    if (filter_idx is None) == (shard_idx is None):
        raise ValueError("Must use filter or shard API, not both.")

    # Get the files for this shard (or all files)
    if isinstance(shard_idx, int) or isinstance(filter_idx, str):
        file = P_PREP / f"packingPrep_{shard_idx if shard_idx else filter_idx}.txt"
        with open(file, "r") as fp:
            files = [l.strip() for l in fp.readlines()]
        if ignore_complete:
            pass  # Already handled in the preparation :)
    else:
        files = (s.sha256 for s in stream_sorel_meta() if s.is_malware)
        files = sorted(islice(files, None))
        if ignore_complete:
            completed = infer_completed_samples()
            files = [f for f in files if f not in completed]
    print(f"{len(files)=}")
    analyze_samples(files)


def prepare(filter_mode: Optional[int], num_shards: Optional[int], ignore_complete: bool) -> None:
    if (filter_mode is None) == (num_shards is None):
        raise ValueError("Must use filter or shard API, not both.")
    print(f"{ignore_complete=}")

    for f in P_PREP.iterdir():
        f.unlink()

    files = (s.sha256 for s in stream_sorel_meta() if s.is_malware)
    files = sorted(islice(files, None))
    print(f"{len(files)=}")
    print(f"mem(files)={int(sys.getsizeof(files[0]) * len(files) // 1e6)}MB")

    if ignore_complete:
        completed = infer_completed_samples()
        print(f"Ignoring {len(completed)} completed files")
        # files = [f for f in files if f not in completed]  # premature filtering fucks up some other stuff
    else:
        completed = set()

    if num_shards:
        shard_size = (len(files) // num_shards) + 1
        print(f"{shard_size=}")
        for shard_idx in range(num_shards):
            idx_start = shard_idx * shard_size
            idx_end = (shard_idx + 1) * shard_size
            shard_file = P_PREP / f"packingPrep_{shard_idx}.txt"
            print(str(shard_file))
            with open(shard_file, "w") as fp:
                for i in range(idx_start, min(idx_end, len(files) - 1)):
                    fp.write(f"{files[i]}\n")
        return

    if filter_mode:
        num_filters = 16 ** filter_mode
        print(f"{num_filters=}")
        start = 0
        finish = len(files)
        for filter_idx in range(16 ** filter_mode):
            filter_ = hex(filter_idx)[2:]
            filter_ = ("0" * (filter_mode - len(filter_))) + filter_
            filter_file = P_PREP / f"packingPrep_{filter_}.txt"
            print(str(filter_file))
            for i in range(start, len(files)):
                if files[i][0:filter_mode] != filter_:
                    if i == start:
                        raise RuntimeError("This should never happen.")
                    finish = i
                    break

            with open(filter_file, "w") as fp:
                for f in files[start:finish]:
                    if f not in completed:
                        fp.write(f"{f}\n")

            start = finish
            finish = len(files)
        return


class PackingMap(UserDict):

    """Map SHA256 to whether the corresponding sample is packed or not.

    Use `fast` mode to load the entire output.json file into memory before
    processing into boolean values. This is the fastest method (~1 min) but requires
    the most memory (16 GB). Use `lazy` mode to process the output.json file line-by-line,
    which is much slower (~20 min). Use `parallel` mode to process the output.json file
    line-by-line with multiple processes (~2 min).
    """

    def __init__(
        self,
        include: tuple[DiecMode] = tuple(DIEC_MODES),
        mode: Literal["fast", "lazy", "parallel"] = "parallel",
        num_workers: Optional[int] = 32,
    ) -> None:
        self.include = tuple(include)
        self.num_workers = num_workers

        if mode == "fast":
            d = self.get_packing_map_fast()
        if mode == "lazy":
            d = self.get_packing_map_lazy()
        if mode == "parallel":
            d = self.get_packing_map_parallel()

        super().__init__(d)

    def get_packing_map_fast(self):
        with open(P_CONSOLIDATED / "output.json", "r") as fp:
            d = json.load(fp)
        return {sha: self.get_packing_report(v) for sha, v in d.items()}

    def get_packing_map_lazy(
        self,
        file: str = str(P_CONSOLIDATED / "output.json"),
        disable_tqdm: bool = False,
    ):

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
                if i == total - 1:  # Skip the final bracket
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
                    blob = "".join(blob)[:-1]
                    try:
                        d = json.loads(blob)
                    except json.JSONDecodeError:
                        print(f"JSONDecodeError: {sha}")
                        print(f"*****{blob}*****")
                        raise
                    p = self.get_packing_report(d)
                    packing_map[sha] = p

                    blob = []
                    brace_op = 0
                    brace_cl = 0

        return packing_map

    def get_packing_map_parallel(self):
        files = sorted(P_CONSOLIDATED.glob("tmp_*.json"), key=lambda f: int(f.stem.split("_")[1]))

        iterable = [(str(f), True) for f in files]
        with mp.Pool(self.num_workers) as pool:
            results: list[dict] = list(pool.starmap(self.get_packing_map_lazy, iterable))

        packing_map = {}
        for r in results:
            packing_map.update(r)

        return packing_map

    def get_packing_report(
        self,
        report: dict[DiecMode, Optional[dict[Literal["packed", "packer"], bool | str]]],
    ) -> bool:
        for mode in self.include:
            if report[mode] is None:
                continue
            if report[mode]["packed"]:
                return True
        return False


def main():

    parser = ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--consolidate", action="store_true")
    parser.add_argument("--dont_ignore_complete", action="store_true")
    parser.add_argument("--filter_mode", type=int, default=None,
        help="Parallel with 16 ** `filter_mode` processes. Required for --prepare and --run.")
    parser.add_argument("--filter_idx", type=str, default=None,
        help="Required for --run.")
    parser.add_argument("--num_shards", type=int, default=None,
        help="Parallel with `num_shards` processes. Required for --prepare and --run.")
    parser.add_argument("--shard_idx", type=int, default=None,
        help="Required for --run.")
    args = parser.parse_args()

    P_ROOT.mkdir(exist_ok=True)
    P_PREP.mkdir(exist_ok=True)
    P_RAW.mkdir(exist_ok=True)
    for p in P_MODES.values():
        p.mkdir(exist_ok=True)
        for h in HEX:
            (p / h).mkdir(exist_ok=True)
    P_MERGED.mkdir(exist_ok=True)
    for h in HEX:
        (P_MERGED / h).mkdir(exist_ok=True)

    t_0 = time.time()

    if args.prepare:
        prepare(args.filter_mode, args.num_shards, not args.dont_ignore_complete)

    if args.run:
        run(args.filter_idx, args.shard_idx, not args.dont_ignore_complete)

    if args.merge:
        merge(not args.dont_ignore_complete)

    if args.consolidate:
        consolidate()

    print(f"Elapsed time: {time.time() - t_0:.2f} seconds")


def test():
    t = time.time()
    packing_map = PackingMap(mode="fast")
    print(f"Elapsed time: {time.time() - t:.2f} seconds")

    t = time.time()
    packing_map = PackingMap(mode="lazy")
    print(f"Elapsed time: {time.time() - t:.2f} seconds")

    t = time.time()
    packing_map = PackingMap(mode="parallel")
    print(f"Elapsed time: {time.time() - t:.2f} seconds")


if __name__ == "__main__":
    # test()
    main()
