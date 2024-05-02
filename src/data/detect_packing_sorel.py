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

    parallel --bar -j 48 'python src/data/detect_packing_sorel.py --filter_idx={1} --filter_mode=4 > ./logs/packing_python_{1}.log 2>&1' ::: $(printf "%04x\n" {0..65536})

Either way, this will produce a set of JSON files for each sample in the SOREL collection

    | -- P_ROOT
        | -- P_RAW
            | -- recursive
                | -- sha.json
                ...
            | -- deep
                | -- sha.json
                ...
            | -- heuristic
                | -- sha.json
                ...

If you want DO NOT want to ignore files that have already been processed (process them again),
then use the --dont_ignore_complete flag.

    python detect_packing_sorel.py --run --dont_ignore_complete

3. After all the samples have been processed, merge the results from each mode of DIEC
into a single JSON file for each sample.

    python detect_packing_sorel.py --merge

This will produce one JSON files for each sample in the SOREL collection

    | -- P_ROOT
        | -- P_MERGED
            | -- sha.json

4. Finally, consolidate the results into a single file for the entire collection
for usage in downstream ML tasks.

    python detect_packing_sorel.py --consolidate

This will produce a single JSON file for the entire SOREL collection.

    | -- P_ROOT
        | -- P_CONSOLIDATED
            | -- output.json
            | -- output.csv

TODO
----
- Save intermediary files (from run process) as .json not .txt
- Save consolidated results as CSV as well as JSON
"""

from argparse import ArgumentParser
from collections.abc import Iterable
from itertools import chain, islice
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Optional

# pylint: disable=wrong-import-position
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# pylint: enable=wrong-import-position

from tqdm import tqdm

from src.data.cfg import SOREL_BUCKET, SOREL_PREFIX
from src.data.prepare_datasets import s3_dataset_generator
from src.data.utils import read_binary_files_asynch, stream_sorel_meta, Decompressor


DIEC_MODES = ("recursive", "deep", "heuristic")
DIEC_TIMEOUT = 10
DECOMPRESSOR = Decompressor(Decompressor.ZLIB, must_decompress=True)

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
        outfile = P_MODES[mode] / f"{sha}.txt"
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
        c = set(f.stem for f in d.iterdir() if f.stat().st_size > 0)
        if not completed:
            completed = c
        else:
            if all_modes:
                completed = completed.intersection(c)
            else:
                completed = completed.union(c)
    return completed


def consolidate():


    def packeds_decision(packeds: list[bool]) -> bool:
        return any(packeds)


    def packers_decision(packers: list[str]) -> str:
        packers = [p if p != "Packer detected" else "Heuristic" for p in packers]
        packers = [p for p in packers if p != ""]
        return "|".join(packers) if packers != "" else ""


    def parse_values_blob(values: list[dict]) -> tuple[bool, str]:
        packeds: list[bool] = []
        packers: list[str] = []
        for value in values:
            if "values" in value:
                packed, packer = parse_values_blob(value.get("values"))
            elif value.get("type") == "Packer":
                packed = True
                packer = value.get("name", "")
            else:
                packed = False
                packer = ""

            packeds.append(packed)
            packers.append(packer)

        return packeds_decision(packeds), packers_decision(packers)


    def parse_detects_blob(detects: list[dict]) -> tuple[bool, str]:
        packeds: list[bool] = []
        packers: list[str] = []

        for detect in detects:
            packed, packer = parse_values_blob(detect.get("values", []))
            packeds.append(packed)
            packers.append(packer)

        return packeds_decision(packeds), packers_decision(packers)


    output = {}
    pbar = tqdm(P_MERGED.iterdir(), total=sum(1 for _ in P_MERGED.iterdir()))
    for file in pbar:
        sha = file.stem
        pbar.set_description(f"Processing: {sha}")
        with open(file, "r") as f:
            data = json.load(f)

        output[sha] = {}
        for mode, d in data.items():
            if d is None:
                output[sha][mode] = None
                continue
            packed, packer = parse_detects_blob(d.get("detects", []))
            output[sha][mode] = {"packed": packed, "packer": packer}

    with open(P_CONSOLIDATED / "output.json", "w") as f:
        json.dump(output, f, indent=4)

    # TODO: save as CSV


def merge():

    files = islice(chain.from_iterable(d.iterdir() for d in P_MODES.values()), None)
    files = list(tqdm(files, desc="Initial Scan..."))
    shas = set(file.stem for file in files)
    print(f"{len(files)} reports from {len(shas)} unique files.")

    errors = {name: 0 for name in (0, 1, 2)}
    pbar = tqdm(shas)
    for sha in pbar:
        pbar.set_description(f"Processing: {sha}")
        files = {alg: (path / sha).with_suffix(".txt") for alg, path in P_MODES.items()}
        data = {}
        for alg, file in files.items():
            s = str(Path(file.parent.name) / file.name)
            if not file.exists():
                print(f"File not found: {s}")
                d = None
                errors[0] += 1
            elif file.stat().st_size == 0:
                print(f"File is empty: {s}")
                d = None
                errors[1] += 1
            else:

                with open(file, "r") as fp:
                    raw = fp.read()
                content = raw[raw.find("{"):raw.rfind("}") + 1].strip()

                try:
                    d = json.loads(content)
                except json.JSONDecodeError:
                    print(f"JSONDecodeError: {s}")
                    print(f"*****{content}*****")
                    errors[2] += 1
                    d = None

            data[alg] = d

        outfile = (P_MERGED / sha).with_suffix(".json")
        with open(outfile, "w") as fp:
            json.dump(data, fp, indent=4)

    print("ERRORS\n------")
    print(f"\tfile not found: {errors[0]}")
    print(f"\tFile is empty: {errors[1]}")
    print(f"\tJSONDecodeError: {errors[2]}")


def run(filter_idx: Optional[int], shard_idx: Optional[int], ignore_complete: bool) -> None:
    if filter_idx is None == shard_idx is None:
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
    print(f"mem(files)={int(sys.getsizeof(files[0]) * len(files) // 1e6)}MB")

    analyze_samples(files)


def prepare(filter_mode: Optional[int], num_shards: Optional[int], ignore_complete: bool) -> None:
    if filter_mode is None == num_shards is None:
        raise ValueError("Must use filter or shard API, not both.")

    for f in P_PREP.iterdir():
        f.unlink()

    files = (s.sha256 for s in stream_sorel_meta() if s.is_malware)
    files = sorted(islice(files, None))
    print(f"{len(files)=}")
    print(f"mem(files)={int(sys.getsizeof(files[0]) * len(files) // 1e6)}MB")

    if ignore_complete:
        completed = infer_completed_samples()
        files = [f for f in files if f not in completed]

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
                        raise RuntimeError()
                    finish = i
                    break

            with open(filter_file, "w") as fp:
                for f in files[start:finish]:
                    fp.write(f"{f}\n")

            start = finish
            finish = len(files)
        return


def main():

    parser = ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--consolidate", action="store_true")
    parser.add_argument("--dont_ignore_complete", action="store_true")
    parser.add_argument("--filter_mode", type=int, default=None)
    parser.add_argument("--filter_idx", type=str, default=None)
    parser.add_argument("--num_shards", type=int, default=None)
    parser.add_argument("--shard_idx", type=int, default=None)
    args = parser.parse_args()

    P_ROOT.mkdir(exist_ok=True)
    P_PREP.mkdir(exist_ok=True)
    P_RAW.mkdir(exist_ok=True)
    for p in P_MODES.values():
        p.mkdir(exist_ok=True)
    P_MERGED.mkdir(exist_ok=True)

    t_0 = time.time()

    if args.prepare:
        prepare(args.filter_mode, args.num_shards, not args.dont_ignore_complete)

    if args.run:
        run(args.filter_idx, args.shard_idx, not args.dont_ignore_complete)

    if args.merge:
        merge()

    if args.consolidate:
        consolidate()

    print(f"Elapsed time: {time.time() - t_0:.2f} seconds")


if __name__ == "__main__":
    main()
